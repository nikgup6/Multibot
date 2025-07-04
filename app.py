from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import os
from uuid import uuid4
import langid
import ollama
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from typing import List
import requests
from datetime import datetime

# Import our custom modules
from auth import get_current_user, authenticate_user, create_access_token
from database import DatabaseManager
from models import *
from utils import generate_audio_response
import bcrypt

print(hasattr(bcrypt, "__about__"))
# Initialize FastAPI app
app = FastAPI(title="DocChat AI Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database manager
db_manager = DatabaseManager()

# Define directories
persist_directory = "./storage"
uploaded_pdfs_directory = "uploaded_pdfs"
static_directory = "static"
data_directory = "data"

# Create directories
for directory in [
    uploaded_pdfs_directory,
    persist_directory,
    static_directory,
    data_directory,
]:
    os.makedirs(directory, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


# Enhanced Ollama Embeddings Class
class EnhancedOllamaEmbeddings:
    def __init__(
        self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                embedding = self._embed(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error embedding document: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 384)  # Adjust dimension as needed
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._embed(text)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return [0.0] * 384  # Fallback zero vector

    def _embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]


# Initialize embeddings
ef = EnhancedOllamaEmbeddings(base_url="http://localhost:11434", model="llama3.2:3b")


# Enhanced RAG Pipeline
class EnhancedRAGPipeline:
    def __init__(self, embeddings_function):
        self.embeddings_function = embeddings_function
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Increased chunk size
            chunk_overlap=50,  # Increased overlap
            separators=["\n\n", "\n", ". ", " "],  # Better separators
        )

    def process_document(self, pdf_path: str, document_id: str) -> Chroma:
        """Process a PDF document and create vector store"""
        try:
            # Load PDF
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            if not documents:
                raise ValueError("No content found in PDF")

            # Split documents
            docs = self.text_splitter.split_documents(documents)

            if not docs:
                raise ValueError("No chunks created from document")

            # Create vector store
            persist_path = os.path.join(persist_directory, f"doc_{document_id}")
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings_function,
                persist_directory=persist_path,
            )

            return vectorstore

        except Exception as e:
            print(f"Error processing document: {e}")
            raise

    def get_vectorstore(self, document_id: str) -> Chroma:
        """Load existing vector store"""
        persist_path = os.path.join(persist_directory, f"doc_{document_id}")
        if not os.path.exists(persist_path):
            raise FileNotFoundError(
                f"Vector store not found for document {document_id}"
            )

        return Chroma(
            persist_directory=persist_path, embedding_function=self.embeddings_function
        )

    def generate_response(
        self, query: str, document_id: str, language: str = "en"
    ) -> str:
        """Generate response using RAG pipeline"""
        try:
            # Get vector store
            vectorstore = self.get_vectorstore(document_id)

            # Create retrieval chain
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},  # Retrieve top 5 chunks
            )

            # Enhanced prompt template
            PROMPT_TEMPLATE = """You are an intelligent document assistant. Answer the question based on the provided context.

Context:
{context}

Question: {input}

Instructions:
- Provide accurate, detailed answers based solely on the context
- If the context doesn't contain enough information, clearly state this
- Use a helpful and professional tone
- Structure your response clearly with relevant details
- If asked for a summary, provide key points in a logical order

Answer:"""

            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

            # Initialize language model
            llm = ChatOllama(
                model="llama3.2:3b",
                temperature=0.1,  # Lower temperature for more focused responses
                base_url="http://localhost:11434",
            )

            # Create chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Generate response
            response = retrieval_chain.invoke({"input": query})

            return response.get("answer", "Sorry, I could not generate a response.")

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while processing your question. Please try again."


# Initialize RAG pipeline
rag_pipeline = EnhancedRAGPipeline(ef)

# Routes


@app.get("/")
def index(request: Request):
    """Serve main page"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/login")
def login_page(request: Request):
    """Serve login page"""
    return templates.TemplateResponse("auth_page.html", {"request": request})


# Authentication Routes
@app.post("/api/auth/register")
async def register(user_data: UserCreate):
    """Register new user"""
    try:
        # Check if user already exists
        if db_manager.get_user_by_email(user_data.email):
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user
        user_id = db_manager.create_user(user_data)
        user = db_manager.get_user_by_id(user_id)

        return {"message": "User created successfully", "user": user}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/auth/login")
async def login(login_data: UserLogin):
    """Login user"""
    try:
        user = authenticate_user(login_data.email, login_data.password, db_manager)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token = create_access_token(data={"sub": user["email"]})

        return {
            "token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "fullName": user["full_name"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


# Document Routes
@app.post("/api/documents/upload")
async def upload_pdf(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    """Upload and process PDF document"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Generate document ID
        document_id = str(uuid4())

        # Save PDF file
        pdf_path = os.path.join(uploaded_pdfs_directory, f"{document_id}.pdf")
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process document with RAG pipeline
        vectorstore = rag_pipeline.process_document(pdf_path, document_id)

        # Save document metadata
        db_manager.create_document(
            {
                "id": document_id,
                "user_id": current_user["id"],
                "filename": file.filename,
                "file_path": pdf_path,
                "file_size": len(content),
                "upload_date": datetime.now().isoformat(),
            }
        )

        return {
            "message": "PDF uploaded and processed successfully",
            "document_id": document_id,
        }

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {str(e)}")


@app.get("/api/documents/list")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List user's documents"""
    try:
        documents = db_manager.get_user_documents(current_user["id"])

        # Format documents for frontend
        formatted_docs = []
        for doc in documents:
            formatted_docs.append(
                {
                    "id": doc["id"],
                    "name": doc["filename"],
                    "size": db_manager._format_file_size(int(doc["file_size"])),
                    "uploadedAt": doc["upload_date"],
                }
            )

        return formatted_docs

    except Exception as e:
        print(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: str, current_user: dict = Depends(get_current_user)
):
    """Delete a document"""
    try:
        # Check if document belongs to user
        document = db_manager.get_document_by_id(document_id)
        if not document or document["user_id"] != current_user["id"]:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete physical files
        pdf_path = document["file_path"]
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        # Delete vector store
        vector_path = os.path.join(persist_directory, f"doc_{document_id}")
        if os.path.exists(vector_path):
            import shutil

            shutil.rmtree(vector_path)

        # Delete from database
        db_manager.delete_document(document_id)

        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


# Chat Routes
@app.post("/api/chat/ask")
async def ask_question(
    chat_request: ChatRequest, current_user: dict = Depends(get_current_user)
):
    """Ask question about document"""
    try:
        # Verify document ownership
        document = db_manager.get_document_by_id(chat_request.document_id)
        if not document or document["user_id"] != current_user["id"]:
            raise HTTPException(status_code=404, detail="Document not found")

        # Generate response using enhanced RAG pipeline
        response_text = rag_pipeline.generate_response(
            chat_request.query, chat_request.document_id, chat_request.language
        )

        # Generate audio response
        audio_url = None
        try:
            audio_filename = generate_audio_response(
                response_text, chat_request.language
            )
            audio_url = f"/static/{audio_filename}"
        except Exception as audio_error:
            print(f"Audio generation error: {audio_error}")

        # Save chat history
        db_manager.save_chat_history(
            {
                "user_id": current_user["id"],
                "document_id": chat_request.document_id,
                "query": chat_request.query,
                "response": response_text,
                "language": chat_request.language,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return {"response": response_text, "audio_url": audio_url}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process question")


@app.get("/api/chat/history/{document_id}")
async def get_chat_history(
    document_id: str, current_user: dict = Depends(get_current_user)
):
    """Get chat history for a document"""
    try:
        # Verify document ownership
        document = db_manager.get_document_by_id(document_id)
        if not document or document["user_id"] != current_user["id"]:
            raise HTTPException(status_code=404, detail="Document not found")

        history = db_manager.get_chat_history(current_user["id"], document_id)
        return history

    except HTTPException:
        raise
    except Exception as e:
        print(f"History error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat history")


# Legacy routes (for backward compatibility)
@app.post("/upload-pdf")
async def legacy_upload_pdf(file: UploadFile = File(...)):
    """Legacy upload endpoint"""
    try:
        document_id = "CPA2019"  # Fixed ID for legacy support
        pdf_path = os.path.join(uploaded_pdfs_directory, "CPA2019.pdf")

        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        return {"message": "PDF uploaded and processed successfully."}
    except Exception as e:
        print(f"Error during PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {e}")


@app.post("/ask")
async def legacy_ask(request: Request):
    """Legacy ask endpoint"""
    try:
        data = await request.json()
        user_input = data["query"]

        # Use legacy document ID
        document_id = "CPA2019"

        # Process with RAG pipeline
        response_text = rag_pipeline.generate_response(user_input, document_id)

        return JSONResponse(content={"response": response_text})

    except Exception as e:
        print(f"Legacy ask error: {e}")
        return JSONResponse(
            content={"error": "Failed to process question"}, status_code=500
        )


@app.get("/list-col")
def list_collections():
    """List ChromaDB collections"""
    try:
        client = chromadb.PersistentClient(path="./storage")
        collections = client.list_collections()
        collection_names = [collection.name for collection in collections]
        return collection_names
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
