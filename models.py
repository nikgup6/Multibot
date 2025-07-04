from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class UserCreate(BaseModel):
    fullName: str = Field(
        ..., min_length=2, max_length=100, description="Full name of the user"
    )
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(
        ...,
        min_length=6,
        max_length=128,
        description="Password must be at least 6 characters",
    )

    @validator("fullName")
    def validate_full_name(cls, v):
        if not v.strip():
            raise ValueError("Full name cannot be empty")
        return v.strip()

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters long")
        return v


class UserLogin(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    id: str
    fullName: str
    email: str
    created_at: str
    last_login: Optional[str] = None
    is_active: bool = True

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    fullName: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None

    @validator("fullName")
    def validate_full_name(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Full name cannot be empty")
        return v.strip() if v else v


class DocumentUpload(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    file_size: int = Field(..., gt=0, description="File size in bytes")


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_size: str
    upload_date: str
    is_processed: bool = True
    user_id: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    id: str
    name: str
    size: str
    uploadedAt: str
    isProcessed: bool = True


class ChatRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=2000, description="User question or query"
    )
    document_id: str = Field(..., description="ID of the document to query")
    language: str = Field(default="en", description="Response language code")

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator("language")
    def validate_language(cls, v):
        supported_languages = [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
            "ar",
            "hi",
        ]
        if v not in supported_languages:
            return "en"  # Default to English if unsupported
        return v


class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    language: str = "en"


class ChatHistory(BaseModel):
    query: str
    response: str
    timestamp: str
    language: str = "en"


class ChatHistoryResponse(BaseModel):
    history: List[ChatHistory]
    total_count: int


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds
    user: Dict[str, Any]


class APIResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error: str
    status_code: int


class LanguageSupport(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"
    ARABIC = "ar"
    HINDI = "hi"


class AudioRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="en")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class AudioResponse(BaseModel):
    audio_url: str
    filename: str
    duration: Optional[float] = None


class UserStats(BaseModel):
    document_count: int = 0
    chat_count: int = 0
    total_file_size: int = 0
    formatted_file_size: str = "0 B"
    last_activity: Optional[str] = None


class DocumentStats(BaseModel):
    total_documents: int = 0
    total_size: int = 0
    formatted_total_size: str = "0 B"
    recent_uploads: List[DocumentResponse] = []


class SystemHealth(BaseModel):
    status: str = "healthy"
    ollama_status: bool = True
    database_status: bool = True
    storage_available: str = "0 B"
    uptime: str = "0s"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    document_ids: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=50)

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    document_id: str
    document_name: str
    relevance_score: float
    snippet: str
    page_number: Optional[int] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    query: str
    processing_time: float


class BulkDeleteRequest(BaseModel):
    document_ids: List[str] = Field(..., min_items=1)

    @validator("document_ids")
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("At least one document ID is required")
        return list(set(v))  # Remove duplicates


class ExportRequest(BaseModel):
    document_id: str
    format: str = Field(default="json", pattern="^(json|csv|txt)$")
    include_chat_history: bool = True


class ExportResponse(BaseModel):
    download_url: str
    filename: str
    format: str
    expires_at: str


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=128)

    @validator("new_password")
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError("New password must be at least 6 characters long")
        return v


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=128)

    @validator("new_password")
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError("New password must be at least 6 characters long")
        return v


class ConfigUpdate(BaseModel):
    max_file_size: Optional[int] = Field(None, gt=0)
    supported_formats: Optional[List[str]] = None
    max_chat_history: Optional[int] = Field(None, gt=0)
    session_timeout: Optional[int] = Field(None, gt=0)


class HealthCheck(BaseModel):
    status: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    services: Dict[str, bool]
    version: str = "1.0.0"
