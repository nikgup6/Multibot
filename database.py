import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from uuid import uuid4
from auth import get_password_hash


class DatabaseManager:
    def __init__(self):
        self.data_dir = "data"
        self.users_file = os.path.join(self.data_dir, "users.csv")
        self.documents_file = os.path.join(self.data_dir, "documents.csv")
        self.chat_history_file = os.path.join(self.data_dir, "chat_history.csv")

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize CSV files
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""

        # Users CSV
        if not os.path.exists(self.users_file):
            with open(self.users_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "id",
                        "full_name",
                        "email",
                        "password_hash",
                        "created_at",
                        "last_login",
                        "is_active",
                    ]
                )

        # Documents CSV
        if not os.path.exists(self.documents_file):
            with open(self.documents_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "id",
                        "user_id",
                        "filename",
                        "file_path",
                        "file_size",
                        "upload_date",
                        "is_processed",
                    ]
                )

        # Chat History CSV
        if not os.path.exists(self.chat_history_file):
            with open(self.chat_history_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "id",
                        "user_id",
                        "document_id",
                        "query",
                        "response",
                        "language",
                        "timestamp",
                    ]
                )

    # User Management
    def create_user(self, user_data) -> str:
        """Create a new user"""
        user_id = str(uuid4())
        password_hash = get_password_hash(user_data.password)

        with open(self.users_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    user_id,
                    user_data.fullName,
                    user_data.email,
                    password_hash,
                    datetime.now().isoformat(),
                    "",
                    "true",
                ]
            )

        return user_id

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        try:
            with open(self.users_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["email"] == email:
                        return row
        except FileNotFoundError:
            pass
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        try:
            with open(self.users_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["id"] == user_id:
                        return row
        except FileNotFoundError:
            pass
        return None

    def update_user_login(self, email: str):
        """Update user's last login time"""
        users = []

        # Read all users
        try:
            with open(self.users_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                users = list(reader)
        except FileNotFoundError:
            return

        # Update the specific user
        for user in users:
            if user["email"] == email:
                user["last_login"] = datetime.now().isoformat()
                break

        # Write back to file
        with open(self.users_file, "w", newline="", encoding="utf-8") as f:
            if users:
                writer = csv.DictWriter(f, fieldnames=users[0].keys())
                writer.writeheader()
                writer.writerows(users)

    # Document Management
    def create_document(self, doc_data: Dict) -> str:
        """Create a new document record"""
        with open(self.documents_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    doc_data["id"],
                    doc_data["user_id"],
                    doc_data["filename"],
                    doc_data["file_path"],
                    doc_data["file_size"],
                    doc_data["upload_date"],
                    "true",
                ]
            )

        return doc_data["id"]

    def get_user_documents(self, user_id: str) -> List[Dict]:
        """Get all documents for a user"""
        documents = []

        try:
            with open(self.documents_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["user_id"] == user_id:
                        documents.append(row)
        except FileNotFoundError:
            pass

        return documents

    def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """Get document by ID"""
        try:
            with open(self.documents_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["id"] == document_id:
                        return row
        except FileNotFoundError:
            pass
        return None

    def delete_document(self, document_id: str):
        """Delete a document record"""
        documents = []

        # Read all documents except the one to delete
        try:
            with open(self.documents_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                documents = [row for row in reader if row["id"] != document_id]
        except FileNotFoundError:
            return

        # Write back to file
        with open(self.documents_file, "w", newline="", encoding="utf-8") as f:
            if documents:
                writer = csv.DictWriter(f, fieldnames=documents[0].keys())
                writer.writeheader()
                writer.writerows(documents)

    # Chat History Management
    def save_chat_history(self, chat_data: Dict):
        """Save chat interaction"""
        chat_id = str(uuid4())

        with open(self.chat_history_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    chat_id,
                    chat_data["user_id"],
                    chat_data["document_id"],
                    chat_data["query"],
                    chat_data["response"],
                    chat_data["language"],
                    chat_data["timestamp"],
                ]
            )

    def get_chat_history(self, user_id: str, document_id: str) -> List[Dict]:
        """Get chat history for a specific document"""
        history = []

        try:
            with open(self.chat_history_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["user_id"] == user_id and row["document_id"] == document_id:
                        history.append(
                            {
                                "query": row["query"],
                                "response": row["response"],
                                "timestamp": row["timestamp"],
                                "language": row["language"],
                            }
                        )
        except FileNotFoundError:
            pass

        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        return history

    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        # Count documents
        documents = self.get_user_documents(user_id)
        doc_count = len(documents)

        # Count chat interactions
        chat_count = 0
        try:
            with open(self.chat_history_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["user_id"] == user_id:
                        chat_count += 1
        except FileNotFoundError:
            pass

        return {
            "document_count": doc_count,
            "chat_count": chat_count,
            "total_file_size": sum(int(doc["file_size"]) for doc in documents),
        }

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB"]
        import math

        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
