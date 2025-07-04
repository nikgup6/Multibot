import os
import math
import shutil
import hashlib
import tempfile
import mimetypes
from uuid import uuid4
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import langid
from gtts import gTTS
from deep_translator import GoogleTranslator
import whisper
import requests
from fastapi import HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUtils:
    """Utility class for file operations"""

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {e}")
            return ""

    @staticmethod
    def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
        """Validate if file type is allowed"""
        if allowed_types is None:
            allowed_types = [".pdf"]

        file_ext = Path(filename).suffix.lower()
        return file_ext in allowed_types

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Clean filename by removing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename

    @staticmethod
    def ensure_directory(directory: str) -> None:
        """Ensure directory exists, create if it doesn't"""
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_available_disk_space(path: str) -> int:
        """Get available disk space in bytes"""
        try:
            statvfs = os.statvfs(path)
            return statvfs.f_frsize * statvfs.f_bavail
        except Exception:
            return 0

    @staticmethod
    def cleanup_old_files(directory: str, max_age_days: int = 7) -> int:
        """Clean up files older than specified days"""
        if not os.path.exists(directory):
            return 0

        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0

        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")

        return cleaned_count


class AudioUtils:
    """Utility class for audio operations"""

    @staticmethod
    def generate_audio_response(
        text: str, language: str = "en", slow: bool = False
    ) -> str:
        """Generate audio response using Google Text-to-Speech"""
        try:
            # Clean text for TTS
            clean_text = AudioUtils._clean_text_for_tts(text)

            if not clean_text.strip():
                raise ValueError("No text to convert to speech")

            # Create TTS object
            tts = gTTS(text=clean_text, lang=language, slow=slow)

            # Generate unique filename
            audio_filename = (
                f"response_{uuid4().hex[:8]}_{int(datetime.now().timestamp())}.mp3"
            )
            audio_path = os.path.join("static", audio_filename)

            # Ensure static directory exists
            FileUtils.ensure_directory("static")

            # Save audio file
            tts.save(audio_path)

            logger.info(f"Audio response generated: {audio_filename}")
            return audio_filename

        except Exception as e:
            logger.error(f"Error generating audio response: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate audio: {str(e)}"
            )

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """Clean text for better TTS output"""
        # Remove or replace problematic characters
        replacements = {
            "&": " and ",
            "@": " at ",
            "#": " hash ",
            "%": " percent ",
            "*": " ",
            "_": " ",
            "|": " ",
            "\\": " ",
            "/": " ",
            "[": " ",
            "]": " ",
            "{": " ",
            "}": " ",
            "<": " ",
            ">": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove multiple spaces
        import re

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def get_audio_duration(file_path: str) -> Optional[float]:
        """Get audio file duration in seconds"""
        try:
            import mutagen

            audio_file = mutagen.File(file_path)
            if audio_file is not None:
                return audio_file.info.length
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
        return None

    @staticmethod
    def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
        """Transcribe audio file using Whisper"""
        try:
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""


class LanguageUtils:
    """Utility class for language operations"""

    SUPPORTED_LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ar": "Arabic",
        "hi": "Hindi",
        "nl": "Dutch",
        "pl": "Polish",
        "tr": "Turkish",
    }

    @staticmethod
    def detect_language(text: str) -> Tuple[str, float]:
        """Detect language of text using langid"""
        try:
            lang, confidence = langid.classify(text)
            return lang, confidence
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en", 0.0

    @staticmethod
    def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
        """Translate text using Google Translator"""
        try:
            if source_lang == target_lang:
                return text

            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return text

    @staticmethod
    def is_supported_language(lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in LanguageUtils.SUPPORTED_LANGUAGES

    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Get language name from code"""
        return LanguageUtils.SUPPORTED_LANGUAGES.get(lang_code, lang_code)


class ValidationUtils:
    """Utility class for validation operations"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {"is_valid": True, "score": 0, "issues": []}

        if len(password) < 6:
            result["issues"].append("Password must be at least 6 characters long")
            result["is_valid"] = False
        else:
            result["score"] += 1

        if len(password) >= 8:
            result["score"] += 1

        if any(c.isupper() for c in password):
            result["score"] += 1
        else:
            result["issues"].append("Password should contain uppercase letters")

        if any(c.islower() for c in password):
            result["score"] += 1
        else:
            result["issues"].append("Password should contain lowercase letters")

        if any(c.isdigit() for c in password):
            result["score"] += 1
        else:
            result["issues"].append("Password should contain numbers")

        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            result["score"] += 1
        else:
            result["issues"].append("Password should contain special characters")

        return result

    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if not text:
            return ""

        # Remove HTML tags
        import re

        text = re.sub(r"<[^>]+>", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]

        return text.strip()


class SystemUtils:
    """Utility class for system operations"""

    @staticmethod
    def check_ollama_status(base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get basic system information"""
        import psutil

        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "load_average": os.getloadavg()
                if hasattr(os, "getloadavg")
                else [0, 0, 0],
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}

    @staticmethod
    def create_backup(source_dir: str, backup_dir: str) -> str:
        """Create a backup of the source directory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)

            FileUtils.ensure_directory(backup_dir)
            shutil.copytree(source_dir, backup_path)

            logger.info(f"Backup created at: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise


class DateTimeUtils:
    """Utility class for date and time operations"""

    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime to string"""
        try:
            return dt.strftime(format_str)
        except Exception:
            return str(dt)

    @staticmethod
    def parse_datetime(dt_str: str) -> Optional[datetime]:
        """Parse datetime string"""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def get_time_ago(dt: datetime) -> str:
        """Get human readable time ago string"""
        now = datetime.now()
        diff = now - dt

        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"


class CacheUtils:
    """Simple in-memory cache utility"""

    _cache = {}
    _cache_timestamps = {}

    @classmethod
    def set(cls, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cache value with TTL in seconds"""
        cls._cache[key] = value
        cls._cache_timestamps[key] = datetime.now() + timedelta(seconds=ttl)

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """Get cache value"""
        if key not in cls._cache:
            return None

        if datetime.now() > cls._cache_timestamps[key]:
            cls.delete(key)
            return None

        return cls._cache[key]

    @classmethod
    def delete(cls, key: str) -> None:
        """Delete cache entry"""
        cls._cache.pop(key, None)
        cls._cache_timestamps.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all cache"""
        cls._cache.clear()
        cls._cache_timestamps.clear()

    @classmethod
    def cleanup_expired(cls) -> int:
        """Clean up expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, timestamp in cls._cache_timestamps.items() if now > timestamp
        ]

        for key in expired_keys:
            cls.delete(key)

        return len(expired_keys)


# Convenience functions for backward compatibility
def generate_audio_response(text: str, language: str = "en") -> str:
    """Generate audio response - backward compatibility function"""
    return AudioUtils.generate_audio_response(text, language)


def format_file_size(size_bytes: int) -> str:
    """Format file size - backward compatibility function"""
    return FileUtils.format_file_size(size_bytes)


def detect_language(text: str) -> Tuple[str, float]:
    """Detect language - backward compatibility function"""
    return LanguageUtils.detect_language(text)


def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """Translate text - backward compatibility function"""
    return LanguageUtils.translate_text(text, target_lang, source_lang)
