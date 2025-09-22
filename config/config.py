"""
Configuration Manager for Thai RAG Chatbot
จัดการการตั้งค่าต่างๆ ของระบบ
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# โหลด environment variables
load_dotenv()


class Config:
    """คลาสสำหรับจัดการ configuration"""
    
    # =============================================
    # Ollama Configuration
    # =============================================
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
    
    # =============================================
    # Model Configuration
    # =============================================
    DEFAULT_EMBEDDING_MODEL = os.getenv(
        "DEFAULT_EMBEDDING_MODEL", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemma2:2b")
    
    # Model Parameters
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
    
    # =============================================
    # File and Storage Configuration
    # =============================================
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vectorstore")
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
    
    # File size limit (50MB)
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "52428800"))
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = os.getenv(
        "SUPPORTED_EXTENSIONS", 
        ".txt,.pdf,.docx,.doc,.pptx,.ppt"
    ).split(",")
    
    # =============================================
    # Streamlit Configuration
    # =============================================
    STREAMLIT_PAGE_TITLE = os.getenv("STREAMLIT_PAGE_TITLE", "Thai RAG Chatbot")
    STREAMLIT_PAGE_ICON = os.getenv("STREAMLIT_PAGE_ICON", "🤖")
    STREAMLIT_LAYOUT = os.getenv("STREAMLIT_LAYOUT", "wide")
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "50"))
    
    # =============================================
    # Thai Language Processing
    # =============================================
    THAI_TOKENIZER = os.getenv("THAI_TOKENIZER", "newmm")
    THAI_ENCODINGS = os.getenv("THAI_ENCODINGS", "utf-8,cp874,iso-8859-11").split(",")
    
    # =============================================
    # Performance Configuration
    # =============================================
    DEFAULT_SEARCH_K = int(os.getenv("DEFAULT_SEARCH_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    MAX_CHAT_HISTORY_LENGTH = int(os.getenv("MAX_CHAT_HISTORY_LENGTH", "50"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    
    @classmethod
    def get_embedding_models(cls) -> Dict[str, str]:
        """รายการโมเดล embedding ที่รองรับ"""
        return {
            "fast": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "balanced": "sentence-transformers/distiluse-base-multilingual-cased",
            "best": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "thai_optimized": "sentence-transformers/LaBSE"
        }
    
    @classmethod
    def get_llm_models(cls) -> Dict[str, str]:
        """รายการโมเดล LLM ที่แนะนำ"""
        return {
            "small_fast": "llama3.1:8b",
            "balanced": "llama3.1:70b", 
            "large": "llama3.1:405b",
            "coding": "codellama:7b",
            "chat": "mistral:7b"
        }
    
    @classmethod
    def create_directories(cls) -> None:
        """สร้างโฟลเดอร์ที่จำเป็น"""
        dirs_to_create = [
            cls.VECTOR_STORE_PATH,
            cls.DATA_DIR,
            cls.TEMP_DIR
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_file(cls, file_path: str, file_size: int) -> tuple[bool, str]:
        """ตรวจสอบไฟล์ว่าถูกต้องหรือไม่"""
        file_ext = Path(file_path).suffix.lower()
        
        # ตรวจสอบขนาดไฟล์
        if file_size > cls.MAX_FILE_SIZE:
            return False, f"ไฟล์ใหญ่เกินไป (สูงสุด {cls.MAX_FILE_SIZE / 1024 / 1024:.1f} MB)"
        
        # ตรวจสอบประเภทไฟล์
        if file_ext not in cls.SUPPORTED_EXTENSIONS:
            return False, f"ไม่รองรับไฟล์ประเภท {file_ext}"
        
        return True, "ไฟล์ถูกต้อง"
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """สรุปการตั้งค่าปัจจุบัน"""
        return {
            "ollama": {
                "base_url": cls.OLLAMA_BASE_URL,
                "timeout": cls.OLLAMA_TIMEOUT
            },
            "models": {
                "default_embedding": cls.DEFAULT_EMBEDDING_MODEL,
                "default_llm": cls.DEFAULT_LLM_MODEL,
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS
            },
            "storage": {
                "vector_store": cls.VECTOR_STORE_PATH,
                "data_dir": cls.DATA_DIR,
                "temp_dir": cls.TEMP_DIR
            },
            "files": {
                "max_size_mb": cls.MAX_FILE_SIZE / 1024 / 1024,
                "supported_extensions": cls.SUPPORTED_EXTENSIONS
            },
            "processing": {
                "chunk_size": cls.DEFAULT_CHUNK_SIZE,
                "chunk_overlap": cls.DEFAULT_CHUNK_OVERLAP,
                "search_k": cls.DEFAULT_SEARCH_K,
                "batch_size": cls.BATCH_SIZE
            }
        }


# สร้างโฟลเดอร์ที่จำเป็นเมื่อ import
Config.create_directories()


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("🔧 การตั้งค่าระบบ Thai RAG Chatbot")
    print("=" * 50)
    
    config_summary = Config.get_config_summary()
    
    for section, settings in config_summary.items():
        print(f"\n📋 {section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print(f"\n📁 โฟลเดอร์ที่สร้าง:")
    print(f"  - {Config.VECTOR_STORE_PATH}")
    print(f"  - {Config.DATA_DIR}")
    print(f"  - {Config.TEMP_DIR}")
    
    print(f"\n🤖 โมเดลที่รองรับ:")
    
    print("  Embedding Models:")
    for name, model in Config.get_embedding_models().items():
        print(f"    - {name}: {model}")
    
    print("  LLM Models:")
    for name, model in Config.get_llm_models().items():
        print(f"    - {name}: {model}")
    
    # ทดสอบการตรวจสอบไฟล์
    print(f"\n✅ ทดสอบการตรวจสอบไฟล์:")
    test_files = [
        ("test.txt", 1000),
        ("test.pdf", 100000000),  # ไฟล์ใหญ่เกินไป
        ("test.xyz", 1000)        # ประเภทไฟล์ไม่รองรับ
    ]
    
    for file_path, file_size in test_files:
        is_valid, message = Config.validate_file(file_path, file_size)
        status = "✅" if is_valid else "❌"
        print(f"  {status} {file_path}: {message}")