import os

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
MODEL_CONFIG = {
    "main_model": "llama-3.3-70b-versatile",
    "classifier_model": "llama-3.3-70b-versatile",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
}

# Directory Configuration
DIRECTORIES = {
    "hr_docs": "./hr_documents",
    "vector_store": "./hr_chroma_db",
    "model_cache": "./model_cache",
    "prompts": "./src/prompts",
    "data": "./src/data"
}

# Token Configuration
TOKEN_CONFIG = {
    "max_tokens_per_minute": 6000,
    "warning_threshold": 5000,
    "reset_interval": 60  # seconds
}

# Memory Configuration
MEMORY_CONFIG = {
    "window_size": 4
}

# Document Processing Configuration
DOC_PROCESSING = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "supported_extensions": ['.pdf', '.docx', '.doc', '.txt']
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Mati Carbon HR AI Assistant",
    "page_icon": "👩‍💼",
    "layout": "wide"
} 