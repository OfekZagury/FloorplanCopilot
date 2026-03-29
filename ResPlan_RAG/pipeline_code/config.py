import os

# ==========================================
# Database & File Paths
# ==========================================
# The path to the downloaded ResPlan dataset
RESPLAN_DATA_PATH = 'ResPlan.pkl'

# The path where the SQLite database will be saved
SQLITE_DB_PATH = 'apartments_raw.db'

# The path where the ChromaDB vector database will be saved
CHROMA_DB_PATH = './chroma_db'

# ==========================================
# API Keys
# ==========================================
# Gemini API Key for Embeddings and LLM generation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# ==========================================
# Application Settings
# ==========================================
# Collection name for ChromaDB
CHROMA_COLLECTION_NAME = 'floorplans'