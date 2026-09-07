import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
CHAT_MODEL_NAME = "openai/gpt-3.5-turbo"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"

DATA_DIR = PROJECT_ROOT / "data"

CORE_KNOWLEDGE_DIR = str(DATA_DIR / "core_knowledge")
USER_MEMORY_DIR = str(DATA_DIR / "user_memory")
USER_INFO_PATH = str(DATA_DIR / "user_info.json")
MEMORY_DECISION_LOG_PATH = str(DATA_DIR / "memory_decision_log.jsonl")

CORE_KNOWLEDGE_COLLECTION = "core_knowledge"
USER_MEMORY_COLLECTION = "user_memory"
