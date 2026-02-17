"""Central configuration for FactorMiner."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)
FACTOR_LIBRARY_PATH = STORAGE_DIR / "factor_library.json"
EXPERIENCE_MEMORY_PATH = STORAGE_DIR / "experience_memory.json"
DATA_CACHE_DIR = STORAGE_DIR / "data_cache"
DATA_CACHE_DIR.mkdir(exist_ok=True)

# --- LLM (Kimi 2.5 via OpenAI-compatible API) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "kimi")
LLM_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")
LLM_API_KEY = os.getenv("KIMI_API_KEY", "")
LLM_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")

# --- Mining Parameters ---
IC_THRESHOLD = 0.02
CORR_THRESHOLD = 0.7
BATCH_SIZE = 10
MAX_ROUNDS = 50
TARGET_LIBRARY_SIZE = 100
FAST_SCREEN_ASSETS = 50
FULL_UNIVERSE_ASSETS = 300

# --- Data ---
DATA_START = "2020-01-01"
DATA_END = "2025-12-31"
DATA_FEATURES = ["open", "high", "low", "close", "volume", "amount", "vwap", "returns"]

# --- Server ---
API_HOST = "0.0.0.0"
API_PORT = 8000
