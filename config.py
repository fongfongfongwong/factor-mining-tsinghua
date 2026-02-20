"""Central configuration for FactorMiner.

Uses a dataclass for structured config with env-var overrides.
Module-level constants are kept for backward compatibility.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class FactorMinerConfig:
    """Structured configuration for the entire system."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # LLM
    llm_provider: str = "kimi"
    llm_model: str = "kimi-k2.5"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.moonshot.cn/v1"

    # Mining (relaxed for large-scale factor generation)
    ic_threshold: float = 0.005
    corr_threshold: float = 0.85
    batch_size: int = 50
    max_rounds: int = 200
    target_library_size: int = 5000
    fast_screen_assets: int = 50
    full_universe_assets: int = 300

    # Data
    data_start: str = "2020-01-01"
    data_end: str = "2025-12-31"
    data_features: list[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "amount", "vwap", "returns",
    ])

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def storage_dir(self) -> Path:
        d = self.base_dir / "storage"
        d.mkdir(exist_ok=True)
        return d

    @property
    def factor_library_path(self) -> Path:
        return self.storage_dir / "factor_library.json"

    @property
    def experience_memory_path(self) -> Path:
        return self.storage_dir / "experience_memory.json"

    @property
    def data_cache_dir(self) -> Path:
        d = self.storage_dir / "data_cache"
        d.mkdir(exist_ok=True)
        return d

    @classmethod
    def from_env(cls) -> "FactorMinerConfig":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "kimi"),
            llm_model=os.getenv("KIMI_MODEL", "kimi-k2.5"),
            llm_api_key=os.getenv("KIMI_API_KEY", ""),
            llm_base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        )


_cfg = FactorMinerConfig.from_env()

# --- Backward-compatible module-level constants ---
BASE_DIR = _cfg.base_dir
STORAGE_DIR = _cfg.storage_dir
FACTOR_LIBRARY_PATH = _cfg.factor_library_path
EXPERIENCE_MEMORY_PATH = _cfg.experience_memory_path
DATA_CACHE_DIR = _cfg.data_cache_dir

LLM_PROVIDER = _cfg.llm_provider
LLM_MODEL = _cfg.llm_model
LLM_API_KEY = _cfg.llm_api_key
LLM_BASE_URL = _cfg.llm_base_url

IC_THRESHOLD = _cfg.ic_threshold
CORR_THRESHOLD = _cfg.corr_threshold
BATCH_SIZE = _cfg.batch_size
MAX_ROUNDS = _cfg.max_rounds
TARGET_LIBRARY_SIZE = _cfg.target_library_size
FAST_SCREEN_ASSETS = _cfg.fast_screen_assets
FULL_UNIVERSE_ASSETS = _cfg.full_universe_assets

DATA_START = _cfg.data_start
DATA_END = _cfg.data_end
DATA_FEATURES = _cfg.data_features

API_HOST = _cfg.api_host
API_PORT = _cfg.api_port
