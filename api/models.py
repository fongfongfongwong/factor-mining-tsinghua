"""Pydantic models for API request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class MiningConfig(BaseModel):
    batch_size: int = Field(default=10, ge=1, le=50, description="Candidates per round")
    max_rounds: int = Field(default=50, ge=1, le=500, description="Maximum mining iterations")
    ic_threshold: float = Field(default=0.02, ge=0.001, le=0.1, description="Minimum IC for admission")
    corr_threshold: float = Field(default=0.7, ge=0.3, le=0.95, description="Maximum correlation budget")
    target_size: int = Field(default=100, ge=5, le=1000, description="Target library size")
    stock_universe: str = Field(default="000300", description="Index code: 000300 or 000905")


class FactorRecord(BaseModel):
    factor_id: str
    expression: str
    ic_mean: float
    ic_std: float
    icir: float
    max_correlation: float
    turnover: float
    logic_description: str
    admitted_at: float
    mining_round: int


class BacktestRequest(BaseModel):
    factor_id: Optional[str] = Field(default=None, description="Single factor ID, or None for full library")
    method: str = Field(default="ic_weighted", description="Combination method: equal, ic_weighted, icir_weighted")


class BacktestResult(BaseModel):
    expression: Optional[str] = None
    ls_returns: list[float] = []
    cumulative_returns: list[float] = []
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    annual_return: float = 0.0
    win_ratio: float = 0.0
    turnover: float = 0.0
    ic_series: list[float] = []
    ic_mean: float = 0.0
    icir: float = 0.0
    group_cumulative_returns: dict = {}


class MiningStatus(BaseModel):
    status: str
    current_round: int
    max_rounds: int
    library_size: int
    target_size: int
    is_running: bool
    admission_rate: float


class MemoryState(BaseModel):
    mining_state: dict = {}
    recommended_directions: list = []
    forbidden_directions: list = []
    strategic_insights: list = []
    recent_rejections: list = []
