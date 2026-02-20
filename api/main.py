"""FastAPI application: REST API + WebSocket for FactorMiner dashboard."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_HOST, API_PORT
from api.models import (
    MiningConfig,
    BacktestRequest,
    BacktestResult,
    MiningStatus,
    MemoryState,
)
from factor_mining.miner import FactorMiner
from factor_mining.factor_library import FactorLibrary
from factor_mining.expression_engine import ExpressionEngine
from factor_mining.experience_memory import ExperienceMemory
from data.stock_data import build_market_panel, calculate_returns, get_stock_list
from backtest.engine import run_factor_backtest, run_library_backtest, run_single_factor_analysis
from backtest.metrics import calc_ic_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
MAX_WS_CONNECTIONS = 10
API_KEY = os.getenv("FACTORMINER_API_KEY", "")


async def verify_api_key(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class AppState:
    def __init__(self):
        self.miner: Optional[FactorMiner] = None
        self.mining_task: Optional[asyncio.Task] = None
        self.ws_connections: list[WebSocket] = []

    async def broadcast_log(self, log_entry: dict):
        message = json.dumps(log_entry, ensure_ascii=False, default=str)
        disconnected = []
        for ws in self.ws_connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.ws_connections.remove(ws)


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FactorMiner API starting up")
    yield
    if _state.miner and _state.miner.is_running:
        _state.miner.stop()
    logger.info("FactorMiner API shutting down")


app = FastAPI(title="FactorMiner", version="1.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/api/mining/start", dependencies=[Depends(verify_api_key)])
async def start_mining(config: MiningConfig = MiningConfig()):
    if _state.miner and _state.miner.is_running:
        return {"error": "Mining is already running"}

    _state.miner = FactorMiner(
        batch_size=config.batch_size,
        max_rounds=config.max_rounds,
        ic_threshold=config.ic_threshold,
        corr_threshold=config.corr_threshold,
        target_size=config.target_size,
        stock_universe=config.stock_universe,
    )
    _state.miner.set_log_callback(_state.broadcast_log)
    _state.mining_task = asyncio.create_task(_state.miner.run())
    return {"status": "started", "config": config.model_dump()}


@app.post("/api/mining/stop", dependencies=[Depends(verify_api_key)])
async def stop_mining():
    if _state.miner and _state.miner.is_running:
        _state.miner.stop()
        return {"status": "stopping"}
    return {"status": "not_running"}


@app.get("/api/mining/status", response_model=MiningStatus)
async def mining_status():
    if _state.miner:
        return MiningStatus(**_state.miner.get_status())
    return MiningStatus(
        status="idle", current_round=0, max_rounds=0,
        library_size=FactorLibrary().size, target_size=0,
        is_running=False, admission_rate=0.0,
    )


@app.get("/api/factors")
async def get_factors():
    library = _state.miner.library if _state.miner else FactorLibrary()
    return {"factors": library.get_all_records(), "total": library.size}


@app.get("/api/factors/{factor_id}/detail")
async def get_factor_detail(factor_id: str):
    library = _state.miner.library if _state.miner else FactorLibrary()
    record = library.get_record_by_id(factor_id)
    if not record:
        return {"error": "Factor not found"}
    result = {"factor": record.to_dict()}
    try:
        codes = get_stock_list()
        panel = build_market_panel(codes, max_stocks=100)
        fwd = calculate_returns(panel)
        analysis = run_single_factor_analysis(record.expression, panel, fwd)
        result["analysis"] = analysis
    except Exception as e:
        logger.exception("Factor detail error")
        result["analysis"] = {"error": "Computation failed"}
    return result


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    library = _state.miner.library if _state.miner else FactorLibrary()
    if library.size == 0:
        return {"error": "Factor library is empty"}
    try:
        codes = get_stock_list()
        panel = build_market_panel(codes, max_stocks=100)
        fwd = calculate_returns(panel)
        engine = ExpressionEngine(panel)

        if request.factor_id:
            record = library.get_record_by_id(request.factor_id)
            if not record:
                return {"error": "Factor not found"}
            signal = engine.evaluate(record.expression)
            bt = run_factor_backtest(signal, fwd)
            ic = calc_ic_series(signal, fwd)
            valid_ic = ic[~np.isnan(ic)]
            return BacktestResult(
                expression=record.expression,
                ls_returns=bt["ls_returns"],
                cumulative_returns=bt["cumulative_returns"],
                sharpe=bt["sharpe"],
                max_drawdown=bt["max_drawdown"],
                annual_return=bt["annual_return"],
                win_ratio=bt["win_ratio"],
                turnover=bt["turnover"],
                ic_series=ic.tolist(),
                ic_mean=float(valid_ic.mean()) if len(valid_ic) > 0 else 0.0,
                icir=float(valid_ic.mean() / (valid_ic.std() + 1e-10)) if len(valid_ic) > 0 else 0.0,
                group_cumulative_returns=bt.get("group_cumulative_returns", {}),
            ).model_dump()
        else:
            signals = {}
            for fr in library.factors:
                try:
                    signals[fr.expression] = engine.evaluate(fr.expression)
                except Exception:
                    continue
            if not signals:
                return {"error": "No valid signals could be computed"}
            bt = run_library_backtest(signals, fwd, method=request.method)
            return BacktestResult(
                expression=f"Library ({len(signals)} factors, {request.method})",
                cumulative_returns=bt.get("cumulative_returns", []),
                sharpe=bt.get("sharpe", 0),
                max_drawdown=bt.get("max_drawdown", 0),
                annual_return=bt.get("annual_return", 0),
                win_ratio=bt.get("win_ratio", 0),
                turnover=bt.get("turnover", 0),
                group_cumulative_returns=bt.get("group_cumulative_returns", {}),
            ).model_dump()
    except Exception as e:
        logger.exception("Backtest error")
        return {"error": "Backtest computation failed"}


@app.get("/api/memory", response_model=MemoryState)
async def get_memory():
    memory = _state.miner.memory if _state.miner else ExperienceMemory()
    return MemoryState(**memory.get_state())


@app.websocket("/ws/mining")
async def websocket_mining(ws: WebSocket):
    if len(_state.ws_connections) >= MAX_WS_CONNECTIONS:
        await ws.close(code=1008, reason="Too many connections")
        return
    await ws.accept()
    _state.ws_connections.append(ws)
    logger.info(f"WebSocket connected ({len(_state.ws_connections)} total)")
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        if ws in _state.ws_connections:
            _state.ws_connections.remove(ws)
        logger.info(f"WebSocket disconnected ({len(_state.ws_connections)} remaining)")


if __name__ == "__main__":
    import uvicorn
    reload = os.getenv("ENV") == "development"
    uvicorn.run("api.main:app", host="127.0.0.1", port=API_PORT, reload=reload)
