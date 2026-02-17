"""FastAPI application: REST API + WebSocket for FactorMiner dashboard."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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
from factor_mining.factor_library import FactorLibrary, compute_ic, compute_icir
from factor_mining.expression_engine import ExpressionEngine
from factor_mining.experience_memory import ExperienceMemory
from data.stock_data import build_market_panel, calculate_returns, get_stock_list
from backtest.engine import run_factor_backtest, run_library_backtest, run_single_factor_analysis
from backtest.metrics import calc_ic_series, factor_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

miner: Optional[FactorMiner] = None
mining_task: Optional[asyncio.Task] = None
ws_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FactorMiner API starting up")
    yield
    if miner and miner.is_running:
        miner.stop()
    logger.info("FactorMiner API shutting down")


app = FastAPI(title="FactorMiner", version="1.0.0", lifespan=lifespan)


async def broadcast_log(log_entry: dict):
    """Send mining log to all connected WebSocket clients."""
    message = json.dumps(log_entry, ensure_ascii=False, default=str)
    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_connections.remove(ws)


# --- Dashboard ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# --- Mining Control ---

@app.post("/api/mining/start")
async def start_mining(config: MiningConfig = MiningConfig()):
    global miner, mining_task

    if miner and miner.is_running:
        return {"error": "Mining is already running"}

    miner = FactorMiner(
        batch_size=config.batch_size,
        max_rounds=config.max_rounds,
        ic_threshold=config.ic_threshold,
        corr_threshold=config.corr_threshold,
        target_size=config.target_size,
        stock_universe=config.stock_universe,
    )
    miner.set_log_callback(broadcast_log)

    mining_task = asyncio.create_task(miner.run())
    return {"status": "started", "config": config.model_dump()}


@app.post("/api/mining/stop")
async def stop_mining():
    global miner
    if miner and miner.is_running:
        miner.stop()
        return {"status": "stopping"}
    return {"status": "not_running"}


@app.get("/api/mining/status", response_model=MiningStatus)
async def mining_status():
    if miner:
        return MiningStatus(**miner.get_status())
    return MiningStatus(
        status="idle",
        current_round=0,
        max_rounds=0,
        library_size=FactorLibrary().size,
        target_size=0,
        is_running=False,
        admission_rate=0.0,
    )


# --- Factor Library ---

@app.get("/api/factors")
async def get_factors():
    library = miner.library if miner else FactorLibrary()
    return {"factors": library.get_all_records(), "total": library.size}


@app.get("/api/factors/{factor_id}/detail")
async def get_factor_detail(factor_id: str):
    library = miner.library if miner else FactorLibrary()
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
        result["analysis"] = {"error": str(e)}

    return result


# --- Backtest ---

@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    library = miner.library if miner else FactorLibrary()

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
            valid_ic = ic[~__import__("numpy").isnan(ic)]

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
                    sig = engine.evaluate(fr.expression)
                    signals[fr.expression] = sig
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
        return {"error": str(e)}


# --- Experience Memory ---

@app.get("/api/memory", response_model=MemoryState)
async def get_memory():
    memory = miner.memory if miner else ExperienceMemory()
    return MemoryState(**memory.get_state())


# --- WebSocket ---

@app.websocket("/ws/mining")
async def websocket_mining(ws: WebSocket):
    await ws.accept()
    ws_connections.append(ws)
    logger.info(f"WebSocket client connected ({len(ws_connections)} total)")
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        if ws in ws_connections:
            ws_connections.remove(ws)
        logger.info(f"WebSocket client disconnected ({len(ws_connections)} remaining)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
