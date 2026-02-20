#!/usr/bin/env python3
"""
FactorMiner CLI Runner（与 factor_investing 数据合并）
=====================================================
数据统一从 factor_investing/data 读取（OHLCV + 基本面）。

Usage:
    python run.py                        # 默认: mining
    python run.py all                    # 合并流程: 数据→生成→筛选→组合→报告
    python run.py pipeline               # 同上，可调参数
    python run.py mine --rounds 20 --batch 10
    python run.py list                   # 列出因子库
    python run.py backtest               # 回测因子库
    python run.py combine                # 仅运行因子组合器
    python run.py evaluate               # 全模型评估：IC 图 + P&L（1亿模拟）
    python run.py report                 # 生成完整金融报告 + 系统审计（Markdown）
    python run.py server                 # 启动 Web 面板
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LLM_API_KEY, IC_THRESHOLD, CORR_THRESHOLD,
    TARGET_LIBRARY_SIZE, FACTOR_LIBRARY_PATH, EXPERIENCE_MEMORY_PATH,
)
from data.stock_data import build_panel_from_parquet, calculate_returns
from factor_mining.expression_engine import ExpressionEngine
from factor_mining.factor_library import FactorLibrary
from factor_mining.miner import FactorMiner
from factor_mining.combiner import FactorCombiner, CombinerConfig
from factor_mining.generator import generate_all_candidates, screen_factors
from backtest.engine import run_factor_backtest, run_library_backtest
from backtest.metrics import calc_ic_series, calc_icir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("factorminer")


def build_data(max_stocks: int = 200, include_fundamentals: bool = False):
    """Load A-share data from factor_investing（合并数据源），可选基本面."""
    logger.info("Loading A-share data from factor_investing ...")
    full_panel = build_panel_from_parquet(
        max_stocks=max_stocks,
        min_days=500,
        include_fundamentals=include_fundamentals,
    )
    fwd_returns = calculate_returns(full_panel)
    M, T = full_panel["close"].shape
    logger.info(f"Full panel: {M} stocks x {T} days")

    fast_n = min(50, M)
    fast_panel = {}
    for key, val in full_panel.items():
        if isinstance(val, np.ndarray) and val.ndim == 2:
            fast_panel[key] = val[:fast_n, :]
        elif isinstance(val, list):
            fast_panel[key] = val[:fast_n]
        else:
            fast_panel[key] = val
    fast_fwd = fwd_returns[:fast_n, :]

    return full_panel, fwd_returns, fast_panel, fast_fwd


async def run_mining(args):
    """Run the Ralph Loop mining cycle via the FactorMiner class."""
    if not LLM_API_KEY:
        logger.error("KIMI_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    full_panel, fwd_returns, fast_panel, fast_fwd = build_data(args.max_stocks)

    miner = FactorMiner(
        batch_size=args.batch,
        max_rounds=args.rounds,
        ic_threshold=args.ic_threshold,
        corr_threshold=args.corr_threshold,
        target_size=args.target_size,
    )
    miner.inject_data(
        fast_panel=fast_panel,
        full_panel=full_panel,
        fast_fwd_returns=fast_fwd,
        full_fwd_returns=fwd_returns,
    )

    await miner.run()

    if miner.library.size > 0:
        print_library(miner.library)


def list_factors_cmd():
    """List all discovered factors."""
    library = FactorLibrary()
    if library.size == 0:
        logger.info("Factor library is empty. Run mining first.")
        return
    print_library(library)


def print_library(library: FactorLibrary):
    """Pretty-print the factor library."""
    print(f"\n{'='*100}")
    print(f"Factor Library: {library.size} factors")
    print(f"{'='*100}")
    print(f"{'#':>3} {'IC':>8} {'ICIR':>8} {'Corr':>8} {'Turn':>8}  Expression")
    print(f"{'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*60}")
    for i, f in enumerate(sorted(library.factors, key=lambda x: -abs(x.ic_mean))):
        print(f"{i+1:>3} {f.ic_mean:>8.4f} {f.icir:>8.3f} {f.max_correlation:>8.3f} {f.turnover:>8.4f}  {f.expression[:60]}")
        if f.logic_description:
            print(f"{'':>39}  -> {f.logic_description[:60]}")
    print()


def run_backtest_cmd(args):
    """Backtest the factor library."""
    library = FactorLibrary()
    if library.size == 0:
        logger.info("Factor library is empty. Run mining first.")
        return

    full_panel, fwd_returns, _, _ = build_data(args.max_stocks)
    engine = ExpressionEngine(full_panel)

    print(f"\n{'='*80}")
    print(f"Backtesting {library.size} factors")
    print(f"{'='*80}")

    print(f"\n{'Expression':60} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} {'WinR':>8}")
    print(f"{'-'*60} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    signals = {}
    for f in library.factors:
        try:
            sig = engine.evaluate(f.expression)
            signals[f.expression] = sig
            bt = run_factor_backtest(sig, fwd_returns)
            print(f"{f.expression[:60]:60} {bt['sharpe']:>8.3f} {bt['annual_return']*100:>7.1f}% {bt['max_drawdown']*100:>7.1f}% {bt['win_ratio']*100:>7.1f}%")
        except Exception as e:
            print(f"{f.expression[:60]:60} ERROR: {e}")

    if signals:
        for method in ["equal", "ic_weighted", "icir_weighted"]:
            bt = run_library_backtest(signals, fwd_returns, method=method)
            print(f"\nLibrary ({method:15}): Sharpe={bt['sharpe']:.3f} AnnRet={bt['annual_return']*100:.1f}% MaxDD={bt['max_drawdown']*100:.1f}% Win={bt['win_ratio']*100:.1f}%")

    print()


def run_combine_cmd(args):
    """Run the factor combiner (LightGBM / HGBR) for IC optimization."""
    from factor_mining.operators import ts_mean, ts_std, ts_rank, ts_delta, cs_rank, ts_corr, ts_skew

    library = FactorLibrary()
    if library.size == 0:
        logger.info("Factor library is empty. Run mining first.")
        return

    full_panel, fwd_returns, _, _ = build_data(args.max_stocks)
    engine = ExpressionEngine(full_panel)
    M, T = full_panel["close"].shape

    signals = {}
    for f in library.factors:
        try:
            signals[f.expression] = engine.evaluate(f.expression)
        except Exception:
            pass

    close = full_panel["close"]
    vol = full_panel["volume"]
    ret = full_panel["returns"]
    high, low, vwap = full_panel["high"], full_panel["low"], full_panel["vwap"]

    for d in [5, 10, 20, 40, 60]:
        r = np.full((M, T), np.nan)
        if d < T:
            r[:, d:] = close[:, d:] / close[:, :-d] - 1
        signals[f"ret_{d}d"] = r
        signals[f"cs_rank_ret_{d}d"] = cs_rank(r)

    for d in [5, 10, 20, 40]:
        signals[f"vol_{d}d"] = ts_std(ret, d)
        vr = vol / (ts_mean(vol, d) + 1e-10)
        signals[f"vol_ratio_{d}"] = vr

    for d in [5, 10, 20, 40]:
        signals[f"corr_cv_{d}"] = ts_corr(close, vol, d)

    hl = (high - low) / (close + 1e-10)
    signals["hl_range"] = hl
    signals["vwap_dev"] = (close - vwap) / (vwap + 1e-10)
    signals["intraday"] = (close - full_panel["open"]) / (full_panel["open"] + 1e-10)

    for d in [10, 20, 40]:
        signals[f"skew_{d}"] = ts_skew(ret, d)
        signals[f"ts_rank_close_{d}"] = ts_rank(close, d)

    logger.info(f"Total combiner input: {len(signals)} features")

    cfg = CombinerConfig(backend=args.backend, train_window=500, test_step=60, purge_gap=5)
    combiner = FactorCombiner(cfg)
    report = combiner.run(signals, fwd_returns)
    combiner.print_report(report)


def run_full_pipeline_cmd(args):
    """合并流程：加载数据(含基本面)→生成候选→筛选(IC/HLZ/OOS)→组合→报告."""
    import time
    logger.info("=" * 70)
    logger.info("  合并流程 (FactorMiner + factor_investing 数据)")
    logger.info("=" * 70)
    t0 = time.time()

    panel = build_panel_from_parquet(
        max_stocks=args.max_stocks,
        min_days=500,
        include_fundamentals=getattr(args, "fundamentals", False),
    )
    fwd = calculate_returns(panel)
    M, T = panel["close"].shape
    logger.info(f"Panel: {M} 股 x {T} 日, 基本面={getattr(args, 'fundamentals', False)}")

    engine = ExpressionEngine(panel)
    candidates = generate_all_candidates(
        max_depth=getattr(args, "max_depth", 3),
        include_fundamentals=getattr(args, "fundamentals", False),
    )
    logger.info(f"生成候选: {len(candidates)}")

    survivors = screen_factors(
        candidates,
        engine,
        fwd,
        ic_threshold=getattr(args, "ic_threshold", 0.015),
        use_hlz=getattr(args, "use_hlz", True),
        use_oos=getattr(args, "use_oos", True),
        max_corr=getattr(args, "max_corr", 0.75),
        max_keep=getattr(args, "max_keep", 2000),
        num_workers=getattr(args, "workers", 1),
        positive_ic_only=getattr(args, "positive_ic_only", True),
    )

    if not survivors:
        logger.warning("无因子通过筛选，可降低 --ic-threshold 或放宽 --no-hlz/--no-oos")
        logger.info(f"合并流程结束 ({time.time()-t0:.1f}s)")
        return

    signals = {expr: data["signal"] for expr, data in survivors.items()}
    cfg = CombinerConfig(
        backend=getattr(args, "backend", "auto"),
        train_window=500,
        test_step=60,
        purge_gap=5,
    )
    combiner = FactorCombiner(cfg)
    report = combiner.run(signals, fwd)
    combiner.print_report(report)
    logger.info(f"合并流程结束 ({time.time()-t0:.1f}s), 幸存因子 {len(survivors)}, 组合 IC={report.ic_mean:.4f}")


def main():
    parser = argparse.ArgumentParser(description="FactorMiner CLI")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    mine_p = sub.add_parser("mine", help="Run the Ralph Loop mining cycle")
    mine_p.add_argument("--rounds", type=int, default=3, help="Number of mining rounds (default: 3)")
    mine_p.add_argument("--batch", type=int, default=5, help="Candidates per round (default: 5)")
    mine_p.add_argument("--max-stocks", type=int, default=200, help="Max stocks in panel (default: 200)")
    mine_p.add_argument("--ic-threshold", type=float, default=IC_THRESHOLD)
    mine_p.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    mine_p.add_argument("--target-size", type=int, default=TARGET_LIBRARY_SIZE)

    sub.add_parser("list", help="List discovered factors")

    bt_p = sub.add_parser("backtest", help="Backtest the factor library")
    bt_p.add_argument("--max-stocks", type=int, default=200)

    comb_p = sub.add_parser("combine", help="仅运行因子组合器 (LightGBM/HGBR/Ridge)")
    comb_p.add_argument("--max-stocks", type=int, default=300)
    comb_p.add_argument("--backend", type=str, default="auto", choices=["auto", "lightgbm", "hgbr", "ridge"])

    pipe_p = sub.add_parser("pipeline", help="合并流程: 生成→筛选→组合 (可调参数)")
    pipe_p.add_argument("--max-stocks", type=int, default=500)
    pipe_p.add_argument("--max-depth", type=int, default=3)
    pipe_p.add_argument("--fundamentals", action="store_true", help="使用 factor_investing 基本面数据")
    pipe_p.add_argument("--ic-threshold", type=float, default=0.015)
    pipe_p.add_argument("--max-keep", type=int, default=2000)
    pipe_p.add_argument("--max-corr", type=float, default=0.75)
    pipe_p.add_argument("--no-hlz", dest="use_hlz", action="store_false", default=True)
    pipe_p.add_argument("--no-oos", dest="use_oos", action="store_false", default=True)
    pipe_p.add_argument("--workers", type=int, default=1)
    pipe_p.add_argument("--allow-negative-ic", dest="positive_ic_only", action="store_false", default=True, help="默认仅保留正IC因子")
    pipe_p.add_argument("--backend", type=str, default="auto")

    reinit_p = sub.add_parser("reinit", help="清空因子库与经验记忆，重新开始（保证只做正收益）")
    reinit_p.add_argument("--yes", action="store_true", help="确认执行")

    all_p = sub.add_parser("all", help="合并流程（默认）: 数据+基本面 → 生成→筛选→组合")
    all_p.add_argument("--max-stocks", type=int, default=300)
    all_p.add_argument("--max-depth", type=int, default=2, help="2=快, 3=更多候选")
    all_p.add_argument("--fundamentals", action="store_true")
    all_p.add_argument("--ic-threshold", type=float, default=0.015)
    all_p.add_argument("--max-keep", type=int, default=500)
    all_p.add_argument("--backend", type=str, default="auto")

    eval_p = sub.add_parser("evaluate", help="Full evaluation: IC plots + P&L (1亿 RMB simulation)")
    eval_p.add_argument("--max-stocks", type=int, default=300)
    eval_p.add_argument("--fundamentals", action="store_true")
    eval_p.add_argument("--notional", type=float, default=100_000_000, help="Daily notional RMB (default 1e8)")
    eval_p.add_argument("--output-dir", type=str, default=None)
    eval_p.add_argument("--target-vol", type=float, default=0.05, help="Target annual vol (default 0.05)")
    eval_p.add_argument("--max-dd", type=float, default=0.05, help="Max drawdown cap (default 0.05)")
    eval_p.add_argument("--vol-lookback", type=int, default=20)

    report_p = sub.add_parser("report", help="Generate full Financial Report + System Audit (Markdown)")
    report_p.add_argument("--max-stocks", type=int, default=300)
    report_p.add_argument("--fundamentals", action="store_true")
    report_p.add_argument("--notional", type=float, default=100_000_000)
    report_p.add_argument("--output-dir", type=str, default=None)
    report_p.add_argument("--target-vol", type=float, default=0.05)
    report_p.add_argument("--max-dd", type=float, default=0.05)
    report_p.add_argument("--vol-lookback", type=int, default=20)

    srv_p = sub.add_parser("server", help="Start the web dashboard")
    srv_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command is None:
        args.command = "mine"
        args.rounds = 3
        args.batch = 5
        args.max_stocks = 200
        args.ic_threshold = IC_THRESHOLD
        args.corr_threshold = CORR_THRESHOLD
        args.target_size = TARGET_LIBRARY_SIZE

    if args.command == "mine":
        asyncio.run(run_mining(args))
    elif args.command == "list":
        list_factors_cmd()
    elif args.command == "backtest":
        run_backtest_cmd(args)
    elif args.command == "combine":
        run_combine_cmd(args)
    elif args.command == "pipeline":
        run_full_pipeline_cmd(args)
    elif args.command == "all":
        # 合并流程：补全 all 的默认参数后走 pipeline 逻辑
        args.max_keep = getattr(args, "max_keep", 500)
        args.max_corr = 0.75
        args.use_hlz = True
        args.use_oos = True
        args.workers = 1
        run_full_pipeline_cmd(args)
    elif args.command == "reinit":
        if not getattr(args, "yes", False):
            logger.error("Re-init will clear factor library and experience memory. Run with --yes to confirm.")
            sys.exit(1)
        for name, path in [("factor_library", FACTOR_LIBRARY_PATH), ("experience_memory", EXPERIENCE_MEMORY_PATH)]:
            p = Path(path)
            if p.exists():
                p.write_text("[]" if "library" in name else "{}")
                logger.info("Cleared %s", p)
        logger.info("Re-init done. Run pipeline or all to repopulate with positive-IC-only factors.")

    elif args.command == "evaluate":
        from evaluate_model import run_evaluation
        run_evaluation(
            max_stocks=args.max_stocks,
            include_fundamentals=args.fundamentals,
            daily_notional_rmb=args.notional,
            output_dir=Path(args.output_dir) if getattr(args, "output_dir") else None,
            target_vol_ann=getattr(args, "target_vol", 0.05),
            max_dd_target=getattr(args, "max_dd", 0.05),
            vol_lookback=getattr(args, "vol_lookback", 20),
        )
    elif args.command == "report":
        from generate_report import run_report
        run_report(
            max_stocks=args.max_stocks,
            include_fundamentals=args.fundamentals,
            notional=getattr(args, "notional", 100_000_000),
            output_dir=getattr(args, "output_dir"),
            target_vol_ann=getattr(args, "target_vol", 0.05),
            max_dd_target=getattr(args, "max_dd", 0.05),
            vol_lookback=getattr(args, "vol_lookback", 20),
        )
    elif args.command == "server":
        import uvicorn
        uvicorn.run("api.main:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
