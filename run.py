#!/usr/bin/env python3
"""
FactorMiner CLI Runner
======================
Runs the Ralph Loop mining cycle using existing A-share data from
factor_investing/data/processed/daily_20180101_20241231.parquet.

Usage:
    python run.py                        # Run mining (3 rounds, batch=5 for quick test)
    python run.py --rounds 20 --batch 10 # Full mining run
    python run.py --list                 # List discovered factors
    python run.py --backtest             # Backtest the factor library
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    IC_THRESHOLD, CORR_THRESHOLD,
    TARGET_LIBRARY_SIZE, FACTOR_LIBRARY_PATH, EXPERIENCE_MEMORY_PATH,
)
from data.stock_data import build_panel_from_parquet, calculate_returns
from factor_mining.expression_engine import ExpressionEngine
from factor_mining.factor_library import FactorLibrary, compute_ic, compute_icir
from factor_mining.experience_memory import ExperienceMemory
from backtest.engine import run_factor_backtest, run_library_backtest
from backtest.metrics import calc_ic_series, factor_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("factorminer")


def build_data(max_stocks: int = 200):
    """Load A-share data and build panels for fast screening and full validation."""
    logger.info("Loading A-share data from factor_investing parquet...")
    full_panel = build_panel_from_parquet(max_stocks=max_stocks, min_days=500)
    fwd_returns = calculate_returns(full_panel)
    M, T = full_panel["close"].shape
    logger.info(f"Full panel: {M} stocks x {T} days")

    # Fast screening subset (first 50 stocks)
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
    """Run the Ralph Loop mining cycle."""
    if not LLM_API_KEY:
        logger.error("KIMI_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    full_panel, fwd_returns, fast_panel, fast_fwd = build_data(args.max_stocks)

    library = FactorLibrary()
    memory = ExperienceMemory()

    fast_engine = ExpressionEngine(fast_panel)
    full_engine = ExpressionEngine(full_panel)

    from openai import OpenAI
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    from factor_mining.miner import GENERATION_PROMPT
    from factor_mining.expression_engine import validate_expression
    from factor_mining.factor_library import compute_correlation, FactorRecord
    import re

    logger.info(f"Starting mining: {args.rounds} rounds, batch_size={args.batch}")
    logger.info(f"IC threshold={args.ic_threshold}, corr threshold={args.corr_threshold}")
    logger.info(f"Current library size: {library.size}")

    total_admitted = 0
    total_candidates = 0

    for round_num in range(1, args.rounds + 1):
        if library.size >= args.target_size:
            logger.info(f"Target size {args.target_size} reached!")
            break

        memory.update_round(round_num)
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num}/{args.rounds}")
        logger.info(f"{'='*60}")

        # --- 1. Retrieve ---
        ctx = memory.retrieve(library.size)
        prompt = GENERATION_PROMPT.format(
            batch_size=args.batch,
            library_size=library.size,
            **ctx,
        )

        # --- 2. Generate ---
        logger.info(f"Calling Claude to generate {args.batch} candidates...")
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            continue

        # Parse candidates
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in LLM response")
            continue

        try:
            raw_candidates = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            continue

        candidates = []
        for item in raw_candidates:
            expr = item.get("expression", "").strip()
            logic = item.get("logic", "").strip()
            if expr:
                valid, err = validate_expression(expr)
                if valid:
                    candidates.append({"expression": expr, "logic": logic})
                else:
                    logger.debug(f"  Invalid: {expr} -> {err}")

        logger.info(f"Parsed {len(candidates)} valid candidates from LLM")
        total_candidates += len(candidates)

        # --- 3. Evaluate ---
        batch_results = []
        for i, cand in enumerate(candidates):
            expr = cand["expression"]
            result = {
                "expression": expr,
                "logic": cand.get("logic", ""),
                "admitted": False,
                "reason": "",
                "ic_mean": 0.0,
            }

            # Stage 1: Fast IC screen
            try:
                fast_signal = fast_engine.evaluate(expr)
            except Exception as e:
                result["reason"] = f"Eval error: {e}"
                batch_results.append(result)
                continue

            nan_ratio = np.isnan(fast_signal).sum() / fast_signal.size
            if nan_ratio > 0.5:
                result["reason"] = f"Too many NaNs ({nan_ratio:.0%})"
                batch_results.append(result)
                continue

            fast_ic = compute_ic(fast_signal, fast_fwd)
            valid_ic = fast_ic[~np.isnan(fast_ic)]
            if len(valid_ic) < 10:
                result["reason"] = "Insufficient IC data"
                batch_results.append(result)
                continue

            ic_mean = float(np.mean(valid_ic))
            if abs(ic_mean) < args.ic_threshold:
                result["reason"] = f"Fast IC too low: |{ic_mean:.4f}| < {args.ic_threshold}"
                result["ic_mean"] = ic_mean
                batch_results.append(result)
                continue

            logger.info(f"  [{i+1}/{len(candidates)}] {expr[:60]} -> fast IC={ic_mean:.4f} PASS")

            # Stage 2+3: Full validation + correlation check
            try:
                full_signal = full_engine.evaluate(expr)
            except Exception as e:
                result["reason"] = f"Full eval error: {e}"
                batch_results.append(result)
                continue

            full_ic = compute_ic(full_signal, fwd_returns)
            full_valid_ic = full_ic[~np.isnan(full_ic)]
            if len(full_valid_ic) < 10:
                result["reason"] = "Insufficient full IC data"
                batch_results.append(result)
                continue

            full_ic_mean = float(np.mean(full_valid_ic))
            full_ic_std = float(np.std(full_valid_ic))
            full_icir = compute_icir(full_ic)
            result["ic_mean"] = full_ic_mean

            if abs(full_ic_mean) < args.ic_threshold:
                result["reason"] = f"Full IC too low: |{full_ic_mean:.4f}|"
                batch_results.append(result)
                continue

            # Correlation check
            library.clear_signal_cache()
            max_corr = 0.0
            max_corr_idx = -1
            for j, existing in enumerate(library.factors):
                try:
                    existing_signal = full_engine.evaluate(existing.expression)
                    corr = abs(compute_correlation(full_signal, existing_signal))
                    if corr > max_corr:
                        max_corr = corr
                        max_corr_idx = j
                except Exception:
                    continue

            if max_corr >= args.corr_threshold:
                # Stage 2.5: Replacement check
                if max_corr_idx >= 0:
                    existing_ic = abs(library.factors[max_corr_idx].ic_mean)
                    if abs(full_ic_mean) > existing_ic * 1.1:
                        result["admitted"] = True
                        result["reason"] = f"Replacement: IC={full_ic_mean:.4f} > {existing_ic:.4f}"
                    else:
                        result["reason"] = f"Correlated ({max_corr:.3f}) and not stronger"
                        batch_results.append(result)
                        continue
                else:
                    result["reason"] = f"Too correlated: {max_corr:.3f}"
                    batch_results.append(result)
                    continue
            else:
                result["admitted"] = True
                result["reason"] = f"Admitted: IC={full_ic_mean:.4f}, max_corr={max_corr:.3f}"

            if result["admitted"]:
                # Compute turnover
                positions = full_signal[:, 1:] - full_signal[:, :-1]
                turnover = float(np.nanmean(np.nanmean(np.abs(positions), axis=0)))

                record = FactorRecord(
                    expression=expr,
                    ic_mean=full_ic_mean,
                    ic_std=full_ic_std,
                    icir=full_icir,
                    max_correlation=max_corr,
                    turnover=turnover,
                    logic_description=cand.get("logic", ""),
                    mining_round=round_num,
                )
                replace_idx = max_corr_idx if max_corr >= args.corr_threshold else None
                library.admit(record, replace_idx)
                total_admitted += 1
                logger.info(f"  >>> ADMITTED: {expr[:60]}")
                logger.info(f"      IC={full_ic_mean:.4f} ICIR={full_icir:.3f} corr={max_corr:.3f}")

            batch_results.append(result)

        # --- 4. Distill ---
        memory.formation(batch_results)
        admitted_this_round = sum(1 for r in batch_results if r["admitted"])
        logger.info(f"\nRound {round_num} summary: {admitted_this_round}/{len(batch_results)} admitted")
        logger.info(f"Library size: {library.size}")

    logger.info(f"\n{'='*60}")
    logger.info(f"MINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total candidates: {total_candidates}")
    logger.info(f"Total admitted: {total_admitted}")
    logger.info(f"Library size: {library.size}")
    logger.info(f"Factor library saved to: {FACTOR_LIBRARY_PATH}")
    logger.info(f"Experience memory saved to: {EXPERIENCE_MEMORY_PATH}")

    if library.size > 0:
        print_library(library)


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

    # Individual factor backtests
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
        # Library-level backtest
        for method in ["equal", "ic_weighted", "icir_weighted"]:
            bt = run_library_backtest(signals, fwd_returns, method=method)
            print(f"\nLibrary ({method:15}): Sharpe={bt['sharpe']:.3f} AnnRet={bt['annual_return']*100:.1f}% MaxDD={bt['max_drawdown']*100:.1f}% Win={bt['win_ratio']*100:.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="FactorMiner CLI")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Mine
    mine_p = sub.add_parser("mine", help="Run the Ralph Loop mining cycle")
    mine_p.add_argument("--rounds", type=int, default=3, help="Number of mining rounds (default: 3)")
    mine_p.add_argument("--batch", type=int, default=5, help="Candidates per round (default: 5)")
    mine_p.add_argument("--max-stocks", type=int, default=200, help="Max stocks in panel (default: 200)")
    mine_p.add_argument("--ic-threshold", type=float, default=IC_THRESHOLD)
    mine_p.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    mine_p.add_argument("--target-size", type=int, default=TARGET_LIBRARY_SIZE)

    # List
    sub.add_parser("list", help="List discovered factors")

    # Backtest
    bt_p = sub.add_parser("backtest", help="Backtest the factor library")
    bt_p.add_argument("--max-stocks", type=int, default=200)

    # Server
    srv_p = sub.add_parser("server", help="Start the web dashboard")
    srv_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command is None:
        # Default: quick mining test
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
    elif args.command == "server":
        import uvicorn
        uvicorn.run("api.main:app", host="0.0.0.0", port=args.port, reload=True)


if __name__ == "__main__":
    main()
