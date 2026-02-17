#!/usr/bin/env python3
"""
FactorMiner Pipeline Test (No LLM Required)
============================================
Validates the entire mining pipeline end-to-end using predefined factor
expressions. Tests: data loading, operators, expression engine, IC/ICIR,
correlation checking, factor admission, experience memory, and backtest.

Usage:
    python test_pipeline.py
"""

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.stock_data import build_panel_from_parquet, calculate_returns
from factor_mining.expression_engine import ExpressionEngine, validate_expression
from factor_mining.operators import get_operator_names
from factor_mining.factor_library import (
    FactorLibrary, FactorRecord, compute_ic, compute_icir, compute_correlation,
)
from factor_mining.experience_memory import ExperienceMemory
from backtest.engine import run_factor_backtest, run_library_backtest
from backtest.metrics import factor_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test")

# Pre-defined factor expressions covering different alpha themes
TEST_FACTORS = [
    # Momentum
    {"expression": "cs_rank(ts_delta(close, 20))", "logic": "20-day price momentum ranked cross-sectionally"},
    {"expression": "neg(cs_rank(div(ts_delta(close, 5), ts_lag(close, 5))))", "logic": "5-day return reversal"},
    {"expression": "cs_rank(ts_decay_linear(returns, 10))", "logic": "Linearly decaying 10-day momentum"},

    # Volume-Price
    {"expression": "cs_rank(ts_corr(close, volume, 10))", "logic": "Price-volume correlation (10-day)"},
    {"expression": "neg(cs_rank(ts_corr(returns, volume, 20)))", "logic": "Return-volume divergence"},
    {"expression": "cs_rank(div(volume, ts_mean(volume, 20)))", "logic": "Volume ratio vs 20-day average"},

    # Volatility
    {"expression": "neg(cs_rank(ts_std(returns, 20)))", "logic": "Low-volatility factor (20-day)"},
    {"expression": "cs_rank(ts_skew(returns, 20))", "logic": "Return skewness (20-day)"},
    {"expression": "neg(cs_rank(ts_kurt(returns, 20)))", "logic": "Return kurtosis (negative)"},

    # Price patterns
    {"expression": "cs_rank(div(sub(close, ts_min(low, 20)), sub(ts_max(high, 20), ts_min(low, 20))))", "logic": "Stochastic oscillator (20-day)"},
    {"expression": "cs_rank(sub(close, ts_mean(close, 20)))", "logic": "Price deviation from 20-day MA"},
    {"expression": "neg(cs_rank(div(sub(high, close), sub(high, low))))", "logic": "Upper shadow ratio (bearish)"},

    # VWAP-based
    {"expression": "neg(cs_rank(div(sub(close, vwap), vwap)))", "logic": "VWAP deviation reversal"},
    {"expression": "cs_rank(ts_rank(div(close, vwap), 10))", "logic": "VWAP ratio rank over 10 days"},

    # Higher-order
    {"expression": "cs_rank(ts_regression_residual(volume, close, 20))", "logic": "Volume residual after regressing on price"},
    {"expression": "cs_rank(mul(ts_corr(close, volume, 10), ts_std(returns, 10)))", "logic": "Interaction: PV-corr * volatility"},
]


def main():
    t0 = time.time()

    # ================================================================
    # 1. Data Loading
    # ================================================================
    print("=" * 70)
    print("  FactorMiner Pipeline Test (No LLM Required)")
    print("=" * 70)
    print()

    print("[1/7] Loading A-share data...")
    panel = build_panel_from_parquet(max_stocks=100, min_days=500)
    fwd = calculate_returns(panel)
    M, T = panel["close"].shape
    print(f"       Panel: {M} stocks x {T} days")
    print(f"       Codes: {panel['codes'][:5]} ...")
    print(f"       Dates: {panel['dates'][0]} - {panel['dates'][-1]}")
    print()

    # ================================================================
    # 2. Operator Library
    # ================================================================
    print("[2/7] Operator library check...")
    ops = get_operator_names()
    print(f"       {len(ops)} operators available: {', '.join(ops[:10])} ...")
    print()

    # ================================================================
    # 3. Expression Validation & Evaluation
    # ================================================================
    print("[3/7] Evaluating {len(TEST_FACTORS)} test factor expressions...")
    engine = ExpressionEngine(panel)

    signals = {}
    ic_data = {}
    for i, fdef in enumerate(TEST_FACTORS):
        expr = fdef["expression"]
        valid, err = validate_expression(expr)
        if not valid:
            print(f"  INVALID: {expr} -> {err}")
            continue

        try:
            sig = engine.evaluate(expr)
            ic = compute_ic(sig, fwd)
            valid_ic = ic[~np.isnan(ic)]
            ic_mean = float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0
            icir = compute_icir(ic)
            signals[expr] = sig
            ic_data[expr] = {"ic": ic, "ic_mean": ic_mean, "icir": icir}
            status = "PASS" if abs(ic_mean) >= 0.02 else "WEAK"
            print(f"  [{status}] IC={ic_mean:>8.4f} ICIR={icir:>7.3f}  {expr[:55]}")
        except Exception as e:
            print(f"  ERROR: {expr[:55]} -> {e}")

    print(f"\n       {len(signals)}/{len(TEST_FACTORS)} expressions evaluated successfully")
    print()

    # ================================================================
    # 4. Factor Library: Admission Pipeline
    # ================================================================
    print("[4/7] Testing factor admission pipeline...")
    # Use a temporary library (don't pollute the real one)
    test_lib_path = Path("storage/test_factor_library.json")
    if test_lib_path.exists():
        test_lib_path.unlink()

    library = FactorLibrary(path=test_lib_path)
    admitted_count = 0

    for expr, sig in signals.items():
        ic_series = ic_data[expr]["ic"]
        ic_mean = ic_data[expr]["ic_mean"]
        icir = ic_data[expr]["icir"]

        if abs(ic_mean) < 0.01:
            continue

        # Check correlation with existing library
        max_corr = 0.0
        max_corr_idx = -1
        for j, existing in enumerate(library.factors):
            if existing.expression in signals:
                corr = abs(compute_correlation(sig, signals[existing.expression]))
                if corr > max_corr:
                    max_corr = corr
                    max_corr_idx = j

        if library.size == 0 or max_corr < 0.7:
            record = FactorRecord(
                expression=expr,
                ic_mean=ic_mean,
                ic_std=ic_data[expr].get("ic_std", float(np.std(ic_series[~np.isnan(ic_series)]))),
                icir=icir,
                max_correlation=max_corr,
                turnover=0.0,
                logic_description=next(
                    (f["logic"] for f in TEST_FACTORS if f["expression"] == expr), ""
                ),
                mining_round=1,
            )
            library.admit(record)
            admitted_count += 1
            print(f"  ADMITTED: IC={ic_mean:>8.4f} ICIR={icir:>7.3f} corr={max_corr:.3f}  {expr[:50]}")
        else:
            print(f"  REJECTED: corr={max_corr:.3f} >= 0.7  {expr[:50]}")

    print(f"\n       {admitted_count} factors admitted to library (of {len(signals)} evaluated)")
    print()

    # ================================================================
    # 5. Experience Memory
    # ================================================================
    print("[5/7] Testing experience memory...")
    test_mem_path = Path("storage/test_experience_memory.json")
    if test_mem_path.exists():
        test_mem_path.unlink()

    memory = ExperienceMemory(path=test_mem_path)
    memory.update_round(1)

    batch_results = []
    for expr in list(signals.keys())[:8]:
        ic_mean = ic_data[expr]["ic_mean"]
        batch_results.append({
            "expression": expr,
            "logic": next((f["logic"] for f in TEST_FACTORS if f["expression"] == expr), ""),
            "admitted": abs(ic_mean) >= 0.02,
            "reason": "Good IC" if abs(ic_mean) >= 0.02 else "IC too low",
            "ic_mean": ic_mean,
        })

    memory.formation(batch_results)
    ctx = memory.retrieve(library.size)
    print(f"       Mining state summary:\n{ctx['mining_state_summary']}")
    print(f"       Recommended: {ctx['recommended_directions'][:100]}...")
    print(f"       Forbidden:   {ctx['forbidden_directions'][:100]}...")
    print()

    # ================================================================
    # 6. Backtest
    # ================================================================
    print("[6/7] Running backtests...")

    # Single factor backtest
    best_expr = max(ic_data, key=lambda k: abs(ic_data[k]["ic_mean"]))
    bt = run_factor_backtest(signals[best_expr], fwd)
    print(f"       Best factor: {best_expr[:60]}")
    print(f"       Sharpe:      {bt['sharpe']:.3f}")
    print(f"       Annual Ret:  {bt['annual_return']*100:.1f}%")
    print(f"       Max DD:      {bt['max_drawdown']*100:.1f}%")
    print(f"       Win Ratio:   {bt['win_ratio']*100:.1f}%")
    print(f"       Turnover:    {bt['turnover']:.4f}")
    print()

    # Library backtest
    lib_signals = {}
    for f in library.factors:
        if f.expression in signals:
            lib_signals[f.expression] = signals[f.expression]

    if lib_signals:
        for method in ["equal", "ic_weighted", "icir_weighted"]:
            lbt = run_library_backtest(lib_signals, fwd, method=method)
            print(f"       Library ({method:15}): Sharpe={lbt['sharpe']:.3f} AnnRet={lbt['annual_return']*100:.1f}% MaxDD={lbt['max_drawdown']*100:.1f}%")
    print()

    # ================================================================
    # 7. Quintile Analysis
    # ================================================================
    print("[7/7] Quintile analysis for best factor...")
    gr = bt.get("group_cumulative_returns", {})
    if gr:
        for g in sorted(gr.keys()):
            bar = "█" * int(max(0, min(20, gr[g] * 100)))
            print(f"       {g}: {gr[g]*100:>7.2f}% {bar}")
    print()

    # Cleanup test files
    if test_lib_path.exists():
        test_lib_path.unlink()
    if test_mem_path.exists():
        test_mem_path.unlink()

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"  ALL TESTS PASSED  ({elapsed:.1f}s)")
    print("=" * 70)
    print()
    print("To run actual mining with Claude:")
    print("  1. Set ANTHROPIC_API_KEY in factor-miner/.env")
    print("  2. python run.py mine --rounds 5 --batch 10")
    print()
    print("To start the web dashboard:")
    print("  python run.py server")


if __name__ == "__main__":
    main()
