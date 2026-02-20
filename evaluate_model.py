#!/usr/bin/env python3
"""
Full Model Evaluation: IC analysis + P&L simulation (1亿 RMB daily volume).

Produces:
- IC time series, distribution, rolling mean
- Strategy cumulative & daily P&L (scaled to 1亿 RMB)
- Drawdown, quintile returns
- All plots saved to evaluation/output/
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.stock_data import build_panel_from_parquet, calculate_returns
from factor_mining.expression_engine import ExpressionEngine
from factor_mining.factor_library import FactorLibrary
from factor_mining.operators import (
    ts_mean, ts_std, ts_rank, ts_delta, ts_corr, ts_skew, ts_kurt,
    ts_max, ts_min, ts_argmax, ts_argmin, ts_decay_linear, ts_regression_residual,
    cs_rank, cs_zscore,
)
from factor_mining.combiner import FactorCombiner, CombinerConfig
from backtest.engine import run_factor_backtest, run_library_backtest
from backtest.metrics import calc_ic_series, calc_icir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

# 假设日交易规模 1 亿元（用于 P&L 模拟）
DAILY_NOTIONAL_RMB = 100_000_000  # 1亿


SCREENED_FACTORS_PATH = Path(__file__).resolve().parent / "storage" / "screened_factors_22k.json"


def build_signals(
    panel: dict,
    fwd: np.ndarray,
    use_library: bool = True,
    max_mined: int = 2000,
) -> dict[str, np.ndarray]:
    """Build signal dict: mined factors (top by |IC|) + exhaustive engineered features (200+).

    Sources:
    1. Top `max_mined` mined factors from screened_factors_22k.json (sorted by |IC|)
    2. Legacy factor library
    3. 200+ systematically engineered features
    """
    import json as _json

    engine = ExpressionEngine(panel)
    M, T = panel["close"].shape
    close = panel["close"]
    opn = panel["open"]
    high = panel["high"]
    low = panel["low"]
    vol = panel["volume"]
    amount = panel["amount"]
    vwap = panel["vwap"]
    ret = panel["returns"]
    signals = {}

    if use_library:
        library = FactorLibrary()
        for f in library.factors:
            try:
                signals[f.expression] = engine.evaluate(f.expression)
            except Exception:
                pass

    # Load top mined factors from exhaustive screening
    if SCREENED_FACTORS_PATH.exists():
        with open(SCREENED_FACTORS_PATH, "r") as _f:
            all_mined = _json.load(_f)
        all_mined.sort(key=lambda x: -abs(x["ic_mean"]))
        top_mined = all_mined[:max_mined]
        loaded = 0
        for entry in top_mined:
            expr = entry["expression"]
            if expr in signals:
                continue
            try:
                signals[expr] = engine.evaluate(expr)
                loaded += 1
            except Exception:
                pass
        logger.info(f"Loaded {loaded} mined factors from {SCREENED_FACTORS_PATH.name} (top {max_mined} by |IC|)")

    WINDOWS = [3, 5, 10, 20, 40, 60, 120]
    SHORT_WINDOWS = [3, 5, 10, 20, 40]

    # --- 1. Momentum / Reversal (multi-horizon) ---
    for d in WINDOWS:
        if d >= T:
            continue
        r = np.full((M, T), np.nan)
        r[:, d:] = close[:, d:] / close[:, :-d] - 1
        signals[f"ret_{d}d"] = r
        signals[f"cs_rank_ret_{d}d"] = cs_rank(r)
        signals[f"cs_zscore_ret_{d}d"] = cs_zscore(r)

    # --- 2. Volatility (level, ratio, change, rank) ---
    for d in SHORT_WINDOWS:
        v = ts_std(ret, d)
        signals[f"vol_{d}d"] = v
        signals[f"cs_rank_vol_{d}d"] = cs_rank(v)
        vr = vol / (ts_mean(vol, d) + 1e-10)
        signals[f"vol_ratio_{d}"] = vr
        signals[f"cs_rank_vol_ratio_{d}"] = cs_rank(vr)
    for d in [10, 20, 40]:
        signals[f"vol_change_{d}d"] = ts_delta(ts_std(ret, d), 5)
        signals[f"cs_rank_vol_change_{d}d"] = cs_rank(ts_delta(ts_std(ret, d), 5))

    # --- 3. Volume dynamics ---
    for d in SHORT_WINDOWS:
        vm = ts_mean(vol, d)
        signals[f"vol_ma_{d}"] = vm
        signals[f"vol_over_ma_{d}"] = vol / (vm + 1e-10)
        signals[f"cs_rank_vol_over_ma_{d}"] = cs_rank(vol / (vm + 1e-10))
        am = ts_mean(amount, d)
        signals[f"amt_over_ma_{d}"] = amount / (am + 1e-10)
        signals[f"vol_trend_{d}"] = ts_delta(vm, d)

    # --- 4. Price-volume correlation ---
    for d in SHORT_WINDOWS:
        signals[f"corr_cv_{d}"] = ts_corr(close, vol, d)
        signals[f"corr_rv_{d}"] = ts_corr(ret, vol, d)
        signals[f"cs_rank_corr_cv_{d}"] = cs_rank(ts_corr(close, vol, d))
        signals[f"cs_rank_corr_rv_{d}"] = cs_rank(ts_corr(ret, vol, d))

    # --- 5. Price microstructure ---
    hl = (high - low) / (close + 1e-10)
    signals["hl_range"] = hl
    signals["cs_rank_hl_range"] = cs_rank(hl)
    signals["vwap_dev"] = (close - vwap) / (vwap + 1e-10)
    signals["cs_rank_vwap_dev"] = cs_rank((close - vwap) / (vwap + 1e-10))
    signals["intraday"] = (close - opn) / (opn + 1e-10)
    signals["cs_rank_intraday"] = cs_rank((close - opn) / (opn + 1e-10))
    body = np.abs(close - opn)
    wick = high - low
    body_ratio = body / (wick + 1e-10)
    signals["body_ratio"] = body_ratio
    signals["cs_rank_body_ratio"] = cs_rank(body_ratio)
    upper_shadow = (high - np.maximum(close, opn)) / (wick + 1e-10)
    signals["upper_shadow"] = upper_shadow
    lower_shadow = (np.minimum(close, opn) - low) / (wick + 1e-10)
    signals["lower_shadow"] = lower_shadow
    typical_price = (high + low + close) / 3.0
    signals["typical_dev"] = (close - typical_price) / (typical_price + 1e-10)

    for d in SHORT_WINDOWS:
        signals[f"hl_range_ma_{d}"] = ts_mean(hl, d)
        signals[f"cs_rank_hl_ma_{d}"] = cs_rank(ts_mean(hl, d))

    # --- 6. Higher moments (skew, kurtosis) ---
    for d in [5, 10, 20, 40, 60]:
        signals[f"skew_{d}"] = ts_skew(ret, d)
        signals[f"cs_rank_skew_{d}"] = cs_rank(ts_skew(ret, d))
    for d in [10, 20, 40]:
        signals[f"kurt_{d}"] = ts_kurt(ret, d)
        signals[f"cs_rank_kurt_{d}"] = cs_rank(ts_kurt(ret, d))

    # --- 7. Rank / relative position ---
    for d in SHORT_WINDOWS:
        signals[f"ts_rank_close_{d}"] = ts_rank(close, d)
        signals[f"ts_rank_vol_{d}"] = ts_rank(vol, d)
        signals[f"ts_rank_ret_{d}"] = ts_rank(ret, d)

    # --- 8. Min/Max/Argmin/Argmax features ---
    for d in [10, 20, 40, 60]:
        signals[f"close_highD{d}"] = close / (ts_max(close, d) + 1e-10)
        signals[f"close_lowD{d}"] = close / (ts_min(close, d) + 1e-10)
        signals[f"argmax_close_{d}"] = ts_argmax(close, d).astype(float) / d
        signals[f"argmin_close_{d}"] = ts_argmin(close, d).astype(float) / d
        signals[f"argmax_vol_{d}"] = ts_argmax(vol, d).astype(float) / d

    # --- 9. Decay-weighted momentum ---
    for d in [5, 10, 20, 40]:
        signals[f"decay_ret_{d}"] = ts_decay_linear(ret, d)
        signals[f"cs_rank_decay_ret_{d}"] = cs_rank(ts_decay_linear(ret, d))
        signals[f"decay_vol_{d}"] = ts_decay_linear(vol, d)

    # --- 10. Composite: risk-adjusted momentum, momentum/vol ratio ---
    for d in [5, 10, 20, 40]:
        mom = np.full((M, T), np.nan)
        if d < T:
            mom[:, d:] = close[:, d:] / close[:, :-d] - 1
        v = ts_std(ret, d)
        risk_adj = mom / (v + 1e-10)
        signals[f"risk_adj_mom_{d}"] = risk_adj
        signals[f"cs_rank_risk_adj_mom_{d}"] = cs_rank(risk_adj)

    # --- 11. Regression residuals (price vs volume) ---
    for d in [10, 20]:
        try:
            resid = ts_regression_residual(close, vol, d)
            signals[f"resid_cv_{d}"] = resid
            signals[f"cs_rank_resid_cv_{d}"] = cs_rank(resid)
        except Exception:
            pass

    # --- 12. Fundamental factors (if available in panel) ---
    fundamental_fields = ["total_mv", "pe", "pb", "dv_ttm", "bvps", "roe_pct", "basic_eps"]
    for f in fundamental_fields:
        if f in panel and isinstance(panel[f], np.ndarray):
            arr = panel[f]
            signals[f"cs_rank_{f}"] = cs_rank(arr)
            signals[f"cs_zscore_{f}"] = cs_zscore(arr)
            if f in ("pe", "total_mv"):
                signals[f"cs_rank_neg_{f}"] = cs_rank(-arr)

    logger.info(f"Built {len(signals)} engineered signals")
    return signals


def run_evaluation(
    max_stocks: int = 300,
    include_fundamentals: bool = False,
    daily_notional_rmb: float = DAILY_NOTIONAL_RMB,
    output_dir: Path = None,
    combine_backend: str = "auto",
    target_vol_ann: float = 0.05,
    max_dd_target: float = 0.05,
    vol_lookback: int = 20,
) -> dict:
    """Run full evaluation: load data, combine factors, backtest, compute IC & P&L, plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots. pip install matplotlib")
        plt = None

    output_dir = output_dir or Path(__file__).resolve().parent / "evaluation" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading panel ...")
    panel = build_panel_from_parquet(
        max_stocks=max_stocks,
        min_days=500,
        include_fundamentals=include_fundamentals,
    )
    fwd = calculate_returns(panel)
    M, T = panel["close"].shape
    dates = panel.get("dates", np.arange(T))

    logger.info("Building signals ...")
    signals = build_signals(panel, fwd)
    if not signals:
        raise RuntimeError("No signals available. Run mine or pipeline first.")

    # Combined signal: IC-weighted, with optional risk controls (vol target + DD cap)
    logger.info("Combining factors (IC-weighted) with risk controls (vol=%.0f%%, max_dd=%.0f%%) ...", target_vol_ann * 100, max_dd_target * 100)
    bt_lib = run_library_backtest(
        signals,
        fwd,
        method="ic_weighted",
        target_vol_ann=target_vol_ann if target_vol_ann > 0 else None,
        max_dd_target=max_dd_target if max_dd_target > 0 else None,
        vol_lookback=vol_lookback,
    )
    if bt_lib.get("error"):
        raise RuntimeError(bt_lib["error"])

    # Recompute combined signal (signed IC: negative-IC factors contribute reversed)
    ic_series_map = {name: calc_ic_series(sig, fwd) for name, sig in signals.items()}
    names = list(signals.keys())
    ic_means = []
    ic_signs = []
    for n in names:
        ic = ic_series_map[n]
        mean_ic = float(np.nanmean(ic)) if np.any(np.isfinite(ic)) else 0.0
        ic_means.append(abs(mean_ic))
        ic_signs.append(1.0 if mean_ic >= 0 else -1.0)
    total = sum(ic_means) or 1.0
    weights = np.array([m / total for m in ic_means])
    signs = np.array(ic_signs)
    combined = np.zeros((M, T))
    for w, sgn, sig in zip(weights, signs, signals.values()):
        combined += (w * sgn) * np.nan_to_num(sig, nan=0.0)

    ic_series = calc_ic_series(combined, fwd)
    ls_returns = np.array(bt_lib["ls_returns"])
    cum_returns = np.array(bt_lib["cumulative_returns"])
    group_returns = bt_lib.get("group_returns", {})

    valid_ls = ls_returns[~np.isnan(ls_returns)]
    n_days = len(valid_ls)
    cum_one = np.cumprod(1 + valid_ls) if n_days else np.array([1.0])
    peak = np.maximum.accumulate(cum_one)
    drawdown = (cum_one - peak) / (peak + 1e-10)

    # P&L in RMB: 每日盈亏 = 日规模 * 当日收益率；累计盈亏 = 日规模 * (累计净值 - 1)
    daily_pnl_rmb = np.full(T, np.nan)
    daily_pnl_rmb[~np.isnan(ls_returns)] = daily_notional_rmb * ls_returns[~np.isnan(ls_returns)]
    cum_full = np.ones(T)
    for t in range(1, T):
        if np.isfinite(ls_returns[t - 1]):
            cum_full[t] = cum_full[t - 1] * (1 + ls_returns[t - 1])
        else:
            cum_full[t] = cum_full[t - 1]
    cum_pnl_rmb = daily_notional_rmb * (cum_full - 1.0)

    # Summary metrics
    ic_mean = float(np.nanmean(ic_series))
    ic_std = float(np.nanstd(ic_series))
    icir = calc_icir(ic_series)
    ic_positive_ratio = float(np.nanmean(ic_series > 0))

    summary = {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_positive_ratio": ic_positive_ratio,
        "sharpe": bt_lib["sharpe"],
        "annual_return": bt_lib["annual_return"],
        "max_drawdown": bt_lib["max_drawdown"],
        "win_ratio": bt_lib["win_ratio"],
        "turnover": bt_lib["turnover"],
        "n_factors": len(signals),
        "n_days": n_days,
        "daily_notional_rmb": daily_notional_rmb,
        "total_pnl_rmb": float(cum_pnl_rmb[-1]),
        "avg_daily_pnl_rmb": float(np.nanmean(daily_pnl_rmb)) if n_days else 0.0,
    }

    # Print report
    logger.info("=" * 60)
    logger.info("  Model Evaluation Report")
    logger.info("=" * 60)
    logger.info("  IC Mean:    %+.4f   IC Std:   %.4f", summary["ic_mean"], summary["ic_std"])
    logger.info("  ICIR:       %+.3f   IC>0 %%:  %.1f%%", summary["icir"], summary["ic_positive_ratio"] * 100)
    logger.info("  Sharpe:     %.3f   Ann Ret:  %.2f%%", summary["sharpe"], summary["annual_return"] * 100)
    logger.info("  Max DD:     %.2f%%   Win %%:   %.1f%%", summary["max_drawdown"] * 100, summary["win_ratio"] * 100)
    logger.info("  Turnover:   %.4f", summary["turnover"])
    logger.info("  --- P&L (假设日交易规模 %.0f 万元 = %.2f 亿) ---", daily_notional_rmb / 1e4, daily_notional_rmb / 1e8)
    logger.info("  累计盈亏:   %+.2f 万元", summary["total_pnl_rmb"] / 1e4)
    logger.info("  日均盈亏:   %+.2f 万元", summary["avg_daily_pnl_rmb"] / 1e4)
    logger.info("=" * 60)

    if plt is None:
        return {"summary": summary, "plots": []}

    # ---- Plots ----
    date_idx = np.arange(T)
    if hasattr(dates[0], "item"):
        try:
            date_idx = np.array([int(str(d).replace("-", "")[:8]) for d in dates])
        except Exception:
            pass

    fig_dir = output_dir
    plots_saved = []

    # 1) IC time series
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(date_idx, np.nan_to_num(ic_series, nan=0), alpha=0.7, label="Daily IC")
    roll = 20
    if T >= roll:
        ic_roll = np.convolve(np.nan_to_num(ic_series, nan=0), np.ones(roll) / roll, mode="same")
        ax.plot(date_idx, ic_roll, color="C1", lw=2, label=f"IC {roll}d rolling mean")
    ax.axhline(0, color="gray", ls="--")
    ax.set_title("Information Coefficient (IC) Time Series")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = fig_dir / "ic_series.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p1))

    # 2) IC distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    valid_ic = ic_series[~np.isnan(ic_series)]
    if len(valid_ic) > 0:
        ax.hist(valid_ic, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(ic_mean, color="red", ls="--", lw=2, label=f"Mean IC = {ic_mean:+.4f}")
    ax.set_title("IC Distribution")
    ax.set_xlabel("IC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = fig_dir / "ic_distribution.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p2))

    # 3) Cumulative P&L (RMB)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(date_idx, 0, cum_pnl_rmb / 1e4, where=(cum_pnl_rmb >= 0), color="green", alpha=0.5)
    ax.fill_between(date_idx, 0, cum_pnl_rmb / 1e4, where=(cum_pnl_rmb < 0), color="red", alpha=0.5)
    ax.plot(date_idx, cum_pnl_rmb / 1e4, color="black", lw=1)
    ax.axhline(0, color="gray", ls="--")
    ax.set_title(f"Cumulative P&L (万元) — 假设日交易规模 {daily_notional_rmb/1e4:.0f} 万元")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (万元)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = fig_dir / "cumulative_pnl_rmb.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p3))

    # 4) Daily P&L (RMB)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    daily_ok = ~np.isnan(daily_pnl_rmb)
    ax.bar(date_idx[daily_ok], daily_pnl_rmb[daily_ok] / 1e4, color=["green" if x >= 0 else "red" for x in daily_pnl_rmb[daily_ok]], alpha=0.7, width=1)
    ax.axhline(0, color="gray", ls="--")
    ax.set_title(f"Daily P&L (万元) — 日规模 {daily_notional_rmb/1e4:.0f} 万元")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily P&L (万元)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p4 = fig_dir / "daily_pnl_rmb.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p4))

    # 5) Drawdown
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(range(len(drawdown)), drawdown * 100, 0, color="red", alpha=0.5)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p5 = fig_dir / "drawdown.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p5))

    # 6) Quintile cumulative returns
    if group_returns:
        fig, ax = plt.subplots(figsize=(10, 4))
        for q, label in enumerate(["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"], 1):
            key = f"Q{q}"
            if key in group_returns:
                gr = np.array(group_returns[key], dtype=float)
                cumq = np.ones(T)
                for t in range(1, T):
                    cumq[t] = cumq[t - 1] * (1 + gr[t]) if np.isfinite(gr[t]) else cumq[t - 1]
                ax.plot(np.arange(T), cumq, label=label)
        ax.set_title("Quintile Cumulative Returns (Q1=low factor, Q5=high factor)")
        ax.set_xlabel("Trading days")
        ax.set_ylabel("Cumulative return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p6 = fig_dir / "quintile_returns.png"
        fig.savefig(p6, dpi=150)
        plt.close(fig)
        plots_saved.append(str(p6))

    # 7) Equity curve (strategy cumulative return, unit)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(range(len(cum_one)), cum_one, color="steelblue", lw=2)
    ax.axhline(1, color="gray", ls="--")
    ax.set_title("Strategy Cumulative Return (1 = 100%)")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p7 = fig_dir / "equity_curve.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p7))

    # 8) Rolling Sharpe (63d)
    roll_sharpe = np.full(T, np.nan)
    for t in range(63, T):
        w = ls_returns[t - 63:t]
        w = w[np.isfinite(w)]
        if len(w) >= 20 and np.std(w) > 1e-10:
            roll_sharpe[t] = np.mean(w) / np.std(w) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(date_idx, roll_sharpe, color="purple", lw=1.5)
    ax.axhline(0, color="gray", ls="--")
    ax.set_title("Rolling Sharpe (63 days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p8 = fig_dir / "rolling_sharpe_63d.png"
    fig.savefig(p8, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p8))

    # 9) Rolling Annualized Volatility (21d)
    roll_vol = np.full(T, np.nan)
    for t in range(21, T):
        w = ls_returns[t - 21:t]
        w = w[np.isfinite(w)]
        if len(w) >= 10:
            roll_vol[t] = np.std(w) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(date_idx, roll_vol * 100, color="darkorange", lw=1.5)
    ax.set_title("Rolling Annualized Volatility (21 days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p9 = fig_dir / "rolling_vol_21d.png"
    fig.savefig(p9, dpi=150)
    plt.close(fig)
    plots_saved.append(str(p9))

    # 10) Monthly P&L (RMB)
    if np.any(np.isfinite(daily_pnl_rmb)):
        date_series = pd.to_datetime([str(d).replace("-", "")[:8] for d in dates], errors="coerce")
        monthly = pd.DataFrame({"date": date_series, "pnl": daily_pnl_rmb})
        monthly = monthly.dropna(subset=["date", "pnl"])
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        monthly_pnl = monthly.groupby("month", as_index=False)["pnl"].sum()

        fig, ax = plt.subplots(figsize=(12, 3.5))
        colors = ["green" if x >= 0 else "red" for x in monthly_pnl["pnl"].values]
        ax.bar(monthly_pnl["month"], monthly_pnl["pnl"] / 1e4, color=colors, alpha=0.75)
        ax.axhline(0, color="gray", ls="--")
        ax.set_title("Monthly P&L (万元)")
        ax.set_xlabel("Month")
        ax.set_ylabel("P&L (万元)")
        step = max(1, len(monthly_pnl) // 15)
        for i, label in enumerate(ax.get_xticklabels()):
            if i % step != 0:
                label.set_visible(False)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        p10 = fig_dir / "monthly_pnl_rmb.png"
        fig.savefig(p10, dpi=150)
        plt.close(fig)
        plots_saved.append(str(p10))

    logger.info("Plots saved to %s: %s", fig_dir, ", ".join(plots_saved))

    # Optional: run LightGBM combiner for comparison
    try:
        cfg = CombinerConfig(backend=combine_backend, train_window=500, test_step=60, purge_gap=5)
        combiner = FactorCombiner(cfg)
        report = combiner.run(signals, fwd)
        logger.info("Combiner (walk-forward) IC mean: %+.4f, ICIR: %+.3f", report.ic_mean, report.icir)
        summary["combiner_ic_mean"] = report.ic_mean
        summary["combiner_icir"] = report.icir
    except Exception as e:
        logger.warning("Combiner run failed: %s", e)

    return {"summary": summary, "plots": plots_saved}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model: IC + P&L (1亿 RMB scale)")
    parser.add_argument("--max-stocks", type=int, default=300)
    parser.add_argument("--fundamentals", action="store_true")
    parser.add_argument("--notional", type=float, default=DAILY_NOTIONAL_RMB, help="Daily notional in RMB (default 1e8)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--target-vol", type=float, default=0.05, help="Target annual vol for vol targeting (default 0.05 = 5%%)")
    parser.add_argument("--max-dd", type=float, default=0.05, help="Max drawdown cap (default 0.05 = 5%%)")
    parser.add_argument("--vol-lookback", type=int, default=20)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else None
    run_evaluation(
        max_stocks=args.max_stocks,
        include_fundamentals=args.fundamentals,
        daily_notional_rmb=args.notional,
        output_dir=out,
        combine_backend=args.backend,
        target_vol_ann=args.target_vol,
        max_dd_target=args.max_dd,
        vol_lookback=args.vol_lookback,
    )


if __name__ == "__main__":
    main()
