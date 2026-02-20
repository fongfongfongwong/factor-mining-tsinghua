#!/usr/bin/env python3
"""
Generate comprehensive financial_report.md covering:
1. Factor analysis (all 19K+ mined, top 2K used)
2. Feature analysis (212 engineered features)
3. Beta / market exposure analysis
4. Return analysis (yearly, monthly, daily)
5. P&L analysis (10亿 initial capital)
6. Drawdown / block analysis (DD events, recovery)

Output: evaluation/output/financial_report.md
"""

import sys, json, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("full_report")

from data.stock_data import build_panel_from_parquet, calculate_returns, _PROCESSED_DAILY
from factor_mining.expression_engine import ExpressionEngine
from factor_mining.factor_library import FactorLibrary
from backtest.engine import run_library_backtest
from backtest.metrics import calc_ic_series, calc_icir, calc_sharpe, calc_max_drawdown, calc_annual_return
from evaluate_model import build_signals, SCREENED_FACTORS_PATH

NOTIONAL = 1_000_000_000  # 10亿
MAX_DD = 0.05


def main():
    out_dir = Path(__file__).resolve().parent / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "financial_report.md"

    logger.info("Loading data ...")
    panel = build_panel_from_parquet(max_stocks=300, min_days=500, include_fundamentals=True)
    fwd = calculate_returns(panel)
    M, T = panel["close"].shape
    dates = panel.get("dates", np.arange(T))
    date_strs = [str(d)[:10].replace("-", "") for d in dates]

    db_meta = {}
    if _PROCESSED_DAILY.exists():
        meta_df = pd.read_parquet(_PROCESSED_DAILY, columns=["ts_code", "trade_date"])
        db_meta["rows"] = len(meta_df)
        db_meta["stocks"] = int(meta_df["ts_code"].nunique())
        db_meta["min_date"] = str(meta_df["trade_date"].min())
        db_meta["max_date"] = str(meta_df["trade_date"].max())

    logger.info("Building signals (2000+ factors) ...")
    signals = build_signals(panel, fwd, use_library=True, max_mined=2000)
    n_signals = len(signals)
    logger.info(f"Total signals: {n_signals}")

    # --- Backtest: DD-cap only ---
    logger.info("Running backtest (DD-cap 5%, no vol targeting) ...")
    bt = run_library_backtest(signals, fwd, method="ic_weighted", max_dd_target=MAX_DD)
    ls_returns = np.array(bt["ls_returns"])
    valid_mask = np.isfinite(ls_returns)
    valid_ls = ls_returns[valid_mask]
    n_days = len(valid_ls)

    cum = np.cumprod(1 + valid_ls)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / (peak + 1e-10)
    vol_ann = float(np.std(valid_ls) * np.sqrt(252))

    # P&L
    daily_pnl = NOTIONAL * ls_returns
    cum_full = np.ones(T)
    for t in range(1, T):
        cum_full[t] = cum_full[t - 1] * (1 + ls_returns[t - 1]) if np.isfinite(ls_returns[t - 1]) else cum_full[t - 1]
    cum_pnl = NOTIONAL * (cum_full - 1.0)
    end_equity = NOTIONAL + cum_pnl[-1]
    peak_equity = NOTIONAL + float(np.nanmax(cum_pnl))

    # --- Section 1: Factor Analysis ---
    logger.info("Computing per-factor IC ...")
    all_mined = []
    if SCREENED_FACTORS_PATH.exists():
        with open(SCREENED_FACTORS_PATH, "r") as f:
            all_mined = json.load(f)

    factor_ics = []
    engine = ExpressionEngine(panel)
    for name, sig in signals.items():
        ic = calc_ic_series(sig, fwd)
        valid_ic = ic[~np.isnan(ic)]
        if len(valid_ic) > 0:
            factor_ics.append({
                "expression": name,
                "ic_mean": float(np.mean(valid_ic)),
                "ic_std": float(np.std(valid_ic)),
                "icir": calc_icir(ic),
            })
    factor_ics.sort(key=lambda x: -abs(x["ic_mean"]))

    pos_ic = sum(1 for f in factor_ics if f["ic_mean"] > 0)
    neg_ic = sum(1 for f in factor_ics if f["ic_mean"] < 0)
    avg_abs_ic = float(np.mean([abs(f["ic_mean"]) for f in factor_ics])) if factor_ics else 0

    # --- Section 3: Beta analysis ---
    logger.info("Computing beta / market exposure ...")
    market_ret = np.nanmean(panel["returns"], axis=0)
    valid_both = np.isfinite(ls_returns) & np.isfinite(market_ret)
    if valid_both.sum() > 50:
        ls_v = ls_returns[valid_both]
        mkt_v = market_ret[valid_both]
        cov_mat = np.cov(ls_v, mkt_v)
        beta = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-10)
        corr_mkt = np.corrcoef(ls_v, mkt_v)[0, 1]
        mkt_ann_ret = float((np.prod(1 + mkt_v) ** (252 / len(mkt_v)) - 1))
        alpha_ann = bt["annual_return"] - beta * mkt_ann_ret
    else:
        beta = corr_mkt = mkt_ann_ret = alpha_ann = 0.0

    # --- Section 4: Yearly / monthly returns ---
    yearly = defaultdict(list)
    monthly = defaultdict(list)
    for t in range(T):
        if not np.isfinite(ls_returns[t]):
            continue
        yr = date_strs[t][:4]
        mo = date_strs[t][:6]
        yearly[yr].append(ls_returns[t])
        monthly[mo].append(ls_returns[t])

    # --- Section 6: Drawdown block analysis ---
    dd_events = []
    in_dd = False
    dd_start = 0
    dd_trough = 0
    dd_trough_val = 0
    eq = 1.0
    pk = 1.0
    for i, r in enumerate(valid_ls):
        eq *= (1 + r)
        pk = max(pk, eq)
        dd_val = (eq - pk) / (pk + 1e-10)
        if dd_val < -0.01 and not in_dd:
            in_dd = True
            dd_start = i
            dd_trough = i
            dd_trough_val = dd_val
        elif in_dd:
            if dd_val < dd_trough_val:
                dd_trough = i
                dd_trough_val = dd_val
            if dd_val >= -0.005:
                dd_events.append({
                    "start": dd_start,
                    "trough": dd_trough,
                    "end": i,
                    "depth": dd_trough_val,
                    "duration": i - dd_start,
                    "recovery": i - dd_trough,
                })
                in_dd = False
    if in_dd:
        dd_events.append({
            "start": dd_start,
            "trough": dd_trough,
            "end": len(valid_ls) - 1,
            "depth": dd_trough_val,
            "duration": len(valid_ls) - 1 - dd_start,
            "recovery": len(valid_ls) - 1 - dd_trough,
        })
    dd_events.sort(key=lambda x: x["depth"])

    # === Build Report ===
    logger.info("Writing financial_report.md ...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    def w(s=""): L.append(s)

    w("# Dragon Alpha — Comprehensive Financial Report")
    w()
    w(f"**Generated**: {now}  ")
    w(f"**Initial Capital**: {NOTIONAL/1e8:.2f} 亿元 (RMB {NOTIONAL:,.0f})  ")
    w(f"**Universe**: {M} stocks x {n_days} trading days ({date_strs[0]}~{date_strs[-1]})  ")
    w(f"**Full Database**: {db_meta.get('rows',0):,} rows | {db_meta.get('stocks',0)} stocks | {db_meta.get('min_date','?')}~{db_meta.get('max_date','?')}  ")
    w(f"**Total Signals**: {n_signals}  ")
    w(f"**Risk Control**: DD Cap {MAX_DD*100:.0f}% (no vol targeting)  ")
    w()
    w("---")
    w()

    # === Executive Summary ===
    w("## 0. Executive Summary")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Sharpe Ratio | **{bt['sharpe']:.3f}** |")
    w(f"| Annualized Return | **{bt['annual_return']*100:.2f}%** |")
    w(f"| Annualized Volatility | {vol_ann*100:.2f}% |")
    w(f"| Max Drawdown | {bt['max_drawdown']*100:.2f}% |")
    w(f"| Win Rate | {bt['win_ratio']*100:.1f}% |")
    w(f"| Turnover | {bt['turnover']:.4f} |")
    w(f"| Beta (vs market) | {beta:.4f} |")
    w(f"| Alpha (annualized) | {alpha_ann*100:+.2f}% |")
    w(f"| Cumulative PnL | **{cum_pnl[-1]/1e8:+.2f} 亿元** |")
    w(f"| Ending Equity | **{end_equity/1e8:.2f} 亿元** |")
    w(f"| Signals Used | {n_signals} |")
    w()
    w("---")
    w()

    # === Section 1: Factor Analysis ===
    w("## 1. Factor Analysis (All Mined Factors)")
    w()
    w(f"### 1.1 Mined Factor Universe")
    w()
    w(f"- **Candidates generated**: 22,110 (depth-3 formulaic alpha)")
    w(f"- **Passed IC > 0.005**: {len(all_mined):,}")
    w(f"- **Top {min(2000, len(all_mined))} by |IC| used in model**")
    w()

    # IC distribution of mined
    if all_mined:
        mined_ics = [x["ic_mean"] for x in all_mined]
        mined_pos = sum(1 for x in mined_ics if x > 0)
        mined_neg = sum(1 for x in mined_ics if x < 0)
        w("| Stat | Value |")
        w("|------|-------|")
        w(f"| Total mined factors | {len(all_mined):,} |")
        w(f"| Positive IC | {mined_pos:,} |")
        w(f"| Negative IC | {mined_neg:,} |")
        w(f"| Mean \\|IC\\| | {np.mean([abs(x) for x in mined_ics]):.4f} |")
        w(f"| Max IC | {max(mined_ics):+.4f} |")
        w(f"| Min IC | {min(mined_ics):+.4f} |")
        w(f"| Median \\|IC\\| | {np.median([abs(x) for x in mined_ics]):.4f} |")
        w()

    w("### 1.2 Top 50 Factors by |IC| (Used in Model)")
    w()
    w("| # | Expression | IC Mean | ICIR |")
    w("|---|-----------|---------|------|")
    for i, f in enumerate(factor_ics[:50], 1):
        expr = f["expression"][:65] + ("..." if len(f["expression"]) > 65 else "")
        w(f"| {i} | `{expr}` | {f['ic_mean']:+.4f} | {f['icir']:+.3f} |")
    w()

    w(f"### 1.3 Factor IC Summary (Model Signals)")
    w()
    w(f"- Total signals in model: **{n_signals}**")
    w(f"- Positive IC: {pos_ic} ({pos_ic/n_signals*100:.1f}%)")
    w(f"- Negative IC: {neg_ic} ({neg_ic/n_signals*100:.1f}%)")
    w(f"- Average |IC|: {avg_abs_ic:.4f}")
    w()
    w("---")
    w()

    # === Section 2: Feature Analysis ===
    w("## 2. Feature / Signal Analysis")
    w()
    w("### 2.1 Signal Categories")
    w()
    categories = defaultdict(list)
    for f in factor_ics:
        expr = f["expression"]
        if expr.startswith("ret_") or "momentum" in expr.lower() or "ret" in expr[:10]:
            categories["Momentum/Reversal"].append(f)
        elif "vol_" in expr or "vol(" in expr or "std" in expr:
            categories["Volatility"].append(f)
        elif "corr" in expr or "cov" in expr:
            categories["Price-Volume Correlation"].append(f)
        elif "rank" in expr and ("close" in expr or "volume" in expr):
            categories["Rank/Relative"].append(f)
        elif "skew" in expr or "kurt" in expr:
            categories["Higher Moments"].append(f)
        elif "hl_" in expr or "vwap" in expr or "intraday" in expr or "body" in expr or "shadow" in expr:
            categories["Microstructure"].append(f)
        elif "decay" in expr:
            categories["Decay-Weighted"].append(f)
        elif "risk_adj" in expr:
            categories["Risk-Adjusted"].append(f)
        elif "resid" in expr or "regression" in expr:
            categories["Regression Residual"].append(f)
        elif any(fund in expr for fund in ["pe", "pb", "bvps", "roe", "eps", "mv", "dv_ttm"]):
            categories["Fundamental"].append(f)
        elif "argmax" in expr or "argmin" in expr:
            categories["Timing (Argmax/Argmin)"].append(f)
        else:
            categories["Formulaic Alpha (Mined)"].append(f)

    w("| Category | Count | Mean |IC| | Best IC |")
    w("|----------|-------|---------|---------|")
    for cat in sorted(categories.keys()):
        fs = categories[cat]
        cnt = len(fs)
        avg_ic = np.mean([abs(f["ic_mean"]) for f in fs]) if fs else 0
        best = max(fs, key=lambda x: abs(x["ic_mean"])) if fs else {"ic_mean": 0}
        w(f"| {cat} | {cnt} | {avg_ic:.4f} | {best['ic_mean']:+.4f} |")
    w()
    w("---")
    w()

    # === Section 3: Beta / Market Exposure ===
    w("## 3. Beta & Market Exposure Analysis")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Strategy Beta (vs equal-weight market) | **{beta:.4f}** |")
    w(f"| Correlation with Market | {corr_mkt:.4f} |")
    w(f"| Market Ann Return | {mkt_ann_ret*100:.2f}% |")
    w(f"| Strategy Ann Return | {bt['annual_return']*100:.2f}% |")
    w(f"| Pure Alpha (annualized) | **{alpha_ann*100:+.2f}%** |")
    w()
    beta_verdict = "market-neutral" if abs(beta) < 0.1 else ("low-beta" if abs(beta) < 0.3 else "has market exposure")
    w(f"**Interpretation**: Strategy is **{beta_verdict}** (beta={beta:.4f}). "
      f"Long-short construction inherently hedges market risk.")
    w()
    w("---")
    w()

    # === Section 4: Return Analysis ===
    w("## 4. Return Analysis")
    w()
    w("### 4.1 Yearly Breakdown")
    w()
    w("| Year | Ann Return | Max DD | Sharpe | Win Rate | Trading Days |")
    w("|------|-----------|--------|--------|----------|-------------|")
    for yr in sorted(yearly.keys()):
        rets = np.array(yearly[yr])
        ann = float((np.prod(1 + rets) ** (252 / len(rets)) - 1) * 100) if len(rets) > 0 else 0
        c = np.cumprod(1 + rets)
        p = np.maximum.accumulate(c)
        mdd = float(np.min((c - p) / (p + 1e-10)) * 100)
        sh = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))
        wr = float(np.mean(rets > 0) * 100)
        w(f"| {yr} | {ann:+.2f}% | {mdd:.2f}% | {sh:.2f} | {wr:.1f}% | {len(rets)} |")
    w()

    w("### 4.2 Monthly Returns (万元)")
    w()
    w("| Month | Return % | PnL (万元) |")
    w("|-------|---------|-----------|")
    for mo in sorted(monthly.keys()):
        rets = np.array(monthly[mo])
        mo_ret = float((np.prod(1 + rets) - 1) * 100)
        mo_pnl = float(NOTIONAL * (np.prod(1 + rets) - 1) / 1e4)
        w(f"| {mo[:4]}-{mo[4:]} | {mo_ret:+.2f}% | {mo_pnl:+,.0f} |")
    w()

    w("### 4.3 Return Distribution")
    w()
    w("| Stat | Value |")
    w("|------|-------|")
    w(f"| Mean Daily Return | {np.mean(valid_ls)*100:+.4f}% |")
    w(f"| Std Daily Return | {np.std(valid_ls)*100:.4f}% |")
    w(f"| Skewness | {float(pd.Series(valid_ls).skew()):.4f} |")
    w(f"| Kurtosis | {float(pd.Series(valid_ls).kurtosis()):.4f} |")
    w(f"| Best Day | {np.max(valid_ls)*100:+.4f}% |")
    w(f"| Worst Day | {np.min(valid_ls)*100:+.4f}% |")
    w(f"| Positive Days | {np.sum(valid_ls>0)} / {n_days} ({np.mean(valid_ls>0)*100:.1f}%) |")
    w()
    w("---")
    w()

    # === Section 5: P&L Analysis ===
    w("## 5. P&L Analysis (10 亿 Initial Capital)")
    w()
    w("### 5.1 Financial Statement")
    w()
    w("| Item | Value |")
    w("|------|-------|")
    w(f"| Initial Capital | {NOTIONAL/1e8:.2f} 亿元 |")
    w(f"| Ending Equity | **{end_equity/1e8:.2f} 亿元** |")
    w(f"| Peak Equity | {peak_equity/1e8:.2f} 亿元 |")
    w(f"| Net Trading P&L | **{cum_pnl[-1]/1e8:+.2f} 亿元** |")
    w(f"| ROI | {cum_pnl[-1]/NOTIONAL*100:+.2f}% |")
    w(f"| Annualized Return | {bt['annual_return']*100:.2f}% |")
    w(f"| Avg Daily P&L | {float(np.nanmean(daily_pnl[valid_mask]))/1e4:+.2f} 万元 |")
    w(f"| Max Daily P&L | {float(np.nanmax(daily_pnl[valid_mask]))/1e4:+.2f} 万元 |")
    w(f"| Min Daily P&L | {float(np.nanmin(daily_pnl[valid_mask]))/1e4:+.2f} 万元 |")
    w()

    w("### 5.2 Cumulative P&L Milestones")
    w()
    w("| Milestone | Trading Day | Date |")
    w("|-----------|------------|------|")
    milestones = [1e8, 5e8, 10e8, 20e8, 30e8, 50e8, 100e8]
    valid_idx = 0
    for t in range(T):
        if not np.isfinite(ls_returns[t]):
            continue
        for m in milestones:
            if cum_pnl[t] >= m and cum_pnl[max(0, t - 1)] < m:
                w(f"| +{m/1e8:.0f} 亿元 | {valid_idx} | {date_strs[t]} |")
        valid_idx += 1
    w()
    w("---")
    w()

    # === Section 6: Drawdown / Block Analysis ===
    w("## 6. Drawdown Block Analysis")
    w()
    w(f"Max drawdown: **{bt['max_drawdown']*100:.2f}%**")
    w()
    w(f"### 6.1 Drawdown Events (depth > 1%)")
    w()
    w(f"Total events: **{len(dd_events)}**")
    w()
    if dd_events:
        w("| # | Depth | Duration (days) | Recovery (days) | Start Day | Trough Day |")
        w("|---|-------|----------------|----------------|-----------|------------|")
        for i, ev in enumerate(dd_events[:20], 1):
            w(f"| {i} | {ev['depth']*100:.2f}% | {ev['duration']} | {ev['recovery']} | {ev['start']} | {ev['trough']} |")
        w()

    # Drawdown statistics
    if dd_events:
        depths = [abs(e["depth"]) for e in dd_events]
        durations = [e["duration"] for e in dd_events]
        recoveries = [e["recovery"] for e in dd_events]
        w("### 6.2 Drawdown Statistics")
        w()
        w("| Stat | Depth | Duration | Recovery |")
        w("|------|-------|----------|----------|")
        w(f"| Mean | {np.mean(depths)*100:.2f}% | {np.mean(durations):.1f} days | {np.mean(recoveries):.1f} days |")
        w(f"| Median | {np.median(depths)*100:.2f}% | {np.median(durations):.1f} days | {np.median(recoveries):.1f} days |")
        w(f"| Max | {np.max(depths)*100:.2f}% | {max(durations)} days | {max(recoveries)} days |")
        w(f"| Count | {len(dd_events)} | | |")
        w()

    # DD-cap triggered days
    zero_days = int(np.sum(ls_returns[valid_mask] == 0))
    w("### 6.3 DD-Cap Trigger Analysis")
    w()
    w(f"- DD-Cap triggered (zero-return) days: **{zero_days}** / {n_days} ({zero_days/n_days*100:.1f}%)")
    w(f"- Strategy active days: **{n_days - zero_days}** / {n_days} ({(n_days-zero_days)/n_days*100:.1f}%)")
    w()
    w("---")
    w()

    # === Charts Index ===
    w("## 7. Chart Index")
    w()
    w("| File | Description |")
    w("|------|-------------|")
    w("| `ic_series.png` | IC time series + 20d rolling mean |")
    w("| `ic_distribution.png` | IC distribution histogram |")
    w("| `cumulative_pnl_rmb.png` | Cumulative P&L (万元) |")
    w("| `daily_pnl_rmb.png` | Daily P&L bars |")
    w("| `drawdown.png` | Drawdown curve |")
    w("| `quintile_returns.png` | Quintile cumulative returns |")
    w("| `equity_curve.png` | Strategy equity curve |")
    w("| `rolling_sharpe_63d.png` | 63-day rolling Sharpe |")
    w("| `rolling_vol_21d.png` | 21-day rolling annualized vol |")
    w("| `monthly_pnl_rmb.png` | Monthly P&L bars |")
    w()

    # Write
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    logger.info(f"Report written: {report_path}")
    logger.info(f"Sections: 0-Executive Summary, 1-Factors, 2-Features, 3-Beta, 4-Returns, 5-P&L, 6-Drawdown, 7-Charts")
    print(f"\n=== DONE: {report_path} ===")


if __name__ == "__main__":
    main()
