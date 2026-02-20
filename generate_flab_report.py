#!/usr/bin/env python3
"""
FLAB A-Share R&D — Full IC Committee Report Generator.

Follows the flab-ashare-rd SKILL Phase 5 framework:
- Section 1: Executive Summary (traffic light)
- Section 2: Strategy Spec
- Section 3: Data Quality
- Section 4: Model Performance (IC, ICIR, DSR, factor attribution)
- Section 5: P&L Analysis (4-Level: Signal vs Backtest-with-costs)
- Section 6: Risk Analytics (regime, drawdown, tail, concentration)
- Section 7: Robustness (OOS degradation, cost sensitivity, parameter sensitivity)
- Section 8: Execution Analysis (turnover, cost breakdown)
- Section 9: Recommendations & Go/No-Go

TP/SL is skipped: cross-sectional multi-factor L/S with 2000+ signals and daily
rebalance uses portfolio-level DD Cap, not per-stock TP/SL.

Output: docs/reports/report_dragon_alpha_backtest_YYYY-MM-DD.md
"""

import sys, json, logging, math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flab_report")

from data.stock_data import build_panel_from_parquet, calculate_returns, _PROCESSED_DAILY
from backtest.engine import run_library_backtest, run_factor_backtest, _apply_drawdown_cap
from backtest.metrics import calc_ic_series, calc_icir, calc_sharpe, calc_max_drawdown, calc_annual_return
from evaluate_model import build_signals, SCREENED_FACTORS_PATH

NOTIONAL = 1_000_000_000
MAX_DD = 0.05

# A-share cost model (round-trip)
COMMISSION_BPS = 3      # ~0.03% each way
STAMP_TAX_BPS = 5       # 0.05% on sells only → ~0.025% per trade
IMPACT_BPS = 10          # market impact estimate
TOTAL_ONE_WAY_BPS = COMMISSION_BPS + STAMP_TAX_BPS / 2 + IMPACT_BPS  # ~15.5 bps
TOTAL_ROUND_TRIP_BPS = 2 * TOTAL_ONE_WAY_BPS  # ~31 bps


def main():
    report_dir = Path(__file__).resolve().parent / "docs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = report_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    report_path = report_dir / f"report_dragon_alpha_backtest_{date_tag}.md"

    logger.info("Loading data ...")
    panel = build_panel_from_parquet(max_stocks=300, min_days=500, include_fundamentals=True)
    fwd = calculate_returns(panel)
    M, T = panel["close"].shape
    dates = panel.get("dates", np.arange(T))
    date_strs = [str(d)[:10].replace("-", "") for d in dates]

    logger.info("Building signals ...")
    signals = build_signals(panel, fwd, use_library=True, max_mined=2000)
    N = len(signals)
    logger.info(f"Signals: {N}")

    # ============================================================
    # LEVEL 1: Signal P&L (zero costs, no DD cap)
    # ============================================================
    logger.info("Level 1: Signal P&L (zero cost, no risk control) ...")
    bt_signal = run_library_backtest(signals, fwd, method="ic_weighted")

    # ============================================================
    # LEVEL 2: Backtest P&L (with A-share costs + DD cap)
    # ============================================================
    logger.info("Level 2: Backtest P&L (with costs + DD cap 5%) ...")
    bt_backtest = run_library_backtest(signals, fwd, method="ic_weighted", max_dd_target=MAX_DD)

    ls_sig = np.array(bt_signal["ls_returns"])
    ls_bt = np.array(bt_backtest["ls_returns"])
    valid_sig = ls_sig[np.isfinite(ls_sig)]
    valid_bt = ls_bt[np.isfinite(ls_bt)]
    n_days = len(valid_bt)

    # Apply transaction costs to Level 2
    turnover = bt_backtest["turnover"]
    daily_cost = turnover * TOTAL_ROUND_TRIP_BPS / 10000
    ls_bt_net = np.where(np.isfinite(ls_bt), ls_bt - daily_cost, ls_bt)
    valid_bt_net = ls_bt_net[np.isfinite(ls_bt_net)]

    # ============================================================
    # IC / ICIR (combined signal)
    # ============================================================
    logger.info("Computing combined IC series ...")
    from backtest.mlx_accel import mlx_ic_weighted_combine, HAS_MLX
    if HAS_MLX and N > 50:
        combined_sig = mlx_ic_weighted_combine(signals, fwd)
    else:
        names = list(signals.keys())
        ic_map = {n: calc_ic_series(signals[n], fwd) for n in names}
        combined_sig = np.zeros((M, T))
        for n in names:
            ic_v = ic_map[n][~np.isnan(ic_map[n])]
            m = float(np.mean(ic_v)) if len(ic_v) > 0 else 0.0
            w = abs(m)
            s = 1.0 if m >= 0 else -1.0
            combined_sig += w * s * np.nan_to_num(signals[n])

    ic_series = calc_ic_series(combined_sig, fwd)
    valid_ic = ic_series[~np.isnan(ic_series)]
    ic_mean = float(np.mean(valid_ic))
    ic_std = float(np.std(valid_ic))
    icir = ic_mean / (ic_std + 1e-10)

    # ============================================================
    # Deflated Sharpe Ratio (Bailey et al.)
    # ============================================================
    n_trials = 22110  # candidates tested
    sharpe_raw = bt_signal["sharpe"]
    T_obs = len(valid_sig)
    skew = float(pd.Series(valid_sig).skew())
    kurt = float(pd.Series(valid_sig).kurtosis())
    # DSR approximation: SR* = SR * sqrt(1 - skew*SR/3 + (kurt-1)*SR^2/12)
    sr = sharpe_raw / np.sqrt(252)  # daily
    sr_adj = sr * np.sqrt(max(0.01, 1 - skew * sr / 3 + (kurt - 1) * sr ** 2 / 12))
    # E[max SR] under null ≈ sqrt(2 * ln(n_trials)) for iid normal
    e_max_sr = np.sqrt(2 * np.log(max(n_trials, 2))) / np.sqrt(252) * np.sqrt(252)
    dsr = (sharpe_raw - e_max_sr) / max(0.01, np.sqrt(1 + (skew * sharpe_raw) / 3 + ((kurt - 1) / 4) * sharpe_raw ** 2))

    # ============================================================
    # OOS Degradation
    # ============================================================
    split = T // 2
    is_ret = valid_sig[:split] if split < len(valid_sig) else valid_sig[:len(valid_sig)//2]
    oos_ret = valid_sig[split:] if split < len(valid_sig) else valid_sig[len(valid_sig)//2:]
    sharpe_is = float(np.mean(is_ret) / (np.std(is_ret) + 1e-10) * np.sqrt(252))
    sharpe_oos = float(np.mean(oos_ret) / (np.std(oos_ret) + 1e-10) * np.sqrt(252))
    oos_degradation = sharpe_oos / (sharpe_is + 1e-10)

    # ============================================================
    # Cost Sensitivity
    # ============================================================
    cost_results = []
    for mult in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        cost = turnover * TOTAL_ROUND_TRIP_BPS * mult / 10000
        ret_net = valid_bt - cost
        sh = float(np.mean(ret_net) / (np.std(ret_net) + 1e-10) * np.sqrt(252))
        ann = float((np.prod(1 + ret_net) ** (252 / len(ret_net)) - 1))
        cost_results.append({"mult": mult, "sharpe": sh, "ann_ret": ann})

    # ============================================================
    # Regime Analysis
    # ============================================================
    market_ret = np.nanmean(panel["returns"], axis=0)
    # Simple regime: bull/bear/high-vol/low-vol by rolling 60d
    regime_results = {}
    for t_start in range(60, T, 60):
        t_end = min(t_start + 60, T)
        mkt_window = market_ret[t_start:t_end]
        strat_window = ls_bt[t_start:t_end]
        mkt_w = mkt_window[np.isfinite(mkt_window)]
        strat_w = strat_window[np.isfinite(strat_window)]
        if len(mkt_w) < 20 or len(strat_w) < 20:
            continue
        mkt_cum = float(np.prod(1 + mkt_w) - 1)
        mkt_vol = float(np.std(mkt_w) * np.sqrt(252))
        if mkt_cum > 0.05:
            regime = "Bull"
        elif mkt_cum < -0.05:
            regime = "Bear"
        elif mkt_vol > 0.25:
            regime = "High Vol"
        else:
            regime = "Low Vol"
        if regime not in regime_results:
            regime_results[regime] = {"returns": [], "count": 0}
        regime_results[regime]["returns"].extend(strat_w.tolist())
        regime_results[regime]["count"] += 1

    # ============================================================
    # Beta / factor attribution
    # ============================================================
    valid_both = np.isfinite(ls_bt) & np.isfinite(market_ret)
    ls_v = ls_bt[valid_both]
    mkt_v = market_ret[valid_both]
    if len(ls_v) > 50:
        cov = np.cov(ls_v, mkt_v)
        beta = cov[0, 1] / (cov[1, 1] + 1e-10)
        corr_mkt = np.corrcoef(ls_v, mkt_v)[0, 1]
    else:
        beta = corr_mkt = 0.0

    # ============================================================
    # Calmar, profit factor, tail risk
    # ============================================================
    cum_bt = np.cumprod(1 + valid_bt)
    peak_bt = np.maximum.accumulate(cum_bt)
    dd_bt = (cum_bt - peak_bt) / (peak_bt + 1e-10)
    max_dd = float(np.min(dd_bt))
    calmar = bt_backtest["annual_return"] / (abs(max_dd) + 1e-10)
    gross_profit = float(np.sum(valid_bt[valid_bt > 0]))
    gross_loss = float(abs(np.sum(valid_bt[valid_bt < 0])))
    profit_factor = gross_profit / (gross_loss + 1e-10)
    # CVaR 95%
    sorted_ret = np.sort(valid_bt)
    cvar_idx = int(len(sorted_ret) * 0.05)
    cvar_95 = float(np.mean(sorted_ret[:max(cvar_idx, 1)]))

    # ============================================================
    # Build Report
    # ============================================================
    logger.info("Writing FLAB report ...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    def w(s=""): L.append(s)

    # --- Section 1: Executive Summary ---
    sharpe_bt = bt_backtest["sharpe"]
    ann_ret_bt = bt_backtest["annual_return"]
    sharpe_net = float(np.mean(valid_bt_net) / (np.std(valid_bt_net) + 1e-10) * np.sqrt(252))
    ann_ret_net = float((np.prod(1 + valid_bt_net) ** (252 / len(valid_bt_net)) - 1))

    # Traffic light
    if sharpe_net > 2.0 and abs(max_dd) < 0.06 and oos_degradation > 0.5:
        traffic = "GREEN"
        verdict = "Deploy to paper trading"
    elif sharpe_net > 1.0 and abs(max_dd) < 0.10:
        traffic = "YELLOW"
        verdict = "Fix simulation issues, then re-evaluate"
    else:
        traffic = "RED"
        verdict = "Does not meet the bar — archive or pivot"

    w(f"# Dragon Alpha — FLAB IC Committee Report")
    w()
    w(f"**Report Date**: {now}  ")
    w(f"**Strategy**: Cross-sectional multi-factor long/short, A-share  ")
    w(f"**Signals**: {N} (top 2000 mined + 212 engineered)  ")
    w(f"**Risk Control**: DD Cap {MAX_DD*100:.0f}% (TP/SL not applicable for cross-sectional strategy)  ")
    w(f"**Verdict**: **{traffic}** — {verdict}  ")
    w()
    w("---")
    w()

    # --- Section 2: Strategy Spec ---
    w("## 2. Strategy Specification")
    w()
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w(f"| Universe | {M} A-share stocks (top by data availability) |")
    w(f"| Period | {date_strs[0]} ~ {date_strs[-1]} ({n_days} trading days) |")
    w(f"| Rebalance | Daily |")
    w(f"| Alpha Model | IC-signed weighted combination of {N} signals |")
    w(f"| Portfolio | Long top quintile (Q5), short bottom quintile (Q1), equal weight |")
    w(f"| Risk | DD Cap {MAX_DD*100:.0f}% (zero exposure when drawdown would breach) |")
    w(f"| Costs | Commission {COMMISSION_BPS}bps + Stamp tax {STAMP_TAX_BPS}bps + Impact {IMPACT_BPS}bps = {TOTAL_ROUND_TRIP_BPS:.0f}bps RT |")
    w(f"| Initial Capital | {NOTIONAL/1e8:.0f} 亿元 |")
    w()
    w("---")
    w()

    # --- Section 3: Data Quality ---
    w("## 3. Data Quality")
    w()
    w("| Check | Status |")
    w("|-------|--------|")
    w(f"| Source | Local AKShare parquet (`factor_investing/data/processed/`) |")
    w(f"| Forward returns | close[t+1]/close[t]-1, no lookahead |")
    w(f"| Point-in-time fundamentals | report_lag=45 days via merge_asof |")
    w(f"| Survivorship | Stocks with ≥500 days history; possible bias if delisted excluded |")
    w(f"| NaN ratio | Checked per-factor; >50% NaN factors excluded |")
    w()
    w("---")
    w()

    # --- Section 4: Model Performance ---
    w("## 4. Model Performance (IC Committee Metrics)")
    w()
    w("| Metric | Value | Bar | Status |")
    w("|--------|-------|-----|--------|")
    w(f"| IC (daily) | {ic_mean:+.4f} | >0.02 | {'PASS' if ic_mean > 0.02 else 'FAIL'} |")
    w(f"| ICIR | {icir:+.3f} | >0.1 | {'PASS' if icir > 0.1 else 'FAIL'} |")
    w(f"| Sharpe (gross) | {sharpe_raw:.3f} | >1.0 | {'PASS' if sharpe_raw > 1.0 else 'FAIL'} |")
    w(f"| Sharpe (net of costs) | {sharpe_net:.3f} | >1.0 | {'PASS' if sharpe_net > 1.0 else 'FAIL'} |")
    w(f"| Deflated Sharpe Ratio | {dsr:.3f} | >0 | {'PASS' if dsr > 0 else 'FAIL'} |")
    w(f"| Max Drawdown | {max_dd*100:.2f}% | <15% | {'PASS' if abs(max_dd) < 0.15 else 'FAIL'} |")
    w(f"| Calmar Ratio | {calmar:.2f} | >1.0 | {'PASS' if calmar > 1.0 else 'FAIL'} |")
    w(f"| Profit Factor | {profit_factor:.2f} | >1.5 | {'PASS' if profit_factor > 1.5 else 'FAIL'} |")
    w(f"| Win Rate | {bt_backtest['win_ratio']*100:.1f}% | — | — |")
    w(f"| Beta | {beta:.4f} | ~0 | {'PASS' if abs(beta) < 0.1 else 'WARN'} |")
    w(f"| CVaR 95% | {cvar_95*100:.2f}% | — | — |")
    w()
    w(f"**Deflated Sharpe details**: {n_trials:,} candidates tested. E[max SR under null] ≈ {e_max_sr:.2f}. "
      f"DSR adjusts for multiple testing + non-normality (skew={skew:.2f}, kurtosis={kurt:.2f}).")
    w()
    w("---")
    w()

    # --- Section 5: P&L Analysis (4-Level) ---
    w("## 5. P&L Analysis (4-Level Framework)")
    w()
    cum_sig = np.cumprod(1 + valid_sig)
    cum_bt_arr = np.cumprod(1 + valid_bt)
    cum_net = np.cumprod(1 + valid_bt_net)

    w("| Level | Sharpe | Ann Return | Max DD | Cumul PnL |")
    w("|-------|--------|-----------|--------|-----------|")
    w(f"| **L1: Signal** (zero cost, no DD cap) | {bt_signal['sharpe']:.3f} | {bt_signal['annual_return']*100:.2f}% | {bt_signal['max_drawdown']*100:.2f}% | {NOTIONAL*(cum_sig[-1]-1)/1e8:+.2f} 亿 |")
    w(f"| **L2: Backtest** (DD cap, no cost) | {bt_backtest['sharpe']:.3f} | {bt_backtest['annual_return']*100:.2f}% | {bt_backtest['max_drawdown']*100:.2f}% | {NOTIONAL*(cum_bt_arr[-1]-1)/1e8:+.2f} 亿 |")
    w(f"| **L2b: Backtest** (DD cap + costs) | {sharpe_net:.3f} | {ann_ret_net*100:.2f}% | — | {NOTIONAL*(cum_net[-1]-1)/1e8:+.2f} 亿 |")
    w(f"| L3: Paper | — | — | — | (not yet deployed) |")
    w(f"| L4: Live | — | — | — | (not yet deployed) |")
    w()
    l1_pnl = NOTIONAL * (cum_sig[-1] - 1)
    l2_pnl = NOTIONAL * (cum_net[-1] - 1)
    retention = l2_pnl / (l1_pnl + 1e-10) * 100
    w(f"**Alpha Retention** (L2b / L1): **{retention:.1f}%** (target: >60%)")
    w()

    gap_12 = (1 - cum_net[-1] / cum_sig[-1]) * 100
    w(f"**Gap Analysis**: L1→L2b = {gap_12:.1f}% P&L lost to DD cap + costs + turnover")
    w()
    w("---")
    w()

    # --- Section 6: Risk Analytics ---
    w("## 6. Risk Analytics")
    w()
    w("### 6a. Regime Performance")
    w()
    w("| Regime | Periods | Ann Return | Sharpe |")
    w("|--------|---------|-----------|--------|")
    for regime in ["Bull", "Bear", "High Vol", "Low Vol"]:
        if regime in regime_results:
            rets = np.array(regime_results[regime]["returns"])
            ann = float((np.prod(1 + rets) ** (252 / len(rets)) - 1) * 100)
            sh = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))
            w(f"| {regime} | {regime_results[regime]['count']} | {ann:+.2f}% | {sh:.2f} |")
        else:
            w(f"| {regime} | 0 | — | — |")
    w()

    w("### 6b. Tail Risk")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Best Day | {np.max(valid_bt)*100:+.2f}% |")
    w(f"| Worst Day | {np.min(valid_bt)*100:+.2f}% |")
    w(f"| CVaR 95% (daily) | {cvar_95*100:.2f}% |")
    w(f"| Skewness | {skew:.3f} |")
    w(f"| Kurtosis | {kurt:.3f} |")
    w()
    w("---")
    w()

    # --- Section 7: Robustness ---
    w("## 7. Robustness Checks")
    w()
    w("### 7a. OOS Degradation")
    w()
    w(f"| Split | Sharpe |")
    w(f"|-------|--------|")
    w(f"| In-Sample (first half) | {sharpe_is:.3f} |")
    w(f"| Out-of-Sample (second half) | {sharpe_oos:.3f} |")
    w(f"| **OOS / IS ratio** | **{oos_degradation:.2f}** (target: >0.6) |")
    w()

    w("### 7b. Cost Sensitivity")
    w()
    w("| Cost Multiplier | Sharpe | Ann Return |")
    w("|----------------|--------|-----------|")
    for cr in cost_results:
        w(f"| {cr['mult']:.1f}x | {cr['sharpe']:.3f} | {cr['ann_ret']*100:.2f}% |")
    break_even = next((cr["mult"] for cr in cost_results if cr["sharpe"] < 1.0), ">3.0x")
    w()
    w(f"**Break-even**: Sharpe drops below 1.0 at **{break_even}** cost multiplier.")
    w()
    w("---")
    w()

    # --- Section 8: Execution Analysis ---
    w("## 8. Execution Analysis")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Daily Turnover | {turnover:.4f} |")
    w(f"| Est. Annual Turnover | {turnover*252:.1f}x |")
    w(f"| Cost per Trade (RT) | {TOTAL_ROUND_TRIP_BPS:.0f} bps |")
    w(f"| Daily Cost Drag | {daily_cost*10000:.2f} bps |")
    w(f"| Annual Cost Drag | {daily_cost*252*100:.2f}% |")
    w()
    w("**Note**: TP/SL is not applicable for this cross-sectional multi-factor strategy. "
      "Risk control is via portfolio-level DD Cap (5%). "
      "A-share T+1 rule is implicitly handled by daily rebalance (positions held overnight).")
    w()
    w("---")
    w()

    # --- Section 9: Recommendations ---
    w("## 9. Recommendations & Go/No-Go")
    w()
    w(f"### Decision: **{traffic}**")
    w()
    checks = [
        ("Sharpe (net) > 1.0", sharpe_net > 1.0),
        ("Max DD < 15%", abs(max_dd) < 0.15),
        ("OOS degradation > 0.6", oos_degradation > 0.6),
        ("Calmar > 1.0", calmar > 1.0),
        ("Profit Factor > 1.5", profit_factor > 1.5),
        ("Alpha Retention > 60%", retention > 60),
        ("Beta < 0.1", abs(beta) < 0.1),
        ("DSR > 0", dsr > 0),
    ]
    w("| Check | Result |")
    w("|-------|--------|")
    for name, passed in checks:
        w(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    n_pass = sum(1 for _, p in checks if p)
    w()
    w(f"**Score**: {n_pass}/{len(checks)} checks passed.")
    w()

    if traffic == "GREEN":
        w("**Next Steps**: Deploy to paper trading for 60+ trading days. Monitor daily signal/position/P&L alignment.")
    elif traffic == "YELLOW":
        w("**Next Steps**: Address FAIL items above. Re-run backtest after fixes. Focus on cost sensitivity and OOS stability.")
    else:
        w("**Next Steps**: Research new signal sources or model architectures. Current approach does not generate sufficient risk-adjusted alpha after costs.")
    w()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    logger.info(f"FLAB report: {report_path}")
    logger.info(f"Verdict: {traffic} — {n_pass}/{len(checks)} checks passed")
    print(f"\n=== {report_path} ===")


if __name__ == "__main__":
    main()
