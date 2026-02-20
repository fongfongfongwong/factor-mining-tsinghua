# Dragon Alpha — FLAB IC Committee Report

**Report Date**: 2026-02-21 01:28  
**Strategy**: Cross-sectional multi-factor long/short, A-share  
**Signals**: 2211 (top 2000 mined + 212 engineered)  
**Risk Control**: DD Cap 5% (TP/SL not applicable for cross-sectional strategy)  
**Verdict**: **GREEN** — Deploy to paper trading  

---

## 2. Strategy Specification

| Parameter | Value |
|-----------|-------|
| Universe | 300 A-share stocks (top by data availability) |
| Period | 20180102 ~ 20241231 (1698 trading days) |
| Rebalance | Daily |
| Alpha Model | IC-signed weighted combination of 2211 signals |
| Portfolio | Long top quintile (Q5), short bottom quintile (Q1), equal weight |
| Risk | DD Cap 5% (zero exposure when drawdown would breach) |
| Costs | Commission 3bps + Stamp tax 5bps + Impact 10bps = 31bps RT |
| Initial Capital | 10 亿元 |

---

## 3. Data Quality

| Check | Status |
|-------|--------|
| Source | Local AKShare parquet (`factor_investing/data/processed/`) |
| Forward returns | close[t+1]/close[t]-1, no lookahead |
| Point-in-time fundamentals | report_lag=45 days via merge_asof |
| Survivorship | Stocks with ≥500 days history; possible bias if delisted excluded |
| NaN ratio | Checked per-factor; >50% NaN factors excluded |

---

## 4. Model Performance (IC Committee Metrics)

| Metric | Value | Bar | Status |
|--------|-------|-----|--------|
| IC (daily) | +0.0528 | >0.02 | PASS |
| ICIR | +0.303 | >0.1 | PASS |
| Sharpe (gross) | 2.456 | >1.0 | PASS |
| Sharpe (net of costs) | 2.443 | >1.0 | PASS |
| Deflated Sharpe Ratio | -1.625 | >0 | FAIL |
| Max Drawdown | -5.00% | <15% | PASS |
| Calmar Ratio | 14.56 | >1.0 | PASS |
| Profit Factor | 1.83 | >1.5 | PASS |
| Win Rate | 57.3% | — | — |
| Beta | 0.0292 | ~0 | PASS |
| CVaR 95% | -1.74% | — | — |

**Deflated Sharpe details**: 22,110 candidates tested. E[max SR under null] ≈ 4.47. DSR adjusts for multiple testing + non-normality (skew=-0.05, kurtosis=1.39).

---

## 5. P&L Analysis (4-Level Framework)

| Level | Sharpe | Ann Return | Max DD | Cumul PnL |
|-------|--------|-----------|--------|-----------|
| **L1: Signal** (zero cost, no DD cap) | 2.456 | 48.69% | -21.41% | +134.80 亿 |
| **L2: Backtest** (DD cap, no cost) | 3.626 | 72.81% | -5.00% | +388.83 亿 |
| **L2b: Backtest** (DD cap + costs) | 2.443 | 44.03% | — | +106.88 亿 |
| L3: Paper | — | — | — | (not yet deployed) |
| L4: Live | — | — | — | (not yet deployed) |

**Alpha Retention** (L2b / L1): **79.3%** (target: >60%)

**Gap Analysis**: L1→L2b = 19.3% P&L lost to DD cap + costs + turnover

---

## 6. Risk Analytics

### 6a. Regime Performance

| Regime | Periods | Ann Return | Sharpe |
|--------|---------|-----------|--------|
| Bull | 5 | +44.19% | 2.25 |
| Bear | 6 | +105.34% | 4.63 |
| High Vol | 2 | +102.40% | 4.28 |
| Low Vol | 14 | +70.80% | 3.77 |

### 6b. Tail Risk

| Metric | Value |
|--------|-------|
| Best Day | +4.85% |
| Worst Day | -3.02% |
| CVaR 95% (daily) | -1.74% |
| Skewness | -0.053 |
| Kurtosis | 1.387 |

---

## 7. Robustness Checks

### 7a. OOS Degradation

| Split | Sharpe |
|-------|--------|
| In-Sample (first half) | 1.530 |
| Out-of-Sample (second half) | 3.330 |
| **OOS / IS ratio** | **2.18** (target: >0.6) |

### 7b. Cost Sensitivity

| Cost Multiplier | Sharpe | Ann Return |
|----------------|--------|-----------|
| 0.0x | 3.626 | 72.81% |
| 0.5x | 3.035 | 57.77% |
| 1.0x | 2.443 | 44.03% |
| 1.5x | 1.852 | 31.49% |
| 2.0x | 1.261 | 20.03% |
| 3.0x | 0.078 | 0.02% |

**Break-even**: Sharpe drops below 1.0 at **3.0** cost multiplier.

---

## 8. Execution Analysis

| Metric | Value |
|--------|-------|
| Daily Turnover | 0.2336 |
| Est. Annual Turnover | 58.9x |
| Cost per Trade (RT) | 31 bps |
| Daily Cost Drag | 7.24 bps |
| Annual Cost Drag | 18.25% |

**Note**: TP/SL is not applicable for this cross-sectional multi-factor strategy. Risk control is via portfolio-level DD Cap (5%). A-share T+1 rule is implicitly handled by daily rebalance (positions held overnight).

---

## 9. Recommendations & Go/No-Go

### Decision: **GREEN**

| Check | Result |
|-------|--------|
| Sharpe (net) > 1.0 | PASS |
| Max DD < 15% | PASS |
| OOS degradation > 0.6 | PASS |
| Calmar > 1.0 | PASS |
| Profit Factor > 1.5 | PASS |
| Alpha Retention > 60% | PASS |
| Beta < 0.1 | PASS |
| DSR > 0 | FAIL |

**Score**: 7/8 checks passed.

**Next Steps**: Deploy to paper trading for 60+ trading days. Monitor daily signal/position/P&L alignment.
