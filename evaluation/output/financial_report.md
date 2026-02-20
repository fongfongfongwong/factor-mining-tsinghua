# Dragon Alpha — Comprehensive Financial Report

**Generated**: 2026-02-20 08:38  
**Initial Capital**: 10.00 亿元 (RMB 1,000,000,000)  
**Universe**: 300 stocks x 1698 trading days (20180102~20241231)  
**Full Database**: 0 rows | 0 stocks | ?~?  
**Total Signals**: 2211  
**Risk Control**: DD Cap 5% (no vol targeting)  

---

## 0. Executive Summary

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **3.625** |
| Annualized Return | **72.79%** |
| Annualized Volatility | 15.43% |
| Max Drawdown | -5.00% |
| Win Rate | 57.3% |
| Turnover | 0.2336 |
| Beta (vs market) | 0.0292 |
| Alpha (annualized) | +72.68% |
| Cumulative PnL | **+388.50 亿元** |
| Ending Equity | **398.50 亿元** |
| Signals Used | 2211 |

---

## 1. Factor Analysis (All Mined Factors)

### 1.1 Mined Factor Universe

- **Candidates generated**: 22,110 (depth-3 formulaic alpha)
- **Passed IC > 0.005**: 19,618
- **Top 2000 by |IC| used in model**

| Stat | Value |
|------|-------|
| Total mined factors | 19,618 |
| Positive IC | 5,315 |
| Negative IC | 14,303 |
| Mean \|IC\| | 0.0251 |
| Max IC | +0.0663 |
| Min IC | -0.0802 |
| Median \|IC\| | 0.0207 |

### 1.2 Top 50 Factors by |IC| (Used in Model)

| # | Expression | IC Mean | ICIR |
|---|-----------|---------|------|
| 1 | `cs_rank(ts_decay_linear(div(close, low), 3))` | -0.0802 | -0.558 |
| 2 | `ts_decay_linear(div(close, low), 3)` | -0.0802 | -0.558 |
| 3 | `cs_rank(ts_decay_linear(div(close, low), 5))` | -0.0785 | -0.528 |
| 4 | `ts_decay_linear(div(close, low), 5)` | -0.0785 | -0.528 |
| 5 | `cs_rank(ts_product(div(close, low), 3))` | -0.0746 | -0.516 |
| 6 | `ts_product(div(close, low), 3)` | -0.0746 | -0.516 |
| 7 | `cs_rank(ts_mean(div(close, low), 3))` | -0.0745 | -0.516 |
| 8 | `cs_rank(ts_sum(div(close, low), 3))` | -0.0745 | -0.516 |
| 9 | `ts_mean(div(close, low), 3)` | -0.0745 | -0.516 |
| 10 | `ts_sum(div(close, low), 3)` | -0.0745 | -0.516 |
| 11 | `cs_rank(ts_decay_linear(div(close, low), 10))` | -0.0732 | -0.462 |
| 12 | `ts_decay_linear(div(close, low), 10)` | -0.0732 | -0.462 |
| 13 | `cs_rank(ts_product(div(close, low), 5))` | -0.0721 | -0.476 |
| 14 | `ts_product(div(close, low), 5)` | -0.0721 | -0.476 |
| 15 | `cs_rank(ts_mean(div(close, low), 5))` | -0.0719 | -0.475 |
| 16 | `cs_rank(ts_sum(div(close, low), 5))` | -0.0719 | -0.475 |
| 17 | `ts_mean(div(close, low), 5)` | -0.0719 | -0.475 |
| 18 | `ts_sum(div(close, low), 5)` | -0.0719 | -0.475 |
| 19 | `cs_rank(ts_decay_linear(div(high, low), 3))` | -0.0710 | -0.430 |
| 20 | `ts_decay_linear(div(high, low), 3)` | -0.0710 | -0.430 |
| 21 | `cs_rank(ts_decay_linear(div(high, vwap), 3))` | -0.0708 | -0.428 |
| 22 | `ts_decay_linear(div(high, vwap), 3)` | -0.0708 | -0.428 |
| 23 | `cs_rank(ts_decay_linear(div(high, low), 5))` | -0.0702 | -0.409 |
| 24 | `ts_decay_linear(div(high, low), 5)` | -0.0702 | -0.409 |
| 25 | `cs_rank(ts_decay_linear(div(high, vwap), 5))` | -0.0694 | -0.401 |
| 26 | `ts_decay_linear(div(high, vwap), 5)` | -0.0694 | -0.401 |
| 27 | `cs_rank(ts_mean(div(high, low), 3))` | -0.0686 | -0.410 |
| 28 | `cs_rank(ts_sum(div(high, low), 3))` | -0.0686 | -0.410 |
| 29 | `ts_mean(div(high, low), 3)` | -0.0686 | -0.410 |
| 30 | `ts_sum(div(high, low), 3)` | -0.0686 | -0.410 |
| 31 | `cs_rank(ts_product(div(high, low), 3))` | -0.0685 | -0.410 |
| 32 | `ts_product(div(high, low), 3)` | -0.0685 | -0.410 |
| 33 | `hl_range_ma_3` | -0.0682 | -0.406 |
| 34 | `cs_rank_hl_ma_3` | -0.0682 | -0.406 |
| 35 | `cs_rank(ts_mean(div(high, vwap), 3))` | -0.0678 | -0.404 |
| 36 | `cs_rank(ts_sum(div(high, vwap), 3))` | -0.0678 | -0.404 |
| 37 | `ts_mean(div(high, vwap), 3)` | -0.0678 | -0.404 |
| 38 | `ts_sum(div(high, vwap), 3)` | -0.0678 | -0.404 |
| 39 | `cs_rank(ts_product(div(high, vwap), 3))` | -0.0677 | -0.404 |
| 40 | `ts_product(div(high, vwap), 3)` | -0.0677 | -0.404 |
| 41 | `cs_rank(ts_decay_linear(div(close, low), 20))` | -0.0674 | -0.398 |
| 42 | `ts_decay_linear(div(close, low), 20)` | -0.0674 | -0.398 |
| 43 | `ts_decay_linear(div(high, open), 10)` | -0.0667 | -0.429 |
| 44 | `cs_rank(ts_max(div(close, low), 3))` | -0.0666 | -0.477 |
| 45 | `ts_max(div(close, low), 3)` | -0.0666 | -0.477 |
| 46 | `cs_rank(ts_max(div(high, low), 3))` | -0.0665 | -0.426 |
| 47 | `ts_max(div(high, low), 3)` | -0.0665 | -0.426 |
| 48 | `cs_rank(ts_mean(div(high, low), 5))` | -0.0665 | -0.381 |
| 49 | `cs_rank(ts_sum(div(high, low), 5))` | -0.0665 | -0.381 |
| 50 | `ts_mean(div(high, low), 5)` | -0.0665 | -0.381 |

### 1.3 Factor IC Summary (Model Signals)

- Total signals in model: **2211**
- Positive IC: 351 (15.9%)
- Negative IC: 1850 (83.7%)
- Average |IC|: 0.0523

---

## 2. Feature / Signal Analysis

### 2.1 Signal Categories

| Category | Count | Mean |IC| | Best IC |
|----------|-------|---------|---------|
| Decay-Weighted | 198 | 0.0541 | -0.0802 |
| Formulaic Alpha (Mined) | 666 | 0.0528 | -0.0746 |
| Fundamental | 86 | 0.0507 | -0.0649 |
| Higher Moments | 54 | 0.0442 | +0.0637 |
| Microstructure | 210 | 0.0552 | -0.0708 |
| Momentum/Reversal | 17 | 0.0435 | -0.0525 |
| Price-Volume Correlation | 209 | 0.0507 | -0.0644 |
| Rank/Relative | 272 | 0.0541 | -0.0802 |
| Regression Residual | 4 | 0.0083 | -0.0089 |
| Risk-Adjusted | 8 | 0.0379 | -0.0398 |
| Timing (Argmax/Argmin) | 8 | 0.0214 | +0.0284 |
| Volatility | 469 | 0.0520 | -0.0631 |

---

## 3. Beta & Market Exposure Analysis

| Metric | Value |
|--------|-------|
| Strategy Beta (vs equal-weight market) | **0.0292** |
| Correlation with Market | 0.0431 |
| Market Ann Return | 3.69% |
| Strategy Ann Return | 72.79% |
| Pure Alpha (annualized) | **+72.68%** |

**Interpretation**: Strategy is **market-neutral** (beta=0.0292). Long-short construction inherently hedges market risk.

---

## 4. Return Analysis

### 4.1 Yearly Breakdown

| Year | Ann Return | Max DD | Sharpe | Win Rate | Trading Days |
|------|-----------|--------|--------|----------|-------------|
| 2018 | +54.55% | -4.77% | 3.27 | 60.5% | 243 |
| 2019 | +26.41% | -4.94% | 2.06 | 52.0% | 244 |
| 2020 | +42.69% | -4.79% | 2.32 | 49.0% | 243 |
| 2021 | +103.56% | -4.99% | 3.66 | 52.7% | 243 |
| 2022 | +134.89% | -3.84% | 5.59 | 63.2% | 242 |
| 2023 | +82.05% | -4.80% | 5.45 | 64.0% | 242 |
| 2024 | +90.18% | -4.29% | 3.71 | 59.8% | 241 |

### 4.2 Monthly Returns (万元)

| Month | Return % | PnL (万元) |
|-------|---------|-----------|
| 2018-01 | -3.21% | -3,206 |
| 2018-02 | +0.60% | +603 |
| 2018-03 | +6.73% | +6,727 |
| 2018-04 | +0.96% | +959 |
| 2018-05 | +3.16% | +3,161 |
| 2018-06 | -0.44% | -444 |
| 2018-07 | +8.01% | +8,011 |
| 2018-08 | +5.31% | +5,308 |
| 2018-09 | +4.58% | +4,585 |
| 2018-10 | +2.94% | +2,939 |
| 2018-11 | +9.18% | +9,175 |
| 2018-12 | +5.62% | +5,625 |
| 2019-01 | +1.23% | +1,225 |
| 2019-02 | -0.97% | -971 |
| 2019-03 | +6.68% | +6,678 |
| 2019-04 | +10.18% | +10,181 |
| 2019-05 | +2.91% | +2,907 |
| 2019-06 | +2.64% | +2,644 |
| 2019-07 | +3.25% | +3,250 |
| 2019-08 | -2.50% | -2,497 |
| 2019-09 | +2.35% | +2,353 |
| 2019-10 | -0.27% | -275 |
| 2019-11 | +1.25% | +1,245 |
| 2019-12 | -3.09% | -3,092 |
| 2020-01 | +0.02% | +20 |
| 2020-02 | +6.03% | +6,029 |
| 2020-03 | +13.67% | +13,668 |
| 2020-04 | -1.38% | -1,382 |
| 2020-05 | +1.84% | +1,844 |
| 2020-06 | -2.15% | -2,155 |
| 2020-07 | +4.65% | +4,653 |
| 2020-08 | +7.48% | +7,482 |
| 2020-09 | +3.93% | +3,925 |
| 2020-10 | +0.22% | +219 |
| 2020-11 | +3.73% | +3,727 |
| 2020-12 | -2.13% | -2,130 |
| 2021-01 | +5.69% | +5,690 |
| 2021-02 | +4.78% | +4,776 |
| 2021-03 | +13.67% | +13,670 |
| 2021-04 | +3.12% | +3,121 |
| 2021-05 | -0.93% | -927 |
| 2021-06 | +4.06% | +4,065 |
| 2021-07 | -1.51% | -1,514 |
| 2021-08 | +4.08% | +4,079 |
| 2021-09 | +13.27% | +13,269 |
| 2021-10 | -0.19% | -185 |
| 2021-11 | +7.44% | +7,444 |
| 2021-12 | +19.10% | +19,096 |
| 2022-01 | +11.91% | +11,914 |
| 2022-02 | +4.47% | +4,468 |
| 2022-03 | +9.21% | +9,213 |
| 2022-04 | +6.58% | +6,580 |
| 2022-05 | +8.55% | +8,547 |
| 2022-06 | +0.53% | +532 |
| 2022-07 | +5.41% | +5,412 |
| 2022-08 | +17.44% | +17,444 |
| 2022-09 | +11.25% | +11,246 |
| 2022-10 | +6.82% | +6,815 |
| 2022-11 | +4.34% | +4,337 |
| 2022-12 | -0.38% | -382 |
| 2023-01 | +4.48% | +4,478 |
| 2023-02 | +3.95% | +3,952 |
| 2023-03 | -1.36% | -1,358 |
| 2023-04 | +5.69% | +5,692 |
| 2023-05 | +2.57% | +2,572 |
| 2023-06 | +2.78% | +2,779 |
| 2023-07 | +10.78% | +10,780 |
| 2023-08 | +5.29% | +5,289 |
| 2023-09 | +3.95% | +3,949 |
| 2023-10 | +8.04% | +8,044 |
| 2023-11 | +4.60% | +4,598 |
| 2023-12 | +8.69% | +8,686 |
| 2024-01 | +2.72% | +2,718 |
| 2024-02 | +8.13% | +8,125 |
| 2024-03 | +7.07% | +7,066 |
| 2024-04 | +8.07% | +8,075 |
| 2024-05 | -0.68% | -681 |
| 2024-06 | +4.77% | +4,767 |
| 2024-07 | +4.79% | +4,790 |
| 2024-08 | +7.45% | +7,448 |
| 2024-09 | -0.17% | -168 |
| 2024-10 | +2.33% | +2,333 |
| 2024-11 | +13.33% | +13,333 |
| 2024-12 | +6.07% | +6,074 |

### 4.3 Return Distribution

| Stat | Value |
|------|-------|
| Mean Daily Return | +0.2220% |
| Std Daily Return | 0.9719% |
| Skewness | 0.3174 |
| Kurtosis | 0.8143 |
| Best Day | +4.8502% |
| Worst Day | -3.0231% |
| Positive Days | 973 / 1698 (57.3%) |

---

## 5. P&L Analysis (10 亿 Initial Capital)

### 5.1 Financial Statement

| Item | Value |
|------|-------|
| Initial Capital | 10.00 亿元 |
| Ending Equity | **398.50 亿元** |
| Peak Equity | 398.50 亿元 |
| Net Trading P&L | **+388.50 亿元** |
| ROI | +3885.00% |
| Annualized Return | 72.79% |
| Avg Daily P&L | +221.97 万元 |
| Max Daily P&L | +4850.23 万元 |
| Min Daily P&L | -3023.09 万元 |

### 5.2 Cumulative P&L Milestones

| Milestone | Trading Day | Date |
|-----------|------------|------|
| +1 亿元 | 137 | 20180726 |
| +5 亿元 | 232 | 20181214 |
| +5 亿元 | 237 | 20181221 |
| +5 亿元 | 241 | 20181227 |
| +5 亿元 | 277 | 20190226 |
| +10 亿元 | 521 | 20200227 |
| +20 亿元 | 764 | 20210226 |
| +20 亿元 | 768 | 20210304 |
| +30 亿元 | 908 | 20210927 |
| +30 亿元 | 910 | 20210929 |
| +50 亿元 | 993 | 20220208 |
| +50 亿元 | 996 | 20220211 |
| +100 亿元 | 1157 | 20221012 |

---

## 6. Drawdown Block Analysis

Max drawdown: **-5.00%**

### 6.1 Drawdown Events (depth > 1%)

Total events: **115**

| # | Depth | Duration (days) | Recovery (days) | Start Day | Trough Day |
|---|-------|----------------|----------------|-----------|------------|
| 1 | -5.00% | 20 | 7 | 718 | 731 |
| 2 | -4.99% | 14 | 9 | 917 | 922 |
| 3 | -4.94% | 52 | 35 | 378 | 395 |
| 4 | -4.93% | 81 | 44 | 439 | 476 |
| 5 | -4.84% | 33 | 6 | 798 | 825 |
| 6 | -4.81% | 26 | 24 | 867 | 869 |
| 7 | -4.80% | 21 | 7 | 1308 | 1322 |
| 8 | -4.79% | 44 | 31 | 654 | 667 |
| 9 | -4.78% | 20 | 12 | 594 | 602 |
| 10 | -4.77% | 7 | 3 | 20 | 24 |
| 11 | -4.70% | 36 | 9 | 545 | 572 |
| 12 | -4.69% | 11 | 4 | 185 | 192 |
| 13 | -4.62% | 28 | 3 | 837 | 862 |
| 14 | -4.41% | 9 | 7 | 940 | 942 |
| 15 | -4.29% | 12 | 8 | 1491 | 1495 |
| 16 | -4.27% | 24 | 15 | 1266 | 1275 |
| 17 | -4.24% | 16 | 4 | 1544 | 1556 |
| 18 | -4.20% | 9 | 7 | 1634 | 1636 |
| 19 | -4.08% | 16 | 6 | 100 | 110 |
| 20 | -4.04% | 6 | 4 | 1468 | 1470 |

### 6.2 Drawdown Statistics

| Stat | Depth | Duration | Recovery |
|------|-------|----------|----------|
| Mean | 2.39% | 7.8 days | 4.6 days |
| Median | 1.94% | 4.0 days | 3.0 days |
| Max | 5.00% | 81 days | 44 days |
| Count | 115 | | |

### 6.3 DD-Cap Trigger Analysis

- DD-Cap triggered (zero-return) days: **60** / 1698 (3.5%)
- Strategy active days: **1638** / 1698 (96.5%)

---

## 7. Chart Index

| File | Description |
|------|-------------|
| `ic_series.png` | IC time series + 20d rolling mean |
| `ic_distribution.png` | IC distribution histogram |
| `cumulative_pnl_rmb.png` | Cumulative P&L (万元) |
| `daily_pnl_rmb.png` | Daily P&L bars |
| `drawdown.png` | Drawdown curve |
| `quintile_returns.png` | Quintile cumulative returns |
| `equity_curve.png` | Strategy equity curve |
| `rolling_sharpe_63d.png` | 63-day rolling Sharpe |
| `rolling_vol_21d.png` | 21-day rolling annualized vol |
| `monthly_pnl_rmb.png` | Monthly P&L bars |

---

## 8. Research References

### Alpha Mining / Factor Generation

| Paper | Link | Key Contribution |
|-------|------|-----------------|
| **AlphaPROBE** (Guo et al., PKU + Zhengren Quant, 2026) | [arXiv:2602.11917](https://arxiv.org/abs/2602.11917) / [Code](https://github.com/gta0804/AlphaPROBE) | DAG-based factor mining with Bayesian retriever + multi-agent generator; principled evolution avoiding redundant search |
| **QuantaAlpha** (2026) | [arXiv:2602.07085](https://arxiv.org/abs/2602.07085) | Evolutionary LLM-driven alpha mining, IC=0.1501 on CSI300, 27.75% ann return |
| **AlphaForge** (AAAI 2025) | [arXiv:2406.18394](https://arxiv.org/abs/2406.18394) | Generative-predictive NN for formulaic alpha factor diversity |
| **FactorMiner** (Tsinghua, 2026) | [arXiv:2602.14670](https://arxiv.org/abs/2602.14670) | LLM-driven Ralph Loop for factor mining (our system's inspiration) |
| **Gu, Kelly, Xiu** (RFS 2020) | [DOI](https://academic.oup.com/rfs/article/33/5/2223/5758276) | Empirical Asset Pricing via ML — tree models + NNs dominate |
| **WorldQuant 101 Alphas** (2015) | [arXiv:1601.00991](https://arxiv.org/abs/1601.00991) | 101 formulaic alphas, avg pairwise corr 15.9% |

### Risk & Evaluation

| Paper | Link | Key Contribution |
|-------|------|-----------------|
| **Harvey, Liu, Zhu** (RFS 2016) | SSRN 2326253 | Factor Zoo — t>3 for new factors, 27-53% false discoveries |
| **Deflated Sharpe Ratio** (Bailey et al.) | SSRN 2460551 | Corrects selection bias + non-normality in backtest Sharpe |
| **Probability of Backtest Overfitting** | SSRN 2326253 | MinBTL framework — max 45 trials on 5yr data |
| **Almgren & Chriss** (2001) | J. Risk 3:5-40 | Optimal execution / market impact model |
| **López de Prado** (2018) | Wiley | Purged K-Fold + Embargo for time-series CV |
