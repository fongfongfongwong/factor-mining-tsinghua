# 最大回撤控制与系统评估框架

## 1. 目标

- **最大回撤**：作为可交易的 return 系统，最大回撤不应超过 **5%**。当前实现通过 **波动率目标 (volatility targeting)** + **回撤上限 (drawdown cap)** 在回测中强制满足该约束。
- **评估视角**：从 **交易员 (Trader)** 与 **量化研究员 (Quant Researcher)** 两个角度评估系统是否达标。

---

## 2. 文献与实务依据

### 2.1 回撤控制

- **Optimal Portfolio Strategy to Control Maximum Drawdown** (SSRN 2053854)：使用 Rolling Economic Drawdown (REDD) 与常数回望窗口，在给定最大损失限制下构建有效前沿；动态资产配置在 20 年样本内表现稳健。
- **Drawdowns** (Van Hemert et al., SSRN 3583864)：回撤统计被机构广泛用于组合管理、申赎决策；实施回撤控制会同时影响收益与风险，需谨慎校准。
- **Harnessing Volatility Targeting** (Research Affiliates)：波动率目标在低波时加仓、高波时减仓；**条件波动率目标**（仅在极端波动状态调整）可更好降低回撤与尾部风险，避免传统 vol targeting 在部分市场放大回撤。

### 2.2 仓位与风险

- **Fractional Kelly / Risk-Constrained Kelly** (Stanford, arXiv 1603.06183)：全 Kelly 回撤大；多数实务采用 fractional Kelly 或 **风险约束 Kelly**，在给定回撤概率上限下优化增长与回撤的权衡。
- 实务中：先定最大回撤与波动上限，再谈收益；回撤超阈时减仓至零直至恢复。

### 2.3 量化研究员如何评估回测

- **All that Glitters Is Not Gold** (SSRN 2745220)：Sharpe 与 OOS 表现几乎无相关；**波动率、最大回撤、对冲/组合构造**等对实盘预测更有信息量。
- **Deflated Sharpe Ratio** (Bailey et al., SSRN 2460551)：校正多重检验与选择偏差、非正态性；策略尝试次数越多，需对 Sharpe 做 deflation，避免 false positive。
- **Telling the Good from the Bad and the Ugly** (SSRN 2346600)：评估需考虑 backtest overfitting 概率、性能退化潜力、最小回测长度、风险收益可持续性。
- 约 5 年日频数据下，若不做多重检验校正，仅约 45 种策略变异就可能导致 Sharpe≥1 由偶然产生；需 OOS 或 deflated 指标验证。

---

## 3. 交易员视角（Trader View）

1. **风险优先**：先设最大回撤（如 5%）与波动上限，再谈收益；本系统通过 vol targeting + DD cap 在回测中满足回撤≤5%。
2. **仓位管理**：采用波动率目标（相当于 fractional/constrained exposure），回撤超阈时自动将当日敞口归零直至净值恢复。
3. **可交易性**：信号为截面多空、日频调仓；实盘需加入成本、冲击、融券约束后再评估净收益与可执行性。

---

## 4. 量化研究员视角（Researcher View）

1. **选择偏差**：组合使用大量因子/衍生信号，存在多重检验与选择偏差；应报告 Deflated Sharpe 或严格 OOS 划分。
2. **评估维度**：除 IC/Sharpe 外，重点看最大回撤、波动率、回撤持续期；文献表明 max drawdown、vol 对 OOS 更具预测力。
3. **过拟合**：回测长度与尝试次数需匹配；仅当 OOS 稳定、回撤可控时，才具备上实盘条件。

---

## 5. 实现要点（本代码库）

- **backtest/engine.py**
  - `_apply_vol_targeting(ls_returns, target_vol_ann, vol_lookback)`：用过去收益的滚动波动率，将当日收益缩放至目标年化波动（仅用过去数据，无 lookahead）。
  - `_apply_drawdown_cap(ls_returns, max_dd_target)`：当累计净值从峰值回撤超过 `max_dd_target` 时，将当日收益置零，直至净值恢复。
  - `run_factor_backtest(..., target_vol_ann=0.05, max_dd_target=0.05)`：先 vol targeting，再 DD cap；所有绩效指标基于约束后收益计算。
- **evaluate_model / generate_report**：默认 `target_vol_ann=0.05`、`max_dd_target=0.05`；报告中标明「风险与回撤目标」「交易员视角」「研究员视角」，并标注是否达标（最大回撤≤5%）。

---

## 6. 参考文献（可进一步查阅）

- SSRN 2053854 — Optimal Portfolio Strategy to Control Maximum Drawdown  
- SSRN 3583864 — Drawdowns (Van Hemert et al.)  
- Research Affiliates — Harnessing Volatility Targeting in Multi-Asset Portfolios  
- SSRN 2460551 — The Deflated Sharpe Ratio (Bailey et al.)  
- SSRN 2745220 — All that Glitters Is Not Gold: Backtest vs OOS  
- SSRN 2346600 — Telling the Good from the Bad and the Ugly: Evaluating Backtested Strategies  
