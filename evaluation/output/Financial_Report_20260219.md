# FactorMiner 多因子策略 — 金融报告

**报告日期**：2026-02-19  
**数据路径**：/Users/fongyeungwong/factor_investing/data/processed/daily_20180101_20241231.parquet  
**初始资金**：10.00 亿元人民币  
**建模股票数量**：300  |  **交易日数**：1698  
**全量本地数据库**：7,448,654 行 | 5577 股票 | 20180102 ~ 20241231  

---

## 1. 执行摘要

| 指标 | 数值 |
|------|------|
| 组合 IC 均值 | +0.0640 |
| 组合 IC 标准差 | 0.1655 |
| ICIR | +0.387 |
| IC>0 比例 | 65.7% |
| 夏普比率 | 2.737 |
| 年化收益 | 52.65% |
| 最大回撤 | -4.96% |
| 胜率 | 55.2% |
| 日均换手 | 1.1022 |
| 参与因子数 | 39 |

**P&L 模拟（假设日交易规模 100000 万元 = 10.00 亿元）**

| 项目 | 数值 |
|------|------|
| 累计盈亏 | **+1,628,850.01 万元** |
| 日均盈亏 | +173.00 万元 |
| 资金回报率 (ROI) | +1628.85% |

### 1.1 Financial Statement（简化）

| Statement Item | Value |
|---|---:|
| Initial Capital | 10.00 亿元 |
| Ending Equity | 172.89 亿元 |
| Peak Equity | 174.59 亿元 |
| Net Trading P&L | +162.8850 亿元 |
| Annualized Return | 52.65% |
| Max Drawdown | -4.96% |

---

## 2. 数据与方法论

- **数据来源**：本地 `factor_investing/data/processed/` 日频行情；可选 `raw/` 下基本面（balance/income/daily_basic）。
- **收益定义**：前向 1 日收益率（close[t+1]/close[t]-1），无 lookahead。
- **组合方式**：IC 加权（带符号），负 IC 因子自动反向参与，保证组合方向与收益正相关。
- **回测规则**：按组合信号截面排序，做多前 20%（Q5）、做空后 20%（Q1），组内等权。
- **模型栈**：39 个信号（library + engineered features）→ IC-signed aggregation → 长短组合回测。
- **风险控制**：波动率目标 0% 年化，最大回撤上限 5%。

---

## 3. 因子与 IC

| 因子/表达式 | IC 均值 | ICIR |
|------------|--------|------|
| cs_rank(neg(ts_mean(returns, 5))) | +0.0400 | +0.233 |
| skew_10 | -0.0182 | -0.148 |
| cs_rank(ts_delta(ts_std(returns, 20), 5)) | -0.0206 | -0.173 |
| skew_20 | -0.0228 | -0.191 |
| corr_cv_40 | -0.0242 | -0.194 |
| cs_rank(ts_corr(returns, ts_delta(volume, 5), 20)) | -0.0250 | -0.236 |
| skew_40 | -0.0280 | -0.241 |
| cs_rank(ts_mean(div(sub(close, open), sub(high, low)), 20)) | -0.0296 | -0.196 |
| vol_ratio_5 | -0.0314 | -0.279 |
| corr_cv_20 | -0.0322 | -0.263 |
| corr_cv_5 | -0.0328 | -0.309 |
| cs_rank_ret_60d | -0.0330 | -0.187 |
| ret_60d | -0.0330 | -0.187 |
| cs_rank(div(ts_delta(close, 10), ts_std(close, 20))) | -0.0337 | -0.212 |
| cs_rank(ts_corr(close, volume, 10)) | -0.0356 | -0.316 |
| corr_cv_10 | -0.0356 | -0.316 |
| intraday | -0.0365 | -0.240 |
| ret_5d | -0.0388 | -0.225 |
| cs_rank_ret_5d | -0.0388 | -0.225 |
| vol_ratio_10 | -0.0391 | -0.330 |
| ts_rank_close_10 | -0.0393 | -0.247 |
| ts_rank_close_20 | -0.0398 | -0.243 |
| ret_10d | -0.0401 | -0.234 |
| cs_rank_ret_10d | -0.0401 | -0.234 |
| cs_rank_ret_40d | -0.0402 | -0.226 |
| ret_40d | -0.0402 | -0.226 |
| ret_20d | -0.0410 | -0.232 |
| cs_rank_ret_20d | -0.0410 | -0.232 |
| ts_rank_close_40 | -0.0423 | -0.248 |
| vol_40d | -0.0442 | -0.234 |
| vol_ratio_20 | -0.0457 | -0.368 |
| vol_5d | -0.0487 | -0.320 |
| cs_rank(ts_cov(close, volume, 20)) | -0.0490 | -0.342 |
| vol_20d | -0.0491 | -0.272 |
| vol_10d | -0.0521 | -0.310 |
| vwap_dev | -0.0541 | -0.374 |
| vol_ratio_40 | -0.0548 | -0.418 |
| cs_rank(ts_mean(div(sub(high, low), close), 15)) | -0.0570 | -0.304 |
| hl_range | -0.0661 | -0.437 |

---

## 4. 分组收益（五分组累计）

| 分组 | 累计收益 |
|------|----------|
| Q1 (低→高因子值) | -70.01% |
| Q2 (低→高因子值) | 25.50% |
| Q3 (低→高因子值) | 109.22% |
| Q4 (低→高因子值) | 118.03% |
| Q5 (低→高因子值) | 62.38% |

---

## 5. 风险与回撤目标（≤5%）

- **目标最大回撤**：≤ 5%。当前回测已施加：波动率目标 0% 年化 + 回撤上限 5%（超则当日敞口归零）。
- **实际最大回撤**：-4.96%。**达标**。
- 依据：机构常用回撤控制（Rolling Economic Drawdown、动态资产配置）；波动率目标可降低极端波动期敞口（Research Affiliates, 条件波动率目标）。

---

## 6. 交易员视角（Trader View）

- **风险优先**：先设最大回撤与波动上限，再谈收益；本系统通过 vol targeting + DD cap 将回撤控制在目标内。
- **仓位管理**：采用 fractional Kelly / 波动率目标，避免满仓导致回撤过大；回撤超阈时自动减仓至零直至恢复。
- **可交易性**：信号为截面多空、日频调仓；需在实盘中加入成本、冲击、融券约束后再评估净收益。

---

## 7. 量化研究员视角（Researcher View）

- **多重检验与选择偏差**：当前组合使用 **39 个**因子/衍生信号；大量尝试会带来 selection bias。建议报告 Deflated Sharpe Ratio（Bailey et al.）或严格 OOS 划分。
- **评估维度**：除 Sharpe/IC 外，应关注最大回撤、波动率、回撤持续期；文献表明 Sharpe 对 OOS 预测力有限，而 max drawdown、vol 更具信息量。
- **过拟合**：回测长度与尝试次数需匹配；仅当 OOS 表现稳定、回撤可控时，策略才具备可上实盘的条件。

---

## 8. 假设与局限

- P&L 按日规模 100000 万元复利累计，未扣除交易成本、冲击、融券成本。
- 回测为历史模拟，不保证未来表现。
- 因子库与衍生信号共同参与组合；新数据或新股票需重新评估。

---

## 9. 图表索引（Trading Company Pack）

- `ic_series.png` — IC 时间序列与滚动均值
- `ic_distribution.png` — IC 分布
- `cumulative_pnl_rmb.png` — 累计盈亏（万元）
- `daily_pnl_rmb.png` — 每日盈亏
- `drawdown.png` — 回撤
- `quintile_returns.png` — 五分组累计收益曲线
- `equity_curve.png` — 策略净值
- `rolling_sharpe_63d.png` — 63日滚动夏普
- `rolling_vol_21d.png` — 21日滚动年化波动率
- `monthly_pnl_rmb.png` — 月度 P&L（万元）
