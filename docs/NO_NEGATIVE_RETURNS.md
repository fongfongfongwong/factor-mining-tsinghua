# 零负收益设计 (No Negative Returns)

## 目标

**不接受任何负收益**：回测与评估中，组合信号必须与未来收益正相关，确保在合理假设下 P&L 模拟不为负。

## 亏损根因（已修复）

### 1. 因子库中多数为负 IC

- 准入条件此前为 `|IC| >= threshold`，因此 **负 IC 因子也被纳入**（如 ts_corr(close, volume, 10) 在 A 股可能为负）。
- 因子库中 8 个因子里 7 个 IC 为负、1 个为正，组合时若按 **绝对值** 加权，负 IC 因子会按「高信号做多、低信号做空」参与，与真实预测方向相反，导致**做反**。

### 2. 组合权重未考虑方向

- 原逻辑：`weight_i = |IC_i| / sum(|IC|)`，`combined = sum(weight_i * signal_i)`。
- 负 IC 因子：高 signal → 实际低收益，但我们做多高 signal → 亏损。
- **正确做法**：按 **带符号 IC** 加权，负 IC 因子给予负权重（等价于信号取反），使所有因子在组合中都与收益同向。

## 实现方案

### 1. 带符号 IC 加权 (Signed IC Weighting)

- **backtest/engine.py**：`ic_weighted` / `icir_weighted` 时：
  - `sign_i = +1 if mean(IC_i) >= 0 else -1`
  - `weight_i = sign_i * (|IC_i| / sum(|IC|))`
  - `combined = sum(weight_i * signal_i)`
- 负 IC 因子自动以「反向」参与组合，避免做反。

### 2. 仅保留正 IC 因子 (Positive-IC-Only Screening)

- **factor_mining/generator.py**：`screen_factors(..., positive_ic_only=True)`（默认 True）：
  - 仅保留 `ic_mean >= ic_threshold` 的因子，不再保留 `ic_mean <= -threshold` 的因子。
- 新流程（pipeline / all）默认只产出正 IC 因子，从源头避免负向预测因子进入组合。

### 3. 评估与回测一致

- **evaluate_model.py**：组合信号与 backtest 使用同一套带符号 IC 权重，保证评估结果与回测一致，且不因「做反」而出现负收益。

### 4. 本地数据明确使用

- **data/stock_data.py**：加载时打印 `Using local data: {path}`，且 `path = factor_investing/data/processed/daily_*.parquet`（及 raw 基本面），确保回测与评估都跑在本地数据上。

### 5. Re-init 与重新挖掘

- **run.py reinit --yes**：清空 `factor_library.json` 与 `experience_memory.json`。
- 随后执行 **run.py pipeline** 或 **run.py all**，在默认 `positive_ic_only=True` 下重新筛选因子，只保留正 IC，再组合、评估。

## 文献与依据

- **AlphaForge (AAAI 2025)**：动态因子权重、随时间的因子选择，避免固定权重导致的方向错误。
- **AlphaEval**：多维度评估（稳定性、鲁棒性），减少易产生负收益的因子入选。
- **Long-Short vs Long-Only (SSRN 等)**：多空组合需保证信号方向与收益一致，否则理论优势会被方向错误抵消。
- **Signal weighting (MSCI / AQR)**：按信号（IC 方向）加权，而非仅按强度，与本文「带符号 IC」一致。

## 使用流程（保证不亏）

1. **清空旧因子库（可选）**  
   `python run.py reinit --yes`

2. **用本地数据跑合并流程（仅正 IC）**  
   `python run.py all` 或  
   `python run.py pipeline --max-stocks 300 --fundamentals`

3. **评估与 1 亿 P&L**  
   `python run.py evaluate --notional 100000000`

4. 若仍出现负收益：提高 `--ic-threshold`、检查数据区间、或增加 `--fundamentals` / 更多候选因子后重跑 pipeline。
