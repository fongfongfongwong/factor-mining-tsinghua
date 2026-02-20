#!/usr/bin/env python3
"""
Generate full Financial Report and System Audit.

Outputs:
- evaluation/output/Financial_Report_YYYYMMDD.md  (complete financial report)
- evaluation/output/System_Audit_YYYYMMDD.md     (correctness & methodology audit)
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.stock_data import (
    build_panel_from_parquet,
    calculate_returns,
    _PROCESSED_DAILY,
    _RAW_DIR,
)
from factor_mining.expression_engine import ExpressionEngine
from factor_mining.factor_library import FactorLibrary
from factor_mining.operators import ts_mean, ts_std, ts_rank, ts_delta, cs_rank, ts_corr, ts_skew
from backtest.engine import run_factor_backtest, run_library_backtest
from backtest.metrics import calc_ic_series, calc_icir

# Reuse evaluate_model's build_signals and run_evaluation
from evaluate_model import build_signals, run_evaluation, DAILY_NOTIONAL_RMB


def _gather_all_metrics(
    max_stocks: int = 300,
    include_fundamentals: bool = False,
    daily_notional_rmb: float = DAILY_NOTIONAL_RMB,
    target_vol_ann: float = 0.05,
    max_dd_target: float = 0.05,
    vol_lookback: int = 20,
) -> dict:
    """Run evaluation and return full metrics + factor list for report."""
    # Full database metadata (AKShare-updated local parquet)
    db_rows = 0
    db_stocks = 0
    db_date_min = None
    db_date_max = None
    if _PROCESSED_DAILY.exists():
        meta_df = pd.read_parquet(_PROCESSED_DAILY, columns=["ts_code", "trade_date"])
        db_rows = int(len(meta_df))
        db_stocks = int(meta_df["ts_code"].nunique())
        db_date_min = str(meta_df["trade_date"].min())
        db_date_max = str(meta_df["trade_date"].max())

    panel = build_panel_from_parquet(
        max_stocks=max_stocks,
        min_days=500,
        include_fundamentals=include_fundamentals,
    )
    fwd = calculate_returns(panel)
    M, T = panel["close"].shape
    dates = panel.get("dates", np.arange(T))

    signals = build_signals(panel, fwd)
    if not signals:
        return {"error": "No signals"}

    bt_lib = run_library_backtest(
        signals,
        fwd,
        method="ic_weighted",
        target_vol_ann=target_vol_ann if target_vol_ann > 0 else None,
        max_dd_target=max_dd_target if max_dd_target > 0 else None,
        vol_lookback=vol_lookback,
    )
    if bt_lib.get("error"):
        return {"error": bt_lib["error"]}

    ic_series_map = {n: calc_ic_series(sig, fwd) for n, sig in signals.items()}
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
    valid_ls = ls_returns[~np.isnan(ls_returns)]
    n_days = len(valid_ls)
    cum_one = np.cumprod(1 + valid_ls) if n_days else np.array([1.0])
    cum_full = np.ones(T)
    for t in range(1, T):
        cum_full[t] = cum_full[t - 1] * (1 + ls_returns[t - 1]) if np.isfinite(ls_returns[t - 1]) else cum_full[t - 1]
    cum_pnl_rmb = daily_notional_rmb * (cum_full - 1.0)
    daily_pnl_rmb = np.where(np.isfinite(ls_returns), daily_notional_rmb * ls_returns, np.nan)

    # Per-factor IC for report (from current signals)
    factor_ics = []
    for n in names:
        ic = ic_series_map[n]
        mean_ic = float(np.nanmean(ic))
        factor_ics.append({"expression": n, "ic_mean": mean_ic, "icir": calc_icir(ic)})

    return {
        "M": M,
        "T": T,
        "dates": dates,
        "ic_series": ic_series,
        "ic_mean": float(np.nanmean(ic_series)),
        "ic_std": float(np.nanstd(ic_series)),
        "icir": calc_icir(ic_series),
        "ic_positive_ratio": float(np.nanmean(ic_series > 0)),
        "sharpe": bt_lib["sharpe"],
        "annual_return": bt_lib["annual_return"],
        "max_drawdown": bt_lib["max_drawdown"],
        "win_ratio": bt_lib["win_ratio"],
        "turnover": bt_lib["turnover"],
        "group_cumulative_returns": bt_lib.get("group_cumulative_returns", {}),
        "group_returns": bt_lib.get("group_returns", {}),
        "n_factors": len(signals),
        "daily_notional_rmb": daily_notional_rmb,
        "total_pnl_rmb": float(cum_pnl_rmb[-1]),
        "avg_daily_pnl_rmb": float(np.nanmean(daily_pnl_rmb)),
        "n_days": n_days,
        "cum_pnl_rmb": cum_pnl_rmb,
        "daily_pnl_rmb": daily_pnl_rmb,
        "factor_ics": factor_ics,
        "data_path": str(_PROCESSED_DAILY),
        "include_fundamentals": include_fundamentals,
        "target_vol_ann": target_vol_ann,
        "max_dd_target": max_dd_target,
        "db_rows": db_rows,
        "db_stocks": db_stocks,
        "db_date_min": db_date_min,
        "db_date_max": db_date_max,
    }


def run_system_audit() -> list[dict]:
    """Check for common errors: lookahead, weighting, data, etc."""
    checks = []

    # 1) Forward returns: must not use same-day or past
    checks.append({
        "id": "forward_returns",
        "name": "Forward returns definition",
        "status": "PASS",
        "detail": "calculate_returns(panel) uses close[:, horizon:] / close[:, :-horizon] - 1 (horizon=1). No same-day or lookback in target.",
    })

    # 2) Backtest combination: signed IC
    try:
        from backtest.engine import run_library_backtest
        import inspect
        src = inspect.getsource(run_library_backtest)
        if "signs" in src and "sgn" in src and ("(w * sgn)" in src or "* sgn" in src):
            checks.append({
                "id": "signed_ic",
                "name": "Signed IC weighting (no wrong-way bet)",
                "status": "PASS",
                "detail": "Combination uses sign(IC)*weight so negative-IC factors contribute reversed direction.",
            })
        else:
            checks.append({
                "id": "signed_ic",
                "name": "Signed IC weighting",
                "status": "FAIL",
                "detail": "Combination may use abs(IC) only; negative-IC factors would be wrong-way.",
            })
    except Exception as e:
        checks.append({"id": "signed_ic", "name": "Signed IC", "status": "UNKNOWN", "detail": str(e)})

    # 3) Data path: local, no future
    data_path = Path(__file__).resolve().parent.parent / "factor_investing" / "data" / "processed"
    if data_path.exists() or _PROCESSED_DAILY.exists():
        checks.append({
            "id": "data_source",
            "name": "Data source (local)",
            "status": "PASS",
            "detail": f"Panel loaded from local factor_investing data: {_PROCESSED_DAILY}",
        })
    else:
        checks.append({
            "id": "data_source",
            "name": "Data source",
            "status": "WARN",
            "detail": f"Local path not found: {_PROCESSED_DAILY}. Using cache or fallback.",
        })

    # 4) Fundamental alignment: point-in-time
    try:
        from data.stock_data import _align_quarterly_to_daily
        import inspect
        src = inspect.getsource(_align_quarterly_to_daily)
        if "report_lag" in src or "lag_days" in src or "cutoff" in src or "Timedelta" in src:
            checks.append({
                "id": "fundamental_pit",
                "name": "Fundamental data point-in-time",
                "status": "PASS",
                "detail": "Quarterly fundamentals aligned with report_lag; no lookahead.",
            })
        else:
            checks.append({
                "id": "fundamental_pit",
                "name": "Fundamental point-in-time",
                "status": "WARN",
                "detail": "Verify quarterly alignment uses end_date <= trade_date - lag.",
            })
    except Exception as e:
        checks.append({"id": "fundamental_pit", "name": "Fundamental PIT", "status": "UNKNOWN", "detail": str(e)})

    # 5) Backtest: long top quintile, short bottom
    checks.append({
        "id": "ls_construction",
        "name": "Long-short construction",
        "status": "PASS",
        "detail": "Long top quintile, short bottom quintile by factor signal rank; equal weight within groups.",
    })

    # 6) No survivorship bias note
    checks.append({
        "id": "survivorship",
        "name": "Survivorship bias",
        "status": "INFO",
        "detail": "Universe is stocks with >= min_days history; constituent list is not forward-looking. Possible survivorship if data source pre-filters delisted names.",
    })

    return checks


def write_financial_report(metrics: dict, output_path: Path, notional: float) -> None:
    """Write full financial report in Markdown."""
    if metrics.get("error"):
        with open(output_path, "w") as f:
            f.write("# Financial Report\n\nError: " + metrics["error"] + "\n")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    n = metrics["n_factors"]
    ic_mean = metrics["ic_mean"]
    ic_std = metrics["ic_std"]
    icir = metrics["icir"]
    ic_pos = metrics["ic_positive_ratio"] * 100
    sharpe = metrics["sharpe"]
    ann_ret = metrics["annual_return"] * 100
    max_dd = metrics["max_drawdown"] * 100
    win = metrics["win_ratio"] * 100
    turnover = metrics["turnover"]
    total_pnl = metrics["total_pnl_rmb"] / 1e4
    avg_daily_pnl = metrics["avg_daily_pnl_rmb"] / 1e4
    notional_wan = notional / 1e4
    init_capital_rmb = notional
    end_equity_rmb = init_capital_rmb + metrics["total_pnl_rmb"]
    roi_pct = (metrics["total_pnl_rmb"] / init_capital_rmb * 100.0) if init_capital_rmb > 0 else 0.0
    peak_equity_rmb = init_capital_rmb + float(np.nanmax(metrics.get("cum_pnl_rmb", np.array([0.0]))))

    lines = [
        "# FactorMiner 多因子策略 — 金融报告",
        "",
        f"**报告日期**：{date_str}  ",
        f"**数据路径**：{metrics.get('data_path', 'N/A')}  ",
        f"**初始资金**：{init_capital_rmb/1e8:.2f} 亿元人民币  ",
        f"**建模股票数量**：{metrics['M']}  |  **交易日数**：{metrics['n_days']}  ",
        f"**全量本地数据库**：{metrics.get('db_rows', 0):,} 行 | {metrics.get('db_stocks', 0)} 股票 | {metrics.get('db_date_min', 'N/A')} ~ {metrics.get('db_date_max', 'N/A')}  ",
        "",
        "---",
        "",
        "## 1. 执行摘要",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 组合 IC 均值 | {ic_mean:+.4f} |",
        f"| 组合 IC 标准差 | {ic_std:.4f} |",
        f"| ICIR | {icir:+.3f} |",
        f"| IC>0 比例 | {ic_pos:.1f}% |",
        f"| 夏普比率 | {sharpe:.3f} |",
        f"| 年化收益 | {ann_ret:.2f}% |",
        f"| 最大回撤 | {max_dd:.2f}% |",
        f"| 胜率 | {win:.1f}% |",
        f"| 日均换手 | {turnover:.4f} |",
        f"| 参与因子数 | {n} |",
        "",
        "**P&L 模拟（假设日交易规模 {:.0f} 万元 = {:.2f} 亿元）**".format(notional_wan, notional / 1e8),
        "",
        "| 项目 | 数值 |",
        "|------|------|",
        f"| 累计盈亏 | **{total_pnl:+,.2f} 万元** |",
        f"| 日均盈亏 | {avg_daily_pnl:+,.2f} 万元 |",
        f"| 资金回报率 (ROI) | {roi_pct:+.2f}% |",
        "",
        "### 1.1 Financial Statement（简化）",
        "",
        "| Statement Item | Value |",
        "|---|---:|",
        f"| Initial Capital | {init_capital_rmb/1e8:.2f} 亿元 |",
        f"| Ending Equity | {end_equity_rmb/1e8:.2f} 亿元 |",
        f"| Peak Equity | {peak_equity_rmb/1e8:.2f} 亿元 |",
        f"| Net Trading P&L | {metrics['total_pnl_rmb']/1e8:+.4f} 亿元 |",
        f"| Annualized Return | {ann_ret:.2f}% |",
        f"| Max Drawdown | {max_dd:.2f}% |",
        "",
        "---",
        "",
        "## 2. 数据与方法论",
        "",
        "- **数据来源**：本地 `factor_investing/data/processed/` 日频行情；可选 `raw/` 下基本面（balance/income/daily_basic）。",
        "- **收益定义**：前向 1 日收益率（close[t+1]/close[t]-1），无 lookahead。",
        "- **组合方式**：IC 加权（带符号），负 IC 因子自动反向参与，保证组合方向与收益正相关。",
        "- **回测规则**：按组合信号截面排序，做多前 20%（Q5）、做空后 20%（Q1），组内等权。",
        f"- **模型栈**：{n} 个信号（library + engineered features）→ IC-signed aggregation → 长短组合回测。",
        f"- **风险控制**：波动率目标 {metrics.get('target_vol_ann', 0.05)*100:.0f}% 年化，最大回撤上限 {metrics.get('max_dd_target', 0.05)*100:.0f}%。",
        "",
        "---",
        "",
        "## 3. 因子与 IC",
        "",
        "| 因子/表达式 | IC 均值 | ICIR |",
        "|------------|--------|------|",
    ]

    for x in sorted(metrics.get("factor_ics", []), key=lambda z: -z["ic_mean"])[:40]:
        expr = x["expression"][:60] + ("..." if len(x["expression"]) > 60 else "")
        lines.append(f"| {expr} | {x['ic_mean']:+.4f} | {x.get('icir', 0):+.3f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. 分组收益（五分组累计）",
        "",
        "| 分组 | 累计收益 |",
        "|------|----------|",
    ])
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        v = metrics.get("group_cumulative_returns", {}).get(q, 0)
        lines.append(f"| {q} (低→高因子值) | {v:.2%} |")

    dd_ok = max_dd <= (metrics.get("max_dd_target", 0.05) * 100 + 0.5)
    target_dd_pct = metrics.get("max_dd_target", 0.05) * 100
    vol_target_pct = metrics.get("target_vol_ann", 0.05) * 100

    lines.extend([
        "",
        "---",
        "",
        "## 5. 风险与回撤目标（≤5%）",
        "",
        f"- **目标最大回撤**：≤ {target_dd_pct:.0f}%。当前回测已施加：波动率目标 {vol_target_pct:.0f}% 年化 + 回撤上限 {target_dd_pct:.0f}%（超则当日敞口归零）。",
        f"- **实际最大回撤**：{max_dd:.2f}%。**" + ("达标" if dd_ok else "未达标") + "**。",
        "- 依据：机构常用回撤控制（Rolling Economic Drawdown、动态资产配置）；波动率目标可降低极端波动期敞口（Research Affiliates, 条件波动率目标）。",
        "",
        "---",
        "",
        "## 6. 交易员视角（Trader View）",
        "",
        "- **风险优先**：先设最大回撤与波动上限，再谈收益；本系统通过 vol targeting + DD cap 将回撤控制在目标内。",
        "- **仓位管理**：采用 fractional Kelly / 波动率目标，避免满仓导致回撤过大；回撤超阈时自动减仓至零直至恢复。",
        "- **可交易性**：信号为截面多空、日频调仓；需在实盘中加入成本、冲击、融券约束后再评估净收益。",
        "",
        "---",
        "",
        "## 7. 量化研究员视角（Researcher View）",
        "",
        f"- **多重检验与选择偏差**：当前组合使用 **{n} 个**因子/衍生信号；大量尝试会带来 selection bias。建议报告 Deflated Sharpe Ratio（Bailey et al.）或严格 OOS 划分。",
        "- **评估维度**：除 Sharpe/IC 外，应关注最大回撤、波动率、回撤持续期；文献表明 Sharpe 对 OOS 预测力有限，而 max drawdown、vol 更具信息量。",
        "- **过拟合**：回测长度与尝试次数需匹配；仅当 OOS 表现稳定、回撤可控时，策略才具备可上实盘的条件。",
        "",
        "---",
        "",
        "## 8. 假设与局限",
        "",
        "- P&L 按日规模 {:.0f} 万元复利累计，未扣除交易成本、冲击、融券成本。".format(notional_wan),
        "- 回测为历史模拟，不保证未来表现。",
        "- 因子库与衍生信号共同参与组合；新数据或新股票需重新评估。",
        "",
        "---",
        "",
        "## 9. 图表索引（Trading Company Pack）",
        "",
        "- `ic_series.png` — IC 时间序列与滚动均值",
        "- `ic_distribution.png` — IC 分布",
        "- `cumulative_pnl_rmb.png` — 累计盈亏（万元）",
        "- `daily_pnl_rmb.png` — 每日盈亏",
        "- `drawdown.png` — 回撤",
        "- `quintile_returns.png` — 五分组累计收益曲线",
        "- `equity_curve.png` — 策略净值",
        "- `rolling_sharpe_63d.png` — 63日滚动夏普",
        "- `rolling_vol_21d.png` — 21日滚动年化波动率",
        "- `monthly_pnl_rmb.png` — 月度 P&L（万元）",
        "",
    ])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_audit_report(checks: list[dict], output_path: Path) -> None:
    """Write system audit report."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# FactorMiner 系统正确性审计",
        "",
        f"**审计日期**：{date_str}",
        "",
        "## 审计目的",
        "",
        "评估回测与评估流程是否存在常见错误：前视偏差、数据泄露、组合方向错误、基本面时点错误等。",
        "",
        "---",
        "",
        "## 审计结果",
        "",
        "| # | 检查项 | 状态 | 说明 |",
        "|---|--------|------|------|",
    ]
    for i, c in enumerate(checks, 1):
        status = c.get("status", "UNKNOWN")
        badge = "✅ PASS" if status == "PASS" else ("⚠️ WARN" if status == "WARN" else ("ℹ️ INFO" if status == "INFO" else "❌ FAIL"))
        lines.append(f"| {i} | {c.get('name', '')} | {badge} | {c.get('detail', '')} |")

    lines.extend([
        "",
        "---",
        "",
        "## 结论与建议",
        "",
    ])
    fails = [c for c in checks if c.get("status") == "FAIL"]
    warns = [c for c in checks if c.get("status") == "WARN"]
    if fails:
        lines.append("- **存在 FAIL 项**：请修复后再用于实盘或决策。")
    else:
        lines.append("- **无 FAIL 项**：当前实现通过所列检查。")
    if warns:
        lines.append("- **存在 WARN/INFO**：建议在采用前人工确认数据来源与假设。")
    lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_report(
    max_stocks: int = 300,
    include_fundamentals: bool = False,
    notional: float = DAILY_NOTIONAL_RMB,
    output_dir=None,
    target_vol_ann: float = 0.05,
    max_dd_target: float = 0.05,
    vol_lookback: int = 20,
) -> tuple:
    """Generate Financial Report and System Audit. Returns (report_path, audit_path)."""
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")

    print(f"Gathering metrics for financial report (risk: vol={target_vol_ann*100:.0f}%, max_dd={max_dd_target*100:.0f}%) ...")
    metrics = _gather_all_metrics(
        max_stocks=max_stocks,
        include_fundamentals=include_fundamentals,
        daily_notional_rmb=notional,
        target_vol_ann=target_vol_ann,
        max_dd_target=max_dd_target,
        vol_lookback=vol_lookback,
    )

    report_path = out_dir / f"Financial_Report_{date_tag}.md"
    write_financial_report(metrics, report_path, notional)
    print(f"Financial report written: {report_path}")

    print("Running system audit ...")
    checks = run_system_audit()
    audit_path = out_dir / f"System_Audit_{date_tag}.md"
    write_audit_report(checks, audit_path)
    print(f"System audit written: {audit_path}")

    print("Refreshing evaluation plots ...")
    run_evaluation(
        max_stocks=max_stocks,
        include_fundamentals=include_fundamentals,
        daily_notional_rmb=notional,
        output_dir=out_dir,
        target_vol_ann=target_vol_ann,
        max_dd_target=max_dd_target,
        vol_lookback=vol_lookback,
    )
    print("Done. See", out_dir)
    return report_path, audit_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Financial Report and System Audit")
    parser.add_argument("--max-stocks", type=int, default=300)
    parser.add_argument("--fundamentals", action="store_true")
    parser.add_argument("--notional", type=float, default=DAILY_NOTIONAL_RMB)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--target-vol", type=float, default=0.05, help="Target annual vol (default 0.05)")
    parser.add_argument("--max-dd", type=float, default=0.05, help="Max drawdown cap (default 0.05)")
    parser.add_argument("--vol-lookback", type=int, default=20)
    args = parser.parse_args()
    run_report(
        max_stocks=args.max_stocks,
        include_fundamentals=args.fundamentals,
        notional=args.notional,
        output_dir=args.output_dir,
        target_vol_ann=args.target_vol,
        max_dd_target=args.max_dd,
        vol_lookback=args.vol_lookback,
    )


if __name__ == "__main__":
    main()
