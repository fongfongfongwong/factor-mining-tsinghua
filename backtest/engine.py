"""Factor-based portfolio backtesting engine."""

import logging
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from .metrics import (
    calc_ic_series,
    calc_icir,
    calc_sharpe,
    calc_max_drawdown,
    calc_turnover,
    calc_win_ratio,
    calc_annual_return,
    factor_summary,
)

logger = logging.getLogger(__name__)


def run_factor_backtest(
    factor_signal: np.ndarray,
    returns: np.ndarray,
    n_groups: int = 5,
) -> dict:
    """Long-short quintile portfolio backtest for a single factor.

    Ranks stocks by factor signal each period, goes long top quintile
    and short bottom quintile, then tracks daily P&L.

    Args:
        factor_signal: Factor signal array (M, T).
        returns: Forward returns array (M, T).
        n_groups: Number of quantile groups.

    Returns:
        Dict with equity curve, group returns, metrics.
    """
    M, T = factor_signal.shape
    ls_returns = np.full(T, np.nan)
    group_returns = {g: np.full(T, np.nan) for g in range(n_groups)}
    positions = np.zeros((M, T))

    for t in range(T):
        sig = factor_signal[:, t]
        ret = returns[:, t]
        valid = ~(np.isnan(sig) | np.isnan(ret))

        if valid.sum() < n_groups * 2:
            continue

        sv = sig[valid]
        rv = ret[valid]
        valid_idx = np.where(valid)[0]

        ranks = sp_stats.rankdata(sv)
        n = len(ranks)
        group_size = n / n_groups

        for g in range(n_groups):
            lo = g * group_size
            hi = (g + 1) * group_size
            mask = (ranks > lo) & (ranks <= hi)
            if g == 0:
                mask = ranks <= hi
            if mask.sum() > 0:
                group_returns[g][t] = np.mean(rv[mask])

        top_mask = ranks > (n * (n_groups - 1) / n_groups)
        bot_mask = ranks <= (n / n_groups)

        if top_mask.sum() > 0 and bot_mask.sum() > 0:
            ls_returns[t] = np.mean(rv[top_mask]) - np.mean(rv[bot_mask])

            for idx in valid_idx[top_mask]:
                positions[idx, t] = 1.0 / top_mask.sum()
            for idx in valid_idx[bot_mask]:
                positions[idx, t] = -1.0 / bot_mask.sum()

    valid_ls = ls_returns[~np.isnan(ls_returns)]
    cum = np.cumprod(1 + valid_ls) if len(valid_ls) > 0 else np.array([1.0])

    group_cum = {}
    for g in range(n_groups):
        gv = group_returns[g][~np.isnan(group_returns[g])]
        group_cum[f"Q{g+1}"] = (np.cumprod(1 + gv)[-1] - 1) if len(gv) > 0 else 0.0

    return {
        "ls_returns": ls_returns.tolist(),
        "cumulative_returns": cum.tolist(),
        "sharpe": calc_sharpe(valid_ls),
        "max_drawdown": calc_max_drawdown(cum),
        "annual_return": calc_annual_return(valid_ls),
        "win_ratio": calc_win_ratio(valid_ls),
        "turnover": calc_turnover(positions),
        "group_cumulative_returns": group_cum,
    }


def run_library_backtest(
    signals: dict[str, np.ndarray],
    returns: np.ndarray,
    method: str = "ic_weighted",
    ic_series_map: Optional[dict[str, np.ndarray]] = None,
) -> dict:
    """Combined factor library backtest.

    Args:
        signals: Dict mapping expression -> signal array (M, T).
        returns: Forward returns (M, T).
        method: Combination method ('equal', 'ic_weighted', 'icir_weighted').
        ic_series_map: Optional pre-computed IC series per factor.

    Returns:
        Backtest result dict.
    """
    if not signals:
        return {"error": "No signals provided"}

    expressions = list(signals.keys())
    signal_list = [signals[e] for e in expressions]
    M, T = signal_list[0].shape

    if method == "equal":
        weights = np.ones(len(signal_list)) / len(signal_list)
    elif method == "ic_weighted":
        if ic_series_map is None:
            ic_series_map = {}
            for expr, sig in signals.items():
                ic_series_map[expr] = calc_ic_series(sig, returns)

        ic_means = []
        for expr in expressions:
            ic = ic_series_map.get(expr, np.zeros(T))
            valid = ic[~np.isnan(ic)]
            ic_means.append(abs(float(np.mean(valid))) if len(valid) > 0 else 0.0)

        total = sum(ic_means) or 1.0
        weights = np.array([m / total for m in ic_means])
    elif method == "icir_weighted":
        if ic_series_map is None:
            ic_series_map = {}
            for expr, sig in signals.items():
                ic_series_map[expr] = calc_ic_series(sig, returns)

        icirs = []
        for expr in expressions:
            ic = ic_series_map.get(expr, np.zeros(T))
            icirs.append(abs(calc_icir(ic)))

        total = sum(icirs) or 1.0
        weights = np.array([m / total for m in icirs])
    else:
        weights = np.ones(len(signal_list)) / len(signal_list)

    combined = np.zeros((M, T))
    for w, sig in zip(weights, signal_list):
        nan_mask = np.isnan(sig)
        safe_sig = np.where(nan_mask, 0.0, sig)
        combined += w * safe_sig

    return run_factor_backtest(combined, returns)


def run_single_factor_analysis(
    expression: str,
    data_panel: dict[str, np.ndarray],
    forward_returns: np.ndarray,
) -> dict:
    """Full factor tearsheet for a single expression.

    Args:
        expression: Factor expression string.
        data_panel: Market data panel.
        forward_returns: Forward returns.

    Returns:
        Dict with all metrics and time series for charting.
    """
    from factor_mining.expression_engine import ExpressionEngine

    engine = ExpressionEngine(data_panel)
    try:
        signal = engine.evaluate(expression)
    except Exception as e:
        return {"error": str(e)}

    metrics = factor_summary(signal, forward_returns)
    backtest = run_factor_backtest(signal, forward_returns)

    return {
        **metrics,
        **backtest,
        "expression": expression,
    }
