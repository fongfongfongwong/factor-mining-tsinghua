"""Performance metrics for factor evaluation and backtest analysis."""

import numpy as np
from scipy import stats as sp_stats


def calc_ic_series(signal: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Daily Spearman rank IC time series.

    Args:
        signal: Factor signal (M, T).
        returns: Forward returns (M, T).

    Returns:
        1D array of IC values per timestamp (length T).
    """
    M, T = signal.shape
    ic = np.full(T, np.nan)
    for t in range(T):
        s = signal[:, t]
        r = returns[:, t]
        valid = ~(np.isnan(s) | np.isnan(r))
        if valid.sum() >= 10:
            corr, _ = sp_stats.spearmanr(s[valid], r[valid])
            ic[t] = corr
    return ic


def calc_icir(ic_series: np.ndarray) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < 10:
        return 0.0
    std = np.std(valid)
    if std < 1e-10:
        return 0.0
    return float(np.mean(valid) / std)


def calc_sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio.

    Args:
        returns: 1D array of period returns.
        rf: Risk-free rate per period.
        periods_per_year: Trading periods per year.
    """
    valid = returns[~np.isnan(returns)]
    if len(valid) < 10:
        return 0.0
    excess = valid - rf
    std = np.std(excess)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def calc_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative return series.

    Args:
        cumulative_returns: 1D cumulative return series (starting from 1.0).
    """
    valid = cumulative_returns[~np.isnan(cumulative_returns)]
    if len(valid) < 2:
        return 0.0
    peak = np.maximum.accumulate(valid)
    drawdown = (valid - peak) / (peak + 1e-10)
    return float(np.min(drawdown))


def calc_turnover(positions: np.ndarray) -> float:
    """Average daily turnover.

    Args:
        positions: 2D array (M, T) of portfolio weights.
    """
    if positions.shape[1] < 2:
        return 0.0
    changes = np.abs(positions[:, 1:] - positions[:, :-1])
    daily_turnover = np.nanmean(np.nansum(changes, axis=0))
    return float(daily_turnover)


def calc_win_ratio(returns: np.ndarray) -> float:
    """Fraction of profitable periods."""
    valid = returns[~np.isnan(returns)]
    if len(valid) == 0:
        return 0.0
    return float(np.sum(valid > 0) / len(valid))


def calc_annual_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized return."""
    valid = returns[~np.isnan(returns)]
    if len(valid) < 2:
        return 0.0
    total = np.prod(1 + valid) - 1
    n_years = len(valid) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((1 + total) ** (1 / n_years) - 1)


def factor_summary(
    signal: np.ndarray,
    forward_returns: np.ndarray,
) -> dict:
    """Compute all metrics for a factor signal.

    Args:
        signal: Factor signal (M, T).
        forward_returns: Forward returns (M, T).

    Returns:
        Dict of all metrics.
    """
    ic = calc_ic_series(signal, forward_returns)
    valid_ic = ic[~np.isnan(ic)]

    M, T = signal.shape
    n_groups = 5
    ls_returns = np.full(T, np.nan)
    for t in range(T):
        s = signal[:, t]
        r = forward_returns[:, t]
        valid = ~(np.isnan(s) | np.isnan(r))
        if valid.sum() >= n_groups * 2:
            sv = s[valid]
            rv = r[valid]
            ranks = sp_stats.rankdata(sv)
            n = len(ranks)
            top = ranks > (n * (n_groups - 1) / n_groups)
            bot = ranks <= (n / n_groups)
            if top.sum() > 0 and bot.sum() > 0:
                ls_returns[t] = np.mean(rv[top]) - np.mean(rv[bot])

    valid_ls = ls_returns[~np.isnan(ls_returns)]
    cum = np.cumprod(1 + valid_ls) if len(valid_ls) > 0 else np.array([1.0])

    return {
        "ic_mean": float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0,
        "ic_std": float(np.std(valid_ic)) if len(valid_ic) > 0 else 0.0,
        "icir": calc_icir(ic),
        "sharpe": calc_sharpe(valid_ls),
        "max_drawdown": calc_max_drawdown(cum),
        "annual_return": calc_annual_return(valid_ls),
        "win_ratio": calc_win_ratio(valid_ls),
        "ic_series": ic.tolist(),
        "ls_returns": ls_returns.tolist(),
        "cumulative_returns": cum.tolist(),
    }
