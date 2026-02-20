"""Performance metrics for factor evaluation and backtest analysis."""

import numpy as np
from scipy import stats as sp_stats


def _fast_rank(x: np.ndarray) -> np.ndarray:
    """Rank along axis 0 (per column), NaN-aware, using argsort (much faster than scipy)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[1]):
        col = x[:, t]
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            continue
        vals = col[mask]
        order = np.argsort(np.argsort(vals)).astype(np.float64)
        out[mask, t] = order
    return out


def calc_ic_series(signal: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Daily Spearman rank IC time series (vectorized, ~10x faster than scipy loop).

    Computes Pearson correlation of ranks per cross-section.
    """
    M, T = signal.shape
    valid = ~(np.isnan(signal) | np.isnan(returns))
    sig_masked = np.where(valid, signal, np.nan)
    ret_masked = np.where(valid, returns, np.nan)
    sig_rank = _fast_rank(sig_masked)
    ret_rank = _fast_rank(ret_masked)

    ic = np.full(T, np.nan)
    for t in range(T):
        mask = ~(np.isnan(sig_rank[:, t]) | np.isnan(ret_rank[:, t]))
        n = mask.sum()
        if n < 10:
            continue
        a = sig_rank[mask, t]
        b = ret_rank[mask, t]
        a_dm = a - a.mean()
        b_dm = b - b.mean()
        denom = np.sqrt((a_dm ** 2).sum() * (b_dm ** 2).sum())
        if denom < 1e-10:
            continue
        ic[t] = (a_dm * b_dm).sum() / denom
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
    """Compute all metrics for a factor signal, reusing run_factor_backtest."""
    from .engine import run_factor_backtest

    ic = calc_ic_series(signal, forward_returns)
    valid_ic = ic[~np.isnan(ic)]
    bt = run_factor_backtest(signal, forward_returns)

    return {
        "ic_mean": float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0,
        "ic_std": float(np.std(valid_ic)) if len(valid_ic) > 0 else 0.0,
        "icir": calc_icir(ic),
        "sharpe": bt["sharpe"],
        "max_drawdown": bt["max_drawdown"],
        "annual_return": bt["annual_return"],
        "win_ratio": bt["win_ratio"],
        "ic_series": ic.tolist(),
        "ls_returns": bt["ls_returns"],
        "cumulative_returns": bt["cumulative_returns"],
    }
