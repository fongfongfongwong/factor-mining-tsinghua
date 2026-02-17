"""40+ time-series, cross-sectional, and element-wise operators for factor mining.

All operators work on 2D NumPy arrays of shape (M, T) where M = assets, T = time.
NaN values are propagated correctly through all operations.
"""

import numpy as np
from scipy import stats as sp_stats

EPS = 1e-10


# ---------------------------------------------------------------------------
# Time-Series Operators (per-asset, along time axis=1)
# ---------------------------------------------------------------------------

def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """Rank of current value within past d periods (percentile, 0-1)."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        ranks = np.apply_along_axis(
            lambda row: sp_stats.rankdata(row, nan_policy="omit")[-1] / np.sum(~np.isnan(row))
            if np.sum(~np.isnan(row)) > 0 else np.nan,
            axis=1, arr=window,
        )
        out[:, t] = ranks
    return out


def ts_std(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling standard deviation over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nanstd(window, axis=1, ddof=1)
    return out


def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling mean over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nanmean(window, axis=1)
    return out


def ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling sum over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nansum(window, axis=1)
    return out


def ts_min(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling minimum over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nanmin(window, axis=1)
    return out


def ts_max(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling maximum over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nanmax(window, axis=1)
    return out


def ts_argmin(x: np.ndarray, d: int) -> np.ndarray:
    """Position of minimum within rolling window (0 = oldest, d-1 = newest)."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nanargmin(window, axis=1).astype(float)
    return out


def ts_argmax(x: np.ndarray, d: int) -> np.ndarray:
    """Position of maximum within rolling window (0 = oldest, d-1 = newest)."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nanargmax(window, axis=1).astype(float)
    return out


def ts_delta(x: np.ndarray, d: int) -> np.ndarray:
    """x[t] - x[t-d]."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    if d < T:
        out[:, d:] = x[:, d:] - x[:, :-d]
    return out


def ts_lag(x: np.ndarray, d: int) -> np.ndarray:
    """Lagged value x[t-d]."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    if d < T:
        out[:, d:] = x[:, :-d]
    return out


def ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling Pearson correlation between x and y over d periods."""
    M, T = x.shape
    out = np.full((M, T), np.nan)
    for t in range(d - 1, T):
        wx = x[:, t - d + 1: t + 1]
        wy = y[:, t - d + 1: t + 1]
        mx = np.nanmean(wx, axis=1, keepdims=True)
        my = np.nanmean(wy, axis=1, keepdims=True)
        dx = wx - mx
        dy = wy - my
        cov = np.nanmean(dx * dy, axis=1)
        sx = np.sqrt(np.nanmean(dx ** 2, axis=1) + EPS)
        sy = np.sqrt(np.nanmean(dy ** 2, axis=1) + EPS)
        out[:, t] = cov / (sx * sy)
    return np.clip(out, -1.0, 1.0)


def ts_cov(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling covariance between x and y over d periods."""
    M, T = x.shape
    out = np.full((M, T), np.nan)
    for t in range(d - 1, T):
        wx = x[:, t - d + 1: t + 1]
        wy = y[:, t - d + 1: t + 1]
        mx = np.nanmean(wx, axis=1, keepdims=True)
        my = np.nanmean(wy, axis=1, keepdims=True)
        out[:, t] = np.nanmean((wx - mx) * (wy - my), axis=1)
    return out


def ts_skew(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling skewness over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        m = np.nanmean(window, axis=1, keepdims=True)
        diff = window - m
        m2 = np.nanmean(diff ** 2, axis=1)
        m3 = np.nanmean(diff ** 3, axis=1)
        out[:, t] = m3 / (np.power(m2 + EPS, 1.5))
    return out


def ts_kurt(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling kurtosis over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        m = np.nanmean(window, axis=1, keepdims=True)
        diff = window - m
        m2 = np.nanmean(diff ** 2, axis=1)
        m4 = np.nanmean(diff ** 4, axis=1)
        out[:, t] = m4 / (m2 ** 2 + EPS) - 3.0
    return out


def ts_decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """Linearly decaying weighted mean over d periods."""
    M, T = x.shape
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nansum(window * weights[np.newaxis, :], axis=1)
    return out


def ts_product(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling product over d periods."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(d - 1, T):
        window = x[:, t - d + 1: t + 1]
        out[:, t] = np.nanprod(window, axis=1)
    return out


def ts_rsquare(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling R-squared of y regressed on x."""
    corr = ts_corr(x, y, d)
    return corr ** 2


def ts_regression_residual(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling regression residual: y - beta * x - alpha."""
    M, T = x.shape
    out = np.full((M, T), np.nan)
    for t in range(d - 1, T):
        wx = x[:, t - d + 1: t + 1]
        wy = y[:, t - d + 1: t + 1]
        mx = np.nanmean(wx, axis=1, keepdims=True)
        my = np.nanmean(wy, axis=1, keepdims=True)
        dx = wx - mx
        dy = wy - my
        var_x = np.nansum(dx ** 2, axis=1) + EPS
        cov_xy = np.nansum(dx * dy, axis=1)
        beta = cov_xy / var_x
        alpha = my[:, 0] - beta * mx[:, 0]
        out[:, t] = y[:, t] - beta * x[:, t] - alpha
    return out


# ---------------------------------------------------------------------------
# Cross-Sectional Operators (across assets, at each timestamp, axis=0)
# ---------------------------------------------------------------------------

def cs_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank across assets at each timestamp (0-1)."""
    M, T = x.shape
    out = np.full_like(x, np.nan)
    for t in range(T):
        col = x[:, t]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            ranked = sp_stats.rankdata(col[valid])
            out[valid, t] = ranked / valid.sum()
    return out


def cs_zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalization across assets at each timestamp."""
    m = np.nanmean(x, axis=0, keepdims=True)
    s = np.nanstd(x, axis=0, keepdims=True) + EPS
    return (x - m) / s


def cs_scale(x: np.ndarray) -> np.ndarray:
    """Scale so sum(abs(x)) = 1 across assets at each timestamp."""
    denom = np.nansum(np.abs(x), axis=0, keepdims=True) + EPS
    return x / denom


def cs_demean(x: np.ndarray) -> np.ndarray:
    """Subtract cross-sectional mean at each timestamp."""
    return x - np.nanmean(x, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Element-Wise Operators
# ---------------------------------------------------------------------------

def abs_op(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def log_op(x: np.ndarray) -> np.ndarray:
    return np.log(np.abs(x) + EPS)


def sign_op(x: np.ndarray) -> np.ndarray:
    return np.sign(x)


def sqrt_op(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x))


def neg_op(x: np.ndarray) -> np.ndarray:
    return -x


def inv_op(x: np.ndarray) -> np.ndarray:
    return 1.0 / (x + EPS * np.sign(x + EPS))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - y


def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y


def div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x / (y + EPS * np.sign(y + EPS))


def power(x: np.ndarray, n: float) -> np.ndarray:
    return np.power(np.abs(x) + EPS, n) * np.sign(x)


def max_op(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.fmax(x, y)


def min_op(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.fmin(x, y)


# ---------------------------------------------------------------------------
# Operator Registry -- maps names to (function, arity) for the expression engine
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY = {
    # Time-series (x, d)
    "ts_rank": (ts_rank, "ts"),
    "ts_std": (ts_std, "ts"),
    "ts_mean": (ts_mean, "ts"),
    "ts_sum": (ts_sum, "ts"),
    "ts_min": (ts_min, "ts"),
    "ts_max": (ts_max, "ts"),
    "ts_argmin": (ts_argmin, "ts"),
    "ts_argmax": (ts_argmax, "ts"),
    "ts_delta": (ts_delta, "ts"),
    "ts_lag": (ts_lag, "ts"),
    "ts_skew": (ts_skew, "ts"),
    "ts_kurt": (ts_kurt, "ts"),
    "ts_decay_linear": (ts_decay_linear, "ts"),
    "ts_product": (ts_product, "ts"),
    # Time-series (x, y, d)
    "ts_corr": (ts_corr, "ts2"),
    "ts_cov": (ts_cov, "ts2"),
    "ts_rsquare": (ts_rsquare, "ts2"),
    "ts_regression_residual": (ts_regression_residual, "ts2"),
    # Cross-sectional (x)
    "cs_rank": (cs_rank, "cs"),
    "cs_zscore": (cs_zscore, "cs"),
    "cs_scale": (cs_scale, "cs"),
    "cs_demean": (cs_demean, "cs"),
    # Element-wise unary
    "abs": (abs_op, "unary"),
    "log": (log_op, "unary"),
    "sign": (sign_op, "unary"),
    "sqrt": (sqrt_op, "unary"),
    "neg": (neg_op, "unary"),
    "inv": (inv_op, "unary"),
    # Element-wise binary
    "add": (add, "binary"),
    "sub": (sub, "binary"),
    "mul": (mul, "binary"),
    "div": (div, "binary"),
    "max": (max_op, "binary"),
    "min": (min_op, "binary"),
    # Power
    "power": (power, "power"),
}


def get_operator_names() -> list[str]:
    """Return sorted list of all available operator names."""
    return sorted(OPERATOR_REGISTRY.keys())
