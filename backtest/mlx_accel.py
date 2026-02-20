"""MLX (Apple Silicon GPU) accelerated computations for factor backtesting.

Uses Metal GPU via mlx for:
1. Batch rank-IC computation across all factors simultaneously
2. Ridge regression for adaptive factor combination
3. Correlation matrix computation for dedup

M2 Max with 96GB unified memory can hold thousands of (300, 1699) factor arrays on GPU.
"""

import logging
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)


def _np_to_mx(arr: np.ndarray) -> "mx.array":
    """Convert numpy array to MLX, handling NaN -> 0 for GPU ops."""
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return mx.array(clean)


def _rank_2d(x: "mx.array") -> "mx.array":
    """Rank along axis 0 per column (GPU-accelerated argsort-of-argsort)."""
    return mx.argsort(mx.argsort(x, axis=0), axis=0).astype(mx.float32)


def mlx_batch_ic(
    signals_3d: np.ndarray,
    returns_2d: np.ndarray,
) -> np.ndarray:
    """Compute mean Spearman rank-IC for all factors simultaneously on GPU.

    Args:
        signals_3d: (N_factors, M_stocks, T_days) — all factor signals stacked.
        returns_2d: (M_stocks, T_days) — forward returns.

    Returns:
        1D array of shape (N_factors,) with mean IC per factor.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available")

    N, M, T = signals_3d.shape
    logger.info(f"MLX batch IC: {N} factors x {M} stocks x {T} days on GPU")

    sig_mx = _np_to_mx(signals_3d)  # (N, M, T)
    ret_mx = _np_to_mx(returns_2d)  # (M, T)

    # Validity mask: both signal and return are non-zero (proxy for non-NaN)
    sig_orig = np.array(signals_3d)
    ret_orig = np.array(returns_2d)

    ic_means = np.zeros(N, dtype=np.float32)

    CHUNK_T = 50
    for t_start in range(0, T, CHUNK_T):
        t_end = min(t_start + CHUNK_T, T)
        chunk_ics = []

        for t in range(t_start, t_end):
            ret_col = ret_mx[:, t]  # (M,)
            sig_col = sig_mx[:, :, t]  # (N, M)

            ret_rank = mx.argsort(mx.argsort(ret_col)).astype(mx.float32)  # (M,)
            ret_rank = ret_rank - mx.mean(ret_rank)

            sig_ranks = mx.argsort(mx.argsort(sig_col, axis=1), axis=1).astype(mx.float32)  # (N, M)
            sig_ranks = sig_ranks - mx.mean(sig_ranks, axis=1, keepdims=True)

            num = mx.sum(sig_ranks * ret_rank[None, :], axis=1)  # (N,)
            denom = mx.sqrt(mx.sum(sig_ranks ** 2, axis=1) * mx.sum(ret_rank ** 2))  # (N,)
            ic_t = num / (denom + 1e-10)  # (N,)
            chunk_ics.append(ic_t)

        chunk_stack = mx.stack(chunk_ics, axis=1)  # (N, chunk_T)
        mx.eval(chunk_stack)
        ic_means += np.array(chunk_stack).sum(axis=1)

    ic_means /= T
    logger.info(f"MLX batch IC done: mean |IC| = {np.mean(np.abs(ic_means)):.4f}")
    return ic_means


def mlx_ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """Solve ridge regression on GPU: w = (X'X + alpha*I)^{-1} X'y.

    Args:
        X: (n_samples, n_features) design matrix.
        y: (n_samples,) target.
        alpha: regularization.

    Returns:
        w: (n_features,) coefficients.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available")

    X_mx = _np_to_mx(X)
    y_mx = _np_to_mx(y)
    n_feat = X_mx.shape[1]

    XtX = X_mx.T @ X_mx + alpha * mx.eye(n_feat)
    Xty = X_mx.T @ y_mx

    w = mx.linalg.solve(XtX, Xty[:, None])[:, 0]
    mx.eval(w)
    return np.array(w)


def mlx_ic_weighted_combine(
    signals: dict[str, np.ndarray],
    returns: np.ndarray,
) -> np.ndarray:
    """IC-weighted combination using MLX GPU for IC + weighted sum.

    Memory-efficient: processes factors in chunks, never stacks all at once.
    Returns combined signal (M, T).
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available")

    names = list(signals.keys())
    N = len(names)
    M, T = signals[names[0]].shape

    # Compute IC in chunks to avoid OOM
    CHUNK = 200
    ic_means = np.zeros(N, dtype=np.float32)
    ret_mx = _np_to_mx(returns)

    for c_start in range(0, N, CHUNK):
        c_end = min(c_start + CHUNK, N)
        chunk_sigs = np.stack([signals[names[i]] for i in range(c_start, c_end)], axis=0)
        chunk_ic = mlx_batch_ic(chunk_sigs, returns)
        ic_means[c_start:c_end] = chunk_ic

    abs_ic = np.abs(ic_means)
    total = abs_ic.sum() or 1.0
    weights = abs_ic / total
    signs = np.where(ic_means >= 0, 1.0, -1.0)

    combined = np.zeros((M, T), dtype=np.float64)
    for i, name in enumerate(names):
        w = float(weights[i] * signs[i])
        if abs(w) < 1e-10:
            continue
        combined += w * np.nan_to_num(signals[name], nan=0.0)

    logger.info(f"MLX IC-weighted combine: {N} factors, mean |IC|={np.mean(abs_ic):.4f}")
    return combined
