"""AlphaPROBE-style Adaptive Factor Combination (GPU-optimized).

Implements ideas from AlphaPROBE (arXiv:2602.11917):
1. Pre-compute all factor IC series (MLX GPU batch) — done once
2. Bayesian Factor Retrieval via lookup (no recomputation)
3. Rolling Ridge Regression on GPU to dynamically re-weight top-k factors

Drop-in replacement for static IC-weighted combination.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def adaptive_combine_signals(
    signals: dict[str, np.ndarray],
    returns: np.ndarray,
    lookback: int = 120,
    top_k: int = 50,
    ridge_alpha: float = 1e-4,
    min_train_days: int = 60,
    refit_interval: int = 5,
    exploration_weight: float = 0.1,
) -> np.ndarray:
    """AlphaPROBE-style adaptive factor combination via rolling ridge regression.

    Optimized pipeline:
    1. Pre-compute per-factor daily IC series once (vectorized)
    2. At each refit step: select top-k by rolling IC*ICIR + exploration bonus
    3. Fit ridge regression on rolling window → predict combined signal
    """
    names = list(signals.keys())
    if not names:
        raise ValueError("No signals provided")

    first = signals[names[0]]
    M, T = first.shape
    N = len(names)

    # Step 1: Pre-compute all factor IC series (vectorized, ~seconds for 2K factors)
    logger.info(f"Adaptive: pre-computing IC series for {N} factors ...")
    from .metrics import calc_ic_series
    ic_matrix = np.full((N, T), np.nan)
    for i, name in enumerate(names):
        ic_matrix[i] = calc_ic_series(signals[name], returns)
    logger.info("Adaptive: IC pre-computation done")

    combined = np.full((M, T), np.nan)
    visit_counts = np.zeros(N)
    last_w = None
    last_sel_idx = None

    n_refits = 0
    for t in range(min_train_days, T):
        # Reuse last weights between refit steps
        if t % refit_interval != 0 and last_w is not None and last_sel_idx is not None:
            X_t = np.column_stack([
                np.nan_to_num(signals[names[j]][:, t], nan=0.0) for j in last_sel_idx
            ])
            combined[:, t] = X_t @ last_w
            continue

        # Step 2: Bayesian retrieval via pre-computed IC lookup
        start = max(0, t - lookback)
        ic_window = ic_matrix[:, start:t]  # (N, window)
        ic_mean = np.nanmean(ic_window, axis=1)  # (N,)
        ic_std = np.nanstd(ic_window, axis=1)    # (N,)
        ic_std = np.where(ic_std < 1e-10, 1.0, ic_std)
        icir = np.abs(ic_mean) / ic_std
        exploit = np.abs(ic_mean) * icir
        explore = exploration_weight / (1 + visit_counts)
        scores = exploit + explore

        sel_idx = np.argsort(-scores)[:top_k]
        visit_counts[sel_idx] += 1
        selected_names = [names[j] for j in sel_idx]
        n_factors = len(sel_idx)

        # Step 3: Build training data for ridge regression
        n_train = t - start
        X_train = np.zeros((M * n_train, n_factors))
        y_train = np.zeros(M * n_train)
        valid_mask = np.ones(M * n_train, dtype=bool)

        for j_local, j_global in enumerate(sel_idx):
            sig = signals[names[j_global]]
            for dt, tt in enumerate(range(start, t)):
                X_train[dt * M:(dt + 1) * M, j_local] = np.nan_to_num(sig[:, tt], nan=0.0)

        for dt, tt in enumerate(range(start, t)):
            y_train[dt * M:(dt + 1) * M] = returns[:, tt]
            nan_mask = np.isnan(returns[:, tt])
            valid_mask[dt * M:(dt + 1) * M] &= ~nan_mask

        X_v = X_train[valid_mask]
        y_v = y_train[valid_mask]

        if len(y_v) < n_factors + 10:
            continue

        # Ridge solve (MLX GPU if available)
        try:
            from .mlx_accel import mlx_ridge_solve, HAS_MLX
            if HAS_MLX:
                w = mlx_ridge_solve(X_v, y_v, alpha=ridge_alpha)
            else:
                raise ImportError
        except Exception:
            XtX = X_v.T @ X_v + ridge_alpha * np.eye(n_factors)
            Xty = X_v.T @ y_v
            try:
                w = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                continue

        last_w = w
        last_sel_idx = sel_idx
        n_refits += 1

        X_t = np.column_stack([
            np.nan_to_num(signals[names[j]][:, t], nan=0.0) for j in sel_idx
        ])
        combined[:, t] = X_t @ w

    n_valid = int(np.sum(np.any(np.isfinite(combined), axis=0)))
    logger.info(f"Adaptive: {N} signals -> top-{top_k}/step, {n_refits} refits, "
                f"{n_valid}/{T} valid days")
    return combined
