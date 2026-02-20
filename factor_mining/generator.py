"""Exhaustive Factor Generator: enumerates all valid expressions up to a given depth.

Generates thousands of formulaic alpha candidates from the operator library,
screens them via IC, HLZ, OOS, and correlation dedup, then returns survivors.

Research basis:
- QuantaAlpha (arXiv:2602.07085): IC=0.1501 via evolutionary LLM + diversity
- LightGBM + 101 factors: IC=0.153 on 2893 A-shares
- AlphaForge (AAAI 2025): generative-predictive NN for factor diversity
- Key insight: IC>=0.1 requires 100+ diverse factors + non-linear combination
"""

import logging
import time
import itertools
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from .operators import OPERATOR_REGISTRY
from .expression_engine import ExpressionEngine
from .factor_library import hlz_threshold, ic_t_statistic, oos_validate
from backtest.metrics import calc_ic_series, calc_icir

logger = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap", "returns"]
FUNDAMENTAL_FIELDS = ["bvps", "roe_pct", "basic_eps", "n_income", "revenue", "total_mv", "pe", "pb", "dv_ttm"]
ALL_FIELDS = FIELDS + FUNDAMENTAL_FIELDS
WINDOWS = [3, 5, 10, 20, 40, 60, 120]

TS_OPS = [k for k, (_, a) in OPERATOR_REGISTRY.items() if a == "ts"]
TS2_OPS = [k for k, (_, a) in OPERATOR_REGISTRY.items() if a == "ts2"]
CS_OPS = [k for k, (_, a) in OPERATOR_REGISTRY.items() if a == "cs"]
UNARY_OPS = [k for k, (_, a) in OPERATOR_REGISTRY.items() if a == "unary"]
BINARY_OPS = [k for k, (_, a) in OPERATOR_REGISTRY.items() if a == "binary"]


def _generate_depth1() -> list[str]:
    """Depth 1: single operator applied to a field."""
    exprs = []
    for op in TS_OPS:
        for field in FIELDS:
            for w in WINDOWS:
                exprs.append(f"{op}({field}, {w})")
    for op in CS_OPS:
        for field in FIELDS:
            exprs.append(f"{op}({field})")
    for op in UNARY_OPS:
        for field in FIELDS:
            exprs.append(f"{op}({field})")
    return exprs


def _generate_depth2() -> list[str]:
    """Depth 2: CS/unary wrapping depth-1 TS expressions, or TS on binary combos."""
    exprs = []
    for outer in CS_OPS + UNARY_OPS:
        for inner in TS_OPS:
            for field in FIELDS:
                for w in WINDOWS:
                    exprs.append(f"{outer}({inner}({field}, {w}))")

    for op in TS_OPS:
        for f1 in FIELDS:
            for f2 in FIELDS:
                if f1 >= f2:
                    continue
                for w in WINDOWS:
                    exprs.append(f"{op}(div({f1}, {f2}), {w})")
                    exprs.append(f"{op}(sub({f1}, {f2}), {w})")

    for op in TS2_OPS:
        for f1 in FIELDS:
            for f2 in FIELDS:
                if f1 == f2:
                    continue
                for w in [5, 10, 20]:
                    exprs.append(f"{op}({f1}, {f2}, {w})")

    for outer in CS_OPS:
        for op in TS2_OPS:
            for f1 in ["close", "volume", "returns"]:
                for f2 in ["close", "volume", "returns", "amount"]:
                    if f1 == f2:
                        continue
                    for w in [10, 20]:
                        exprs.append(f"{outer}({op}({f1}, {f2}, {w}))")

    return exprs


def _generate_depth3_selective() -> list[str]:
    """Depth 3: expanded high-value patterns for large-scale factor mining."""
    exprs = []
    core_fields = ["close", "volume", "returns", "high", "low", "amount", "vwap"]
    core_windows = [3, 5, 10, 20, 40]

    # Pattern A: cs_rank(ts1(ts2(f1, f2, w2), w1))
    for ts1 in ["ts_rank", "ts_std", "ts_mean", "ts_delta", "ts_decay_linear"]:
        for ts2 in ["ts_corr", "ts_cov"]:
            for f1 in ["close", "returns", "high", "low"]:
                for f2 in ["volume", "amount", "vwap"]:
                    for w1 in core_windows:
                        for w2 in core_windows:
                            exprs.append(f"cs_rank({ts1}({ts2}({f1}, {f2}, {w2}), {w1}))")

    # Pattern B: cs_rank(ts_op(binary(f1, f2), w))
    for ts_op in TS_OPS:
        for f1 in core_fields:
            for f2 in core_fields:
                if f1 >= f2:
                    continue
                for w in core_windows:
                    exprs.append(f"cs_rank({ts_op}(div({f1}, {f2}), {w}))")
                    exprs.append(f"cs_rank({ts_op}(sub({f1}, {f2}), {w}))")

    # Pattern C: sub(cs(ts(f, w)), cs(f)) — cross-sectional deviation of TS feature
    for cs in CS_OPS:
        for ts in ["ts_rank", "ts_std", "ts_skew", "ts_kurt", "ts_mean", "ts_delta"]:
            for f in core_fields:
                for w in core_windows:
                    exprs.append(f"sub({cs}({ts}({f}, {w})), {cs}({f}))")

    # Pattern D: ts_delta(cs_rank(f), w) — momentum of cross-sectional rank
    for f in core_fields:
        for w in core_windows:
            exprs.append(f"ts_delta(cs_rank({f}), {w})")
            exprs.append(f"ts_rank(cs_rank({f}), {w})")
            exprs.append(f"ts_std(cs_rank({f}), {w})")

    # Pattern E: cs_rank(ts_regression_residual(f1, f2, w))
    for f1 in ["close", "returns"]:
        for f2 in ["volume", "amount", "vwap"]:
            for w in [10, 20, 40]:
                exprs.append(f"cs_rank(ts_regression_residual({f1}, {f2}, {w}))")

    # Pattern F: cs_rank(div(ts_delta(f, w1), ts_std(f, w2)))
    for f in ["close", "volume", "returns"]:
        for w1 in [3, 5, 10, 20]:
            for w2 in [10, 20, 40]:
                exprs.append(f"cs_rank(div(ts_delta({f}, {w1}), ts_std({f}, {w2})))")

    # Pattern G: decay-weighted composites
    for f in ["returns", "volume"]:
        for w in core_windows:
            exprs.append(f"cs_rank(ts_decay_linear({f}, {w}))")
            exprs.append(f"sub(cs_rank(ts_decay_linear({f}, {w})), cs_rank(ts_mean({f}, {w})))")

    return exprs


def _generate_fundamental_value() -> list[str]:
    """Value factors from factor_investing: EP, BP, SIZE, ROE."""
    exprs = []
    for f in ["bvps", "roe_pct", "basic_eps", "total_mv", "pe", "pb"]:
        exprs.append(f"cs_rank({f})")
        exprs.append(f"cs_rank(div({f}, close))")
    for f1 in ["basic_eps", "bvps", "roe_pct"]:
        for f2 in ["close", "total_mv"]:
            if f1 != f2:
                exprs.append(f"cs_rank(div({f1}, {f2}))")
    exprs.append("cs_rank(neg(total_mv))")
    exprs.append("cs_rank(neg(pe))")
    return exprs


def generate_all_candidates(max_depth: int = 3, include_fundamentals: bool = False) -> list[str]:
    """Generate candidate factor expressions. With include_fundamentals=True uses ALL_FIELDS (2000+ with depth 3)."""
    all_exprs = set()
    fields = ALL_FIELDS if include_fundamentals else FIELDS

    all_exprs.update(_generate_depth1())
    if max_depth >= 2:
        all_exprs.update(_generate_depth2())
    if max_depth >= 3:
        all_exprs.update(_generate_depth3_selective())
    if include_fundamentals:
        all_exprs.update(_generate_fundamental_value())

    logger.info(f"Total: {len(all_exprs)} candidates")
    return sorted(all_exprs)


def screen_factors(
    candidates: list[str],
    engine: ExpressionEngine,
    forward_returns: np.ndarray,
    ic_threshold: float = 0.005,
    use_hlz: bool = False,
    use_oos: bool = False,
    max_corr: float = 0.85,
    max_keep: int = 10000,
    progress_interval: int = 500,
    num_workers: int = 1,
    positive_ic_only: bool = False,
) -> dict[str, dict]:
    """Screen candidates by IC, HLZ, OOS, and greedy correlation dedup.

    Returns dict of {expression: {signal, ic_mean, icir, t_stat, oos_ic}} for survivors.
    """
    t0 = time.time()
    n = len(candidates)
    t_thresh = hlz_threshold(n) if use_hlz else 1.96

    passed = []

    for i, expr in enumerate(candidates):
        if i > 0 and i % progress_interval == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            logger.info(f"  Screening [{i}/{n}] {len(passed)} passed, {rate:.0f} expr/s")

        try:
            signal = engine.evaluate(expr)
        except Exception:
            continue

        nan_ratio = np.isnan(signal).sum() / signal.size
        if nan_ratio > 0.5:
            continue

        ic = calc_ic_series(signal, forward_returns)
        valid_ic = ic[~np.isnan(ic)]
        if len(valid_ic) < 50:
            continue

        ic_mean = float(np.mean(valid_ic))
        if abs(ic_mean) < ic_threshold:
            continue
        if positive_ic_only and ic_mean < 0:
            continue

        if use_hlz:
            t_stat = ic_t_statistic(ic)
            if abs(t_stat) < t_thresh:
                continue
        else:
            t_stat = ic_t_statistic(ic)

        if use_oos:
            oos = oos_validate(signal, forward_returns)
            if oos["oos_valid"] and oos["degradation"] > 0.7:
                continue
            oos_ic = oos["oos_ic"]
        else:
            oos_ic = ic_mean

        passed.append({
            "expression": expr,
            "signal": signal,
            "ic_mean": ic_mean,
            "icir": calc_icir(ic),
            "t_stat": t_stat,
            "oos_ic": oos_ic,
            "_ic_ts": np.nan_to_num(ic, nan=0.0),
        })

    logger.info(f"IC+HLZ+OOS: {len(passed)}/{n} passed in {time.time()-t0:.1f}s")

    passed.sort(key=lambda x: -x["ic_mean"])
    if len(passed) <= max_keep:
        survivors = passed
    else:
        survivors = _greedy_corr_dedup(passed, max_corr, max_keep)

    logger.info(f"After correlation dedup: {len(survivors)} survivors")

    return {
        s["expression"]: s for s in survivors
    }


def _greedy_corr_dedup(
    candidates: list[dict],
    max_corr: float,
    max_keep: int,
) -> list[dict]:
    """Greedy forward selection using fast IC-series correlation (not per-cross-section Spearman).

    For each candidate, compute its IC time series and check Pearson correlation
    against IC series of already-selected factors. This is O(T) per pair instead of O(M*T).
    """
    from backtest.metrics import calc_ic_series
    selected = []
    selected_ic_ts = []

    for cand in candidates:
        if len(selected) >= max_keep:
            break

        ic_ts = cand.get("_ic_ts")
        if ic_ts is None:
            continue

        ic_valid = np.nan_to_num(ic_ts, nan=0.0)
        too_corr = False

        check_range = selected_ic_ts[-100:]
        for existing_ic in check_range:
            r = np.corrcoef(ic_valid, existing_ic)[0, 1]
            if np.isfinite(r) and abs(r) > max_corr:
                too_corr = True
                break

        if not too_corr:
            selected.append(cand)
            selected_ic_ts.append(ic_valid)

    return selected
