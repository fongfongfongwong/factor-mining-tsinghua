"""Factor Library: manages discovered factors with admission, dedup, and persistence.

Implements the Orthogonal Library concept from the FactorMiner paper (Section 3.2).
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from config import FACTOR_LIBRARY_PATH, IC_THRESHOLD, CORR_THRESHOLD
from backtest.metrics import calc_ic_series as compute_ic, calc_icir as compute_icir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HLZ Multiple Testing Correction (Harvey, Liu, Zhu 2016)
# ---------------------------------------------------------------------------

def hlz_threshold(n_tests: int, significance: float = 0.05) -> float:
    """Compute the HLZ-adjusted t-statistic threshold.

    Harvey-Liu-Zhu (2016) "...and the Cross-Section of Expected Returns"
    argues that with the modern factor zoo, a t-stat > 3.0 is the minimum
    for a new factor to be credible after accounting for multiple testing.

    Returns:
        Minimum |t-stat| required for factor admission.
    """
    if n_tests <= 1:
        return 1.96
    bonferroni_p = significance / n_tests
    from scipy.stats import norm
    bonferroni_t = float(norm.ppf(1 - bonferroni_p / 2))
    return max(bonferroni_t, 3.0)


def ic_t_statistic(ic_series: np.ndarray) -> float:
    """Compute t-statistic for H0: mean IC = 0."""
    valid = ic_series[~np.isnan(ic_series)]
    n = len(valid)
    if n < 10:
        return 0.0
    mean_ic = np.mean(valid)
    std_ic = np.std(valid, ddof=1)
    if std_ic < 1e-10:
        return 0.0
    return float(mean_ic / (std_ic / np.sqrt(n)))


# ---------------------------------------------------------------------------
# Out-of-Sample (OOS) Validation
# ---------------------------------------------------------------------------

def oos_validate(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    train_frac: float = 0.6,
    purge_gap: int = 5,
) -> dict:
    """Split signal into in-sample and out-of-sample, measure IC degradation.

    Returns dict with is_ic, oos_ic, degradation (fraction of IC lost OOS).
    """
    M, T = signal.shape
    split_t = int(T * train_frac)
    oos_start = split_t + purge_gap

    if oos_start >= T - 20:
        return {"is_ic": 0.0, "oos_ic": 0.0, "degradation": 1.0, "oos_valid": False}

    is_ic_series = compute_ic(signal[:, :split_t], forward_returns[:, :split_t])
    oos_ic_series = compute_ic(signal[:, oos_start:], forward_returns[:, oos_start:])

    is_valid = is_ic_series[~np.isnan(is_ic_series)]
    oos_valid = oos_ic_series[~np.isnan(oos_ic_series)]

    is_ic = float(np.mean(is_valid)) if len(is_valid) > 0 else 0.0
    oos_ic = float(np.mean(oos_valid)) if len(oos_valid) > 0 else 0.0

    if abs(is_ic) > 1e-10:
        degradation = 1.0 - abs(oos_ic) / abs(is_ic)
    else:
        degradation = 1.0

    return {
        "is_ic": is_ic,
        "oos_ic": oos_ic,
        "degradation": degradation,
        "oos_valid": len(oos_valid) >= 20,
    }


def compute_correlation(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """Time-average cross-sectional Spearman correlation (Eq. 3 from paper)."""
    M, T = signal_a.shape
    corrs = []
    for t in range(T):
        a = signal_a[:, t]
        b = signal_b[:, t]
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() >= 10:
            corr, _ = sp_stats.spearmanr(a[valid], b[valid])
            if np.isfinite(corr):
                corrs.append(corr)
    if not corrs:
        return 0.0
    return float(np.mean(corrs))


class FactorRecord:
    """A single factor record in the library."""

    def __init__(
        self,
        expression: str,
        ic_mean: float = 0.0,
        ic_std: float = 0.0,
        icir: float = 0.0,
        max_correlation: float = 0.0,
        turnover: float = 0.0,
        logic_description: str = "",
        admitted_at: float = 0.0,
        mining_round: int = 0,
        factor_id: str = "",
    ):
        self.expression = expression
        self.ic_mean = ic_mean
        self.ic_std = ic_std
        self.icir = icir
        self.max_correlation = max_correlation
        self.turnover = turnover
        self.logic_description = logic_description
        self.admitted_at = admitted_at or time.time()
        self.mining_round = mining_round
        self.factor_id = factor_id or f"F{int(self.admitted_at * 1000) % 1000000:06d}"

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "expression": self.expression,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "max_correlation": self.max_correlation,
            "turnover": self.turnover,
            "logic_description": self.logic_description,
            "admitted_at": self.admitted_at,
            "mining_round": self.mining_round,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactorRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class FactorLibrary:
    """Manages the growing collection of validated factors."""

    def __init__(self, path: Path = FACTOR_LIBRARY_PATH):
        self.path = path
        self.factors: list[FactorRecord] = []
        self._signal_cache: dict[str, np.ndarray] = {}
        self._cache_valid = False
        self._dirty = False
        self.load()

    @property
    def size(self) -> int:
        return len(self.factors)

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.factors = [FactorRecord.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.factors)} factors from {self.path}")
            except Exception as e:
                logger.error(f"Failed to load factor library: {e}")
                self.factors = []

    def save(self):
        with open(self.path, "w") as f:
            json.dump([fr.to_dict() for fr in self.factors], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.factors)} factors to {self.path}")

    def get_signal(
        self,
        expression: str,
        engine,
    ) -> Optional[np.ndarray]:
        """Evaluate a factor expression and cache the result."""
        if expression in self._signal_cache:
            return self._signal_cache[expression]
        try:
            signal = engine.evaluate(expression)
            self._signal_cache[expression] = signal
            return signal
        except Exception as e:
            logger.warning(f"Failed to evaluate '{expression}': {e}")
            return None

    def clear_signal_cache(self):
        self._signal_cache.clear()
        self._cache_valid = False

    def check_admission(
        self,
        candidate_signal: np.ndarray,
        candidate_ic_series: np.ndarray,
        ic_threshold: float = IC_THRESHOLD,
        corr_threshold: float = CORR_THRESHOLD,
        engine=None,
        forward_returns: Optional[np.ndarray] = None,
        use_hlz: bool = True,
        use_oos: bool = True,
        n_total_tested: int = 0,
    ) -> tuple[bool, str, Optional[int]]:
        """Full admission pipeline with HLZ correction and OOS validation.

        Stages:
          1. IC threshold check
          2. HLZ multiple testing t-stat check (if use_hlz)
          3. OOS degradation check (if use_oos and forward_returns provided)
          4. Correlation deduplication
          5. Replacement check

        Returns:
            (admitted, reason, replace_index)
        """
        valid_ic = candidate_ic_series[~np.isnan(candidate_ic_series)]
        if len(valid_ic) < 10:
            return False, "Insufficient IC data points", None

        abs_ic_mean = abs(float(np.mean(valid_ic)))
        if abs_ic_mean < ic_threshold:
            return False, f"IC too low: {abs_ic_mean:.4f} < {ic_threshold}", None

        # Stage: HLZ multiple testing correction
        if use_hlz:
            n_tests = max(n_total_tested, self.size + 1, 10)
            t_stat = ic_t_statistic(candidate_ic_series)
            t_threshold = hlz_threshold(n_tests)
            if abs(t_stat) < t_threshold:
                return False, (
                    f"HLZ rejected: |t|={abs(t_stat):.2f} < {t_threshold:.2f} "
                    f"(n_tests={n_tests})"
                ), None

        # Stage: OOS validation
        if use_oos and forward_returns is not None:
            oos = oos_validate(candidate_signal, forward_returns)
            if oos["oos_valid"] and oos["degradation"] > 0.7:
                return False, (
                    f"OOS degradation too high: {oos['degradation']:.0%} "
                    f"(IS IC={oos['is_ic']:.4f}, OOS IC={oos['oos_ic']:.4f})"
                ), None

        if self.size == 0:
            return True, f"First factor, IC={abs_ic_mean:.4f}", None

        max_corr = 0.0
        max_corr_idx = -1
        for i, existing in enumerate(self.factors):
            if existing.expression in self._signal_cache:
                existing_signal = self._signal_cache[existing.expression]
            elif engine:
                existing_signal = self.get_signal(existing.expression, engine)
            else:
                continue

            if existing_signal is None:
                continue

            corr = abs(compute_correlation(candidate_signal, existing_signal))
            if corr > max_corr:
                max_corr = corr
                max_corr_idx = i

        if max_corr < corr_threshold:
            return True, f"Admitted: IC={abs_ic_mean:.4f}, max_corr={max_corr:.4f}", None

        if max_corr_idx >= 0:
            existing_ic = abs(self.factors[max_corr_idx].ic_mean)
            if abs_ic_mean > existing_ic * 1.1:
                return True, (
                    f"Replacement: IC={abs_ic_mean:.4f} > {existing_ic:.4f}, "
                    f"corr={max_corr:.4f}"
                ), max_corr_idx

        return False, f"Too correlated: max_corr={max_corr:.4f} >= {corr_threshold}", None

    def admit(
        self,
        record: FactorRecord,
        replace_index: Optional[int] = None,
    ):
        """Add a factor to the library, optionally replacing an existing one."""
        if replace_index is not None and 0 <= replace_index < len(self.factors):
            old = self.factors[replace_index]
            logger.info(
                f"Replacing {old.expression} (IC={old.ic_mean:.4f}) "
                f"with {record.expression} (IC={record.ic_mean:.4f})"
            )
            self._signal_cache.pop(old.expression, None)
            self.factors[replace_index] = record
        else:
            self.factors.append(record)
            logger.info(f"Admitted factor #{self.size}: {record.expression} (IC={record.ic_mean:.4f})")
        self._dirty = True
        self.save()

    def ensure_signals_cached(self, engine) -> None:
        """Compute signals for all library factors if not already cached."""
        if self._cache_valid:
            return
        for fr in self.factors:
            if fr.expression not in self._signal_cache:
                try:
                    self._signal_cache[fr.expression] = engine.evaluate(fr.expression)
                except Exception:
                    pass
        self._cache_valid = True

    def get_all_records(self) -> list[dict]:
        return [fr.to_dict() for fr in self.factors]

    def get_record_by_id(self, factor_id: str) -> Optional[FactorRecord]:
        for fr in self.factors:
            if fr.factor_id == factor_id:
                return fr
        return None
