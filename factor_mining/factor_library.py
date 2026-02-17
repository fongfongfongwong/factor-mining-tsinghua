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

logger = logging.getLogger(__name__)


def compute_ic(signal: np.ndarray, forward_returns: np.ndarray) -> np.ndarray:
    """Compute Spearman rank IC per timestamp (Eq. 2 from paper).

    Args:
        signal: Factor signal array (M, T).
        forward_returns: Forward returns array (M, T).

    Returns:
        1D array of IC values per timestamp (length T).
    """
    M, T = signal.shape
    ic_series = np.full(T, np.nan)
    for t in range(T):
        s = signal[:, t]
        r = forward_returns[:, t]
        valid = ~(np.isnan(s) | np.isnan(r))
        if valid.sum() >= 10:
            corr, _ = sp_stats.spearmanr(s[valid], r[valid])
            ic_series[t] = corr
    return ic_series


def compute_icir(ic_series: np.ndarray) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < 10:
        return 0.0
    std = np.std(valid)
    if std < 1e-10:
        return 0.0
    return float(np.mean(valid) / std)


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

    def check_admission(
        self,
        candidate_signal: np.ndarray,
        candidate_ic_series: np.ndarray,
        ic_threshold: float = IC_THRESHOLD,
        corr_threshold: float = CORR_THRESHOLD,
        engine=None,
    ) -> tuple[bool, str, Optional[int]]:
        """Full admission pipeline for a candidate factor.

        Returns:
            (admitted, reason, replace_index)
            - admitted: True if factor passes all checks
            - reason: Description of outcome
            - replace_index: If not None, index of existing factor to replace
        """
        valid_ic = candidate_ic_series[~np.isnan(candidate_ic_series)]
        if len(valid_ic) < 10:
            return False, "Insufficient IC data points", None

        abs_ic_mean = abs(float(np.mean(valid_ic)))
        if abs_ic_mean < ic_threshold:
            return False, f"IC too low: {abs_ic_mean:.4f} < {ic_threshold}", None

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
            if old.expression in self._signal_cache:
                del self._signal_cache[old.expression]
            self.factors[replace_index] = record
        else:
            self.factors.append(record)
            logger.info(f"Admitted factor #{self.size}: {record.expression} (IC={record.ic_mean:.4f})")
        self.save()

    def get_all_signals(self, engine) -> dict[str, np.ndarray]:
        """Compute signals for all factors in the library."""
        signals = {}
        for fr in self.factors:
            sig = self.get_signal(fr.expression, engine)
            if sig is not None:
                signals[fr.expression] = sig
        return signals

    def get_all_records(self) -> list[dict]:
        return [fr.to_dict() for fr in self.factors]

    def get_record_by_id(self, factor_id: str) -> Optional[FactorRecord]:
        for fr in self.factors:
            if fr.factor_id == factor_id:
                return fr
        return None
