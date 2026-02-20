"""Factor Combiner: combines mined factors into a high-IC composite signal.

Uses LightGBM (or Ridge fallback) to learn non-linear factor interactions,
with walk-forward training to prevent lookahead bias.

Key insight from research:
- Individual formulaic factors typically have |IC| ~ 0.03-0.06
- GBDT/MLP combination routinely achieves |IC| ~ 0.05-0.10+
- This is because non-linear interactions between factors carry alpha
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class CombinerConfig:
    train_window: int = 500
    test_step: int = 60
    purge_gap: int = 5
    backend: str = "auto"
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 50
    ridge_alpha: float = 1.0


@dataclass
class CombinerReport:
    ic_series: np.ndarray = field(default_factory=lambda: np.array([]))
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    ic_positive_ratio: float = 0.0
    n_test_days: int = 0
    feature_importance: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    backend_used: str = ""
    individual_ics: dict = field(default_factory=dict)
    improvement_pct: float = 0.0


class FactorCombiner:
    """Walk-forward factor combination for IC maximization.

    Takes a dict of factor signals {name: (M, T)} and forward returns (M, T),
    trains a model to predict returns from cross-sectional factor values,
    and measures the combined signal's IC.
    """

    def __init__(self, config: CombinerConfig = None):
        self.config = config or CombinerConfig()
        self._resolve_backend()

    def _resolve_backend(self):
        cfg = self.config
        if cfg.backend == "auto":
            if HAS_LGB:
                cfg.backend = "lightgbm"
            elif HAS_SKLEARN:
                cfg.backend = "hgbr"
            else:
                raise RuntimeError("No ML backend available. Install lightgbm or scikit-learn.")

    def run(
        self,
        signals: dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> CombinerReport:
        """Walk-forward evaluation of factor combination.

        Args:
            signals: Dict mapping factor name -> signal array (M, T).
            forward_returns: Forward returns (M, T).

        Returns:
            CombinerReport with IC metrics and feature importance.
        """
        t0 = time.time()
        cfg = self.config
        factor_names = list(signals.keys())
        signal_list = [signals[n] for n in factor_names]
        M, T = signal_list[0].shape
        n_factors = len(factor_names)

        logger.info(f"Combining {n_factors} factors, {M} stocks x {T} days, backend={cfg.backend}")

        factor_matrix = np.stack(signal_list, axis=-1)

        first_test = cfg.train_window + cfg.purge_gap
        if first_test >= T:
            raise ValueError(f"Not enough data: need {first_test} days but have {T}")

        individual_ics = {}
        for name, sig in signals.items():
            ics = []
            for t in range(T):
                s, r = sig[:, t], forward_returns[:, t]
                valid = ~(np.isnan(s) | np.isnan(r))
                if valid.sum() >= 20:
                    rho, _ = sp_stats.spearmanr(s[valid], r[valid])
                    if np.isfinite(rho):
                        ics.append(rho)
            individual_ics[name] = float(np.mean(ics)) if ics else 0.0

        all_ic = []
        all_importances = np.zeros(n_factors)
        n_folds = 0

        test_start = first_test
        while test_start < T:
            test_end = min(test_start + cfg.test_step, T)
            train_start = max(0, test_start - cfg.train_window)
            train_end = test_start - cfg.purge_gap

            X_train, y_train = self._build_xy(factor_matrix, forward_returns, train_start, train_end)

            if len(X_train) < 100:
                test_start = test_end
                continue

            model, imp = self._fit(X_train, y_train)
            all_importances += imp
            n_folds += 1

            for t in range(test_start, test_end):
                X_test = factor_matrix[:, t, :]
                y_test = forward_returns[:, t]
                valid = ~(np.isnan(y_test) | np.any(np.isnan(X_test), axis=1))
                if valid.sum() < 20:
                    continue

                pred = self._predict(model, X_test[valid])
                rho, _ = sp_stats.spearmanr(pred, y_test[valid])
                if np.isfinite(rho):
                    all_ic.append(rho)

            test_start = test_end

        ic_array = np.array(all_ic)
        report = CombinerReport()
        report.ic_series = ic_array
        report.n_test_days = len(ic_array)
        report.backend_used = cfg.backend
        report.individual_ics = individual_ics
        report.elapsed_seconds = time.time() - t0

        if len(ic_array) > 0:
            report.ic_mean = float(np.mean(ic_array))
            report.ic_std = float(np.std(ic_array))
            report.icir = report.ic_mean / report.ic_std if report.ic_std > 1e-10 else 0.0
            report.ic_positive_ratio = float(np.mean(ic_array > 0))

        if n_folds > 0:
            avg_imp = all_importances / n_folds
            total_imp = avg_imp.sum() or 1.0
            report.feature_importance = {
                factor_names[i]: float(avg_imp[i] / total_imp)
                for i in range(n_factors)
            }

        best_individual = max(abs(v) for v in individual_ics.values()) if individual_ics else 0
        if best_individual > 0:
            report.improvement_pct = (abs(report.ic_mean) - best_individual) / best_individual * 100

        return report

    def _build_xy(
        self,
        factor_matrix: np.ndarray,
        forward_returns: np.ndarray,
        t_start: int,
        t_end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build training data from cross-sectional slices."""
        Xs, ys = [], []
        for t in range(t_start, t_end):
            x = factor_matrix[:, t, :]
            y = forward_returns[:, t]
            valid = ~(np.isnan(y) | np.any(np.isnan(x), axis=1))
            if valid.sum() >= 20:
                Xs.append(x[valid])
                ys.append(y[valid])
        if not Xs:
            return np.array([]), np.array([])
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Train model, return (model, feature_importance)."""
        cfg = self.config
        if cfg.backend == "lightgbm" and HAS_LGB:
            return self._fit_lgb(X, y)
        elif cfg.backend == "hgbr" and HAS_SKLEARN:
            return self._fit_hgbr(X, y)
        else:
            return self._fit_ridge(X, y)

    def _fit_lgb(self, X: np.ndarray, y: np.ndarray):
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.colsample_bytree,
            "bagging_fraction": self.config.subsample,
            "bagging_freq": 5,
            "min_child_samples": self.config.min_child_samples,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": -1,
        }
        n = len(y)
        val_size = int(n * 0.15)
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[val_size:], idx[:val_size]

        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)
        model = lgb.train(
            params, dtrain,
            num_boost_round=self.config.n_estimators,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=0),
            ],
        )
        importance = model.feature_importance(importance_type="gain").astype(float)
        return model, importance

    def _fit_hgbr(self, X: np.ndarray, y: np.ndarray):
        """HistGradientBoosting: fast, built-in, handles NaN natively."""
        model = HistGradientBoostingRegressor(
            max_iter=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_child_samples,
            l2_regularization=1.0,
            max_bins=128,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.15,
            random_state=42,
        )
        X_clean = np.nan_to_num(X, nan=0.0)
        model.fit(X_clean, y)
        importance = np.zeros(X.shape[1])
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_.astype(float)
        return ("hgbr", model), importance

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray):
        scaler = StandardScaler()
        X_clean = np.nan_to_num(X, nan=0.0)
        X_scaled = scaler.fit_transform(X_clean)
        model = Ridge(alpha=self.config.ridge_alpha)
        model.fit(X_scaled, y)
        importance = np.abs(model.coef_)
        return ("ridge", model, scaler), importance

    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0.0)
        if self.config.backend == "lightgbm" and HAS_LGB:
            return model.predict(X_clean)
        elif isinstance(model, tuple) and model[0] == "hgbr":
            return model[1].predict(X_clean)
        elif isinstance(model, tuple) and model[0] == "ridge":
            _, model_obj, scaler = model
            return model_obj.predict(scaler.transform(X_clean))
        else:
            return model.predict(X_clean)

    @staticmethod
    def print_report(report: CombinerReport):
        W = 72
        print()
        print("=" * W)
        print("  Factor Combiner Report".center(W))
        print("=" * W)

        print(f"\n  Backend:          {report.backend_used}")
        print(f"  Test Days:        {report.n_test_days}")
        print(f"  Elapsed:          {report.elapsed_seconds:.1f}s")

        print(f"\n  Combined IC Mean: {report.ic_mean:+.4f}")
        print(f"  Combined IC Std:  {report.ic_std:.4f}")
        print(f"  Combined ICIR:    {report.icir:+.3f}")
        print(f"  IC Positive %:    {report.ic_positive_ratio:.1%}")

        print(f"\n  --- Individual Factor ICs ---")
        for name, ic in sorted(report.individual_ics.items(), key=lambda x: -abs(x[1])):
            print(f"    {name[:55]:55s}  IC={ic:+.4f}")

        if report.feature_importance:
            print(f"\n  --- Feature Importance ---")
            for name, imp in sorted(report.feature_importance.items(), key=lambda x: -x[1]):
                bar = "#" * int(imp * 40)
                print(f"    {name[:45]:45s}  {imp:.1%}  {bar}")

        delta = "+" if report.improvement_pct > 0 else ""
        print(f"\n  Improvement vs best individual: {delta}{report.improvement_pct:.1f}%")

        if abs(report.ic_mean) >= 0.05 and abs(report.icir) >= 0.5:
            verdict = "GOOD - Production candidate"
        elif abs(report.ic_mean) >= 0.03:
            verdict = "PASS - Usable with tuning"
        else:
            verdict = "WEAK - Needs more/better factors"
        print(f"  Verdict: {verdict}")
        print("=" * W)
        print()
