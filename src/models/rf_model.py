"""Random Forest and Gradient Boosting models for capacity prediction.

Includes quantile regression for confidence intervals.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_rf_pipeline(n_estimators: int = 200, max_depth: int = 10) -> Pipeline:
    """Create a Random Forest regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_quantile_gb(alpha: float = 0.5, n_estimators: int = 200) -> Pipeline:
    """Create a Gradient Boosting quantile regressor for confidence intervals.

    alpha=0.5 for median, 0.05 for lower bound, 0.95 for upper bound.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )),
    ])
