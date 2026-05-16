"""Linear Regression and Ridge baseline models for capacity prediction."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_linear_pipeline(alpha: float = 1.0) -> Pipeline:
    """Create a Ridge regression pipeline with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
