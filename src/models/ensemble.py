"""Ensemble strategies for combining multiple battery capacity predictors.

Four strategies that tell the innovation story:

    Strategy 1: Weighted Ensemble
        Weight models by inverse validation error. Simple but effective baseline.

    Strategy 2: Stacking Meta-Learner
        Train a Ridge regression on base model outputs. Learns optimal
        combination weights from data.

    Strategy 3: Lifecycle-Adaptive Switching
        Different models dominate at different degradation stages:
        - Early life (SOH > 0.95): PINN dominates (physics governs fresh batteries)
        - Mid life (0.80 < SOH < 0.95): Stacking blend (data patterns emerge)
        - Late life (SOH < 0.80): RF/Transformer dominate (nonlinear degradation)

    Strategy 4: Physics-Constrained Ensemble
        Ensemble prediction with PINN as a hard constraint:
        - Clip predictions to PINN's physics-valid range
        - Use PINN uncertainty to reweight other models
        - Enforce monotonicity from physics model

Each strategy outputs predictions + confidence intervals + a "strategy_weight"
vector showing how much each base model contributes (for dashboard visualization).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# Base model names in fixed order
BASE_MODELS = ["linear", "rf", "transformer", "pinn"]


@dataclass(frozen=True)
class EnsembleResult:
    """Output from an ensemble strategy."""

    predictions: np.ndarray          # (n_samples,) final predictions
    lower: np.ndarray                # (n_samples,) lower CI bound
    upper: np.ndarray                # (n_samples,) upper CI bound
    weights: np.ndarray              # (n_samples, n_models) per-sample model weights
    strategy_name: str               # human-readable name
    meta_info: dict = field(default_factory=dict)  # extra info for visualization


def _valid_mask(preds: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean mask where ALL base models have valid (non-NaN) predictions."""
    masks = [~np.isnan(preds[m]) for m in BASE_MODELS if m in preds]
    if not masks:
        return np.array([], dtype=bool)
    combined = masks[0]
    for m in masks[1:]:
        combined = combined & m
    return combined


def _fill_nan_with_fallback(
    preds: dict[str, np.ndarray],
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Fill NaN predictions with the mean of valid models at each sample."""
    filled = {}
    for model_name in BASE_MODELS:
        if model_name in preds:
            filled[model_name] = preds[model_name].copy()
        else:
            filled[model_name] = np.full(n_samples, np.nan)

    # For each sample, fill NaN with mean of valid models
    for i in range(n_samples):
        valid_vals = [
            filled[m][i] for m in BASE_MODELS if not np.isnan(filled[m][i])
        ]
        if valid_vals:
            mean_val = np.mean(valid_vals)
            for m in BASE_MODELS:
                if np.isnan(filled[m][i]):
                    filled[m][i] = mean_val

    return filled


# ============================================================
# Strategy 1: Weighted Ensemble (inverse-error weighting)
# ============================================================
def weighted_ensemble(
    preds: dict[str, np.ndarray],
    errors: dict[str, float],
    ci_lower: dict[str, np.ndarray] | None = None,
    ci_upper: dict[str, np.ndarray] | None = None,
) -> EnsembleResult:
    """Combine models weighted by inverse RMSE.

    Models with lower error get higher weight.
    Weight_i = (1/RMSE_i) / sum(1/RMSE_j)

    Args:
        preds: {model_name: predictions} for each base model
        errors: {model_name: RMSE} validation errors
        ci_lower/ci_upper: optional confidence interval bounds per model
    """
    n_samples = max(len(v) for v in preds.values())
    filled = _fill_nan_with_fallback(preds, n_samples)

    # Compute inverse-error weights (filter out NaN/zero errors)
    raw_weights = {}
    for m in BASE_MODELS:
        if m in errors and not np.isnan(errors[m]) and errors[m] > 0:
            raw_weights[m] = 1.0 / errors[m]
        else:
            raw_weights[m] = 0.0

    total_w = sum(raw_weights.values())
    if total_w == 0:
        total_w = 1.0

    norm_weights = {m: raw_weights[m] / total_w for m in BASE_MODELS}

    # Weighted prediction
    final_pred = np.zeros(n_samples)
    weight_matrix = np.zeros((n_samples, len(BASE_MODELS)))
    for j, m in enumerate(BASE_MODELS):
        w = norm_weights[m]
        final_pred += w * filled[m]
        weight_matrix[:, j] = w

    # Confidence interval: weighted combination of individual CIs
    if ci_lower and ci_upper:
        lb = np.zeros(n_samples)
        ub = np.zeros(n_samples)
        for m in BASE_MODELS:
            w = norm_weights[m]
            if m in ci_lower and m in ci_upper:
                lb_m = np.where(np.isnan(ci_lower[m]), final_pred, ci_lower[m])
                ub_m = np.where(np.isnan(ci_upper[m]), final_pred, ci_upper[m])
                lb += w * lb_m
                ub += w * ub_m
            else:
                lb += w * filled[m]
                ub += w * filled[m]
    else:
        # Estimate CI from model disagreement
        all_preds = np.column_stack([filled[m] for m in BASE_MODELS])
        std = np.nanstd(all_preds, axis=1)
        lb = final_pred - 1.96 * std
        ub = final_pred + 1.96 * std

    return EnsembleResult(
        predictions=final_pred,
        lower=lb,
        upper=ub,
        weights=weight_matrix,
        strategy_name="Weighted Ensemble",
        meta_info={"model_weights": norm_weights},
    )


# ============================================================
# Strategy 2: Stacking Meta-Learner
# ============================================================
def stacking_ensemble(
    train_preds: dict[str, np.ndarray],
    train_targets: np.ndarray,
    test_preds: dict[str, np.ndarray],
    ci_lower: dict[str, np.ndarray] | None = None,
    ci_upper: dict[str, np.ndarray] | None = None,
) -> EnsembleResult:
    """Train a Ridge meta-learner on base model outputs.

    The meta-learner learns optimal blending weights from the training
    predictions of each base model (out-of-fold predictions).

    Args:
        train_preds: {model_name: train_predictions} used to fit meta-learner
        train_targets: actual training targets
        test_preds: {model_name: test_predictions} to generate final output
    """
    n_train = len(train_targets)
    n_test = max(len(v) for v in test_preds.values())

    # Build feature matrices from base model predictions
    train_filled = _fill_nan_with_fallback(train_preds, n_train)
    test_filled = _fill_nan_with_fallback(test_preds, n_test)

    # Only use samples where all models have valid predictions for training
    X_train = np.column_stack([train_filled[m] for m in BASE_MODELS])
    X_test = np.column_stack([test_filled[m] for m in BASE_MODELS])

    # Filter valid training rows
    valid_train = ~np.any(np.isnan(X_train), axis=1)
    X_train_clean = X_train[valid_train]
    y_train_clean = train_targets[valid_train]

    if len(X_train_clean) < 5:
        logger.warning("Stacking: too few valid training samples, falling back to equal weights")
        equal_weights = {m: 1.0 / len(BASE_MODELS) for m in BASE_MODELS}
        return weighted_ensemble(test_preds, {m: 1.0 for m in BASE_MODELS}, ci_lower, ci_upper)

    # Fit Ridge meta-learner (non-negative coefficients preferred)
    meta = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(X_train_clean, y_train_clean)

    final_pred = meta.predict(X_test)

    # Extract learned weights (coefficients show model contribution)
    coefs = meta.coef_
    coef_abs = np.abs(coefs)
    coef_sum = coef_abs.sum()
    if coef_sum > 0:
        norm_coefs = coef_abs / coef_sum
    else:
        norm_coefs = np.ones(len(BASE_MODELS)) / len(BASE_MODELS)

    weight_matrix = np.tile(norm_coefs, (n_test, 1))

    # CI from prediction variance
    all_test = np.column_stack([test_filled[m] for m in BASE_MODELS])
    std = np.nanstd(all_test, axis=1)
    lb = final_pred - 1.96 * std
    ub = final_pred + 1.96 * std

    learned_weights = {m: float(norm_coefs[j]) for j, m in enumerate(BASE_MODELS)}

    return EnsembleResult(
        predictions=final_pred,
        lower=lb,
        upper=ub,
        weights=weight_matrix,
        strategy_name="Stacking Meta-Learner",
        meta_info={
            "model_weights": learned_weights,
            "ridge_coefs": coefs.tolist(),
            "ridge_intercept": float(meta.intercept_),
        },
    )


# ============================================================
# Strategy 3: Lifecycle-Adaptive Switching
# ============================================================
def lifecycle_adaptive_ensemble(
    preds: dict[str, np.ndarray],
    soh_values: np.ndarray,
    errors: dict[str, float],
    ci_lower: dict[str, np.ndarray] | None = None,
    ci_upper: dict[str, np.ndarray] | None = None,
    early_threshold: float = 0.95,
    late_threshold: float = 0.80,
) -> EnsembleResult:
    """Adaptively switch model weights based on battery lifecycle stage.

    Insight: different models excel at different degradation phases:
    - Early life (SOH > 0.95): Physics dominates → PINN gets highest weight
    - Mid life (0.80 < SOH < 0.95): All models blend → stacking-like weights
    - Late life (SOH < 0.80): Data-driven models capture nonlinearity → RF/Transformer up

    Smooth transitions via sigmoid blending, not hard switches.

    Args:
        preds: base model predictions
        soh_values: SOH at each prediction point (capacity / rated)
        errors: per-model RMSE for fallback weighting
    """
    n_samples = len(soh_values)
    filled = _fill_nan_with_fallback(preds, n_samples)

    # Define phase-specific weight profiles
    # [linear, rf, transformer, pinn]
    early_weights = np.array([0.15, 0.10, 0.15, 0.60])   # PINN dominates
    mid_weights = np.array([0.25, 0.30, 0.20, 0.25])      # balanced blend
    late_weights = np.array([0.15, 0.40, 0.30, 0.15])      # RF + Transformer

    # Smooth blending via sigmoid transitions
    weight_matrix = np.zeros((n_samples, len(BASE_MODELS)))
    for i in range(n_samples):
        soh = soh_values[i]

        # Sigmoid blend: early → mid transition
        alpha_early = 1.0 / (1.0 + np.exp(-20 * (soh - early_threshold)))
        # Sigmoid blend: mid → late transition
        alpha_late = 1.0 / (1.0 + np.exp(20 * (soh - late_threshold)))

        # Blend: early_weights * alpha_early + late_weights * alpha_late
        #        + mid_weights * (1 - alpha_early - alpha_late)
        mid_frac = max(0, 1.0 - alpha_early - alpha_late)
        w = alpha_early * early_weights + alpha_late * late_weights + mid_frac * mid_weights
        w = w / w.sum()  # normalize
        weight_matrix[i] = w

    # Weighted prediction per sample
    final_pred = np.zeros(n_samples)
    for j, m in enumerate(BASE_MODELS):
        final_pred += weight_matrix[:, j] * filled[m]

    # CI from weighted model disagreement
    all_preds = np.column_stack([filled[m] for m in BASE_MODELS])
    disagreement = np.zeros(n_samples)
    for i in range(n_samples):
        weighted_var = sum(
            weight_matrix[i, j] * (all_preds[i, j] - final_pred[i]) ** 2
            for j in range(len(BASE_MODELS))
        )
        disagreement[i] = np.sqrt(weighted_var)

    lb = final_pred - 1.96 * disagreement
    ub = final_pred + 1.96 * disagreement

    # Compute average weights per phase for visualization
    early_mask = soh_values > early_threshold
    mid_mask = (soh_values <= early_threshold) & (soh_values > late_threshold)
    late_mask = soh_values <= late_threshold

    phase_avg_weights = {}
    for phase_name, mask in [("early", early_mask), ("mid", mid_mask), ("late", late_mask)]:
        if mask.any():
            phase_avg_weights[phase_name] = {
                m: float(weight_matrix[mask, j].mean())
                for j, m in enumerate(BASE_MODELS)
            }

    return EnsembleResult(
        predictions=final_pred,
        lower=lb,
        upper=ub,
        weights=weight_matrix,
        strategy_name="Lifecycle-Adaptive",
        meta_info={
            "phase_weights": phase_avg_weights,
            "early_threshold": early_threshold,
            "late_threshold": late_threshold,
            "phase_counts": {
                "early": int(early_mask.sum()),
                "mid": int(mid_mask.sum()),
                "late": int(late_mask.sum()),
            },
        },
    )


# ============================================================
# Strategy 4: Physics-Constrained Ensemble
# ============================================================
def physics_constrained_ensemble(
    preds: dict[str, np.ndarray],
    pinn_uncertainty: np.ndarray,
    physics_baseline: np.ndarray,
    rated_capacity: float,
    errors: dict[str, float],
    ci_lower: dict[str, np.ndarray] | None = None,
    ci_upper: dict[str, np.ndarray] | None = None,
) -> EnsembleResult:
    """Ensemble with PINN as a physics constraint layer.

    Innovation: Use the physics model not just as a predictor, but as a
    constraint that filters and adjusts other models' outputs:

    1. Clip all predictions to physics-valid range [0, rated * 1.05]
    2. Penalize predictions that deviate too far from physics baseline
    3. Enforce approximate monotonicity (smooth with physics trend)
    4. Use PINN uncertainty to reweight: high PINN confidence → trust PINN more

    Args:
        preds: base model predictions
        pinn_uncertainty: aleatoric uncertainty from PINN (lower = more confident)
        physics_baseline: ECM-based capacity at each cycle
        rated_capacity: nominal rated capacity
        errors: per-model RMSE
    """
    n_samples = max(len(v) for v in preds.values())
    filled = _fill_nan_with_fallback(preds, n_samples)

    # Step 1: Physics-valid clipping
    cap_max = rated_capacity * 1.05
    for m in BASE_MODELS:
        filled[m] = np.clip(filled[m], 0.0, cap_max)

    # Step 2: PINN confidence-based reweighting
    # Normalize uncertainty to [0, 1] range
    if len(pinn_uncertainty) > 0 and not np.all(np.isnan(pinn_uncertainty)):
        u_min = np.nanmin(pinn_uncertainty)
        u_max = np.nanmax(pinn_uncertainty)
        u_range = u_max - u_min if u_max > u_min else 1.0
        pinn_confidence = 1.0 - (pinn_uncertainty - u_min) / u_range
        pinn_confidence = np.clip(pinn_confidence, 0.1, 0.9)
    else:
        pinn_confidence = np.full(n_samples, 0.5)

    # Pad to n_samples if needed
    if len(pinn_confidence) < n_samples:
        padded = np.full(n_samples, 0.5)
        padded[: len(pinn_confidence)] = pinn_confidence
        pinn_confidence = padded

    # Step 3: Dynamic weighting based on physics proximity
    weight_matrix = np.zeros((n_samples, len(BASE_MODELS)))

    # Base weights from inverse error
    base_w = np.zeros(len(BASE_MODELS))
    for j, m in enumerate(BASE_MODELS):
        if m in errors and not np.isnan(errors[m]) and errors[m] > 0:
            base_w[j] = 1.0 / errors[m]
    if base_w.sum() > 0:
        base_w = base_w / base_w.sum()
    else:
        base_w = np.ones(len(BASE_MODELS)) / len(BASE_MODELS)

    pinn_idx = BASE_MODELS.index("pinn")

    for i in range(n_samples):
        w = base_w.copy()

        # Boost PINN weight when it's confident
        conf = pinn_confidence[i]
        pinn_boost = 1.0 + 2.0 * conf  # 1x to 3x boost
        w[pinn_idx] *= pinn_boost

        # Penalize models that deviate far from physics baseline
        if i < len(physics_baseline) and not np.isnan(physics_baseline[i]):
            for j, m in enumerate(BASE_MODELS):
                deviation = abs(filled[m][i] - physics_baseline[i])
                # Soft penalty: models far from physics lose weight
                penalty = np.exp(-deviation / (rated_capacity * 0.1))
                w[j] *= penalty

        w = w / w.sum()
        weight_matrix[i] = w

    # Step 4: Weighted prediction
    final_pred = np.zeros(n_samples)
    for j, m in enumerate(BASE_MODELS):
        final_pred += weight_matrix[:, j] * filled[m]

    # Step 5: Enforce approximate monotonicity (smooth with EMA)
    # Only smooth if we have enough points
    if n_samples > 5:
        smoothed = final_pred.copy()
        alpha = 0.3  # EMA smoothing factor
        for i in range(1, n_samples):
            if smoothed[i] > smoothed[i - 1] + 0.005:
                # Capacity increase exceeds tolerance → smooth towards previous
                smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i - 1]
        final_pred = smoothed

    # CI
    all_preds = np.column_stack([filled[m] for m in BASE_MODELS])
    std = np.nanstd(all_preds, axis=1)
    lb = np.clip(final_pred - 1.96 * std, 0, cap_max)
    ub = np.clip(final_pred + 1.96 * std, 0, cap_max)

    return EnsembleResult(
        predictions=final_pred,
        lower=lb,
        upper=ub,
        weights=weight_matrix,
        strategy_name="Physics-Constrained Ensemble",
        meta_info={
            "avg_pinn_confidence": float(np.nanmean(pinn_confidence)),
            "avg_weights": {
                m: float(weight_matrix[:, j].mean())
                for j, m in enumerate(BASE_MODELS)
            },
        },
    )


# ============================================================
# Orchestrator: run all strategies and pick the best
# ============================================================
def run_all_ensemble_strategies(
    train_preds: dict[str, np.ndarray],
    train_targets: np.ndarray,
    test_preds: dict[str, np.ndarray],
    test_targets: np.ndarray,
    errors: dict[str, float],
    soh_values: np.ndarray,
    pinn_uncertainty: np.ndarray,
    physics_baseline: np.ndarray,
    rated_capacity: float,
    ci_lower: dict[str, np.ndarray] | None = None,
    ci_upper: dict[str, np.ndarray] | None = None,
) -> dict[str, EnsembleResult]:
    """Run all 4 ensemble strategies and return results.

    Args:
        train_preds: base model predictions on training set (for stacking)
        train_targets: actual training targets
        test_preds: base model predictions on test set
        test_targets: actual test targets
        errors: {model_name: RMSE} on validation set
        soh_values: SOH at each test point
        pinn_uncertainty: aleatoric uncertainty from PINN
        physics_baseline: ECM capacity prediction at each test cycle
        rated_capacity: nominal capacity
    """
    results: dict[str, EnsembleResult] = {}

    # Strategy 1: Weighted
    results["weighted"] = weighted_ensemble(
        test_preds, errors, ci_lower, ci_upper
    )

    # Strategy 2: Stacking
    results["stacking"] = stacking_ensemble(
        train_preds, train_targets, test_preds, ci_lower, ci_upper
    )

    # Strategy 3: Lifecycle-Adaptive
    results["lifecycle"] = lifecycle_adaptive_ensemble(
        test_preds, soh_values, errors, ci_lower, ci_upper
    )

    # Strategy 4: Physics-Constrained
    results["physics_constrained"] = physics_constrained_ensemble(
        test_preds, pinn_uncertainty, physics_baseline,
        rated_capacity, errors, ci_lower, ci_upper
    )

    # Log comparison
    for name, res in results.items():
        valid = ~np.isnan(res.predictions) & ~np.isnan(test_targets)
        if valid.any():
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = float(np.sqrt(mean_squared_error(
                test_targets[valid], res.predictions[valid]
            )))
            r2 = float(r2_score(test_targets[valid], res.predictions[valid]))
            logger.info(
                "  Ensemble %-22s: RMSE=%.4f, R2=%.4f",
                res.strategy_name, rmse, r2,
            )

    return results
