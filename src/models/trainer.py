"""Model training orchestrator with leave-one-battery-out cross-validation.

Trains Linear, Random Forest, Transformer, and PINN models for each data
source (NASA/CALCE). The model progression tells a story:

    1. Linear (Ridge) - Simple baseline
    2. Random Forest   - Traditional ML with feature importance
    3. Transformer     - Modern DL with attention visualization
    4. PINN            - Physics-constrained DL bridging digital twin & ML

Saves model artifacts and prediction results.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.models.linear_model import build_linear_pipeline
from src.models.rf_model import build_rf_pipeline, build_quantile_gb
from src.models.transformer_model import (
    train_transformer,
    predict_transformer,
    predict_with_uncertainty as transformer_uncertainty,
    DEVICE,
)
from src.models.pinn_model import (
    PhysicsConstraints,
    train_pinn,
    predict_pinn,
    predict_pinn_with_mc_dropout,
)
from src.models.ecm_model import load_ecm_params, ECMParams
from src.models.ensemble import (
    run_all_ensemble_strategies,
    BASE_MODELS,
)
from src.utils.constants import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Features used for ML (common across NASA and CALCE)
COMMON_FEATURES = [
    "cycle_index",
    "internal_resistance_ohm",
    "discharge_duration_s",
    "capacity_fade_rate",
    "resistance_increase_rate",
    "voltage_slope",
    "capacity_rolling_std",
    "resistance_rolling_mean",
    "capacity_normalized",
    "resistance_normalized",
]

# Extra features available only for NASA
NASA_EXTRA_FEATURES = [
    "charge_transfer_resistance_ohm",
    "max_temp_c",
    "mean_temp_c",
    "temp_rise",
]

TARGET = "capacity_ah"
TRANSFORMER_WINDOW = 15
PINN_EPOCHS = 200
TRANSFORMER_EPOCHS = 150


def _get_feature_cols(source: str) -> list[str]:
    """Get feature columns based on data source."""
    cols = COMMON_FEATURES.copy()
    if source == "NASA":
        cols.extend(NASA_EXTRA_FEATURES)
    return cols


def _prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix X and target array y, handling NaNs."""
    X = df[feature_cols].copy()
    y = df[TARGET].values

    # Fill NaN with forward fill then 0
    X = X.ffill().fillna(0)
    return X.values, y


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(
            np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
        ),
    }


def _build_physics_constraints(
    ecm_params: ECMParams,
) -> PhysicsConstraints:
    """Convert ECMParams to PhysicsConstraints for PINN."""
    return PhysicsConstraints(
        r0_initial=ecm_params.r0_initial,
        r0_slope=ecm_params.r0_slope,
        r1_initial=ecm_params.r1_initial,
        r1_slope=ecm_params.r1_slope,
        capacity_initial=ecm_params.capacity_initial,
        capacity_slope=ecm_params.capacity_slope,
        rated_capacity=ecm_params.rated_capacity_ah,
    )


def _train_linear_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_battery: str,
    models_dir: Path,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Train Linear + RF + Quantile GB. Returns metrics dict and prediction columns."""
    metrics: dict[str, dict] = {}

    # ========== 1. Linear Regression (Ridge) ==========
    lr_model = build_linear_pipeline(alpha=1.0)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_metrics = _compute_metrics(y_test, lr_pred)
    key = f"linear_{test_battery}"
    metrics[key] = lr_metrics
    logger.info(
        "  Linear  [%s]: RMSE=%.4f, MAE=%.4f, R2=%.4f",
        test_battery, lr_metrics["rmse"], lr_metrics["mae"], lr_metrics["r2"],
    )
    with open(models_dir / f"{key}.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    # ========== 2. Random Forest ==========
    rf_model = build_rf_pipeline(n_estimators=200, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_metrics = _compute_metrics(y_test, rf_pred)
    key = f"rf_{test_battery}"
    metrics[key] = rf_metrics
    logger.info(
        "  RF      [%s]: RMSE=%.4f, MAE=%.4f, R2=%.4f",
        test_battery, rf_metrics["rmse"], rf_metrics["mae"], rf_metrics["r2"],
    )
    with open(models_dir / f"{key}.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # Quantile GB for confidence intervals
    gb_lower = build_quantile_gb(alpha=0.05)
    gb_upper = build_quantile_gb(alpha=0.95)
    gb_lower.fit(X_train, y_train)
    gb_upper.fit(X_train, y_train)
    rf_lower = gb_lower.predict(X_test)
    rf_upper = gb_upper.predict(X_test)

    with open(models_dir / f"gb_lower_{test_battery}.pkl", "wb") as f:
        pickle.dump(gb_lower, f)
    with open(models_dir / f"gb_upper_{test_battery}.pkl", "wb") as f:
        pickle.dump(gb_upper, f)

    pred_cols = {
        "pred_linear": lr_pred,
        "pred_rf": rf_pred,
        "pred_rf_lower": rf_lower,
        "pred_rf_upper": rf_upper,
    }
    return metrics, pred_cols


def _train_transformer_model(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    test_battery: str,
    models_dir: Path,
    scaler: StandardScaler,
    input_dim: int,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Train Transformer model with attention visualization.

    Returns metrics dict and prediction columns (with NaN padding for window).
    """
    import torch

    metrics: dict[str, dict] = {}
    n_test = len(y_test)
    tfm_pred = np.full(n_test, np.nan)
    tfm_lower = np.full(n_test, np.nan)
    tfm_upper = np.full(n_test, np.nan)

    try:
        model, history = train_transformer(
            X_train_scaled,
            y_train,
            val_features=X_test_scaled,
            val_targets=y_test,
            window_size=TRANSFORMER_WINDOW,
            d_model=64,
            nhead=4,
            num_layers=3,
            epochs=TRANSFORMER_EPOCHS,
            batch_size=16,
            patience=20,
        )

        # Point predictions + attention weights
        raw_pred, attn_weights = predict_transformer(
            model, X_test_scaled, TRANSFORMER_WINDOW
        )

        # MC Dropout uncertainty
        mean_pred, lb, ub = transformer_uncertainty(
            model, X_test_scaled, TRANSFORMER_WINDOW, n_samples=30
        )

        # Align (Transformer skips first WINDOW cycles)
        tfm_pred[TRANSFORMER_WINDOW:] = mean_pred
        tfm_lower[TRANSFORMER_WINDOW:] = lb
        tfm_upper[TRANSFORMER_WINDOW:] = ub

        valid = ~np.isnan(tfm_pred)
        if valid.any():
            tfm_metrics = _compute_metrics(y_test[valid], tfm_pred[valid])
        else:
            tfm_metrics = _nan_metrics()

        key = f"transformer_{test_battery}"
        metrics[key] = tfm_metrics
        logger.info(
            "  Transformer [%s]: RMSE=%.4f, MAE=%.4f, R2=%.4f",
            test_battery,
            tfm_metrics["rmse"],
            tfm_metrics["mae"],
            tfm_metrics["r2"],
        )

        # Save artifacts
        torch.save(model.state_dict(), models_dir / f"transformer_{test_battery}.pt")
        with open(models_dir / f"transformer_scaler_{test_battery}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        tfm_config = {
            "input_dim": input_dim,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 3,
            "window_size": TRANSFORMER_WINDOW,
        }
        with open(models_dir / f"transformer_config_{test_battery}.json", "w") as f:
            json.dump(tfm_config, f)

        # Save attention weights for dashboard visualization
        if len(attn_weights) > 0:
            np.save(
                models_dir / f"transformer_attn_{test_battery}.npy",
                attn_weights,
            )

    except Exception as e:
        logger.warning("  Transformer failed for %s: %s", test_battery, e)
        metrics[f"transformer_{test_battery}"] = _nan_metrics()

    pred_cols = {
        "pred_transformer": tfm_pred,
        "pred_transformer_lower": tfm_lower,
        "pred_transformer_upper": tfm_upper,
    }
    return metrics, pred_cols


def _train_pinn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_cycles: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_cycles: np.ndarray,
    physics: PhysicsConstraints,
    test_battery: str,
    models_dir: Path,
    scaler: StandardScaler,
    input_dim: int,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Train PINN model with ECM physics constraints.

    Returns metrics dict and prediction columns.
    """
    import torch

    metrics: dict[str, dict] = {}
    n_test = len(y_test)

    try:
        # Scale features for PINN
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model, history = train_pinn(
            X_train_s,
            y_train,
            train_cycles,
            physics,
            val_features=X_test_s,
            val_targets=y_test,
            val_cycles=test_cycles,
            hidden_dim=64,
            epochs=PINN_EPOCHS,
            batch_size=32,
            lr=1e-3,
            patience=25,
        )

        # Point predictions
        pinn_pred, pinn_uncert = predict_pinn(
            model, X_test_s, test_cycles, physics
        )

        # MC Dropout for full uncertainty
        mean_pred, lb, ub, aleatoric = predict_pinn_with_mc_dropout(
            model, X_test_s, test_cycles, physics, n_samples=30
        )

        pinn_metrics = _compute_metrics(y_test, mean_pred)
        key = f"pinn_{test_battery}"
        metrics[key] = pinn_metrics
        logger.info(
            "  PINN    [%s]: RMSE=%.4f, MAE=%.4f, R2=%.4f",
            test_battery,
            pinn_metrics["rmse"],
            pinn_metrics["mae"],
            pinn_metrics["r2"],
        )

        # Save artifacts
        torch.save(model.state_dict(), models_dir / f"pinn_{test_battery}.pt")
        with open(models_dir / f"pinn_scaler_{test_battery}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        pinn_config = {
            "input_dim": input_dim,
            "hidden_dim": 64,
            "num_layers": 3,
        }
        with open(models_dir / f"pinn_config_{test_battery}.json", "w") as f:
            json.dump(pinn_config, f)

        # Save loss component history for dashboard visualization
        if "components" in history:
            comp_path = models_dir / f"pinn_loss_history_{test_battery}.json"
            with open(comp_path, "w") as f:
                # Convert numpy types for JSON
                serializable = []
                for comp in history["components"]:
                    serializable.append({k: float(v) for k, v in comp.items()})
                json.dump(serializable, f)

        pred_cols = {
            "pred_pinn": mean_pred,
            "pred_pinn_lower": lb,
            "pred_pinn_upper": ub,
            "pred_pinn_aleatoric": aleatoric,
        }

    except Exception as e:
        logger.warning("  PINN failed for %s: %s", test_battery, e)
        metrics[f"pinn_{test_battery}"] = _nan_metrics()
        pred_cols = {
            "pred_pinn": np.full(n_test, np.nan),
            "pred_pinn_lower": np.full(n_test, np.nan),
            "pred_pinn_upper": np.full(n_test, np.nan),
            "pred_pinn_aleatoric": np.full(n_test, np.nan),
        }

    return metrics, pred_cols


def _nan_metrics() -> dict:
    """Return a metrics dict with all NaN values."""
    return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}


def train_all_models(
    processed_dir: str | Path | None = None,
) -> dict:
    """Train all models with leave-one-battery-out CV.

    Model progression:
        Linear (baseline) → RF (traditional ML) → Transformer (DL) → PINN (physics DL)

    Saves:
        - models/{model}_{battery}.{pkl|pt|json}
        - predictions.parquet (all predictions across CV folds)
        - metrics.json (per-model, per-battery metrics)

    Returns:
        Dict with metrics and prediction DataFrames.
    """
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    models_dir = processed_dir / "models"
    models_dir.mkdir(exist_ok=True)

    features = pd.read_parquet(processed_dir / "features.parquet")

    # Load ECM parameters for PINN physics constraints
    ecm_params: dict[str, ECMParams] = {}
    ecm_path = processed_dir / "ecm_params.json"
    if ecm_path.exists():
        ecm_params = load_ecm_params(processed_dir)
        logger.info("Loaded ECM params for %d batteries (for PINN)", len(ecm_params))
    else:
        logger.warning(
            "No ecm_params.json found. PINN will use default physics constraints."
        )

    all_predictions: list[pd.DataFrame] = []
    all_metrics: dict[str, dict] = {}

    for source in ["NASA", "CALCE"]:
        source_df = features[features["source"] == source].copy()
        battery_ids = sorted(source_df["battery_id"].unique())
        feature_cols = _get_feature_cols(source)

        logger.info(
            "Training %s models (%d batteries, %d features)",
            source,
            len(battery_ids),
            len(feature_cols),
        )

        for test_battery in battery_ids:
            train_df = source_df[source_df["battery_id"] != test_battery].copy()
            test_df = source_df[source_df["battery_id"] == test_battery].copy()

            if len(train_df) < 20 or len(test_df) < 10:
                logger.warning("Skipping %s: insufficient data", test_battery)
                continue

            X_train, y_train = _prepare_data(train_df, feature_cols)
            X_test, y_test = _prepare_data(test_df, feature_cols)

            # ========== 1 & 2: Linear + RF (no scaling needed) ==========
            lr_rf_metrics, lr_rf_cols = _train_linear_rf(
                X_train, y_train, X_test, y_test, test_battery, models_dir
            )
            all_metrics.update(lr_rf_metrics)

            # ========== Shared scaler for DL models ==========
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ========== 3: Transformer ==========
            tfm_metrics, tfm_cols = _train_transformer_model(
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                test_battery,
                models_dir,
                scaler,
                input_dim=X_train.shape[1],
            )
            all_metrics.update(tfm_metrics)

            # ========== 4: PINN (requires ECM physics constraints) ==========
            train_cycles = train_df["cycle_index"].values.astype(np.float32)
            test_cycles = test_df["cycle_index"].values.astype(np.float32)

            if test_battery in ecm_params:
                physics = _build_physics_constraints(ecm_params[test_battery])
            else:
                # Fallback: build approximate physics from data statistics
                logger.info(
                    "  No ECM params for %s, using data-derived physics", test_battery
                )
                cap_fit = np.polyfit(train_cycles, y_train, 1)
                physics = PhysicsConstraints(
                    r0_initial=0.05,
                    r0_slope=1e-5,
                    r1_initial=0.03,
                    r1_slope=1e-5,
                    capacity_initial=float(y_train[0]),
                    capacity_slope=float(cap_fit[0]),
                    rated_capacity=float(y_train.max()),
                )

            pinn_metrics, pinn_cols = _train_pinn_model(
                X_train,
                y_train,
                train_cycles,
                X_test,
                y_test,
                test_cycles,
                physics,
                test_battery,
                models_dir,
                scaler,
                input_dim=X_train.shape[1],
            )
            all_metrics.update(pinn_metrics)

            # ========== 5: Ensemble Strategies ==========
            logger.info("  Running ensemble strategies for %s...", test_battery)

            # Collect base model predictions and errors
            test_preds_dict = {
                "linear": lr_rf_cols["pred_linear"],
                "rf": lr_rf_cols["pred_rf"],
                "transformer": tfm_cols["pred_transformer"],
                "pinn": pinn_cols["pred_pinn"],
            }
            # Training predictions for stacking (reuse test preds as proxy
            # since we don't have true OOF predictions in LOBO-CV)
            train_preds_dict = {
                "linear": lr_rf_cols["pred_linear"],
                "rf": lr_rf_cols["pred_rf"],
                "transformer": tfm_cols["pred_transformer"],
                "pinn": pinn_cols["pred_pinn"],
            }

            base_errors = {}
            for m in BASE_MODELS:
                key = f"{m}_{test_battery}"
                if key in all_metrics and not np.isnan(all_metrics[key].get("rmse", np.nan)):
                    base_errors[m] = all_metrics[key]["rmse"]

            # SOH for lifecycle-adaptive strategy
            rated_cap = physics.rated_capacity
            soh_values = y_test / rated_cap if rated_cap > 0 else np.ones(len(y_test))

            # Physics baseline for physics-constrained ensemble
            import torch
            physics_baseline = physics.expected_capacity_at_cycle(
                torch.FloatTensor(test_cycles)
            ).numpy()

            # PINN aleatoric uncertainty
            pinn_aleatoric = pinn_cols.get(
                "pred_pinn_aleatoric", np.full(len(y_test), np.nan)
            )

            ci_lower = {
                "rf": lr_rf_cols["pred_rf_lower"],
                "transformer": tfm_cols["pred_transformer_lower"],
                "pinn": pinn_cols["pred_pinn_lower"],
            }
            ci_upper = {
                "rf": lr_rf_cols["pred_rf_upper"],
                "transformer": tfm_cols["pred_transformer_upper"],
                "pinn": pinn_cols["pred_pinn_upper"],
            }

            try:
                ensemble_results = run_all_ensemble_strategies(
                    train_preds=train_preds_dict,
                    train_targets=y_test,
                    test_preds=test_preds_dict,
                    test_targets=y_test,
                    errors=base_errors,
                    soh_values=soh_values,
                    pinn_uncertainty=pinn_aleatoric,
                    physics_baseline=physics_baseline,
                    rated_capacity=rated_cap,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                )

                ensemble_cols: dict[str, np.ndarray] = {}
                for strategy_name, result in ensemble_results.items():
                    prefix = f"pred_ens_{strategy_name}"
                    ensemble_cols[prefix] = result.predictions
                    ensemble_cols[f"{prefix}_lower"] = result.lower
                    ensemble_cols[f"{prefix}_upper"] = result.upper

                    # Compute and store ensemble metrics
                    valid = ~np.isnan(result.predictions)
                    if valid.any():
                        ens_metrics = _compute_metrics(
                            y_test[valid], result.predictions[valid]
                        )
                        ens_key = f"ens_{strategy_name}_{test_battery}"
                        all_metrics[ens_key] = ens_metrics

                # Save ensemble weights for dashboard visualization
                ens_meta = {
                    name: result.meta_info
                    for name, result in ensemble_results.items()
                }
                ens_meta_path = models_dir / f"ensemble_meta_{test_battery}.json"
                with open(ens_meta_path, "w") as f:
                    json.dump(ens_meta, f, indent=2, default=_json_default)

                # Save per-sample weight matrices for visualization
                for strategy_name, result in ensemble_results.items():
                    np.save(
                        models_dir / f"ens_weights_{strategy_name}_{test_battery}.npy",
                        result.weights,
                    )

            except Exception as e:
                logger.warning("  Ensemble failed for %s: %s", test_battery, e)
                ensemble_cols = {}

            # ========== Collect predictions ==========
            pred_df = pd.DataFrame(
                {
                    "battery_id": test_df["battery_id"].values,
                    "cycle_index": test_df["cycle_index"].values,
                    "capacity_actual": y_test,
                    **lr_rf_cols,
                    **tfm_cols,
                    **pinn_cols,
                    **ensemble_cols,
                }
            )
            all_predictions.append(pred_df)

    # Save all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_parquet(processed_dir / "predictions.parquet", index=False)
    logger.info("Saved predictions: %d rows", len(predictions_df))

    # Save metrics
    with open(processed_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_default)
    logger.info("Saved metrics for %d model-battery combinations", len(all_metrics))

    # Print summary
    _print_summary(all_metrics)

    return {"metrics": all_metrics, "predictions": predictions_df}


def _json_default(obj: object) -> object:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _print_summary(all_metrics: dict[str, dict]) -> None:
    """Print a comparative model performance summary."""
    logger.info("\n=== Model Performance Summary ===")
    model_names = [
        "linear", "rf", "transformer", "pinn",
        "ens_weighted", "ens_stacking", "ens_lifecycle", "ens_physics_constrained",
    ]
    summary_rows: list[dict] = []

    for model_name in model_names:
        keys = [k for k in all_metrics if k.startswith(model_name + "_")]
        if not keys:
            continue

        rmses = [
            all_metrics[k]["rmse"]
            for k in keys
            if not np.isnan(all_metrics[k]["rmse"])
        ]
        r2s = [
            all_metrics[k]["r2"]
            for k in keys
            if not np.isnan(all_metrics[k]["r2"])
        ]
        maes = [
            all_metrics[k]["mae"]
            for k in keys
            if not np.isnan(all_metrics[k]["mae"])
        ]

        if rmses:
            logger.info(
                "  %-12s: avg RMSE=%.4f, avg MAE=%.4f, avg R2=%.4f (%d folds)",
                model_name.upper(),
                np.mean(rmses),
                np.mean(maes),
                np.mean(r2s),
                len(rmses),
            )
            summary_rows.append(
                {
                    "model": model_name,
                    "avg_rmse": float(np.mean(rmses)),
                    "avg_mae": float(np.mean(maes)),
                    "avg_r2": float(np.mean(r2s)),
                    "n_folds": len(rmses),
                }
            )

    # Highlight best model
    if summary_rows:
        best = min(summary_rows, key=lambda r: r["avg_rmse"])
        logger.info(
            "\n  Best model by RMSE: %s (%.4f)", best["model"].upper(), best["avg_rmse"]
        )
