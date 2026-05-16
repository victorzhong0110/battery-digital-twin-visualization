#!/usr/bin/env python3
"""Train all ML models and generate SHAP explanations.

Pipeline:
    1. Calibrate ECM parameters (needed by PINN physics constraints)
    2. Train all models: Linear → RF → Transformer → PINN
    3. Generate SHAP explanations for interpretable models

Usage:
    python scripts/train_models.py
"""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    start = time.time()

    # Step 1: ECM calibration (prerequisite for PINN)
    logger.info("=" * 60)
    logger.info("Step 1/3: Calibrating ECM parameters...")
    logger.info("=" * 60)
    from src.utils.constants import PROCESSED_DIR

    ecm_path = PROCESSED_DIR / "ecm_params.json"
    if ecm_path.exists():
        logger.info("ECM params already exist, skipping calibration.")
    else:
        from src.models.ecm_model import calibrate_all
        ecm_params = calibrate_all()
        logger.info("Calibrated ECM for %d batteries.", len(ecm_params))

    # Step 2: Train all models
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2/3: Training all models (Linear, RF, Transformer, PINN)...")
    logger.info("=" * 60)
    from src.models.trainer import train_all_models
    results = train_all_models()

    # Step 3: SHAP explanations
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3/3: Generating SHAP explanations...")
    logger.info("=" * 60)
    from src.models.explainer import build_all_explanations
    explanations = build_all_explanations()

    elapsed = time.time() - start
    logger.info("")
    logger.info("All models trained and explained in %.1f seconds.", elapsed)

    # Final summary
    logger.info("\n=== Output Files ===")
    for fname in ["predictions.parquet", "metrics.json", "explanations.json", "ecm_params.json"]:
        fpath = PROCESSED_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            logger.info("  %s: %.1f KB", fname, size_kb)

    model_files = list((PROCESSED_DIR / "models").glob("*"))
    logger.info("  models/: %d files", len(model_files))


if __name__ == "__main__":
    main()
