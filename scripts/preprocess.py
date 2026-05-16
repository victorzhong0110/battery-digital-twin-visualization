#!/usr/bin/env python3
"""Run the full data preprocessing pipeline.

Usage:
    python scripts/preprocess.py
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    start = time.time()

    logger.info("=" * 60)
    logger.info("Step 1/3: Building unified dataset from raw data...")
    logger.info("=" * 60)
    from src.data.unified_schema import build_unified
    unified_df = build_unified()
    logger.info("Unified dataset: %d rows", len(unified_df))

    # Summary stats
    for battery_id, group in unified_df.groupby("battery_id"):
        cap_start = group["capacity_ah"].iloc[0]
        cap_end = group["capacity_ah"].iloc[-1]
        soh_end = group["soh"].iloc[-1]
        logger.info(
            "  %s: %d cycles, capacity %.3f -> %.3f Ah, SOH %.1f%%",
            battery_id, len(group), cap_start, cap_end, soh_end * 100,
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2/3: Extracting features...")
    logger.info("=" * 60)
    from src.data.feature_extract import build_features
    features_df = build_features()
    logger.info("Features: %d rows, %d columns", len(features_df), len(features_df.columns))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3/3: Verification...")
    logger.info("=" * 60)

    from src.utils.constants import PROCESSED_DIR
    for fname in ["unified.parquet", "features.parquet"]:
        fpath = PROCESSED_DIR / fname
        size_mb = fpath.stat().st_size / (1024 * 1024)
        logger.info("  %s: %.2f MB", fname, size_mb)

    curves_dir = PROCESSED_DIR / "curves"
    npz_files = list(curves_dir.glob("*.npz"))
    logger.info("  Curve files: %d batteries", len(npz_files))

    elapsed = time.time() - start
    logger.info("")
    logger.info("Pipeline complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
