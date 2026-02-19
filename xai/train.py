"""Optional CLI training script for XGBoost distress model.

Usage:
    python -m xai.train --data path/to/data.csv --target is_distressed

The CSV should contain columns matching FEATURE_NAMES plus the target column.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost distress model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--target", default="is_distressed", help="Target column name")
    parser.add_argument(
        "--output",
        default="xai/artifacts/distress_model.joblib",
        help="Output model path",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    try:
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error("Training requires: pandas, scikit-learn, xgboost")
        sys.exit(1)

    from xai.distress import XGBoostDistressModel
    from xai.features import FEATURE_NAMES

    # Load data
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    # Validate columns
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        logger.error("Missing feature columns: %s", missing)
        sys.exit(1)
    if args.target not in df.columns:
        logger.error("Target column '%s' not found", args.target)
        sys.exit(1)

    x_data = df[FEATURE_NAMES].values.astype(np.float64)
    y = df[args.target].values.astype(int)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y, test_size=args.test_size, random_state=42, stratify=y
    )
    logger.info("Train: %d, Test: %d", len(x_train), len(x_test))

    # Train
    model = XGBoostDistressModel()
    model.train(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    logger.info("Test Accuracy: %.3f", acc)
    logger.info("Test F1:       %.3f", f1)
    logger.info("Test AUC:      %.3f", auc)

    # Save
    model.save(args.output)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
