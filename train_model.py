"""Train the RandomForest, evaluate against baselines, save model + metrics.

Usage (from project root, after generate_data):
    python -m scripts.train_model [--data data/raw/demo_observations.csv] [--test-days 28]
"""
from __future__ import annotations

import argparse

from backend.app.core.config import load_settings
from backend.app.ml.model_store import save_artifacts
from backend.app.ml.preprocessing import load_observations
from backend.app.ml.training import DEFAULT_TEST_DAYS, train_and_evaluate


def main() -> None:
    s = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/demo_observations.csv")
    ap.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    ap.add_argument("--seed", type=int, default=s.seed)
    args = ap.parse_args()

    result = train_and_evaluate(load_observations(args.data), seed=args.seed, test_days=args.test_days)
    save_artifacts(result, s.model_path, s.metrics_path)

    m = result.metrics
    sp = m["split"]
    print("[DEMO DATA - synthetic, not live sensor data]")
    print(f"Train: {sp['train_rows']:,} rows ({sp['train_start']} -> {sp['train_end']})")
    print(f"Test : {sp['test_rows']:,} rows ({sp['test_start']} -> {sp['test_end']})\n")
    print(f"{'model':22s} {'MAE':>8s} {'RMSE':>8s} {'MAE % cap':>10s}")
    for name, v in m["overall"].items():
        print(f"{name:22s} {v['mae']:8.1f} {v['rmse']:8.1f} {v['mae_pct_of_capacity']:9.2f}%")
    imp = m["improvement_vs_seasonal_naive"]
    print(f"\nRandomForest vs seasonal-naive: MAE {imp['mae_pct']:+.1f}%, RMSE {imp['rmse_pct']:+.1f}% (positive = better)")
    print(f"Saved: {s.model_path}\n       {s.metrics_path}")


if __name__ == "__main__":
    main()
