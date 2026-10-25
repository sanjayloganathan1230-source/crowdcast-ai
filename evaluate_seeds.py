"""Repeat data generation + training + evaluation for several seeds to see how stable the results are.

    python -m scripts.evaluate_seeds [--seeds 1 2 3 4 5] [--out docs/seed_robustness.json]

Nothing is tuned on the test period; every seed uses the same fixed hyperparameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.data_generation import generate_dataset
from backend.app.ml.training import train_and_evaluate

MODELS = ("seasonal_naive_168h", "mean_last_4_weeks", "random_forest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="docs/seed_robustness.json")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        df = generate_dataset(seed=seed)
        r = train_and_evaluate(df, seed=seed)
        tp = r.test_predictions.merge(df[["timestamp", "location_id", "event_indicator"]], on=["timestamp", "location_id"])
        ev, non = tp[tp.event_indicator == 1], tp[tp.event_indicator == 0]
        row = {"seed": seed, "test_rows": len(tp), "event_rows": int(len(ev))}
        for m in MODELS:
            row[f"{m}_mae"] = r.metrics["overall"][m]["mae"]
            row[f"{m}_rmse"] = r.metrics["overall"][m]["rmse"]
        row["rf_mae_improvement_pct"] = r.metrics["improvement_vs_seasonal_naive"]["mae_pct"]
        row["rf_rmse_improvement_pct"] = r.metrics["improvement_vs_seasonal_naive"]["rmse_pct"]
        row["rf_mae_event_hours"] = float(np.mean(np.abs(ev.random_forest - ev.observed))) if len(ev) else None
        row["naive_mae_event_hours"] = float(np.mean(np.abs(ev.seasonal_naive_168h - ev.observed))) if len(ev) else None
        row["rf_bias_event_hours"] = float(np.mean(ev.random_forest - ev.observed)) if len(ev) else None
        row["rf_mae_non_event_hours"] = float(np.mean(np.abs(non.random_forest - non.observed)))
        row["rf_bias_non_event_hours"] = float(np.mean(non.random_forest - non.observed))
        rows.append(row)
        print(f"seed {seed}: MAE naive {row['seasonal_naive_168h_mae']:.0f} | 4-wk mean {row['mean_last_4_weeks_mae']:.0f} | "
              f"RF {row['random_forest_mae']:.0f} ({row['rf_mae_improvement_pct']:+.1f}%) | "
              f"event hrs n={row['event_rows']}: RF MAE {row['rf_mae_event_hours']}, bias {row['rf_bias_event_hours']}")

    summary = {k: [float(np.min([r[k] for r in rows])), float(np.mean([r[k] for r in rows])), float(np.max([r[k] for r in rows]))]
               for k in ("rf_mae_improvement_pct", "rf_rmse_improvement_pct", "random_forest_mae", "seasonal_naive_168h_mae")}
    wins = sum(r["random_forest_mae"] < min(r["seasonal_naive_168h_mae"], r["mean_last_4_weeks_mae"]) for r in rows)
    print(f"\nRF beats BOTH baselines on MAE in {wins}/{len(rows)} seeds")
    for k, (lo, mean, hi) in summary.items():
        print(f"  {k}: min {lo:.1f}  mean {mean:.1f}  max {hi:.1f}")
    out = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": "DEMO DATA (synthetic). Fixed untuned hyperparameters.", "seeds": rows, "summary_min_mean_max": summary,
                               "rf_beats_both_baselines_mae": f"{wins}/{len(rows)}"}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
