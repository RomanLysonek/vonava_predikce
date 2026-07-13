"""Regenerate outputs/results.json from already-computed outputs (submission.csv,
cv_results.csv) without rerunning the walk-forward CV or the NN ensemble. Still
retrains XGBoost/LightGBM on the full dataset (cheap relative to the CV) so
their dashboard pages have a real test-week forecast, and recomputes the two
naive baselines' final predictions (free lookups, no training).

Run (from repo root): uv run python ml/export_results.py
"""

import os

import numpy as np
import pandas as pd

from pipeline import (
    CFG,
    export_results_json,
    load_raw,
    run_final_naive_baselines,
    run_final_tree_forecast,
)


def main() -> None:
    train_raw, test_raw = load_raw(CFG)
    CFG.num_products = int(max(train_raw["ProductId"].max(), test_raw["ProductId"].max()))

    submission = pd.read_csv(os.path.join(CFG.output_dir, "submission.csv"), parse_dates=["DateKey"])
    cv_results = pd.read_csv(os.path.join(CFG.output_dir, "cv_results.csv"))
    
    dev_summary_path = os.path.join(CFG.output_dir, "dev_summary.csv")
    dev_summary = pd.read_csv(dev_summary_path) if os.path.exists(dev_summary_path) else None
    
    benchmark_summary_path = os.path.join(CFG.output_dir, "benchmark_summary.csv")
    benchmark_summary = pd.read_csv(benchmark_summary_path) if os.path.exists(benchmark_summary_path) else None

    # submission.csv already holds the NeuralNet's rounded final predictions,
    # aligned to test_raw's (ProductId, DateKey) rows.
    nn_preds = (
        test_raw[["ProductId", "DateKey"]]
        .merge(submission, on=["ProductId", "DateKey"], how="left")["Quantity"]
        .to_numpy(dtype=float)
    )

    print("Training final XGBoost/LightGBM on full data (dashboard comparison only)...")
    tree_final = run_final_tree_forecast(train_raw, test_raw, CFG)
    naive_final = run_final_naive_baselines(train_raw, test_raw, CFG)

    final_forecasts = {
        "NeuralNet": nn_preds,
        "XGBoost": tree_final["XGBoost"],
        "LightGBM": tree_final["LightGBM"],
        "DynamicRidge": tree_final["DynamicRidge"],
        "SeasonalNaive": naive_final["SeasonalNaive"],
        "MovingAvg28": naive_final["MovingAvg28"],
    }
    export_results_json(train_raw, test_raw, submission, final_forecasts, cv_results, CFG,
                        dev_summary=dev_summary, benchmark_summary=benchmark_summary)


if __name__ == "__main__":
    main()
