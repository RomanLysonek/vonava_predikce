"""Regenerate outputs/results.json exclusively from persisted artifacts.

This exporter never retrains models and never reconstructs full-precision
forecasts from rounded submission.csv. The canonical source is
outputs/final_forecasts.parquet, written by ml/pipeline.py.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from pipeline import (
    CFG,
    ForecastStrategy,
    PrimaryStrategy,
    RuntimeOptions,
    SubmissionModel,
    export_results_json,
    load_raw,
)


def main() -> None:
    train_raw, test_raw = load_raw(CFG)
    CFG.num_products = int(max(train_raw["ProductId"].max(), test_raw["ProductId"].max()))
    out = CFG.output_dir
    submission = pd.read_csv(os.path.join(out, "submission.csv"), parse_dates=["DateKey"])
    cv_results = pd.read_csv(os.path.join(out, "cv_results.csv"))
    cv_results_all_path = os.path.join(out, "cv_results_all.csv")
    cv_results_all = (
        pd.read_csv(cv_results_all_path)
        if os.path.exists(cv_results_all_path)
        else None
    )
    dev_summary = pd.read_csv(os.path.join(out, "dev_summary.csv"))
    benchmark_summary = pd.read_csv(os.path.join(out, "benchmark_summary.csv"))
    final_df = pd.read_parquet(os.path.join(out, "final_forecasts.parquet"))
    final_df["DateKey"] = pd.to_datetime(final_df["DateKey"])
    pair_path = os.path.join(out, "strategy_pair_summary.csv")
    pair_summary = pd.read_csv(pair_path) if os.path.exists(pair_path) else pd.DataFrame()
    horizon_path = os.path.join(out, "strategy_by_horizon.csv")
    strategy_by_horizon = (
        pd.read_csv(horizon_path) if os.path.exists(horizon_path) else pd.DataFrame()
    )

    existing_path = os.path.join(out, "results.json")
    existing = {}
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
    config = existing.get("config", {})
    # Preserve the exact NN runtime metadata from the completed training run.
    # This exporter is artifact-only and must not silently rewrite it to the
    # Config defaults.
    CFG.batch_size = int(config.get("nn_batch_size", CFG.batch_size))
    CFG.reference_batch_size = int(
        config.get("nn_reference_batch_size", CFG.reference_batch_size)
    )
    CFG.nn_lr_scaling = config.get("nn_lr_scaling", CFG.nn_lr_scaling)
    CFG.nn_training_backend = config.get(
        "nn_training_backend", CFG.nn_training_backend
    )
    canonical_strategy = config.get("primary_strategy", "direct")
    submission_model = config.get("submission_model", "NeuralNet")
    forecast_strategy = config.get("forecast_strategy", "direct")
    if cv_results_all is None:
        cv_results_all = cv_results.assign(strategy=canonical_strategy)
    options = RuntimeOptions(
        forecast_strategy=ForecastStrategy(forecast_strategy),
        primary_strategy=PrimaryStrategy(canonical_strategy if canonical_strategy in {"direct", "recursive"} else "auto"),
        submission_model=SubmissionModel(submission_model),
        selection_metric=config.get("selection_metric", "WAPE"),
        nn_batch_size=str(CFG.batch_size),
        nn_lr_scaling=CFG.nn_lr_scaling,
        nn_training_backend=CFG.nn_training_backend,
    )

    forecasts_by_strategy = {}
    raw_forecasts = {}
    for strategy, strategy_df in final_df.groupby("strategy"):
        raw_forecasts[strategy] = {}
        strategy_json = {}
        for model, model_df in strategy_df.groupby("model"):
            aligned = test_raw[["ProductId", "DateKey"]].merge(
                model_df[["ProductId", "DateKey", "prediction_raw"]],
                on=["ProductId", "DateKey"], how="left", validate="one_to_one",
            )
            raw_forecasts[strategy][model] = aligned["prediction_raw"].to_numpy(dtype=float)
            per_product = {}
            for pid, sub in aligned.groupby("ProductId"):
                sub = sub.sort_values("DateKey")
                per_product[str(int(pid))] = {
                    "dates": sub["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
                    "quantity": sub["prediction_raw"].astype(float).tolist(),
                }
            strategy_json[model] = per_product
        forecasts_by_strategy[strategy] = strategy_json

    canonical_forecasts = raw_forecasts[canonical_strategy]
    canonical_model = existing.get("selection", {}).get(
        "canonical_model", submission_model
    )
    export_results_json(
        train_raw, test_raw, submission, canonical_forecasts, cv_results, CFG,
        dev_summary=dev_summary, benchmark_summary=benchmark_summary,
        runtime_options=options, forecasts_by_strategy=forecasts_by_strategy,
        strategy_comparison=pair_summary, canonical_strategy=canonical_strategy,
        canonical_model=canonical_model, cv_results_all=cv_results_all,
        strategy_by_horizon=strategy_by_horizon,
    )


if __name__ == "__main__":
    main()
