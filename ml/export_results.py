"""Regenerate ``outputs/results.json`` exclusively from persisted artifacts.

This exporter never retrains models and never reconstructs full-precision
forecasts from rounded ``submission.csv``.  The canonical source is
``outputs/final_forecasts.parquet``, written by ``ml/pipeline.py`` before the
presentation JSON is exported.

When recovering from a failed JSON export, pass the same runtime arguments as
the completed pipeline run so the rebuilt metadata describes the estimator
that actually produced the persisted artifacts.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

from pipeline import (
    CFG,
    ForecastStrategy,
    PrimaryStrategy,
    SubmissionModel,
    configure_c1_runtime,
    configure_c2_runtime,
    configure_c34_runtime,
    configure_c5_runtime,
    configure_nn_runtime,
    export_results_json,
    load_current_final_audit_artifacts,
    load_raw,
    parse_args,
)
from dashboard_artifacts import publish_static_dashboard


def _read_csv_if_present(path: str, **kwargs) -> pd.DataFrame:
    """Read an optional CSV, treating an empty artifact as no rows.

    Direct-only runs intentionally have no paired-strategy rows, so pandas may
    encounter a zero-byte ``strategy_pair_summary.csv``.  Missing and empty
    optional artifacts are equivalent here; malformed non-empty CSVs must still
    raise rather than being silently ignored.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_existing_results(path: str) -> dict:
    """Return a prior valid payload, or an empty dict after a partial write."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv=None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    options = parse_args(raw_argv)

    train_raw, test_raw = load_raw(CFG)
    CFG.num_products = int(
        max(train_raw["ProductId"].max(), test_raw["ProductId"].max())
    )
    configure_c1_runtime(CFG, options)
    configure_c2_runtime(CFG, options)
    configure_c34_runtime(CFG, options)
    configure_c5_runtime(CFG, options)
    configure_nn_runtime(CFG, options)

    out = CFG.output_dir
    submission = pd.read_csv(
        os.path.join(out, "submission.csv"), parse_dates=["DateKey"]
    )
    cv_results = pd.read_csv(os.path.join(out, "cv_results.csv"))
    cv_results_all = _read_csv_if_present(os.path.join(out, "cv_results_all.csv"))
    dev_summary = pd.read_csv(os.path.join(out, "dev_summary.csv"))
    benchmark_summary = pd.read_csv(os.path.join(out, "benchmark_summary.csv"))
    final_df = pd.read_parquet(os.path.join(out, "final_forecasts.parquet"))
    final_df["DateKey"] = pd.to_datetime(final_df["DateKey"])

    pair_summary = _read_csv_if_present(
        os.path.join(out, "strategy_pair_summary.csv")
    )
    strategy_by_horizon = _read_csv_if_present(
        os.path.join(out, "strategy_by_horizon.csv")
    )
    validation_strata_summary = _read_csv_if_present(
        os.path.join(out, "validation_strata_summary.csv")
    )
    test_aligned_scores = _read_csv_if_present(
        os.path.join(out, "test_aligned_scores.csv")
    )
    prediction_diagnostics = _read_csv_if_present(
        os.path.join(out, "prediction_diagnostics.csv")
    )
    prediction_diagnostics_by_origin = _read_csv_if_present(
        os.path.join(out, "prediction_diagnostics_by_origin.csv"),
        parse_dates=["origin"],
    )
    channel_share_summary = _read_csv_if_present(
        os.path.join(out, "channel_share_summary.csv")
    )
    ensemble_comparison = _read_csv_if_present(
        os.path.join(out, "ensemble_comparison.csv")
    )
    per_product_summary = _read_csv_if_present(
        os.path.join(out, "per_product_summary.csv")
    )
    top_decile_summary = _read_csv_if_present(
        os.path.join(out, "top_decile_summary.csv")
    )
    top_error_rows = _read_csv_if_present(
        os.path.join(out, "top_error_rows.csv"), parse_dates=["origin", "DateKey"]
    )
    ablation_showcase = _read_csv_if_present(
        os.path.join(out, "ablation_showcase.csv")
    )
    (
        final_audit_summary,
        final_audit_test_aligned_scores,
    ) = load_current_final_audit_artifacts(out)
    ensemble_path = os.path.join(out, "ensemble_weights.json")
    ensemble_payload = _load_existing_results(ensemble_path)

    existing_path = os.path.join(out, "results.json")
    existing = _load_existing_results(existing_path)
    if not existing and not raw_argv:
        raise RuntimeError(
            "outputs/results.json is missing or incomplete. Re-run this artifact-only "
            "exporter with the exact pipeline runtime arguments so configuration "
            "metadata is not guessed."
        )

    if options.primary_strategy in {
        PrimaryStrategy.DIRECT,
        PrimaryStrategy.RECURSIVE,
    }:
        canonical_strategy = options.primary_strategy.value
    else:
        canonical_strategy = existing.get("selection", {}).get(
            "canonical_strategy"
        ) or existing.get("config", {}).get("primary_strategy")
        if canonical_strategy not in set(final_df["strategy"].astype(str)):
            available = sorted(final_df["strategy"].astype(str).unique())
            canonical_strategy = "direct" if "direct" in available else available[0]

    if options.submission_model is SubmissionModel.AUTO:
        canonical_model = existing.get("selection", {}).get(
            "canonical_model", "NeuralNet"
        )
    else:
        canonical_model = options.submission_model.value

    if cv_results_all.empty:
        cv_results_all = cv_results.assign(strategy=canonical_strategy)

    forecasts_by_strategy: dict[str, dict] = {}
    raw_forecasts: dict[str, dict] = {}
    for strategy, strategy_df in final_df.groupby("strategy"):
        strategy = str(strategy)
        raw_forecasts[strategy] = {}
        strategy_json = {}
        for model, model_df in strategy_df.groupby("model"):
            aligned = test_raw[["ProductId", "DateKey"]].merge(
                model_df[["ProductId", "DateKey", "prediction_raw"]],
                on=["ProductId", "DateKey"],
                how="left",
                validate="one_to_one",
            )
            raw_forecasts[strategy][str(model)] = aligned[
                "prediction_raw"
            ].to_numpy(dtype=float)
            per_product = {}
            for pid, sub in aligned.groupby("ProductId"):
                sub = sub.sort_values("DateKey")
                per_product[str(int(pid))] = {
                    "dates": sub["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
                    "quantity": sub["prediction_raw"].astype(float).tolist(),
                }
            strategy_json[str(model)] = per_product
        forecasts_by_strategy[strategy] = strategy_json

    if canonical_strategy not in raw_forecasts:
        raise RuntimeError(
            f"No persisted final forecasts for canonical strategy "
            f"{canonical_strategy!r}"
        )
    canonical_forecasts = raw_forecasts[canonical_strategy]
    if canonical_model not in canonical_forecasts:
        raise RuntimeError(
            f"No persisted {canonical_model!r} forecasts for strategy "
            f"{canonical_strategy!r}"
        )

    export_results_json(
        train_raw,
        test_raw,
        submission,
        canonical_forecasts,
        cv_results,
        CFG,
        dev_summary=dev_summary,
        benchmark_summary=benchmark_summary,
        runtime_options=options,
        forecasts_by_strategy=forecasts_by_strategy,
        strategy_comparison=pair_summary,
        canonical_strategy=canonical_strategy,
        canonical_model=canonical_model,
        cv_results_all=cv_results_all,
        strategy_by_horizon=strategy_by_horizon,
        validation_strata_summary=validation_strata_summary,
        test_aligned_scores=test_aligned_scores,
        prediction_diagnostics=prediction_diagnostics,
        prediction_diagnostics_by_origin=prediction_diagnostics_by_origin,
        channel_share_summary=channel_share_summary,
        ensemble_payload=ensemble_payload,
        ensemble_comparison=ensemble_comparison,
        per_product_summary=per_product_summary,
        top_decile_summary=top_decile_summary,
        top_error_rows=top_error_rows,
        ablation_showcase=ablation_showcase,
        final_audit_summary=final_audit_summary,
        final_audit_test_aligned_scores=final_audit_test_aligned_scores,
    )
    publish_static_dashboard(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.path.join(out, "results.json"),
    )
    print("Artifact-only export complete; no model training was performed.")


if __name__ == "__main__":
    main()
