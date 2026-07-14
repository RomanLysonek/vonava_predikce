"""Reduced real-data regression check for the Tier C0.1 recursive NN guard."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from framework import CFG, load_raw, product_reference_dates, sanitize_future_covariates
from pipeline import _recursive_nn_predictions, _recursive_panel_training_data

DEFAULT_ORIGINS = pd.to_datetime([
    "2023-07-01",
    "2024-06-20",
    "2024-11-29",
    "2025-02-10",
])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check C0.1 recursive NN stability")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/c01_recursive_check")
    parser.add_argument("--origins", nargs="*", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be positive and batch size at least 2")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    origins = pd.to_datetime(args.origins) if args.origins else DEFAULT_ORIGINS
    cfg = replace(
        CFG,
        seeds=(args.seed,),
        cv_epochs=args.epochs,
        final_epochs=args.epochs,
        batch_size=args.batch_size,
        nn_lr_scaling="fixed",
        training_window_days=None,
        recency_half_life_days=None,
        baseline_variant="weighted_4321",
        enable_trend_features=False,
    )
    train_raw, _ = load_raw(cfg)
    cfg.num_products = int(train_raw["ProductId"].max())
    rows = []

    for origin in origins:
        fold_train = train_raw[train_raw["DateKey"].le(origin)].copy()
        fold_eval = train_raw[train_raw["DateKey"].between(
            origin + pd.Timedelta(days=1),
            origin + pd.Timedelta(days=cfg.horizon),
        )].copy()
        if fold_train.empty or fold_eval.empty:
            continue
        print(f"\nC0.1 recursive check: {origin.date()}")
        price_ref = fold_train.groupby("ProductId")["PriceLocalVat"].median()
        first_seen, first_available = product_reference_dates(fold_train)
        train_panel = _recursive_panel_training_data(
            fold_train, price_ref, first_seen, first_available, cfg
        )
        path, _, _, _ = _recursive_nn_predictions(
            train_panel,
            fold_train,
            sanitize_future_covariates(fold_eval),
            price_ref,
            first_seen,
            first_available,
            cfg,
            args.epochs,
        )
        evaluated = path.rename(columns={"TargetDateKey": "DateKey"}).merge(
            fold_eval[["ProductId", "DateKey", "Quantity", "ProductAvailable"]],
            on=["ProductId", "DateKey"],
            how="left",
            validate="one_to_one",
        )
        conditional = evaluated[
            evaluated["ProductAvailable"].astype("boolean").fillna(False)
            & evaluated["Quantity"].notna()
        ]
        actual = conditional["Quantity"].to_numpy(dtype=float)
        pred = conditional["prediction"].to_numpy(dtype=float)
        denominator = float(np.abs(actual).sum())
        wape = float(np.abs(actual - pred).sum() / denominator) if denominator else np.nan
        observed_max = float(np.nanmax(actual)) if len(actual) else np.nan
        prediction_max = float(np.nanmax(pred)) if len(pred) else np.nan
        max_ratio = (
            prediction_max / observed_max
            if np.isfinite(observed_max) and observed_max > 0 else np.nan
        )
        passed = bool(
            np.isfinite(pred).all()
            and prediction_max <= 5_000.0
            and (not np.isfinite(max_ratio) or max_ratio <= 20.0)
            and (not np.isfinite(wape) or wape < 2.0)
        )
        rows.append({
            "origin": str(origin.date()),
            "n": int(len(conditional)),
            "WAPE": wape,
            "prediction_max": prediction_max,
            "observed_max": observed_max,
            "prediction_to_observed_max_ratio": max_ratio,
            "residual_guard_count": int(path["residual_guard"].sum()),
            "residual_nonfinite_count": int(path["residual_nonfinite"].sum()),
            "catastrophic_guard_count": int(path["catastrophic_guard"].sum()),
            "fallback_count": int(path["fallback_used"].sum()),
            "residual_raw_min": float(path["residual_raw_min"].min()),
            "residual_raw_max": float(path["residual_raw_max"].max()),
            "passed": passed,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "c01_recursive_check.csv", index=False)
    payload = {
        "schema_version": "c01-recursive-check-v1",
        "config": {
            "origins": [str(pd.Timestamp(x).date()) for x in origins],
            "epochs": args.epochs,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "recursive_safety_multiplier": cfg.recursive_safety_multiplier,
            "recursive_safety_floor": cfg.recursive_safety_floor,
            "residual_guard_quantiles": [
                cfg.nn_residual_guard_lower_quantile,
                cfg.nn_residual_guard_upper_quantile,
            ],
            "residual_guard_margin": cfg.nn_residual_guard_margin,
        },
        "passed": bool(not frame.empty and frame["passed"].all()),
        "results": frame.to_dict(orient="records"),
    }
    with open(output_dir / "c01_recursive_check.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("\n" + frame.to_string(index=False))
    print(f"\nOverall pass: {payload['passed']}")
    if args.strict and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
