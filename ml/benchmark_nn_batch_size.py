"""Quality-aware NN batch-size benchmark for Apple MPS/CUDA/CPU.

This is intentionally a real historical fold, not a synthetic throughput
microbenchmark.  It measures elapsed time and held-out WAPE/MAE for candidate
batch-size/LR-scaling policies, then recommends the fastest candidate whose
WAPE is within a configurable relative tolerance of the historical
512/fixed-LR baseline.

Example (M4 Pro):

    caffeinate -i uv run python ml/benchmark_nn_batch_size.py \
      --batch-sizes 512 1024 2048 4096 \
      --lr-scalings fixed sqrt \
      --epochs 10 \
      --quality-tolerance 0.02

The recommendation is written to ``outputs/nn_batch_benchmark.json``.  The
main pipeline's default ``--nn-batch-size auto`` consumes that file only when
it was measured on the same accelerator type.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from framework import (
    CFG,
    Config,
    add_train_lags,
    build_direct_panel,
    compute_metrics,
    load_raw,
    prepare_features,
    product_reference_dates,
    select_trainable_panel_rows,
)
from models.neural_net import (
    DEVICE,
    effective_learning_rate,
    make_numeric_preprocessor,
    make_tensors,
    nn_performance_signature,
    predict_direct,
    neural_training_target,
    train_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark NN batch size using a real held-out direct fold"
    )
    parser.add_argument(
        "--origin", default="2025-02-10",
        help="Historical forecast origin used for the benchmark",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[512, 1024, 2048, 4096],
    )
    parser.add_argument(
        "--lr-scalings", nargs="+", choices=["fixed", "sqrt", "linear"],
        default=["fixed", "sqrt"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--quality-tolerance", type=float, default=0.02,
        help="Maximum relative WAPE degradation versus 512/fixed",
    )
    parser.add_argument(
        "--output", default="outputs/nn_batch_benchmark.json",
    )
    parser.add_argument(
        "--csv-output", default="outputs/nn_batch_benchmark.csv",
    )
    parser.add_argument(
        "--training-backend",
        choices=["auto", "device_resident", "dataloader"],
        default="auto",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if any(size < 2 for size in args.batch_sizes):
        parser.error("all batch sizes must be at least 2")
    if args.quality_tolerance < 0:
        parser.error("--quality-tolerance must be non-negative")
    return args


def build_benchmark_fold(
    train_raw: pd.DataFrame,
    origin: pd.Timestamp,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    eval_start = origin + pd.Timedelta(days=1)
    eval_end = origin + pd.Timedelta(days=cfg.horizon)
    fold_train_raw = train_raw[train_raw["DateKey"].le(origin)].copy()
    fold_eval_raw = train_raw[
        train_raw["DateKey"].between(eval_start, eval_end)
    ].copy()
    if fold_train_raw.empty or fold_eval_raw.empty:
        raise ValueError(
            f"Origin {origin.date()} has no complete train/evaluation fold"
        )

    price_ref = fold_train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(fold_train_raw)
    train_feat = prepare_features(
        fold_train_raw, price_ref, first_seen, first_available, cfg
    )
    train_feat = add_train_lags(
        train_feat, cfg.lag_windows, baseline_variant=cfg.baseline_variant
    )
    eval_feat = prepare_features(
        fold_eval_raw, price_ref, first_seen, first_available, cfg
    ).reset_index(drop=True)
    panel = build_direct_panel(
        train_feat,
        range(1, cfg.horizon + 1),
        cfg=cfg,
        future_covariates=eval_feat,
    )
    train_panel = select_trainable_panel_rows(
        panel, cutoff=origin, available_only=True, cfg=cfg
    )
    eval_panel = panel[panel["OriginDateKey"].eq(origin)].reset_index(drop=True)

    actual_lookup = fold_eval_raw[["ProductId", "DateKey", "Quantity"]]
    actual = (
        eval_panel[["ProductId", "TargetDateKey"]]
        .rename(columns={"TargetDateKey": "DateKey"})
        .merge(actual_lookup, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
        ["Quantity"]
        .to_numpy(dtype=float)
    )
    return train_panel, eval_panel, actual


def candidate_grid(batch_sizes: list[int], lr_scalings: list[str]):
    seen = set()
    for batch_size in batch_sizes:
        for scaling in lr_scalings:
            # At the reference batch all policies have the same LR; run once.
            key = (batch_size, "fixed" if batch_size == 512 else scaling)
            if key not in seen:
                seen.add(key)
                yield key


def run_candidate(
    train_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    actual: np.ndarray,
    base_cfg: Config,
    batch_size: int,
    lr_scaling: str,
    epochs: int,
    seeds: list[int],
    training_backend: str,
) -> dict:
    cfg = replace(
        base_cfg,
        batch_size=batch_size,
        reference_batch_size=512,
        nn_lr_scaling=lr_scaling,
        nn_training_backend=training_backend,
        seeds=tuple(seeds),
    )
    scaler = make_numeric_preprocessor()
    tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
    y_target = neural_training_target(train_panel, cfg)

    models = []
    seed_stats = []
    started = time.perf_counter()
    for seed in seeds:
        stats = {}
        models.append(
            train_model(
                tensors, y_target, cfg, epochs=epochs, seed=seed,
                stats_out=stats,
            )
        )
        seed_stats.append(stats)
    prediction = predict_direct(models, scaler, eval_panel, cfg)
    elapsed = time.perf_counter() - started
    metrics = compute_metrics(actual, prediction)
    return {
        "batch_size": batch_size,
        "lr_scaling": lr_scaling,
        "effective_learning_rate": effective_learning_rate(cfg),
        "training_backend": seed_stats[0]["backend"] if seed_stats else training_backend,
        "epochs": epochs,
        "seeds": list(seeds),
        "train_rows": len(train_panel),
        "eval_rows": len(eval_panel),
        "elapsed_seconds": elapsed,
        "examples_per_second": sum(s["examples_per_second"] for s in seed_stats) / len(seed_stats),
        "optimizer_steps_per_seed": seed_stats[0]["optimizer_steps"],
        "final_train_loss": sum(s["final_train_loss"] for s in seed_stats) / len(seed_stats),
        "estimated_device_tensor_mb": seed_stats[0]["estimated_device_tensor_mb"],
        **metrics,
    }


def choose_recommendation(results: list[dict], tolerance: float) -> dict:
    reference = next(
        (r for r in results if r["batch_size"] == 512 and r["lr_scaling"] == "fixed"),
        None,
    )
    if reference is None:
        raise RuntimeError("Benchmark must include the 512/fixed reference")
    max_wape = reference["WAPE"] * (1.0 + tolerance)
    eligible = [
        r for r in results
        if np.isfinite(r["WAPE"])
        and np.isfinite(r["elapsed_seconds"])
        and r["WAPE"] <= max_wape
    ]
    selected = min(eligible or [reference], key=lambda r: r["elapsed_seconds"])
    return {
        "batch_size": int(selected["batch_size"]),
        "lr_scaling": selected["lr_scaling"],
        "effective_learning_rate": float(selected["effective_learning_rate"]),
        "quality_tolerance": tolerance,
        "reference_wape": float(reference["WAPE"]),
        "selected_wape": float(selected["WAPE"]),
        "relative_wape_change": float(selected["WAPE"] / reference["WAPE"] - 1.0),
        "speedup_vs_reference": float(reference["elapsed_seconds"] / selected["elapsed_seconds"]),
        "selection_rule": "fastest candidate within relative WAPE tolerance of 512/fixed",
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    base_cfg = replace(CFG)
    train_raw, test_raw = load_raw(base_cfg)
    base_cfg.num_products = int(
        max(train_raw["ProductId"].max(), test_raw["ProductId"].max())
    )
    origin = pd.Timestamp(args.origin)
    train_panel, eval_panel, actual = build_benchmark_fold(
        train_raw, origin, base_cfg
    )

    results = []
    for batch_size, lr_scaling in candidate_grid(
        args.batch_sizes, args.lr_scalings
    ):
        print(
            f"\n=== batch={batch_size}, lr_scaling={lr_scaling}, "
            f"device={DEVICE.type} ==="
        )
        result = run_candidate(
            train_panel, eval_panel, actual, base_cfg,
            batch_size, lr_scaling, args.epochs, args.seeds,
            args.training_backend,
        )
        results.append(result)
        print(
            f"time={result['elapsed_seconds']:.1f}s | "
            f"throughput={result['examples_per_second']:.0f} examples/s | "
            f"WAPE={result['WAPE']:.4f} | MAE={result['MAE']:.3f}"
        )

    recommendation = choose_recommendation(results, args.quality_tolerance)
    environment = {
        "device": DEVICE.type,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "origin": origin.date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = {
        "schema_version": "nn-batch-v1",
        "environment": environment,
        "model_signature": nn_performance_signature(base_cfg),
        "benchmark_config": {
            "epochs": args.epochs,
            "seeds": args.seeds,
            "quality_tolerance": args.quality_tolerance,
            "base_config": asdict(base_cfg),
        },
        "results": results,
        "recommendation": recommendation,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(results).to_csv(args.csv_output, index=False)

    print("\n=== recommendation ===")
    print(json.dumps(recommendation, indent=2))
    print(f"Wrote {args.output} and {args.csv_output}")


if __name__ == "__main__":
    main()
