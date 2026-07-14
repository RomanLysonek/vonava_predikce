"""Staged Tier C1 recency/nonstationarity screening.

This runner deliberately uses direct forecasting, one NN seed, a reduced epoch
budget and four stratified development origins.  It first screens history
windows/half-lives, then baseline variants, then the trend feature group.  The
winner is written as a pipeline-consumable recommendation JSON.

The screen is for ranking candidates, not for final reporting.  Every treatment
is compared with a control under the same batch, seed, epoch and origin policy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from framework import BASELINE_VARIANTS, CFG, Config, load_raw
from pipeline import (
    ForecastStrategy,
    compute_test_aligned_scores,
    run_walk_forward_cv,
    summarize_oof_by_strategy,
    summarize_validation_strata,
)

DEFAULT_SCREENING_ORIGINS = pd.to_datetime([
    "2023-01-10",  # older winter/test-like
    "2024-06-20",  # regular regime
    "2024-11-29",  # holiday-event stress
    "2025-02-10",  # recent winter/test-like
])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run staged Tier C1 screening")
    parser.add_argument("--output-dir", default="outputs/c1_screening")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-tolerance", type=float, default=0.03)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--origins", nargs="*", default=None,
        help="Optional YYYY-MM-DD origins; defaults to four stratified folds",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if args.quality_tolerance < 0:
        parser.error("--quality-tolerance must be nonnegative")
    return args


def _config_key(config: dict) -> str:
    def token(name: str, value) -> str:
        if value is None:
            return "all" if name == "training_window_days" else "none"
        return str(value).lower().replace(".", "p")

    names = (
        "training_window_days",
        "recency_half_life_days",
        "baseline_variant",
        "enable_trend_features",
    )
    return "__".join(
        f"{name}-{token(name, config[name])}" for name in names
    )


def _canonical_config(config: dict) -> dict:
    return {
        "training_window_days": (
            None if config.get("training_window_days") in (None, "all")
            else int(config["training_window_days"])
        ),
        "recency_half_life_days": (
            None if config.get("recency_half_life_days") in (None, "none")
            else float(config["recency_half_life_days"])
        ),
        "baseline_variant": str(config.get("baseline_variant", "weighted_4321")),
        "enable_trend_features": bool(config.get("enable_trend_features", False)),
    }


def _extract_model_rows(
    label: str,
    stage: str,
    config: dict,
    oof: pd.DataFrame,
    elapsed_seconds: float,
) -> list[dict]:
    summary = summarize_oof_by_strategy(oof)
    primary = summary[
        summary["evaluation_regime"].eq("conditional")
        & summary["comparison_population"].eq("common")
        & summary["aggregation"].eq("global")
        & summary["strategy"].eq("direct")
    ]
    strata = summarize_validation_strata(oof)
    aligned = compute_test_aligned_scores(strata, metric="WAPE")
    aligned_lookup = {
        row["model"]: row["test_aligned_score"]
        for _, row in aligned[aligned["strategy"].eq("direct")].iterrows()
    }
    rows = []
    for _, row in primary.iterrows():
        model = str(row["model"])
        rows.append({
            "candidate": label,
            "stage": stage,
            **config,
            "model": model,
            "WAPE": float(row["WAPE"]),
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "BiasRatio": float(row["BiasRatio"]),
            # summarize_oof() exposes diagnostic fields in snake_case.  Keep
            # the screening CSV's public metric label capitalized, but read
            # the canonical summary column here.
            "Coverage": float(row["coverage"]),
            "test_aligned_WAPE": float(aligned_lookup.get(model, np.nan)),
            "elapsed_seconds": float(elapsed_seconds),
            "n_oof_rows": int(len(oof)),
        })
    return rows


def _pick_winner(
    rows: list[dict],
    candidate_names: list[str],
    control_name: str,
    quality_tolerance: float,
) -> str:
    frame = pd.DataFrame(rows)
    nn = frame[
        frame["model"].eq("NeuralNet")
        & frame["candidate"].isin(candidate_names)
    ].copy()
    if nn.empty:
        raise RuntimeError("C1 screening produced no NeuralNet rows")
    control = nn[nn["candidate"].eq(control_name)]
    if control.empty:
        raise RuntimeError(f"Missing C1 control candidate {control_name}")
    control_wape = float(control.iloc[0]["WAPE"])
    eligible = nn[
        np.isfinite(nn["test_aligned_WAPE"])
        & (nn["WAPE"] <= control_wape * (1.0 + quality_tolerance))
    ].copy()
    if eligible.empty:
        eligible = control.copy()
    return str(
        eligible.sort_values(
            ["test_aligned_WAPE", "WAPE", "BiasRatio"],
            key=lambda series: series.abs() if series.name == "BiasRatio" else series,
        ).iloc[0]["candidate"]
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    if args.reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origins = (
        pd.to_datetime(args.origins)
        if args.origins else DEFAULT_SCREENING_ORIGINS
    )
    base_cfg = replace(
        CFG,
        seeds=(args.seed,),
        cv_epochs=args.epochs,
        final_epochs=args.epochs,
        batch_size=args.batch_size,
        reference_batch_size=512,
        nn_lr_scaling="fixed",
        nn_training_backend="auto",
        c2_feature_groups=(),
    )
    train_raw, _ = load_raw(base_cfg)
    base_cfg.num_products = int(train_raw["ProductId"].max())

    all_rows: list[dict] = []
    evaluated: dict[tuple, tuple[str, list[dict]]] = {}
    configs_by_label: dict[str, dict] = {}

    def evaluate(label: str, stage: str, config: dict) -> None:
        config = _canonical_config(config)
        configs_by_label[label] = config
        key = tuple(config.items())
        if key in evaluated:
            _, source_rows = evaluated[key]
            copied = [dict(row, candidate=label, stage=stage) for row in source_rows]
            all_rows.extend(copied)
            print(f"[{stage}] {label}: reused identical candidate")
            return

        cfg = replace(base_cfg, **config)
        candidate_checkpoint = checkpoint_root / _config_key(config)
        print(
            f"\n[{stage}] {label}: window={config['training_window_days'] or 'all'}, "
            f"half_life={config['recency_half_life_days'] or 'none'}, "
            f"baseline={config['baseline_variant']}, "
            f"trend={config['enable_trend_features']}"
        )
        started = time.perf_counter()
        oof = run_walk_forward_cv(
            train_raw,
            origins,
            "development",
            cfg,
            strategy=ForecastStrategy.DIRECT,
            checkpoint_dir=str(candidate_checkpoint),
            resume=args.resume,
        )
        elapsed = time.perf_counter() - started
        rows = _extract_model_rows(label, stage, config, oof, elapsed)
        all_rows.extend(rows)
        evaluated[key] = (label, rows)
        oof_dir = output_dir / "candidate_oof"
        oof_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(oof_dir / f"{_config_key(config)}.csv", index=False)
        pd.DataFrame(all_rows).to_csv(
            output_dir / "c1_screening_results.partial.csv", index=False
        )

    # Stage A: broad recency policy screen.  This is deliberately staged rather
    # than a full window x half-life Cartesian product.
    stage_a = {
        "control": (None, None),
        "half365": (None, 365.0),
        "half180": (None, 180.0),
        "half90": (None, 90.0),
        "window730": (730, None),
        "window730_half180": (730, 180.0),
        "window365": (365, None),
        "window365_half180": (365, 180.0),
    }
    for label, (window, half_life) in stage_a.items():
        evaluate(label, "recency", {
            "training_window_days": window,
            "recency_half_life_days": half_life,
            "baseline_variant": "weighted_4321",
            "enable_trend_features": False,
        })
    winner_a = _pick_winner(
        all_rows, list(stage_a), "control", args.quality_tolerance
    )
    best_recency = configs_by_label[winner_a]
    print(f"\nRecency winner: {winner_a}")

    # Stage B: baseline formulation around the winning recency policy.
    stage_b_names = []
    for variant in sorted(BASELINE_VARIANTS):
        label = f"baseline_{variant}"
        stage_b_names.append(label)
        evaluate(label, "baseline", {
            **best_recency,
            "baseline_variant": variant,
            "enable_trend_features": False,
        })
    control_b = "baseline_weighted_4321"
    winner_b = _pick_winner(
        all_rows, stage_b_names, control_b, args.quality_tolerance
    )
    best_baseline = configs_by_label[winner_b]
    print(f"Baseline winner: {winner_b}")

    # Stage C: grouped trend features on/off around the preceding winner.
    evaluate("trend_off", "trend", {
        **best_baseline,
        "enable_trend_features": False,
    })
    evaluate("trend_on", "trend", {
        **best_baseline,
        "enable_trend_features": True,
    })
    winner_c = _pick_winner(
        all_rows, ["trend_off", "trend_on"], "trend_off",
        args.quality_tolerance,
    )
    recommendation = configs_by_label[winner_c]
    print(f"Trend winner: {winner_c}")

    results = pd.DataFrame(all_rows)
    results.to_csv(output_dir / "c1_screening_results.csv", index=False)
    partial = output_dir / "c1_screening_results.partial.csv"
    if partial.exists():
        partial.unlink()

    repo_relative_recommendation = str(
        (output_dir / "recommendation.json").as_posix()
    )
    command_prefix = (
        "caffeinate -i uv run python ml/pipeline.py "
        "--forecast-strategy direct --primary-strategy direct "
        "--submission-model NeuralNet --selection-metric WAPE "
        "--selection-protocol test-aligned "
        f"--c1-config {repo_relative_recommendation} "
        "--nn-batch-size 512 --nn-lr-scaling fixed "
    )
    command = (
        command_prefix
        + "--reset-checkpoints "
        + "2>&1 | tee pipeline_c1_direct_512_fixed.log"
    )
    resume_command = (
        command_prefix
        + "--resume "
        + "2>&1 | tee -a pipeline_c1_direct_512_fixed.log"
    )
    payload = {
        "schema_version": "c1-screening-v1",
        "screening_policy": {
            "origins": [str(pd.Timestamp(x).date()) for x in origins],
            "epochs": args.epochs,
            "seeds": [args.seed],
            "batch_size": args.batch_size,
            "lr_scaling": "fixed",
            "strategy": "direct",
            "quality_tolerance": args.quality_tolerance,
            "selection_model": "NeuralNet",
            "selection_metric": "test_aligned_WAPE",
        },
        "stage_winners": {
            "recency": winner_a,
            "baseline": winner_b,
            "trend": winner_c,
        },
        "recommendation": {
            "config": recommendation,
            "full_confirmation_command": command,
            "full_confirmation_resume_command": resume_command,
        },
        "base_config": asdict(base_cfg),
    }
    with open(output_dir / "recommendation.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {output_dir / 'c1_screening_results.csv'}")
    print(f"Saved: {output_dir / 'recommendation.json'}")
    print("\nFull confirmation command:\n" + command)
    print("\nResume command after interruption:\n" + resume_command)


if __name__ == "__main__":
    main()
