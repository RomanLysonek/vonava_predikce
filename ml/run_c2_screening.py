"""Staged Tier C2 semantic feature-group screening.

The runner asks which business-semantic feature groups add value around the
C1 policy. It uses direct forecasting, one NN seed, a reduced epoch budget and
four stratified origins. Forward selection avoids the full 2^5 Cartesian
search. Because the full C1 half-life-365 confirmation was effectively tied
with the no-decay baseline on test-aligned WAPE, the final selected C2 feature
set is also rechecked under no decay and a 90-day half-life before a full-run
recommendation is emitted.
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

from framework import (
    BASELINE_VARIANTS,
    C2_FEATURE_GROUPS,
    CFG,
    normalize_c2_feature_groups,
    load_raw,
)
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
    parser = argparse.ArgumentParser(description="Run staged Tier C2 screening")
    parser.add_argument("--output-dir", default="outputs/c2_screening")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-tolerance", type=float, default=0.03)
    parser.add_argument("--min-relative-improvement", type=float, default=0.002)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--training-window-days", default="all")
    parser.add_argument("--recency-half-life-days", default="365")
    parser.add_argument(
        "--sensitivity-half-lives", nargs="*", default=["none", "90"],
        help="Additional C1 half-lives checked for the winning C2 feature set",
    )
    parser.add_argument(
        "--baseline-variant", choices=sorted(BASELINE_VARIANTS),
        default="weighted_4321",
    )
    parser.add_argument("--trend-features", choices=["on", "off"], default="off")
    parser.add_argument(
        "--origins", nargs="*", default=None,
        help="Optional YYYY-MM-DD origins; defaults to four stratified folds",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if args.quality_tolerance < 0 or args.min_relative_improvement < 0:
        parser.error("quality thresholds must be nonnegative")
    return args


def _parse_optional(value, none_token: str, cast):
    if value is None or str(value).lower() == none_token:
        return None
    parsed = cast(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive value or {none_token}")
    return parsed


def _half_token(value) -> str:
    return "none" if value is None else str(value).replace(".", "p")


def _group_key(groups) -> str:
    canonical = normalize_c2_feature_groups(groups)
    return "none" if not canonical else "+".join(canonical)


def _candidate_key(groups, half_life) -> str:
    return f"half-{_half_token(half_life)}__groups-{_group_key(groups)}"


def _extract_model_rows(
    label: str,
    groups: tuple[str, ...],
    half_life: float | None,
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
            "c2_feature_groups": ",".join(groups),
            "n_feature_groups": len(groups),
            "recency_half_life_days": half_life,
            "model": model,
            "WAPE": float(row["WAPE"]),
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "BiasRatio": float(row["BiasRatio"]),
            "Coverage": float(row["coverage"]),
            "test_aligned_WAPE": float(aligned_lookup.get(model, np.nan)),
            "elapsed_seconds": float(elapsed_seconds),
            "n_oof_rows": int(len(oof)),
        })
    return rows


def _nn_row(rows: list[dict], candidate: str) -> pd.Series:
    frame = pd.DataFrame(rows)
    selected = frame[
        frame["model"].eq("NeuralNet") & frame["candidate"].eq(candidate)
    ]
    if selected.empty:
        raise RuntimeError(f"Missing NeuralNet row for {candidate}")
    return selected.iloc[0]


def _best_eligible(
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
    control = nn[nn["candidate"].eq(control_name)]
    if control.empty:
        raise RuntimeError("C2 control result is missing")
    broad_limit = float(control.iloc[0]["WAPE"]) * (1.0 + quality_tolerance)
    eligible = nn[
        np.isfinite(nn["test_aligned_WAPE"])
        & (nn["Coverage"] >= 0.999)
        & (nn["WAPE"] <= broad_limit)
    ].copy()
    if eligible.empty:
        return control_name
    eligible["abs_bias"] = eligible["BiasRatio"].abs()
    return str(
        eligible.sort_values(
            ["test_aligned_WAPE", "WAPE", "abs_bias", "n_feature_groups"]
        ).iloc[0]["candidate"]
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    if args.reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origins = pd.to_datetime(args.origins) if args.origins else DEFAULT_SCREENING_ORIGINS
    base_half_life = _parse_optional(args.recency_half_life_days, "none", float)
    base_cfg = replace(
        CFG,
        seeds=(args.seed,),
        cv_epochs=args.epochs,
        final_epochs=args.epochs,
        batch_size=args.batch_size,
        reference_batch_size=512,
        nn_lr_scaling="fixed",
        nn_training_backend="auto",
        training_window_days=_parse_optional(args.training_window_days, "all", int),
        recency_half_life_days=base_half_life,
        baseline_variant=args.baseline_variant,
        enable_trend_features=args.trend_features == "on",
        c2_feature_groups=(),
    )
    train_raw, _ = load_raw(base_cfg)
    base_cfg.num_products = int(train_raw["ProductId"].max())

    all_rows: list[dict] = []
    evaluated: dict[tuple[tuple[str, ...], float | None], str] = {}
    config_by_label: dict[str, dict] = {}

    def evaluate(label: str, groups, half_life=base_half_life) -> None:
        canonical = normalize_c2_feature_groups(groups)
        half_life = None if half_life is None else float(half_life)
        config_by_label[label] = {
            "c2_feature_groups": canonical,
            "recency_half_life_days": half_life,
        }
        key = (canonical, half_life)
        if key in evaluated:
            source = evaluated[key]
            copied = [
                dict(row, candidate=label)
                for row in all_rows if row["candidate"] == source
            ]
            all_rows.extend(copied)
            print(f"{label}: reused identical candidate {source}")
            return

        cfg = replace(
            base_cfg,
            c2_feature_groups=canonical,
            recency_half_life_days=half_life,
        )
        candidate_checkpoint = checkpoint_root / _candidate_key(canonical, half_life)
        print(
            f"\n{label}: half_life={_half_token(half_life)}, "
            f"C2 groups={_group_key(canonical)}"
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
        rows = _extract_model_rows(label, canonical, half_life, oof, elapsed)
        all_rows.extend(rows)
        evaluated[key] = label
        oof_dir = output_dir / "candidate_oof"
        oof_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(oof_dir / f"{_candidate_key(canonical, half_life)}.csv", index=False)
        pd.DataFrame(all_rows).to_csv(
            output_dir / "c2_screening_results.partial.csv", index=False
        )

    control_name = "control"
    evaluate(control_name, ())

    single_names = []
    for group in C2_FEATURE_GROUPS:
        label = f"single_{group}"
        single_names.append(label)
        evaluate(label, (group,))

    current_name = _best_eligible(
        all_rows,
        [control_name] + single_names,
        control_name,
        args.quality_tolerance,
    )
    current_groups = config_by_label[current_name]["c2_feature_groups"]
    print(f"\nInitial C2 winner: {current_name} ({_group_key(current_groups)})")

    round_index = 1
    while len(current_groups) < len(C2_FEATURE_GROUPS):
        remaining = [g for g in C2_FEATURE_GROUPS if g not in current_groups]
        candidate_names = [current_name]
        for group in remaining:
            candidate_groups = normalize_c2_feature_groups((*current_groups, group))
            label = f"forward_{round_index}_{_group_key(candidate_groups)}"
            candidate_names.append(label)
            evaluate(label, candidate_groups)
        best_name = _best_eligible(
            all_rows, candidate_names, control_name, args.quality_tolerance
        )
        current_score = float(_nn_row(all_rows, current_name)["test_aligned_WAPE"])
        best_score = float(_nn_row(all_rows, best_name)["test_aligned_WAPE"])
        if (
            best_name == current_name
            or not np.isfinite(best_score)
            or best_score > current_score * (1.0 - args.min_relative_improvement)
        ):
            break
        current_name = best_name
        current_groups = config_by_label[current_name]["c2_feature_groups"]
        print(f"Accepted: {current_name} ({_group_key(current_groups)})")
        round_index += 1

    evaluate("all_groups", C2_FEATURE_GROUPS)
    group_winner = _best_eligible(
        all_rows, [control_name, current_name, "all_groups"],
        control_name, args.quality_tolerance,
    )
    winner_groups = config_by_label[group_winner]["c2_feature_groups"]

    # Resolve the remaining C1 uncertainty cheaply under the selected semantic
    # representation rather than blocking C2 on another full baseline run.
    sensitivity_names = [group_winner]
    for value in args.sensitivity_half_lives:
        half_life = _parse_optional(value, "none", float)
        label = f"sensitivity_half_{_half_token(half_life)}"
        sensitivity_names.append(label)
        evaluate(label, winner_groups, half_life=half_life)

    recommendation_name = _best_eligible(
        all_rows,
        [control_name] + sensitivity_names,
        control_name,
        args.quality_tolerance,
    )
    recommendation = config_by_label[recommendation_name]
    recommendation_groups = recommendation["c2_feature_groups"]
    recommendation_half_life = recommendation["recency_half_life_days"]

    results = pd.DataFrame(all_rows)
    results.to_csv(output_dir / "c2_screening_results.csv", index=False)
    partial = output_dir / "c2_screening_results.partial.csv"
    if partial.exists():
        partial.unlink()

    recommendation_path = output_dir / "recommendation.json"
    half_flag = "none" if recommendation_half_life is None else str(recommendation_half_life)
    command_prefix = (
        "caffeinate -i uv run python ml/pipeline.py "
        "--forecast-strategy direct --primary-strategy direct "
        "--submission-model NeuralNet --selection-metric WAPE "
        "--selection-protocol test-aligned "
        f"--training-window-days {args.training_window_days} "
        f"--recency-half-life-days {half_flag} "
        f"--baseline-variant {args.baseline_variant} "
        f"--trend-features {args.trend_features} "
        f"--c2-config {recommendation_path.as_posix()} "
        "--nn-batch-size 512 --nn-lr-scaling fixed "
    )
    payload = {
        "schema_version": "c2-screening-v1",
        "screening_policy": {
            "origins": [str(pd.Timestamp(x).date()) for x in origins],
            "epochs": args.epochs,
            "seeds": [args.seed],
            "batch_size": args.batch_size,
            "lr_scaling": "fixed",
            "strategy": "direct",
            "quality_tolerance": args.quality_tolerance,
            "min_relative_improvement": args.min_relative_improvement,
            "selection_model": "NeuralNet",
            "selection_metric": "test_aligned_WAPE",
        },
        "c1_base": {
            "training_window_days": base_cfg.training_window_days,
            "recency_half_life_days": base_half_life,
            "baseline_variant": base_cfg.baseline_variant,
            "enable_trend_features": base_cfg.enable_trend_features,
        },
        "group_winner": {
            "candidate": group_winner,
            "c2_feature_groups": list(winner_groups),
        },
        "recommendation": {
            "candidate": recommendation_name,
            "config": {
                "c2_feature_groups": list(recommendation_groups),
                "recency_half_life_days": recommendation_half_life,
            },
            "full_confirmation_command": (
                command_prefix
                + "--reset-checkpoints "
                + "2>&1 | tee pipeline_c2_direct_512_fixed.log"
            ),
            "full_confirmation_resume_command": (
                command_prefix
                + "--resume "
                + "2>&1 | tee -a pipeline_c2_direct_512_fixed.log"
            ),
        },
        "base_config": asdict(base_cfg),
    }
    with open(recommendation_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nC2 recommendation: {recommendation_name}")
    print(f"Half-life: {_half_token(recommendation_half_life)}")
    print(f"Groups: {_group_key(recommendation_groups)}")
    print(f"Saved: {output_dir / 'c2_screening_results.csv'}")
    print(f"Saved: {recommendation_path}")
    print("\nFull confirmation command:\n" + payload["recommendation"]["full_confirmation_command"])
    print("\nResume command:\n" + payload["recommendation"]["full_confirmation_resume_command"])


if __name__ == "__main__":
    main()
