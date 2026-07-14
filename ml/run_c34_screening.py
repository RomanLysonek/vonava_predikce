"""Fast combined Tier C3 objective and Tier C4 channel-task screening.

The expensive C2 screen spent most wall time retraining structured models for
NN-only questions. This runner separates the work:

* NN objective/target/channel candidates run with structured models disabled.
* XGBoost/LightGBM target modes are evaluated only three times in a dedicated
  structured-only stage.
* Every candidate uses the same four stratified origins, one seed, 12 epochs,
  and batch 2048/fixed. Full confirmation remains 512/fixed, three seeds.

The C2 all-feature representation is treated as the base. Both no-decay and
365-day recency policies are retained at the gate because their C2 screening
WAPE difference was too small to regard as conclusive.
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
    CFG,
    C2_FEATURE_GROUPS,
    NN_LOSSES,
    NN_TARGET_MODES,
    TREE_TARGET_MODES,
    load_raw,
    normalize_c2_feature_groups,
)
from pipeline import (
    ForecastStrategy,
    compute_test_aligned_scores,
    run_walk_forward_cv_direct,
    summarize_oof_by_strategy,
    summarize_validation_strata,
)

DEFAULT_SCREENING_ORIGINS = pd.to_datetime([
    "2023-01-10",
    "2024-06-20",
    "2024-11-29",
    "2025-02-10",
])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run fast combined C3/C4 screening")
    parser.add_argument("--output-dir", default="outputs/c34_screening")
    parser.add_argument("--c2-config", default="outputs/c2_screening/recommendation.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-tolerance", type=float, default=0.03)
    parser.add_argument("--min-relative-improvement", type=float, default=0.002)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--origins", nargs="*", default=None)
    parser.add_argument(
        "--base-half-lives", nargs="*", default=["none", "365"],
        help="C2 recency policies carried into the C3/C4 gate",
    )
    parser.add_argument(
        "--channel-aux-weights", nargs="*", type=float,
        default=[0.05, 0.10, 0.20, 0.35],
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if args.quality_tolerance < 0 or args.min_relative_improvement < 0:
        parser.error("quality thresholds must be nonnegative")
    if any(weight < 0 for weight in args.channel_aux_weights):
        parser.error("channel auxiliary weights must be nonnegative")
    return args


def _parse_half_life(value):
    if value is None or str(value).lower() == "none":
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("half-life must be positive or none")
    return parsed


def _token(value) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def _load_c2_base(path: str) -> tuple[tuple[str, ...], float | None]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    recommendation = payload.get("recommendation", {}).get("config", {})
    groups = normalize_c2_feature_groups(
        recommendation.get("c2_feature_groups", C2_FEATURE_GROUPS)
    )
    half_life = recommendation.get("recency_half_life_days")
    half_life = None if half_life is None else float(half_life)
    return groups, half_life


def _primary_rows(oof: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_oof_by_strategy(oof)
    return summary[
        summary["evaluation_regime"].eq("conditional")
        & summary["comparison_population"].eq("common")
        & summary["aggregation"].eq("global")
        & summary["strategy"].eq("direct")
    ]


def _aligned_lookup(oof: pd.DataFrame) -> dict[str, float]:
    strata = summarize_validation_strata(oof)
    aligned = compute_test_aligned_scores(strata, metric="WAPE")
    return {
        str(row["model"]): float(row["test_aligned_score"])
        for _, row in aligned[aligned["strategy"].eq("direct")].iterrows()
    }


def _channel_metrics(oof: pd.DataFrame) -> tuple[float, float, int]:
    if not {"actual_AppShare", "pred_AppShare_NeuralNet"}.issubset(oof.columns):
        return np.nan, np.nan, 0
    actual = pd.to_numeric(oof["actual_AppShare"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(
        oof["pred_AppShare_NeuralNet"], errors="coerce"
    ).to_numpy(dtype=float)
    total = pd.to_numeric(oof["actual"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(total) & (total > 0)
    if not mask.any():
        return np.nan, np.nan, 0
    error = np.abs(predicted[mask] - actual[mask])
    weights = total[mask]
    weighted = float(np.average(error, weights=weights)) if weights.sum() > 0 else np.nan
    return float(error.mean()), weighted, int(mask.sum())


def _extract_rows(
    label: str,
    stage: str,
    cfg,
    oof: pd.DataFrame,
    elapsed: float,
    reused_from: str | None = None,
) -> list[dict]:
    aligned = _aligned_lookup(oof)
    share_mae, share_wmae, share_n = _channel_metrics(oof)
    rows = []
    for _, row in _primary_rows(oof).iterrows():
        model = str(row["model"])
        if model not in {"NeuralNet", "XGBoost", "LightGBM"}:
            continue
        rows.append({
            "candidate": label,
            "stage": stage,
            "model": model,
            "recency_half_life_days": cfg.recency_half_life_days,
            "nn_loss": cfg.nn_loss,
            "nn_target_mode": cfg.nn_target_mode,
            "enable_channel_history_features": bool(
                cfg.enable_channel_history_features
            ),
            "channel_aux_weight": float(cfg.channel_aux_weight),
            "tree_target_mode": cfg.tree_target_mode,
            "WAPE": float(row["WAPE"]),
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "BiasRatio": float(row["BiasRatio"]),
            "Coverage": float(row["coverage"]),
            "test_aligned_WAPE": float(aligned.get(model, np.nan)),
            "app_share_MAE": share_mae if model == "NeuralNet" else np.nan,
            "app_share_weighted_MAE": share_wmae if model == "NeuralNet" else np.nan,
            "app_share_n": share_n if model == "NeuralNet" else 0,
            "elapsed_seconds": float(elapsed),
            "reused_from": reused_from,
            "n_oof_rows": int(len(oof)),
        })
    return rows


def _best_nn(
    rows: list[dict], candidates: list[str], control: str,
    quality_tolerance: float,
    min_relative_improvement: float = 0.0,
) -> str:
    frame = pd.DataFrame(rows)
    nn = frame[
        frame["model"].eq("NeuralNet") & frame["candidate"].isin(candidates)
    ].copy()
    reference = nn[nn["candidate"].eq(control)]
    if reference.empty:
        raise RuntimeError(f"Missing NN control {control}")
    broad_limit = float(reference.iloc[0]["WAPE"]) * (1.0 + quality_tolerance)
    eligible = nn[
        np.isfinite(nn["test_aligned_WAPE"])
        & (nn["Coverage"] >= 0.999)
        & (nn["WAPE"] <= broad_limit)
    ].copy()
    if eligible.empty:
        return control
    eligible["abs_bias"] = eligible["BiasRatio"].abs()
    eligible["share_tiebreak"] = eligible["app_share_weighted_MAE"].fillna(np.inf)
    winner_row = eligible.sort_values([
        "test_aligned_WAPE", "WAPE", "abs_bias", "share_tiebreak"
    ]).iloc[0]
    winner = str(winner_row["candidate"])
    if winner == control:
        return control
    reference_score = float(reference.iloc[0]["test_aligned_WAPE"])
    winner_score = float(winner_row["test_aligned_WAPE"])
    relative_improvement = (
        (reference_score - winner_score) / abs(reference_score)
        if np.isfinite(reference_score) and reference_score != 0.0 else 0.0
    )
    return winner if relative_improvement >= min_relative_improvement else control


def _best_tree_mode(
    rows: list[dict],
    quality_tolerance: float,
    min_relative_improvement: float = 0.0,
    model: str | None = None,
) -> str:
    frame = pd.DataFrame(rows)
    tree = frame[frame["stage"].eq("tree_target")].copy()
    if model is not None:
        tree = tree[tree["model"].eq(model)].copy()
    grouped = tree.groupby("tree_target_mode", dropna=False).agg(
        mean_test_aligned=("test_aligned_WAPE", "mean"),
        mean_broad=("WAPE", "mean"),
        max_abs_bias=("BiasRatio", lambda x: float(np.max(np.abs(x)))),
        min_coverage=("Coverage", "min"),
    ).reset_index()
    control = grouped[grouped["tree_target_mode"].eq("log1p")]
    if control.empty:
        raise RuntimeError("Missing log1p tree target control")
    broad_limit = float(control.iloc[0]["mean_broad"]) * (1.0 + quality_tolerance)
    eligible = grouped[
        np.isfinite(grouped["mean_test_aligned"])
        & (grouped["min_coverage"] >= 0.999)
        & (grouped["mean_broad"] <= broad_limit)
    ]
    if eligible.empty:
        return "log1p"
    winner_row = eligible.sort_values([
        "mean_test_aligned", "mean_broad", "max_abs_bias"
    ]).iloc[0]
    winner = str(winner_row["tree_target_mode"])
    if winner == "log1p":
        return "log1p"
    reference_score = float(control.iloc[0]["mean_test_aligned"])
    winner_score = float(winner_row["mean_test_aligned"])
    relative_improvement = (
        (reference_score - winner_score) / abs(reference_score)
        if np.isfinite(reference_score) and reference_score != 0.0 else 0.0
    )
    return winner if relative_improvement >= min_relative_improvement else "log1p"


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    if args.reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, c2_recommended_half = _load_c2_base(args.c2_config)
    half_lives = [_parse_half_life(value) for value in args.base_half_lives]
    if c2_recommended_half not in half_lives:
        half_lives.insert(0, c2_recommended_half)
    # Stable deduplication, including None.
    half_lives = list(dict.fromkeys(half_lives))
    origins = pd.to_datetime(args.origins) if args.origins else DEFAULT_SCREENING_ORIGINS

    base_cfg = replace(
        CFG,
        seeds=(args.seed,),
        cv_epochs=args.epochs,
        final_epochs=args.epochs,
        batch_size=args.batch_size,
        reference_batch_size=512,
        nn_lr_scaling="fixed",
        nn_training_backend="auto",
        training_window_days=None,
        recency_half_life_days=c2_recommended_half,
        baseline_variant="weighted_4321",
        enable_trend_features=False,
        c2_feature_groups=groups,
        nn_loss="huber",
        nn_target_mode="residual",
        enable_channel_history_features=False,
        channel_aux_weight=0.0,
        tree_target_mode="log1p",
    )
    train_raw, _ = load_raw(base_cfg)
    base_cfg.num_products = int(train_raw["ProductId"].max())

    rows: list[dict] = []
    configs: dict[str, dict] = {}
    evaluated: set[str] = set()
    nn_cache: dict[tuple, tuple[str, pd.DataFrame]] = {}

    def persist_partial() -> None:
        pd.DataFrame(rows).to_csv(output_dir / "c34_screening_results.partial.csv", index=False)

    def evaluate_nn(label: str, stage: str, **changes) -> None:
        if label in evaluated:
            return
        cfg = replace(base_cfg, **changes)
        configs[label] = {
            "recency_half_life_days": cfg.recency_half_life_days,
            "nn_loss": cfg.nn_loss,
            "nn_target_mode": cfg.nn_target_mode,
            "nn_combined_mse_weight": cfg.nn_combined_mse_weight,
            "enable_channel_history_features": cfg.enable_channel_history_features,
            "channel_aux_weight": cfg.channel_aux_weight,
            "channel_share_smoothing": cfg.channel_share_smoothing,
            "tree_target_mode": cfg.tree_target_mode,
        }
        print(
            f"\n[{stage}] {label}: half={_token(cfg.recency_half_life_days)}, "
            f"loss={cfg.nn_loss}, target={cfg.nn_target_mode}, "
            f"channel_history={cfg.enable_channel_history_features}, "
            f"channel_aux={cfg.channel_aux_weight}"
        )
        semantic_key = (
            cfg.recency_half_life_days,
            cfg.nn_loss,
            cfg.nn_target_mode,
            float(cfg.nn_combined_mse_weight),
            bool(cfg.enable_channel_history_features),
            float(cfg.channel_aux_weight),
            float(cfg.channel_share_smoothing),
        )
        reused_from = None
        if semantic_key in nn_cache:
            reused_from, cached_oof = nn_cache[semantic_key]
            print(f"    [reuse] statistically identical to {reused_from}")
            oof = cached_oof.copy()
            elapsed = 0.0
        else:
            started = time.perf_counter()
            oof = run_walk_forward_cv_direct(
                train_raw, origins, "development", cfg,
                checkpoint_dir=str(checkpoint_root / label),
                resume=args.resume,
                run_neural=True,
                structured_models=(),
            )
            elapsed = time.perf_counter() - started
            nn_cache[semantic_key] = (label, oof.copy())
        rows.extend(
            _extract_rows(label, stage, cfg, oof, elapsed, reused_from)
        )
        candidate_dir = output_dir / "candidate_oof"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(candidate_dir / f"{label}.csv", index=False)
        evaluated.add(label)
        persist_partial()

    def evaluate_tree(mode: str) -> None:
        label = f"tree_{mode}"
        if label in evaluated:
            return
        cfg = replace(
            base_cfg,
            recency_half_life_days=selected_half,
            enable_channel_history_features=selected_channel_history,
            tree_target_mode=mode,
        )
        configs[label] = {
            "recency_half_life_days": cfg.recency_half_life_days,
            "enable_channel_history_features": cfg.enable_channel_history_features,
            "tree_target_mode": mode,
        }
        print(f"\n[tree_target] {label}")
        started = time.perf_counter()
        oof = run_walk_forward_cv_direct(
            train_raw, origins, "development", cfg,
            checkpoint_dir=str(checkpoint_root / label),
            resume=args.resume,
            run_neural=False,
            structured_models=("XGBoost", "LightGBM"),
        )
        elapsed = time.perf_counter() - started
        rows.extend(_extract_rows(label, "tree_target", cfg, oof, elapsed))
        candidate_dir = output_dir / "candidate_oof"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(candidate_dir / f"{label}.csv", index=False)
        evaluated.add(label)
        persist_partial()

    # C2 confirmation gate without a separate expensive full run.
    base_labels = []
    for half_life in half_lives:
        label = f"base_half_{_token(half_life)}"
        base_labels.append(label)
        evaluate_nn(label, "c2_gate", recency_half_life_days=half_life)
    base_control = base_labels[0]
    base_winner = _best_nn(
        rows, base_labels, base_control, args.quality_tolerance,
        args.min_relative_improvement,
    )
    selected_half = configs[base_winner]["recency_half_life_days"]

    # C3 total-demand loss.
    loss_labels = []
    for loss in NN_LOSSES:
        label = f"loss_{loss}"
        loss_labels.append(label)
        evaluate_nn(
            label, "nn_loss",
            recency_half_life_days=selected_half,
            nn_loss=loss,
            nn_target_mode="residual",
            channel_aux_weight=0.0,
        )
    loss_winner = _best_nn(
        rows, loss_labels, "loss_huber", args.quality_tolerance,
        args.min_relative_improvement,
    )
    selected_loss = configs[loss_winner]["nn_loss"]

    # C3 target formulation under the selected loss.
    target_labels = []
    for target_mode in NN_TARGET_MODES:
        label = f"target_{target_mode}"
        target_labels.append(label)
        evaluate_nn(
            label, "nn_target",
            recency_half_life_days=selected_half,
            nn_loss=selected_loss,
            nn_target_mode=target_mode,
            channel_aux_weight=0.0,
        )
    target_winner = _best_nn(
        rows, target_labels, "target_residual", args.quality_tolerance,
        args.min_relative_improvement,
    )
    selected_target = configs[target_winner]["nn_target_mode"]

    # C4 channel-state representation and auxiliary app-share weights.
    # The history-only candidate separates the value of leakage-safe channel
    # state from the value of the multitask loss itself.
    aux_labels = ["channel_control", "channel_history_only"]
    evaluate_nn(
        "channel_control", "channel_aux",
        recency_half_life_days=selected_half,
        nn_loss=selected_loss,
        nn_target_mode=selected_target,
        enable_channel_history_features=False,
        channel_aux_weight=0.0,
    )
    evaluate_nn(
        "channel_history_only", "channel_aux",
        recency_half_life_days=selected_half,
        nn_loss=selected_loss,
        nn_target_mode=selected_target,
        enable_channel_history_features=True,
        channel_aux_weight=0.0,
    )
    for weight in args.channel_aux_weights:
        label = f"channel_history_aux_{_token(weight)}"
        aux_labels.append(label)
        evaluate_nn(
            label, "channel_aux",
            recency_half_life_days=selected_half,
            nn_loss=selected_loss,
            nn_target_mode=selected_target,
            enable_channel_history_features=True,
            channel_aux_weight=float(weight),
        )
    aux_winner = _best_nn(
        rows, aux_labels, "channel_control", args.quality_tolerance,
        args.min_relative_improvement,
    )
    selected_channel_history = configs[aux_winner][
        "enable_channel_history_features"
    ]
    selected_aux = configs[aux_winner]["channel_aux_weight"]

    # Recheck the complete C3/C4 candidate under alternate C2 half-lives.
    final_sensitivity_labels = []
    for half_life in half_lives:
        label = f"final_half_{_token(half_life)}"
        final_sensitivity_labels.append(label)
        evaluate_nn(
            label, "final_sensitivity",
            recency_half_life_days=half_life,
            nn_loss=selected_loss,
            nn_target_mode=selected_target,
            enable_channel_history_features=selected_channel_history,
            channel_aux_weight=selected_aux,
        )
    final_control = f"final_half_{_token(selected_half)}"
    final_winner = _best_nn(
        rows, final_sensitivity_labels, final_control,
        args.quality_tolerance, args.min_relative_improvement,
    )
    selected_half = configs[final_winner]["recency_half_life_days"]

    # C3 tree target formulations are isolated from the NN candidate loop.
    for mode in TREE_TARGET_MODES:
        evaluate_tree(mode)
    selected_xgboost_mode = _best_tree_mode(
        rows, args.quality_tolerance, args.min_relative_improvement,
        model="XGBoost",
    )
    selected_lightgbm_mode = _best_tree_mode(
        rows, args.quality_tolerance, args.min_relative_improvement,
        model="LightGBM",
    )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "c34_screening_results.csv", index=False)
    partial = output_dir / "c34_screening_results.partial.csv"
    if partial.exists():
        partial.unlink()

    recommendation = {
        "nn_loss": selected_loss,
        "nn_target_mode": selected_target,
        "nn_combined_mse_weight": base_cfg.nn_combined_mse_weight,
        "enable_channel_history_features": selected_channel_history,
        "channel_aux_weight": selected_aux,
        "channel_share_smoothing": base_cfg.channel_share_smoothing,
        # Keep the shared fallback for compatibility, but allow each tree
        # family to retain its own validated objective.
        "tree_target_mode": "log1p",
        "xgboost_target_mode": selected_xgboost_mode,
        "lightgbm_target_mode": selected_lightgbm_mode,
    }
    half_flag = "none" if selected_half is None else str(selected_half)
    recommendation_path = output_dir / "recommendation.json"
    command_prefix = (
        "caffeinate -i uv run python ml/pipeline.py "
        "--forecast-strategy direct --primary-strategy direct "
        "--submission-model NeuralNet --selection-metric WAPE "
        "--selection-protocol test-aligned --training-window-days all "
        f"--recency-half-life-days {half_flag} "
        "--baseline-variant weighted_4321 --trend-features off "
        f"--c2-feature-groups {','.join(groups)} "
        f"--c34-config {recommendation_path.as_posix()} "
        "--nn-batch-size 512 --nn-lr-scaling fixed "
    )
    payload = {
        "schema_version": "c34-screening-v1",
        "screening_policy": {
            "origins": [str(pd.Timestamp(origin).date()) for origin in origins],
            "epochs": args.epochs,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "strategy": ForecastStrategy.DIRECT.value,
            "structured_models_during_nn_screen": [],
            "identical_nn_configurations_reused_in_memory": True,
            "tree_modes_evaluated_separately": list(TREE_TARGET_MODES),
            "quality_tolerance": args.quality_tolerance,
            "min_relative_improvement": args.min_relative_improvement,
        },
        "c2_base": {
            "feature_groups": list(groups),
            "screened_half_lives": half_lives,
            "selected_half_life": selected_half,
        },
        "stage_winners": {
            "c2_gate": base_winner,
            "nn_loss": loss_winner,
            "nn_target": target_winner,
            "channel_aux": aux_winner,
            "final_sensitivity": final_winner,
            "xgboost_target_mode": selected_xgboost_mode,
            "lightgbm_target_mode": selected_lightgbm_mode,
        },
        "recommendation": {
            "candidate": final_winner,
            "config": recommendation,
            "recency_half_life_days": selected_half,
            "c2_feature_groups": list(groups),
            "full_confirmation_command": (
                command_prefix + "--reset-checkpoints "
                "2>&1 | tee pipeline_c34_direct_512_fixed.log"
            ),
            "full_confirmation_resume_command": (
                command_prefix + "--resume "
                "2>&1 | tee -a pipeline_c34_direct_512_fixed.log"
            ),
        },
        "base_config": asdict(base_cfg),
    }
    with open(recommendation_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nC3/C4 recommendation")
    print(json.dumps({
        "recency_half_life_days": selected_half,
        **recommendation,
    }, indent=2))
    print(f"Saved: {output_dir / 'c34_screening_results.csv'}")
    print(f"Saved: {recommendation_path}")
    print("\nFull confirmation command:\n" + payload["recommendation"]["full_confirmation_command"])
    print("\nResume command:\n" + payload["recommendation"]["full_confirmation_resume_command"])


if __name__ == "__main__":
    main()
