"""Notino quantity forecasting pipeline.

The orchestrator supports separately trained direct and recursive strategies,
walk-forward development/recent-benchmark evaluation, conditional-demand and
realized-sales reporting, model/strategy selection from development OOF, final
forecast generation, checkpoint recovery, and dashboard artifact export.

Run from the repository root, for example::

    uv run python ml/pipeline.py --forecast-strategy both --resume

Model implementations live under ``ml/models``. Shared feature engineering,
direct/one-step panels, recursive state transitions, model metadata, and
metrics live in ``ml/framework.py``. Native XGBoost/LightGBM execution remains
isolated from PyTorch in ``ml/tree_worker.py`` on macOS.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from framework import (
    BASELINE_VARIANTS,
    C2_FEATURE_GROUPS,
    NN_LOSSES,
    NN_TARGET_MODES,
    TREE_TARGET_MODES,
    CFG,
    MODEL_META,
    MODEL_ORDER,
    MODEL_SLUGS,
    MODEL_STRATEGY_SUPPORT,
    Config,
    add_train_lags,
    build_direct_panel,
    build_one_step_panel,
    forecast_recursive,
    sanitize_future_covariates,
    compute_baseline,
    compute_metrics,
    direct_panel_feature_names,
    load_raw,
    model_supports_strategy,
    order_models,
    prediction_columns_for_strategy,
    prepare_features,
    product_reference_dates,
    select_trainable_panel_rows,
    normalize_c2_feature_groups,
)
from models.naive_baselines import moving_average_predict, seasonal_naive_predict
from models.neural_net import (
    DEVICE, effective_learning_rate, make_numeric_preprocessor, make_tensors,
    neural_training_target, nn_performance_signature, predict_direct,
    resolve_training_backend, train_model,
)

np.random.seed(CFG.seed)



class ForecastStrategy(str, Enum):
    DIRECT = "direct"
    RECURSIVE = "recursive"
    BOTH = "both"


class PrimaryStrategy(str, Enum):
    AUTO = "auto"
    DIRECT = "direct"
    RECURSIVE = "recursive"


class SubmissionModel(str, Enum):
    NEURAL_NET = "NeuralNet"
    DYNAMIC_RIDGE = "DynamicRidge"
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    AUTO = "auto"


@dataclass(frozen=True)
class RuntimeOptions:
    forecast_strategy: ForecastStrategy = ForecastStrategy.DIRECT
    primary_strategy: PrimaryStrategy = PrimaryStrategy.AUTO
    submission_model: SubmissionModel = SubmissionModel.NEURAL_NET
    selection_metric: str = "WAPE"
    selection_protocol: str = "global"
    resume: bool = False
    reset_checkpoints: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    nn_batch_size: str = "auto"
    nn_lr_scaling: str = "auto"
    nn_training_backend: str = "auto"
    nn_benchmark_file: str = "outputs/nn_batch_benchmark.json"
    c1_config: str | None = None
    training_window_days: str | None = None
    recency_half_life_days: str | None = None
    baseline_variant: str | None = None
    trend_features: str | None = None
    c2_config: str | None = None
    c2_feature_groups: str | None = None
    c34_config: str | None = None
    nn_loss: str | None = None
    nn_target_mode: str | None = None
    nn_combined_mse_weight: float | None = None
    tree_target_mode: str | None = None
    xgboost_target_mode: str | None = None
    lightgbm_target_mode: str | None = None
    channel_history_features: str | None = None
    channel_aux_weight: float | None = None
    channel_share_smoothing: float | None = None


def resolve_strategies(strategy: ForecastStrategy) -> tuple[ForecastStrategy, ...]:
    if strategy is ForecastStrategy.BOTH:
        return (ForecastStrategy.DIRECT, ForecastStrategy.RECURSIVE)
    return (strategy,)


def parse_args(argv=None) -> RuntimeOptions:
    parser = argparse.ArgumentParser(description="Notino quantity forecasting pipeline")
    parser.add_argument("--forecast-strategy", choices=[s.value for s in ForecastStrategy], default="direct")
    parser.add_argument("--primary-strategy", choices=[s.value for s in PrimaryStrategy], default="auto")
    parser.add_argument("--submission-model", choices=[s.value for s in SubmissionModel], default="NeuralNet")
    parser.add_argument("--selection-metric", choices=["WAPE", "MAE", "RMSE"], default="WAPE")
    parser.add_argument(
        "--selection-protocol",
        choices=["global", "test-aligned"],
        default="global",
        help=("Global uses the original conditional/common development metric; "
              "test-aligned uses weighted winter/regular/event strata."),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed per-fold CV checkpoints from an interrupted run",
    )
    parser.add_argument(
        "--reset-checkpoints", action="store_true",
        help="Delete existing CV checkpoints before starting",
    )
    parser.add_argument(
        "--checkpoint-dir", default="outputs/checkpoints",
        help="Directory used for atomic per-fold CV checkpoints",
    )
    parser.add_argument(
        "--nn-batch-size", default="auto",
        help=("Positive integer, or 'auto'. Auto reads the quality-aware "
              "batch benchmark when present; otherwise it preserves 512."),
    )
    parser.add_argument(
        "--nn-lr-scaling", choices=["auto", "fixed", "sqrt", "linear"],
        default="auto",
        help="Learning-rate scaling relative to reference batch size 512",
    )
    parser.add_argument(
        "--nn-training-backend",
        choices=["auto", "device_resident", "dataloader"],
        default="auto",
        help="Auto keeps complete fold tensors on MPS/CUDA and uses DataLoader on CPU",
    )
    parser.add_argument(
        "--nn-benchmark-file", default="outputs/nn_batch_benchmark.json",
        help="Quality-aware batch benchmark used by --nn-batch-size auto",
    )
    parser.add_argument(
        "--c1-config", default=None,
        help=("Recommendation JSON written by ml/run_c1_screening.py. "
              "Explicit C1 CLI options override values from this file."),
    )
    parser.add_argument(
        "--training-window-days", default=None,
        help="Positive integer or 'all' (C1 history-window policy)",
    )
    parser.add_argument(
        "--recency-half-life-days", default=None,
        help="Positive number or 'none' (C1 exponential sample weighting)",
    )
    parser.add_argument(
        "--baseline-variant", choices=sorted(BASELINE_VARIANTS), default=None,
        help="C1 same-weekday baseline formulation",
    )
    parser.add_argument(
        "--trend-features", choices=["on", "off"], default=None,
        help="Enable or disable the C1 drift/trend feature group",
    )
    parser.add_argument(
        "--c2-config", default=None,
        help=("Recommendation JSON written by ml/run_c2_screening.py. "
              "An explicit --c2-feature-groups value overrides it."),
    )
    parser.add_argument(
        "--c2-feature-groups", default=None,
        help=("Comma-separated C2 groups, 'all', or 'none'. Available: "
              + ",".join(C2_FEATURE_GROUPS)),
    )
    parser.add_argument(
        "--c34-config", default=None,
        help=("Recommendation JSON written by ml/run_c34_screening.py. "
              "Explicit C3/C4 CLI options override it."),
    )
    parser.add_argument("--nn-loss", choices=NN_LOSSES, default=None)
    parser.add_argument("--nn-target-mode", choices=NN_TARGET_MODES, default=None)
    parser.add_argument("--nn-combined-mse-weight", type=float, default=None)
    parser.add_argument("--tree-target-mode", choices=TREE_TARGET_MODES, default=None)
    parser.add_argument("--xgboost-target-mode", choices=TREE_TARGET_MODES, default=None)
    parser.add_argument("--lightgbm-target-mode", choices=TREE_TARGET_MODES, default=None)
    parser.add_argument(
        "--channel-history-features", choices=["on", "off"], default=None,
        help="Enable leakage-safe historical app/web composition features",
    )
    parser.add_argument("--channel-aux-weight", type=float, default=None)
    parser.add_argument("--channel-share-smoothing", type=float, default=None)
    args = parser.parse_args(argv)
    if args.nn_batch_size != "auto":
        try:
            parsed_batch_size = int(args.nn_batch_size)
        except ValueError as exc:
            parser.error("--nn-batch-size must be 'auto' or a positive integer")
        if parsed_batch_size < 2:
            parser.error("--nn-batch-size must be at least 2")
        args.nn_batch_size = str(parsed_batch_size)
    if args.training_window_days not in (None, "all"):
        try:
            if int(args.training_window_days) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--training-window-days must be 'all' or a positive integer")
    if args.recency_half_life_days not in (None, "none"):
        try:
            if float(args.recency_half_life_days) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--recency-half-life-days must be 'none' or positive")
    if args.nn_combined_mse_weight is not None and not (
        0.0 <= args.nn_combined_mse_weight <= 1.0
    ):
        parser.error("--nn-combined-mse-weight must be between 0 and 1")
    if args.channel_aux_weight is not None and args.channel_aux_weight < 0.0:
        parser.error("--channel-aux-weight must be nonnegative")
    if args.channel_share_smoothing is not None and args.channel_share_smoothing < 0.0:
        parser.error("--channel-share-smoothing must be nonnegative")
    return RuntimeOptions(
        forecast_strategy=ForecastStrategy(args.forecast_strategy),
        primary_strategy=PrimaryStrategy(args.primary_strategy),
        submission_model=SubmissionModel(args.submission_model),
        selection_metric=args.selection_metric,
        selection_protocol=args.selection_protocol,
        resume=args.resume,
        reset_checkpoints=args.reset_checkpoints,
        checkpoint_dir=args.checkpoint_dir,
        nn_batch_size=args.nn_batch_size,
        nn_lr_scaling=args.nn_lr_scaling,
        nn_training_backend=args.nn_training_backend,
        nn_benchmark_file=args.nn_benchmark_file,
        c1_config=args.c1_config,
        training_window_days=args.training_window_days,
        recency_half_life_days=args.recency_half_life_days,
        baseline_variant=args.baseline_variant,
        trend_features=args.trend_features,
        c2_config=args.c2_config,
        c2_feature_groups=args.c2_feature_groups,
        c34_config=args.c34_config,
        nn_loss=args.nn_loss,
        nn_target_mode=args.nn_target_mode,
        nn_combined_mse_weight=args.nn_combined_mse_weight,
        tree_target_mode=args.tree_target_mode,
        xgboost_target_mode=args.xgboost_target_mode,
        lightgbm_target_mode=args.lightgbm_target_mode,
        channel_history_features=args.channel_history_features,
        channel_aux_weight=args.channel_aux_weight,
        channel_share_smoothing=args.channel_share_smoothing,
    )


def _parse_optional_days(value, *, none_token: str, cast):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == none_token:
        return None
    parsed = cast(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive value or {none_token!r}, got {value!r}")
    return parsed


def configure_c1_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Apply a C1 recommendation plus explicit CLI overrides."""
    source = "C0 defaults"
    recommendation = {}
    if options.c1_config is not None:
        with open(options.c1_config, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid C1 recommendation in {options.c1_config}")
        candidate = payload.get("recommendation", payload)
        if isinstance(candidate, dict) and "config" in candidate:
            candidate = candidate["config"]
        elif "config" in payload:
            candidate = payload["config"]
        recommendation = candidate
        if not isinstance(recommendation, dict):
            raise ValueError(f"Invalid C1 recommendation in {options.c1_config}")
        source = options.c1_config

    def value(name, explicit, default):
        if explicit is not None:
            return explicit, "CLI override"
        if name in recommendation:
            return recommendation[name], source
        return default, "C0 default"

    window_raw, window_source = value(
        "training_window_days", options.training_window_days, cfg.training_window_days
    )
    half_life_raw, half_life_source = value(
        "recency_half_life_days", options.recency_half_life_days,
        cfg.recency_half_life_days,
    )
    baseline, baseline_source = value(
        "baseline_variant", options.baseline_variant, cfg.baseline_variant
    )
    trend_raw, trend_source = value(
        "enable_trend_features", options.trend_features, cfg.enable_trend_features
    )

    cfg.training_window_days = _parse_optional_days(
        window_raw, none_token="all", cast=int
    )
    cfg.recency_half_life_days = _parse_optional_days(
        half_life_raw, none_token="none", cast=float
    )
    if baseline not in BASELINE_VARIANTS:
        raise ValueError(
            f"Unknown baseline_variant={baseline!r}; expected {sorted(BASELINE_VARIANTS)}"
        )
    cfg.baseline_variant = str(baseline)
    if isinstance(trend_raw, str):
        if trend_raw not in {"on", "off"}:
            raise ValueError("trend_features must be 'on' or 'off'")
        cfg.enable_trend_features = trend_raw == "on"
    else:
        cfg.enable_trend_features = bool(trend_raw)

    return {
        "training_window_days": cfg.training_window_days,
        "recency_half_life_days": cfg.recency_half_life_days,
        "baseline_variant": cfg.baseline_variant,
        "enable_trend_features": cfg.enable_trend_features,
        "sources": {
            "training_window_days": window_source,
            "recency_half_life_days": half_life_source,
            "baseline_variant": baseline_source,
            "enable_trend_features": trend_source,
        },
        "config_file": options.c1_config,
    }


def configure_c2_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Apply C2 semantic feature-group recommendations and CLI overrides."""
    recommendation = {}
    source = "C1 baseline"
    if options.c2_config is not None:
        with open(options.c2_config, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid C2 recommendation in {options.c2_config}")
        candidate = payload.get("recommendation", payload)
        if isinstance(candidate, dict) and "config" in candidate:
            candidate = candidate["config"]
        elif "config" in payload:
            candidate = payload["config"]
        recommendation = candidate if isinstance(candidate, dict) else {}
        source = options.c2_config

    if options.c2_feature_groups is not None:
        raw_groups = options.c2_feature_groups
        group_source = "CLI override"
    elif "c2_feature_groups" in recommendation:
        raw_groups = recommendation["c2_feature_groups"]
        group_source = source
    else:
        raw_groups = cfg.c2_feature_groups
        group_source = "C1 baseline"

    cfg.c2_feature_groups = normalize_c2_feature_groups(raw_groups)
    return {
        "c2_feature_groups": list(cfg.c2_feature_groups),
        "source": group_source,
        "config_file": options.c2_config,
    }


def configure_c34_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Apply C3 objective and C4 multitask recommendations plus overrides."""
    recommendation = {}
    source = "C2 defaults"
    if options.c34_config is not None:
        with open(options.c34_config, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid C3/C4 recommendation in {options.c34_config}")
        candidate = payload.get("recommendation", payload)
        if isinstance(candidate, dict) and "config" in candidate:
            candidate = candidate["config"]
        recommendation = candidate if isinstance(candidate, dict) else {}
        source = options.c34_config

    def choose(name, explicit, default):
        if explicit is not None:
            return explicit, "CLI override"
        if name in recommendation:
            return recommendation[name], source
        return default, "C2 default"

    values = {}
    values["nn_loss"], loss_source = choose(
        "nn_loss", options.nn_loss, cfg.nn_loss
    )
    values["nn_target_mode"], target_source = choose(
        "nn_target_mode", options.nn_target_mode, cfg.nn_target_mode
    )
    values["nn_combined_mse_weight"], combined_source = choose(
        "nn_combined_mse_weight",
        options.nn_combined_mse_weight,
        cfg.nn_combined_mse_weight,
    )
    values["tree_target_mode"], tree_source = choose(
        "tree_target_mode", options.tree_target_mode, cfg.tree_target_mode
    )
    values["xgboost_target_mode"], xgb_tree_source = choose(
        "xgboost_target_mode", options.xgboost_target_mode,
        cfg.xgboost_target_mode,
    )
    values["lightgbm_target_mode"], lgb_tree_source = choose(
        "lightgbm_target_mode", options.lightgbm_target_mode,
        cfg.lightgbm_target_mode,
    )
    values["enable_channel_history_features"], channel_history_source = choose(
        "enable_channel_history_features",
        (
            options.channel_history_features == "on"
            if options.channel_history_features is not None else None
        ),
        cfg.enable_channel_history_features,
    )
    values["channel_aux_weight"], aux_source = choose(
        "channel_aux_weight", options.channel_aux_weight, cfg.channel_aux_weight
    )
    values["channel_share_smoothing"], smoothing_source = choose(
        "channel_share_smoothing",
        options.channel_share_smoothing,
        cfg.channel_share_smoothing,
    )

    if values["nn_loss"] not in NN_LOSSES:
        raise ValueError(f"Unknown nn_loss={values['nn_loss']!r}")
    if values["nn_target_mode"] not in NN_TARGET_MODES:
        raise ValueError(f"Unknown nn_target_mode={values['nn_target_mode']!r}")
    if values["tree_target_mode"] not in TREE_TARGET_MODES:
        raise ValueError(f"Unknown tree_target_mode={values['tree_target_mode']!r}")
    for name in ("xgboost_target_mode", "lightgbm_target_mode"):
        if values[name] is not None and values[name] not in TREE_TARGET_MODES:
            raise ValueError(f"Unknown {name}={values[name]!r}")
    combined_weight = float(values["nn_combined_mse_weight"])
    if not 0.0 <= combined_weight <= 1.0:
        raise ValueError("nn_combined_mse_weight must be between 0 and 1")
    channel_history = values["enable_channel_history_features"]
    if not isinstance(channel_history, (bool, np.bool_)):
        raise ValueError("enable_channel_history_features must be boolean")
    aux_weight = float(values["channel_aux_weight"])
    smoothing = float(values["channel_share_smoothing"])
    if not np.isfinite(aux_weight) or aux_weight < 0.0:
        raise ValueError("channel_aux_weight must be finite and nonnegative")
    if not np.isfinite(smoothing) or smoothing < 0.0:
        raise ValueError("channel_share_smoothing must be finite and nonnegative")

    cfg.nn_loss = str(values["nn_loss"])
    cfg.nn_target_mode = str(values["nn_target_mode"])
    cfg.nn_combined_mse_weight = combined_weight
    cfg.tree_target_mode = str(values["tree_target_mode"])
    cfg.xgboost_target_mode = (
        None if values["xgboost_target_mode"] is None
        else str(values["xgboost_target_mode"])
    )
    cfg.lightgbm_target_mode = (
        None if values["lightgbm_target_mode"] is None
        else str(values["lightgbm_target_mode"])
    )
    cfg.enable_channel_history_features = bool(channel_history)
    cfg.channel_aux_weight = aux_weight
    cfg.channel_share_smoothing = smoothing
    return {
        "nn_loss": cfg.nn_loss,
        "nn_target_mode": cfg.nn_target_mode,
        "nn_combined_mse_weight": cfg.nn_combined_mse_weight,
        "tree_target_mode": cfg.tree_target_mode,
        "xgboost_target_mode": cfg.xgboost_target_mode,
        "lightgbm_target_mode": cfg.lightgbm_target_mode,
        "enable_channel_history_features": cfg.enable_channel_history_features,
        "channel_aux_weight": cfg.channel_aux_weight,
        "channel_share_smoothing": cfg.channel_share_smoothing,
        "sources": {
            "nn_loss": loss_source,
            "nn_target_mode": target_source,
            "nn_combined_mse_weight": combined_source,
            "tree_target_mode": tree_source,
            "xgboost_target_mode": xgb_tree_source,
            "lightgbm_target_mode": lgb_tree_source,
            "enable_channel_history_features": channel_history_source,
            "channel_aux_weight": aux_source,
            "channel_share_smoothing": smoothing_source,
        },
        "config_file": options.c34_config,
    }


def configure_nn_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Resolve batch/LR/backend without guessing away model quality.

    Auto mode consumes the recommendation produced by
    ``ml/benchmark_nn_batch_size.py`` only when it was measured on the same
    accelerator type.  Without that artifact the historical 512/fixed policy
    is preserved.
    """
    recommendation = None
    if os.path.exists(options.nn_benchmark_file):
        try:
            with open(options.nn_benchmark_file, encoding="utf-8") as f:
                payload = json.load(f)
            candidate = payload.get("recommendation") or {}
            measured_device = payload.get("environment", {}).get("device")
            measured_signature = payload.get("model_signature")
            current_signature = nn_performance_signature(cfg)
            # JSON converts tuples to lists, so compare through a JSON-normalised
            # representation rather than Python container types.
            signature_matches = (
                json.dumps(measured_signature, sort_keys=True)
                == json.dumps(current_signature, sort_keys=True)
            )
            if (
                payload.get("schema_version") == "nn-batch-v1"
                and measured_device == DEVICE.type
                and signature_matches
                and candidate.get("batch_size")
            ):
                recommendation = candidate
        except (OSError, ValueError, TypeError) as exc:
            print(f"Ignoring unreadable NN benchmark {options.nn_benchmark_file}: {exc}")

    if options.nn_batch_size == "auto":
        if recommendation is not None:
            batch_size = int(recommendation["batch_size"])
            batch_source = options.nn_benchmark_file
        else:
            batch_size = int(cfg.reference_batch_size)
            batch_source = "historical safe fallback"
    else:
        batch_size = int(options.nn_batch_size)
        batch_source = "CLI override"

    if options.nn_lr_scaling == "auto":
        if recommendation is not None and options.nn_batch_size == "auto":
            lr_scaling = str(recommendation.get("lr_scaling", "sqrt"))
        elif batch_size == cfg.reference_batch_size:
            lr_scaling = "fixed"
        else:
            lr_scaling = "sqrt"
    else:
        lr_scaling = options.nn_lr_scaling

    cfg.batch_size = batch_size
    cfg.nn_lr_scaling = lr_scaling
    cfg.nn_training_backend = options.nn_training_backend
    return {
        "batch_size": batch_size,
        "batch_source": batch_source,
        "lr_scaling": lr_scaling,
        "effective_learning_rate": effective_learning_rate(cfg),
        "training_backend": resolve_training_backend(cfg),
        "benchmark_file": options.nn_benchmark_file,
    }


TREE_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_worker.py")

CHECKPOINT_SCHEMA_VERSION = "c34-objective-channel-history-v1"


def _fold_checkpoint_path(
    checkpoint_dir: str | None,
    strategy: str,
    origin_type: str,
    origin: pd.Timestamp,
) -> str | None:
    if not checkpoint_dir:
        return None
    filename = f"{pd.Timestamp(origin).date().isoformat()}.pkl"
    return os.path.join(checkpoint_dir, strategy, origin_type, filename)


def _fold_checkpoint_signature(
    cfg: Config, strategy: str, origin_type: str, origin: pd.Timestamp
) -> dict:
    cfg_signature = asdict(cfg)
    # The execution backend changes throughput, not the estimator definition.
    cfg_signature.pop("nn_training_backend", None)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "strategy": strategy,
        "origin_type": origin_type,
        "origin": pd.Timestamp(origin).isoformat(),
        "cfg": cfg_signature,
    }


def _checkpoint_signature_compatible(actual: dict, expected: dict) -> bool:
    """Require an exact semantic/training-policy checkpoint signature."""
    return actual == expected


def _load_fold_checkpoint(
    checkpoint_dir: str | None,
    strategy: str,
    origin_type: str,
    origin: pd.Timestamp,
    cfg: Config,
) -> dict | None:
    path = _fold_checkpoint_path(checkpoint_dir, strategy, origin_type, origin)
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"    [checkpoint] ignoring unreadable {path}: {exc}")
        return None
    expected = _fold_checkpoint_signature(cfg, strategy, origin_type, origin)
    if not _checkpoint_signature_compatible(payload.get("signature") or {}, expected):
        print(f"    [checkpoint] ignoring stale checkpoint {path}")
        return None
    frame = payload.get("oof")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        print(f"    [checkpoint] ignoring invalid checkpoint {path}")
        return None
    return payload


def _save_fold_checkpoint(
    checkpoint_dir: str | None,
    strategy: str,
    origin_type: str,
    origin: pd.Timestamp,
    cfg: Config,
    oof: pd.DataFrame,
    timing: dict,
) -> None:
    path = _fold_checkpoint_path(checkpoint_dir, strategy, origin_type, origin)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "signature": _fold_checkpoint_signature(cfg, strategy, origin_type, origin),
        "oof": oof,
        "timing": timing,
    }
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


# Broader, seasonally-scattered origins used to make modeling/feature
# decisions (spring/summer lulls, several Januaries, Black Friday windows,
# pre/post-Christmas, a Valentine's-adjacent week -- relevant for a
# cosmetics retailer). Deliberately disjoint from `recent_benchmark_origins`
# below: these are for iteration, the benchmark is for a pseudo-test check,
# and mixing the two would let repeated tuning quietly overfit to the
# benchmark the same way a single reused test set would.
DEVELOPMENT_ORIGINS = pd.to_datetime([
    "2022-02-01", "2022-06-15", "2022-11-20",
    "2023-01-10", "2023-07-01", "2023-11-24", "2023-12-18",
    "2024-02-14", "2024-06-20", "2024-11-29", "2024-12-20",
    "2025-02-10",
])



# Frozen source-level audit origins, disjoint from the current development
# windows and normal recent-benchmark protocol. They are intentionally not
# executed by the normal pipeline or Tier-C screening commands. Run them only
# once after C1-C5 decisions are frozen.
FINAL_AUDIT_ORIGINS = pd.to_datetime([
    "2024-01-17",  # winter/test-like regular week
    "2024-05-15",  # ordinary week
    "2024-11-14",  # pre-Black-Friday stress week
])

VALIDATION_STRATUM_WEIGHTS = {
    "winter_test_like": 0.60,
    "regular": 0.25,
    "holiday_event": 0.15,
}


def classify_validation_stratum(
    origin: pd.Timestamp,
    horizon: int = 7,
) -> str:
    """Classify a forecast window for test-aligned reporting.

    January/February regular weeks provide a larger but still test-like
    winter sample.  Late-November/December windows are event stress tests;
    all remaining development windows are ordinary regular periods.
    """
    target_dates = pd.date_range(
        pd.Timestamp(origin) + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    if target_dates.month.isin([1, 2]).all():
        return "winter_test_like"
    if ((target_dates.month == 11) & (target_dates.day >= 20)).any() or (
        target_dates.month == 12
    ).any():
        return "holiday_event"
    return "regular"


def recent_benchmark_origins(hist_df: pd.DataFrame, cfg: Config = CFG) -> pd.DatetimeIndex:
    """Last `cfg.n_cv_folds` non-overlapping `cfg.horizon`-day origins ending
    at the most recent training data -- the closest pseudo-test periods to
    the actual forecast. Meant as a final model-selection check (a benchmark
    of recent performance), not something to repeatedly re-tune against."""
    max_date = hist_df["DateKey"].max()
    return pd.DatetimeIndex([max_date - pd.Timedelta(days=(i + 1) * cfg.horizon) for i in range(cfg.n_cv_folds)])


# ---------------------------------------------------------------------------
# XGBoost / LightGBM baselines, run out-of-process (see tree_worker.py)
# ---------------------------------------------------------------------------
def run_structured_models(
    train_panel: pd.DataFrame,
    cfg: Config = CFG,
    models: tuple | None = None,
    *,
    strategy: str = "direct",
    eval_panel: pd.DataFrame | None = None,
    history_raw: pd.DataFrame | None = None,
    future_covariates: pd.DataFrame | None = None,
    price_ref: pd.Series | None = None,
    first_seen: pd.Series | None = None,
    first_available: pd.Series | None = None,
) -> dict:
    """Train and predict structured models in the native-library worker."""
    if models is None:
        models = (
            ("XGBoost", "LightGBM")
            if strategy == "recursive"
            else ("XGBoost", "LightGBM", "DynamicRidge")
        )
    job = {
        "cfg": asdict(cfg),
        "strategy": strategy,
        "train_panel": train_panel,
        "models": list(models),
    }
    if strategy == "direct":
        if eval_panel is None:
            raise ValueError("eval_panel is required for direct structured prediction")
        job["eval_panel"] = eval_panel
    elif strategy == "recursive":
        required = (history_raw, future_covariates, price_ref, first_seen)
        if any(value is None for value in required):
            raise ValueError("recursive structured prediction requires history, future covariates and references")
        if first_available is None:
            _, first_available = product_reference_dates(history_raw)
        job.update({
            "history_raw": history_raw,
            "future_covariates": sanitize_future_covariates(future_covariates),
            "price_ref": price_ref,
            "first_seen": first_seen,
            "first_available": first_available,
        })
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    with tempfile.TemporaryDirectory() as tmp:
        job_path = os.path.join(tmp, "job.pkl")
        out_path = os.path.join(tmp, "out.pkl")
        with open(job_path, "wb") as f:
            pickle.dump(job, f)
        try:
            subprocess.run(
                [sys.executable, TREE_WORKER_PATH, job_path, out_path],
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Structured-model worker timed out after 180 seconds for {models}.\n"
                f"Stdout: {exc.stdout}\nStderr: {exc.stderr}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Structured-model worker failed (exit {exc.returncode}) for {models}.\n"
                f"Stdout: {exc.stdout}\nStderr: {exc.stderr}"
            ) from exc
        with open(out_path, "rb") as f:
            return pickle.load(f)


def run_tree_baselines(train_panel: pd.DataFrame, eval_panel: pd.DataFrame, cfg: Config = CFG,
                       models: tuple = ("XGBoost", "LightGBM", "DynamicRidge")) -> dict:
    """Backward-compatible alias for the direct structured-model worker."""
    results = run_structured_models(
        train_panel, cfg, models, strategy="direct", eval_panel=eval_panel
    )
    return {name: np.asarray(preds, dtype=np.float32) for name, preds in results.items()}


def _reindex_predictions(panel: pd.DataFrame, preds: np.ndarray, date_col: str,
                          keys: pd.DataFrame) -> np.ndarray:
    """Realign `preds` (computed in `panel`'s own row order) to exactly
    `keys`'s (ProductId, DateKey) row order, via an explicit key-based
    merge -- two independently-constructed frames should never be assumed
    to share a row order."""
    lookup = panel[["ProductId", date_col]].rename(columns={date_col: "DateKey"}).copy()
    lookup["_pred"] = preds
    aligned = keys[["ProductId", "DateKey"]].merge(lookup, on=["ProductId", "DateKey"], how="left")
    return aligned["_pred"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------
def run_walk_forward_cv_direct(
    hist_df: pd.DataFrame, origins, origin_type: str, cfg: Config = CFG,
    timings: list[dict] | None = None, *, checkpoint_dir: str | None = None,
    resume: bool = False,
    run_neural: bool = True,
    structured_models: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Evaluate at each `origin` date (the last training day): trains only
    on data up to and including `origin` (no leakage) and predicts all
    `cfg.horizon` days directly from the multi-horizon panel (see
    `framework.build_direct_panel`) -- no recursion, since every horizon's
    features are already lookups into observed data, never a value that
    would first need to be predicted.

    Trains the SAME `cfg.seeds`-sized NN ensemble as `run_final_forecast`
    -- CV must score the actual estimator being submitted, not a cheaper
    single-seed stand-in. `cv_epochs` vs `final_epochs` remains a
    deliberate, disclosed compute/accuracy trade-off (cheaper proxy
    training while iterating; the one-time final artifact trains longer)
    -- unlike the seed count, that's not a hidden inconsistency, since
    it's applied identically across every model/fold.

    Returns row-level out-of-fold predictions -- one row per (origin,
    product, date), with per-seed NN columns alongside the ensemble and
    every baseline -- rather than only aggregated metrics, so later
    diagnostics (per-horizon, per-product, paired comparisons, ensemble
    weight fitting) don't require rerunning the CV.

    If `timings` is given, one {origin_type, origin, nn_seconds,
    tree_seconds, fold_seconds} dict is appended per fold -- lets `main()`
    build an `outputs/timings.json` breakdown without this function owning
    the file write itself.
    """
    horizons = range(1, cfg.horizon + 1)
    fold_frames = []

    for origin in origins:
        fold_start = time.perf_counter()
        eval_start = origin + pd.Timedelta(days=1)
        eval_end = origin + pd.Timedelta(days=cfg.horizon)
        fold_train_raw = hist_df[hist_df["DateKey"] <= origin].copy()
        fold_eval_raw = hist_df[(hist_df["DateKey"] >= eval_start) & (hist_df["DateKey"] <= eval_end)].copy()
        if fold_train_raw.empty or fold_eval_raw.empty:
            continue

        if resume:
            cached = _load_fold_checkpoint(
                checkpoint_dir, "direct", origin_type, origin, cfg
            )
            if cached is not None:
                print(
                    f"  [{origin_type}] origin {origin.date()}: "
                    "loaded completed direct fold checkpoint"
                )
                fold_frames.append(cached["oof"])
                if timings is not None and cached.get("timing"):
                    timings.append(cached["timing"])
                continue

        print(f"  [{origin_type}] origin {origin.date()}: eval {eval_start.date()}..{eval_end.date()}")

        price_ref = fold_train_raw.groupby("ProductId")["PriceLocalVat"].median()
        first_seen, first_available = product_reference_dates(fold_train_raw)

        fold_train_feat = prepare_features(
            fold_train_raw, price_ref, first_seen, first_available, cfg
        )
        fold_train_feat = add_train_lags(
            fold_train_feat, cfg.lag_windows,
            baseline_variant=cfg.baseline_variant,
        )
        fold_eval_feat = prepare_features(
            fold_eval_raw, price_ref, first_seen, first_available, cfg
        ).reset_index(drop=True)

        panel = build_direct_panel(fold_train_feat, horizons, cfg=cfg, future_covariates=fold_eval_feat)
        # Leakage-safe training slice: a training row's own target must
        # already be observable as of `origin` -- an origin close to the
        # fold's own cutoff combined with a large horizon would otherwise
        # land on a target date this fold isn't allowed to have seen yet.
        train_panel = select_trainable_panel_rows(
            panel, cutoff=origin, available_only=True, cfg=cfg
        )
        eval_panel = panel[panel["OriginDateKey"] == origin].reset_index(drop=True)

        seed_preds: dict[int, np.ndarray] = {}
        ensemble_output = None
        nn_seconds = 0.0
        if run_neural:
            scaler = make_numeric_preprocessor()
            tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
            y_target = neural_training_target(train_panel, cfg)
            nn_start = time.perf_counter()
            seed_models = [
                train_model(
                    tensors, y_target, cfg, epochs=cfg.cv_epochs, seed=seed
                )
                for seed in cfg.seeds
            ]
            nn_seconds = time.perf_counter() - nn_start
            seed_preds = {
                seed: predict_direct([model], scaler, eval_panel, cfg)
                for seed, model in zip(cfg.seeds, seed_models)
            }
            ensemble_output = predict_direct(
                seed_models, scaler, eval_panel, cfg, return_diagnostics=True
            )

        requested_structured = (
            ("XGBoost", "LightGBM", "DynamicRidge")
            if structured_models is None else tuple(structured_models)
        )
        tree_start = time.perf_counter()
        tree_preds = (
            run_tree_baselines(
                train_panel, eval_panel, cfg, models=requested_structured
            )
            if requested_structured else {}
        )
        tree_seconds = time.perf_counter() - tree_start

        seasonal_pred = seasonal_naive_predict(fold_eval_feat, fold_train_raw, lag_days=cfg.horizon)
        ma_pred = moving_average_predict(fold_eval_feat, fold_train_raw, window=28)
        baseline_pred = compute_baseline(
            fold_eval_feat, fold_train_raw, cfg.baseline_variant
        )

        # Predictions are attached straight onto `eval_panel` (so they're
        # trivially self-consistent with its own ProductId/horizon/target
        # date, whatever internal row order it happens to be in); naive
        # baselines + the real actual/availability come from
        # `fold_eval_feat` and are joined in by explicit (ProductId,
        # DateKey) key rather than assumed row order.
        naive_df = fold_eval_feat[["ProductId", "DateKey", "Quantity", "ProductAvailable"]].copy()
        naive_df["baseline"] = baseline_pred
        naive_df["pred_SeasonalNaive"] = seasonal_pred
        naive_df["pred_MovingAvg28"] = ma_pred

        fold_oof = eval_panel[["ProductId", "horizon", "TargetDateKey"]].rename(columns={"TargetDateKey": "DateKey"})
        fold_oof["origin"] = origin
        fold_oof["origin_type"] = origin_type
        fold_oof["strategy"] = "direct"
        fold_oof["validation_stratum"] = classify_validation_stratum(
            origin, cfg.horizon
        )
        model_names: list[str] = []
        if ensemble_output is not None:
            fold_oof["pred_NeuralNet"] = ensemble_output["prediction"]
            model_names.append("NeuralNet")
            if "app_share" in ensemble_output:
                fold_oof["pred_AppShare_NeuralNet"] = ensemble_output["app_share"]
                fold_oof["pred_QuantityApp_NeuralNet"] = ensemble_output["prediction_app"]
                fold_oof["pred_QuantityWeb_NeuralNet"] = ensemble_output["prediction_web"]
                actual_total = pd.to_numeric(
                    eval_panel.get("target", pd.Series(np.nan, index=eval_panel.index)),
                    errors="coerce",
                ).to_numpy(dtype=float)
                actual_app = pd.to_numeric(
                    eval_panel.get("target_app", pd.Series(np.nan, index=eval_panel.index)),
                    errors="coerce",
                ).to_numpy(dtype=float)
                actual_share = np.divide(
                    actual_app,
                    actual_total,
                    out=np.full(len(eval_panel), np.nan, dtype=float),
                    where=np.isfinite(actual_total) & (actual_total > 0),
                )
                fold_oof["actual_AppShare"] = actual_share
        for name, predictions in tree_preds.items():
            fold_oof[f"pred_{name}"] = predictions
            model_names.append(name)
        for name in model_names:
            fold_oof[f"fallback_{name}"] = False
            fold_oof[f"nonfinite_{name}"] = False
            fold_oof[f"catastrophic_{name}"] = False
            fold_oof[f"residual_guard_{name}"] = False
            fold_oof[f"residual_nonfinite_{name}"] = False
            fold_oof[f"residual_raw_min_{name}"] = np.nan
            fold_oof[f"residual_raw_max_{name}"] = np.nan
            fold_oof[f"safety_limit_{name}"] = np.nan
        for seed, predictions in seed_preds.items():
            fold_oof[f"pred_NeuralNet_seed{seed}"] = predictions
        fold_oof = fold_oof.merge(naive_df, on=["ProductId", "DateKey"], how="left")
        fold_oof = fold_oof.rename(columns={"Quantity": "actual"})
        fold_frames.append(fold_oof)

        fold_seconds = time.perf_counter() - fold_start
        timing_record = {
            "strategy": "direct", "origin_type": origin_type,
            "origin": str(origin.date()),
            "nn_seconds": round(nn_seconds, 2),
            "tree_seconds": round(tree_seconds, 2),
            "fold_seconds": round(fold_seconds, 2),
        }
        print(f"    [timing] {origin_type} {origin.date()}: NN {nn_seconds:.1f}s | "
              f"trees {tree_seconds:.1f}s | fold total {fold_seconds:.1f}s")
        if timings is not None:
            timings.append(timing_record)
        _save_fold_checkpoint(
            checkpoint_dir, "direct", origin_type, origin, cfg, fold_oof, timing_record
        )

    return pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()


def _recursive_panel_training_data(
    fold_train_raw: pd.DataFrame,
    price_ref: pd.Series,
    first_seen: pd.Series,
    first_available: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    panel = build_one_step_panel(
        fold_train_raw, price_ref, first_seen, cfg, first_available
    )
    cutoff = fold_train_raw["DateKey"].max()
    return select_trainable_panel_rows(
        panel, cutoff=cutoff, available_only=True, cfg=cfg
    )


def _recursive_nn_predictions(
    train_panel: pd.DataFrame,
    history_raw: pd.DataFrame,
    future_covariates: pd.DataFrame,
    price_ref: pd.Series,
    first_seen: pd.Series,
    first_available: pd.Series,
    cfg: Config,
    epochs: int,
):
    scaler = make_numeric_preprocessor()
    tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
    y_target = neural_training_target(train_panel, cfg)
    seed_models = [train_model(tensors, y_target, cfg, epochs=epochs, seed=seed) for seed in cfg.seeds]

    seed_paths = {}
    # Diagnostics: each seed gets its own path. The deployed ensemble path below
    # feeds back the natural-scale ensemble mean, as required.
    for seed, model in zip(cfg.seeds, seed_models):
        seed_paths[seed] = forecast_recursive(
            history_raw, future_covariates,
            lambda panel, model=model: predict_direct(
                [model], scaler, panel, cfg,
                recursive_guard=True, return_diagnostics=True,
            ),
            price_ref, first_seen, cfg, first_available,
        )
    ensemble_path = forecast_recursive(
        history_raw, future_covariates,
        lambda panel: predict_direct(
            seed_models, scaler, panel, cfg,
            recursive_guard=True, return_diagnostics=True,
        ),
        price_ref, first_seen, cfg, first_available,
    )
    return ensemble_path, seed_paths, scaler, seed_models


def run_walk_forward_cv_recursive(
    hist_df: pd.DataFrame, origins, origin_type: str, cfg: Config = CFG,
    timings: list[dict] | None = None, *, checkpoint_dir: str | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """One-step training plus genuine recursive seven-day inference."""
    fold_frames = []
    for origin in origins:
        fold_start = time.perf_counter()
        eval_start = origin + pd.Timedelta(days=1)
        eval_end = origin + pd.Timedelta(days=cfg.horizon)
        fold_train_raw = hist_df[hist_df["DateKey"].le(origin)].copy()
        fold_eval_raw = hist_df[hist_df["DateKey"].between(eval_start, eval_end)].copy()
        if fold_train_raw.empty or fold_eval_raw.empty:
            continue
        if resume:
            cached = _load_fold_checkpoint(
                checkpoint_dir, "recursive", origin_type, origin, cfg
            )
            if cached is not None:
                print(
                    f"  [{origin_type}/recursive] origin {origin.date()}: "
                    "loaded completed fold checkpoint"
                )
                fold_frames.append(cached["oof"])
                if timings is not None and cached.get("timing"):
                    timings.append(cached["timing"])
                continue
        print(f"  [{origin_type}/recursive] origin {origin.date()}: eval {eval_start.date()}..{eval_end.date()}")
        price_ref = fold_train_raw.groupby("ProductId")["PriceLocalVat"].median()
        first_seen, first_available = product_reference_dates(fold_train_raw)
        train_panel = _recursive_panel_training_data(
            fold_train_raw, price_ref, first_seen, first_available, cfg
        )
        future_covariates = sanitize_future_covariates(fold_eval_raw)

        nn_start = time.perf_counter()
        ensemble_path, seed_paths, _, _ = _recursive_nn_predictions(
            train_panel, fold_train_raw, future_covariates, price_ref,
            first_seen, first_available, cfg, cfg.cv_epochs
        )
        nn_seconds = time.perf_counter() - nn_start

        tree_start = time.perf_counter()
        structured = run_structured_models(
            train_panel, cfg, models=("XGBoost", "LightGBM"),
            strategy="recursive", history_raw=fold_train_raw,
            future_covariates=future_covariates, price_ref=price_ref,
            first_seen=first_seen, first_available=first_available,
        )
        tree_seconds = time.perf_counter() - tree_start

        eval_feat = prepare_features(
            fold_eval_raw, price_ref, first_seen, first_available, cfg
        ).reset_index(drop=True)
        naive_df = fold_eval_raw[["ProductId", "DateKey", "Quantity", "ProductAvailable"]].copy()
        naive_df["baseline"] = compute_baseline(
            eval_feat, fold_train_raw, cfg.baseline_variant
        )
        naive_df["pred_SeasonalNaive"] = seasonal_naive_predict(eval_feat, fold_train_raw, lag_days=cfg.horizon)
        naive_df["pred_MovingAvg28"] = moving_average_predict(eval_feat, fold_train_raw, window=28)

        renamed_path = ensemble_path.rename(columns={
            "TargetDateKey": "DateKey",
            "forecast_horizon": "horizon",
            "prediction": "pred_NeuralNet",
            "app_share": "pred_AppShare_NeuralNet",
            "prediction_app": "pred_QuantityApp_NeuralNet",
            "prediction_web": "pred_QuantityWeb_NeuralNet",
            "fallback_used": "fallback_NeuralNet",
            "nonfinite_raw": "nonfinite_NeuralNet",
            "catastrophic_guard": "catastrophic_NeuralNet",
            "residual_guard": "residual_guard_NeuralNet",
            "residual_nonfinite": "residual_nonfinite_NeuralNet",
            "residual_raw_min": "residual_raw_min_NeuralNet",
            "residual_raw_max": "residual_raw_max_NeuralNet",
            "safety_limit": "safety_limit_NeuralNet",
        })
        path_columns = [
            "ProductId", "DateKey", "horizon", "pred_NeuralNet",
            "fallback_NeuralNet", "nonfinite_NeuralNet",
            "catastrophic_NeuralNet", "residual_guard_NeuralNet",
            "residual_nonfinite_NeuralNet", "residual_raw_min_NeuralNet",
            "residual_raw_max_NeuralNet", "safety_limit_NeuralNet",
        ]
        path_columns += [
            column for column in (
                "pred_AppShare_NeuralNet",
                "pred_QuantityApp_NeuralNet",
                "pred_QuantityWeb_NeuralNet",
            ) if column in renamed_path.columns
        ]
        fold_oof = renamed_path[path_columns].copy()
        fold_oof["origin"] = origin
        fold_oof["origin_type"] = origin_type
        fold_oof["strategy"] = "recursive"
        fold_oof["validation_stratum"] = classify_validation_stratum(
            origin, cfg.horizon
        )
        for seed, path in seed_paths.items():
            seed_col = path[["ProductId", "TargetDateKey", "prediction"]].rename(
                columns={"TargetDateKey": "DateKey", "prediction": f"pred_NeuralNet_seed{seed}"}
            )
            fold_oof = fold_oof.merge(seed_col, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
        for name, payload in structured.items():
            path = pd.DataFrame(payload)
            diagnostic_columns = [
                "ProductId", "TargetDateKey", "prediction", "fallback_used",
                "nonfinite_raw", "catastrophic_guard", "residual_guard",
                "residual_nonfinite", "residual_raw_min", "residual_raw_max",
                "safety_limit",
            ]
            pred_col = path[diagnostic_columns].rename(columns={
                "TargetDateKey": "DateKey",
                "prediction": f"pred_{name}",
                "fallback_used": f"fallback_{name}",
                "nonfinite_raw": f"nonfinite_{name}",
                "catastrophic_guard": f"catastrophic_{name}",
                "residual_guard": f"residual_guard_{name}",
                "residual_nonfinite": f"residual_nonfinite_{name}",
                "residual_raw_min": f"residual_raw_min_{name}",
                "residual_raw_max": f"residual_raw_max_{name}",
                "safety_limit": f"safety_limit_{name}",
            })
            fold_oof = fold_oof.merge(pred_col, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
        fold_oof = fold_oof.merge(naive_df, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
        fold_oof = fold_oof.rename(columns={"Quantity": "actual"})
        if "pred_AppShare_NeuralNet" in fold_oof.columns:
            channel_actual = fold_eval_raw[[
                "ProductId", "DateKey", "QuantityApp", "QuantityWeb"
            ]].copy()
            total = (
                channel_actual["QuantityApp"].to_numpy(dtype=float)
                + channel_actual["QuantityWeb"].to_numpy(dtype=float)
            )
            channel_actual["actual_AppShare"] = np.divide(
                channel_actual["QuantityApp"].to_numpy(dtype=float),
                total,
                out=np.full(len(channel_actual), np.nan, dtype=float),
                where=np.isfinite(total) & (total > 0.0),
            )
            fold_oof = fold_oof.merge(
                channel_actual[["ProductId", "DateKey", "actual_AppShare"]],
                on=["ProductId", "DateKey"], how="left", validate="one_to_one",
            )
        fold_frames.append(fold_oof)

        fold_seconds = time.perf_counter() - fold_start
        timing_record = {
            "strategy": "recursive", "origin_type": origin_type,
            "origin": str(origin.date()),
            "nn_seconds": round(nn_seconds, 2),
            "tree_seconds": round(tree_seconds, 2),
            "fold_seconds": round(fold_seconds, 2),
        }
        print(
            f"    [timing] {origin_type}/recursive {origin.date()}: "
            f"NN {nn_seconds:.1f}s | structured {tree_seconds:.1f}s | "
            f"fold total {fold_seconds:.1f}s"
        )
        if timings is not None:
            timings.append(timing_record)
        _save_fold_checkpoint(
            checkpoint_dir, "recursive", origin_type, origin, cfg, fold_oof, timing_record
        )
    return pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()


def run_walk_forward_cv(
    hist_df: pd.DataFrame, origins, origin_type: str, cfg: Config = CFG,
    timings: list[dict] | None = None,
    strategy: ForecastStrategy | str = ForecastStrategy.DIRECT, *,
    checkpoint_dir: str | None = None, resume: bool = False,
) -> pd.DataFrame:
    strategy = ForecastStrategy(strategy)
    if strategy is ForecastStrategy.DIRECT:
        return run_walk_forward_cv_direct(
            hist_df, origins, origin_type, cfg, timings,
            checkpoint_dir=checkpoint_dir, resume=resume,
        )
    if strategy is ForecastStrategy.RECURSIVE:
        return run_walk_forward_cv_recursive(
            hist_df, origins, origin_type, cfg, timings,
            checkpoint_dir=checkpoint_dir, resume=resume,
        )
    raise ValueError("run_walk_forward_cv accepts one concrete strategy, not 'both'")


OOF_MODEL_COLUMNS = {
    "NeuralNet": "pred_NeuralNet",
    "XGBoost": "pred_XGBoost",
    "LightGBM": "pred_LightGBM",
    "DynamicRidge": "pred_DynamicRidge",
    "SeasonalNaive": "pred_SeasonalNaive",
    "MovingAvg28": "pred_MovingAvg28",
}


def summarize_oof(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    """B4: Refactored to support common-population evaluation and detailed metrics.
    Produces combinations of:
      - evaluation_regime: 'realized' (all days) vs 'conditional' (available only)
      - comparison_population: 'common' (same rows for all models) vs 'model_specific'
      - aggregation: 'global' (micro) vs 'mean_fold' (macro)
    """
    pred_columns = pred_columns or OOF_MODEL_COLUMNS
    pred_columns = {
        model: column
        for model, column in pred_columns.items()
        if column in oof.columns
    }
    pred_cols = list(pred_columns.values())
    if not pred_cols:
        return pd.DataFrame()
    
    # Base masks for regimes
    regime_masks = {
        "realized": oof["actual"].notna(),
        "conditional": (
            oof["actual"].notna()
            & oof["ProductAvailable"].fillna(False)
        ),
    }
    
    rows = []
    
    for regime_name, regime_mask in regime_masks.items():
        # Rows where ALL models have finite predictions
        common_mask = regime_mask & oof[pred_cols].apply(np.isfinite).all(axis=1)
        
        populations = ["common", "model_specific"]
        for pop_name in populations:
            for model_name, pred_col in pred_columns.items():
                if pred_col not in oof.columns:
                    continue
                
                # Rows for THIS model and THIS population
                if pop_name == "common":
                    mask = common_mask
                else:
                    mask = regime_mask & np.isfinite(oof[pred_col])
                
                scored_df = oof[mask]
                
                # Diagnostics (always regime-relative)
                n_expected = int(regime_mask.sum())
                n_actual = int((regime_mask & oof["actual"].notna()).sum())
                n_predicted = int((regime_mask & np.isfinite(oof[pred_col])).sum())
                n_scored = int(mask.sum())
                coverage = n_predicted / n_expected if n_expected > 0 else 0.0
                
                def add_row(df, agg_name):
                    if df.empty:
                        metrics = {k: np.nan for k in ["MAE", "RMSE", "WAPE", "sMAPE", "RMSLE", "Bias", "BiasRatio", "MAPE"]}
                        n_folds = 0
                    else:
                        if agg_name == "global":
                            metrics = compute_metrics(df["actual"], df[pred_col])
                            n_folds = df["origin"].nunique()
                        else:  # mean_fold
                            fold_metrics = [compute_metrics(g["actual"], g[pred_col]) for _, g in df.groupby("origin")]
                            metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
                            n_folds = len(fold_metrics)
                    
                    rows.append({
                        "model": model_name,
                        "evaluation_regime": regime_name,
                        "comparison_population": pop_name,
                        "aggregation": agg_name,
                        "n_folds": n_folds,
                        "n_expected": n_expected,
                        "n_actual": n_actual,
                        "n_predicted": n_predicted,
                        "n_scored": n_scored,
                        "coverage": coverage,
                        **metrics
                    })
                
                add_row(scored_df, "global")
                add_row(scored_df, "mean_fold")
                
    return pd.DataFrame(rows)


def summarize_oof_by_strategy(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    if "strategy" not in oof.columns:
        out = summarize_oof(oof, pred_columns)
        out["strategy"] = "direct"
        return out
    frames = []
    for strategy, group in oof.groupby("strategy", sort=False):
        strategy_columns = prediction_columns_for_strategy(
            pred_columns or OOF_MODEL_COLUMNS, strategy
        )
        summary = summarize_oof(group, strategy_columns)
        summary["strategy"] = strategy
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_strategy_pairs(oof: pd.DataFrame, evaluation_regime: str = "conditional",
                             pred_columns: dict = None) -> pd.DataFrame:
    pred_columns = pred_columns or OOF_MODEL_COLUMNS
    if set(oof.get("strategy", pd.Series(dtype=str)).dropna().unique()) != {"direct", "recursive"}:
        return pd.DataFrame()
    key_cols = ["origin_type", "origin", "ProductId", "DateKey", "horizon"]
    rows = []
    direct = oof[oof["strategy"].eq("direct")]
    recursive = oof[oof["strategy"].eq("recursive")]
    for model, col in pred_columns.items():
        if not (
            model_supports_strategy(model, "direct")
            and model_supports_strategy(model, "recursive")
        ):
            continue
        if col not in direct or col not in recursive:
            continue
        left = direct[key_cols + ["actual", "ProductAvailable", col]].rename(columns={col: "direct_pred"})
        right = recursive[key_cols + [col]].rename(columns={col: "recursive_pred"})
        paired = left.merge(right, on=key_cols, how="inner", validate="one_to_one")
        mask = paired["actual"].notna()
        if evaluation_regime == "conditional":
            mask &= paired["ProductAvailable"].fillna(False)
        mask &= np.isfinite(paired["direct_pred"]) & np.isfinite(paired["recursive_pred"])
        paired = paired[mask]
        if paired.empty:
            continue
        dm = compute_metrics(paired["actual"], paired["direct_pred"])
        rm = compute_metrics(paired["actual"], paired["recursive_pred"])
        for metric in ("WAPE", "MAE", "RMSE", "Bias", "BiasRatio"):
            dv, rv = float(dm[metric]), float(rm[metric])
            lower_is_better = metric not in {"Bias", "BiasRatio"}
            if lower_is_better:
                winner = "direct" if dv < rv else "recursive" if rv < dv else "tie"
            else:
                winner = "direct" if abs(dv) < abs(rv) else "recursive" if abs(rv) < abs(dv) else "tie"
            rows.append({
                "model": model, "evaluation_regime": evaluation_regime,
                "direct_n": len(direct), "recursive_n": len(recursive), "paired_n": len(paired),
                "metric": metric, "direct_value": dv, "recursive_value": rv,
                "absolute_delta": rv - dv,
                "relative_delta": (rv - dv) / abs(dv) if dv != 0 else np.nan,
                "winner": winner,
            })
    return pd.DataFrame(rows)


def summarize_validation_strata(
    oof: pd.DataFrame,
    pred_columns: dict | None = None,
) -> pd.DataFrame:
    """Metric summaries by strategy and data-generating-process stratum."""
    if oof.empty:
        return pd.DataFrame()
    if "validation_stratum" not in oof.columns:
        work = oof.copy()
        work["validation_stratum"] = [
            classify_validation_stratum(origin)
            for origin in work["origin"]
        ]
    else:
        work = oof
    frames = []
    group_keys = ["strategy", "validation_stratum"]
    if "origin_type" in work.columns:
        group_keys = ["origin_type"] + group_keys
    for keys, group in work.groupby(group_keys, sort=False):
        if "origin_type" in work.columns:
            origin_type, strategy, stratum = keys
        else:
            strategy, stratum = keys
            origin_type = "development"
        columns = prediction_columns_for_strategy(
            pred_columns or OOF_MODEL_COLUMNS, strategy
        )
        summary = summarize_oof(group, columns)
        if summary.empty:
            continue
        summary["origin_type"] = origin_type
        summary["strategy"] = strategy
        summary["validation_stratum"] = stratum
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_test_aligned_scores(
    stratum_summary: pd.DataFrame,
    metric: str = "WAPE",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Weighted development score emphasizing winter/test-like weeks."""
    if stratum_summary.empty:
        return pd.DataFrame()
    weights = weights or VALIDATION_STRATUM_WEIGHTS
    selected = stratum_summary[
        stratum_summary["evaluation_regime"].eq("conditional")
        & stratum_summary["comparison_population"].eq("common")
        & stratum_summary["aggregation"].eq("global")
    ].copy()
    if "origin_type" in selected.columns:
        selected = selected[selected["origin_type"].eq("development")]
    rows = []
    for (strategy, model), group in selected.groupby(["strategy", "model"]):
        available = group[group["validation_stratum"].isin(weights)].copy()
        available = available[np.isfinite(available[metric])]
        if available.empty:
            continue
        available["stratum_weight"] = available["validation_stratum"].map(weights)
        total_weight = float(available["stratum_weight"].sum())
        if total_weight <= 0:
            continue
        score = float(
            np.average(available[metric], weights=available["stratum_weight"])
        )
        rows.append({
            "strategy": strategy,
            "model": model,
            "metric": metric,
            "test_aligned_score": score,
            "weight_sum": total_weight,
            "strata_present": ",".join(sorted(available["validation_stratum"].unique())),
        })
    return pd.DataFrame(rows)


def summarize_channel_share_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """C4 app-share diagnostics by split and strategy.

    Total demand remains the submitted target. This table verifies whether the
    auxiliary head learned a meaningful channel composition without allowing
    share quality to conceal a deterioration in total-demand WAPE.
    """
    required = {"actual_AppShare", "pred_AppShare_NeuralNet", "actual"}
    if not required.issubset(oof.columns):
        return pd.DataFrame()
    rows = []
    group_columns = [column for column in ("origin_type", "strategy") if column in oof]
    grouped = oof.groupby(group_columns, sort=False) if group_columns else [((), oof)]
    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys if group_columns else ()))
        actual = pd.to_numeric(group["actual_AppShare"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(
            group["pred_AppShare_NeuralNet"], errors="coerce"
        ).to_numpy(dtype=float)
        total = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
        mask = (
            np.isfinite(actual) & np.isfinite(predicted)
            & np.isfinite(total) & (total > 0.0)
        )
        n_expected = int((np.isfinite(actual) & np.isfinite(total) & (total > 0.0)).sum())
        if mask.any():
            error = predicted[mask] - actual[mask]
            absolute = np.abs(error)
            weights = total[mask]
            weighted_mae = (
                float(np.average(absolute, weights=weights))
                if weights.sum() > 0 else np.nan
            )
            rows.append({
                **context,
                "model": "NeuralNet",
                "n_expected": n_expected,
                "n_scored": int(mask.sum()),
                "coverage": float(mask.sum() / n_expected) if n_expected else np.nan,
                "app_share_MAE": float(absolute.mean()),
                "app_share_weighted_MAE": weighted_mae,
                "app_share_bias": float(error.mean()),
                "actual_app_share_mean": float(actual[mask].mean()),
                "predicted_app_share_mean": float(predicted[mask].mean()),
            })
        else:
            rows.append({
                **context,
                "model": "NeuralNet",
                "n_expected": n_expected,
                "n_scored": 0,
                "coverage": 0.0 if n_expected else np.nan,
                "app_share_MAE": np.nan,
                "app_share_weighted_MAE": np.nan,
                "app_share_bias": np.nan,
                "actual_app_share_mean": np.nan,
                "predicted_app_share_mean": np.nan,
            })
    return pd.DataFrame(rows)


def _summarize_prediction_diagnostics_grouped(
    oof: pd.DataFrame,
    pred_columns: dict,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in oof.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys))
        strategy = context["strategy"]
        columns = prediction_columns_for_strategy(pred_columns, strategy)
        observed_max = pd.to_numeric(group["actual"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).max()
        for model, column in columns.items():
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            finite_values = values[finite]

            def bool_values(prefix: str) -> np.ndarray:
                name = f"{prefix}_{model}"
                if name not in group:
                    return np.zeros(len(group), dtype=bool)
                return group[name].fillna(False).astype(bool).to_numpy()

            fallback = bool_values("fallback")
            nonfinite_raw = bool_values("nonfinite")
            catastrophic = bool_values("catastrophic")
            residual_guard = bool_values("residual_guard")
            residual_nonfinite = bool_values("residual_nonfinite")

            residual_min_col = f"residual_raw_min_{model}"
            residual_max_col = f"residual_raw_max_{model}"
            safety_limit_col = f"safety_limit_{model}"
            residual_min = (
                pd.to_numeric(group[residual_min_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan).min()
                if residual_min_col in group else np.nan
            )
            residual_max = (
                pd.to_numeric(group[residual_max_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan).max()
                if residual_max_col in group else np.nan
            )
            safety_limit_min = (
                pd.to_numeric(group[safety_limit_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan).min()
                if safety_limit_col in group else np.nan
            )
            prediction_max = float(np.max(finite_values)) if finite_values.size else np.nan
            prediction_p99 = float(np.quantile(finite_values, 0.99)) if finite_values.size else np.nan
            rows.append({
                **context,
                "model": model,
                "n_rows": int(len(group)),
                "n_finite": int(finite.sum()),
                "coverage": float(finite.mean()) if len(group) else np.nan,
                "fallback_count": int(fallback.sum()),
                "fallback_rate": float(fallback.mean()) if len(group) else np.nan,
                "nonfinite_raw_count": int(nonfinite_raw.sum()),
                "catastrophic_guard_count": int(catastrophic.sum()),
                "residual_guard_count": int(residual_guard.sum()),
                "residual_nonfinite_count": int(residual_nonfinite.sum()),
                "residual_guard_rate": (
                    float(residual_guard.mean()) if len(group) else np.nan
                ),
                "residual_raw_min": (
                    float(residual_min) if np.isfinite(residual_min) else np.nan
                ),
                "residual_raw_max": (
                    float(residual_max) if np.isfinite(residual_max) else np.nan
                ),
                "safety_limit_min": (
                    float(safety_limit_min) if np.isfinite(safety_limit_min) else np.nan
                ),
                "prediction_max": prediction_max,
                "prediction_p99": prediction_p99,
                "observed_max": float(observed_max) if np.isfinite(observed_max) else np.nan,
                "prediction_to_observed_max_ratio": (
                    prediction_max / observed_max
                    if np.isfinite(prediction_max) and np.isfinite(observed_max) and observed_max > 0
                    else np.nan
                ),
            })
    return pd.DataFrame(rows)


def summarize_prediction_diagnostics(
    oof: pd.DataFrame,
    pred_columns: dict | None = None,
) -> pd.DataFrame:
    """Aggregate fallback, support-guard and extreme behavior by split."""
    return _summarize_prediction_diagnostics_grouped(
        oof, pred_columns or OOF_MODEL_COLUMNS, ["origin_type", "strategy"]
    )


def summarize_prediction_diagnostics_by_origin(
    oof: pd.DataFrame,
    pred_columns: dict | None = None,
) -> pd.DataFrame:
    """Per-origin diagnostics so isolated recursive explosions stay visible."""
    return _summarize_prediction_diagnostics_grouped(
        oof, pred_columns or OOF_MODEL_COLUMNS,
        ["origin_type", "strategy", "origin"],
    )


def select_primary_strategy(dev_summary: pd.DataFrame, *, model: str, metric: str) -> str:
    supported = MODEL_STRATEGY_SUPPORT.get(model, set())
    if supported == {"direct"}:
        return "direct"
    candidates = dev_summary[
        dev_summary["model"].eq(model)
        & dev_summary["evaluation_regime"].eq("conditional")
        & dev_summary["comparison_population"].eq("common")
        & dev_summary["aggregation"].eq("global")
        & dev_summary["strategy"].isin(["direct", "recursive"])
    ]
    if len(candidates) != 2:
        raise RuntimeError(f"Expected one direct and one recursive development row for {model}")
    return str(candidates.sort_values(metric, ascending=True).iloc[0]["strategy"])


def oof_to_legacy_cv_results(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    """Reshape row-level OOF predictions back into the older
    fold/model/MAE/RMSE/WAPE/Bias/BiasRatio shape.
    B4/Fix: Use common populations per fold/regime for fair comparison."""
    pred_columns = pred_columns or OOF_MODEL_COLUMNS
    strategies = oof.get("strategy", pd.Series("direct", index=oof.index)).dropna().unique()
    if len(strategies) == 1:
        pred_columns = prediction_columns_for_strategy(
            pred_columns, str(strategies[0])
        )
    pred_columns = {
        model: column
        for model, column in pred_columns.items()
        if column in oof.columns
    }
    pred_cols = list(pred_columns.values())

    origins_sorted = sorted(oof["origin"].unique(), reverse=True)
    fold_of_origin = {origin: i for i, origin in enumerate(origins_sorted)}

    regime_masks_base = {
        "realized": oof["actual"].notna(),
        "conditional": (
            oof["actual"].notna()
            & oof["ProductAvailable"].fillna(False)
        ),
    }

    rows = []
    for origin, fold_df in oof.groupby("origin"):
        for regime_name, regime_mask_all in regime_masks_base.items():
            # Regime mask for THIS fold
            regime_mask = regime_mask_all.loc[fold_df.index]

            # Common population: rows where ALL models have finite predictions
            common_mask = regime_mask & fold_df[pred_cols].apply(np.isfinite).all(axis=1)

            for model_name, col in pred_columns.items():
                if col not in fold_df.columns:
                    continue

                scored_df = fold_df[common_mask]
                if scored_df.empty:
                    continue

                rows.append({
                    "fold": fold_of_origin[origin],
                    "model": model_name,
                    "regime": regime_name,
                    "comparison_population": "common",
                    **compute_metrics(scored_df["actual"], scored_df[col])
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final ensemble training + test forecast
# ---------------------------------------------------------------------------
def _prepare_final_direct_panel(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG):
    """Shared by `run_final_forecast` and `run_final_tree_forecast`: builds
    the direct multi-horizon panel for the real forecast -- origin = the
    last training day, targets = the actual test week (covariates from
    `test_raw` itself, since nothing later exists in `train_raw` to look
    up)."""
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(train_raw)

    train_feat = prepare_features(
        train_raw, price_ref, first_seen, first_available, cfg
    )
    train_feat = add_train_lags(
        train_feat, cfg.lag_windows, baseline_variant=cfg.baseline_variant
    )
    test_feat = prepare_features(
        test_raw, price_ref, first_seen, first_available, cfg
    ).reset_index(drop=True)

    horizons = range(1, cfg.horizon + 1)
    panel = build_direct_panel(train_feat, horizons, cfg=cfg, future_covariates=test_feat)

    last_train_date = train_raw["DateKey"].max()
    train_panel = select_trainable_panel_rows(
        panel, cutoff=last_train_date, available_only=True, cfg=cfg
    )
    eval_panel = panel[panel["OriginDateKey"] == last_train_date].reset_index(drop=True)
    return train_panel, eval_panel


def run_final_forecast_direct(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cfg: Config = CFG,
    *,
    return_diagnostics: bool = False,
):
    train_panel, eval_panel = _prepare_final_direct_panel(train_raw, test_raw, cfg)

    scaler = make_numeric_preprocessor()
    tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
    y_target = neural_training_target(train_panel, cfg)

    models = []
    for seed in cfg.seeds:
        seed_start = time.perf_counter()
        print(f"    seed {seed}")
        models.append(train_model(tensors, y_target, cfg, epochs=cfg.final_epochs, seed=seed))
        print(f"      [timing] seed {seed}: {time.perf_counter() - seed_start:.1f}s")

    output = predict_direct(
        models, scaler, eval_panel, cfg, return_diagnostics=True
    )
    preds = output["prediction"]
    preds_aligned = _reindex_predictions(eval_panel, preds, "TargetDateKey", test_raw)

    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(preds_aligned).astype(int)
    if not return_diagnostics:
        return submission, preds_aligned

    diagnostics = {"prediction": preds_aligned}
    for key in ("app_share", "prediction_app", "prediction_web"):
        if key in output:
            diagnostics[key] = _reindex_predictions(
                eval_panel, output[key], "TargetDateKey", test_raw
            )
    return submission, preds_aligned, diagnostics


def run_final_tree_forecast_direct(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG) -> dict:
    """XGBoost/LightGBM trained on ALL history and forecast for the actual
    test week -- purely for dashboard comparison parity with the NN's page.
    Not used for the submission file (the task brief asked for a non-tree
    approach as the actual deliverable).
    """
    train_panel, eval_panel = _prepare_final_direct_panel(train_raw, test_raw, cfg)
    tree_preds = run_tree_baselines(train_panel, eval_panel, cfg)
    return {name: _reindex_predictions(eval_panel, preds, "TargetDateKey", test_raw)
            for name, preds in tree_preds.items()}


def _align_recursive_path(path: pd.DataFrame, test_raw: pd.DataFrame) -> np.ndarray:
    lookup = path[["ProductId", "TargetDateKey", "prediction"]].rename(columns={"TargetDateKey": "DateKey"})
    aligned = test_raw[["ProductId", "DateKey"]].merge(
        lookup, on=["ProductId", "DateKey"], how="left", validate="one_to_one"
    )
    return aligned["prediction"].to_numpy(dtype=float)


def _align_recursive_column(
    path: pd.DataFrame,
    test_raw: pd.DataFrame,
    column: str,
) -> np.ndarray:
    if column not in path.columns:
        return np.full(len(test_raw), np.nan, dtype=float)
    lookup = path[["ProductId", "TargetDateKey", column]].rename(
        columns={"TargetDateKey": "DateKey"}
    )
    aligned = test_raw[["ProductId", "DateKey"]].merge(
        lookup, on=["ProductId", "DateKey"], how="left", validate="one_to_one"
    )
    return aligned[column].to_numpy(dtype=float)


def run_final_forecast_recursive(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG):
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(train_raw)
    train_panel = _recursive_panel_training_data(
        train_raw, price_ref, first_seen, first_available, cfg
    )
    future = sanitize_future_covariates(test_raw)
    path, _, _, _ = _recursive_nn_predictions(
        train_panel, train_raw, future, price_ref, first_seen,
        first_available, cfg, cfg.final_epochs
    )
    preds = _align_recursive_path(path, test_raw)
    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(preds).astype(int)
    return submission, preds, path


def run_final_structured_forecast_recursive(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
                                            cfg: Config = CFG) -> tuple[dict, dict]:
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(train_raw)
    train_panel = _recursive_panel_training_data(
        train_raw, price_ref, first_seen, first_available, cfg
    )
    payloads = run_structured_models(
        train_panel, cfg, models=("XGBoost", "LightGBM"),
        strategy="recursive", history_raw=train_raw,
        future_covariates=test_raw, price_ref=price_ref,
        first_seen=first_seen, first_available=first_available,
    )
    aligned, paths = {}, {}
    for name, payload in payloads.items():
        path = pd.DataFrame(payload)
        paths[name] = path
        aligned[name] = _align_recursive_path(path, test_raw)
    return aligned, paths


def run_final_forecast(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG,
                       strategy: ForecastStrategy | str = ForecastStrategy.DIRECT):
    """Compatibility dispatcher returning submission and NN predictions."""
    strategy = ForecastStrategy(strategy)
    if strategy is ForecastStrategy.DIRECT:
        return run_final_forecast_direct(train_raw, test_raw, cfg)
    submission, preds, _ = run_final_forecast_recursive(train_raw, test_raw, cfg)
    return submission, preds


def run_final_tree_forecast(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG,
                            strategy: ForecastStrategy | str = ForecastStrategy.DIRECT) -> dict:
    strategy = ForecastStrategy(strategy)
    if strategy is ForecastStrategy.DIRECT:
        return run_final_tree_forecast_direct(train_raw, test_raw, cfg)
    aligned, _ = run_final_structured_forecast_recursive(train_raw, test_raw, cfg)
    return aligned


def run_final_naive_baselines(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG) -> dict:
    """Seasonal-naive / moving-average predictions for the actual test week --
    shown on their dashboard pages for comparison, not used for submission."""
    seasonal = seasonal_naive_predict(test_raw, train_raw, lag_days=cfg.horizon)
    moving_avg = moving_average_predict(test_raw, train_raw, window=28)
    return {
        "SeasonalNaive": np.clip(np.nan_to_num(seasonal, nan=0.0), 0, None),
        "MovingAvg28": np.clip(np.nan_to_num(moving_avg, nan=0.0), 0, None),
    }


def plot_forecast(train_raw: pd.DataFrame, submission: pd.DataFrame,
                   product_ids: tuple = (1, 5, 16), lookback_days: int = 60,
                   cfg: Config = CFG) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(product_ids), 1, figsize=(9, 3 * len(product_ids)))
    axes = np.atleast_1d(axes)
    for ax, pid in zip(axes, product_ids):
        hist = train_raw[train_raw["ProductId"] == pid].sort_values("DateKey").tail(lookback_days)
        fut = submission[submission["ProductId"] == pid].sort_values("DateKey")
        ax.plot(hist["DateKey"], hist["Quantity"], label="history", color="steelblue")
        ax.plot(fut["DateKey"], fut["Quantity"], label="forecast", color="darkorange", marker="o")
        ax.axvline(hist["DateKey"].max(), color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Product {pid}")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "forecast_plot.png")
    fig.savefig(out_path, dpi=130)
    print(f"Saved: {out_path}")


def _json_safe(obj):
    """Convert pipeline payload values into strict JSON-compatible scalars.

    DataFrame ``to_dict`` preserves pandas/NumPy scalar types, including
    ``Timestamp`` values in per-origin diagnostics.  The JSON encoder does
    not know how to serialize those objects.  Normalize them once at the
    artifact boundary and reject any remaining non-standard NaN/Infinity
    tokens when writing the file.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, np.datetime64, datetime)):
        timestamp = pd.Timestamp(obj)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def select_primary_summary(
    summary: pd.DataFrame,
    *,
    evaluation_regime: str = "conditional",
    comparison_population: str = "common",
    aggregation: str = "global",
) -> pd.DataFrame:
    """Helper to select a canonical slice of the expanded OOF summary (Tier B Fix)."""
    selected = summary[
        (summary["evaluation_regime"] == evaluation_regime)
        & (summary["comparison_population"] == comparison_population)
        & (summary["aggregation"] == aggregation)
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"Primary evaluation summary is empty for "
            f"{evaluation_regime}/{comparison_population}/{aggregation}"
        )

    if selected["model"].duplicated().any():
        raise RuntimeError(
            f"Primary evaluation summary contains duplicate model rows for "
            f"{evaluation_regime}/{comparison_population}/{aggregation}"
        )

    return selected


def export_results_json(train_raw: pd.DataFrame, test_raw: pd.DataFrame, submission: pd.DataFrame,
                         final_forecasts: dict, cv_results: pd.DataFrame, cfg: Config = CFG,
                         history_lookback: int = 90, path: str | None = None,
                         dev_summary: pd.DataFrame = None, benchmark_summary: pd.DataFrame = None,
                         runtime_options: RuntimeOptions | None = None,
                         forecasts_by_strategy: dict | None = None,
                         strategy_comparison: pd.DataFrame | None = None,
                         canonical_strategy: str = "direct",
                         canonical_model: str = "NeuralNet",
                         cv_results_all: pd.DataFrame | None = None,
                         strategy_by_horizon: pd.DataFrame | None = None,
                         validation_strata_summary: pd.DataFrame | None = None,
                         test_aligned_scores: pd.DataFrame | None = None,
                         prediction_diagnostics: pd.DataFrame | None = None,
                         prediction_diagnostics_by_origin: pd.DataFrame | None = None,
                         channel_share_summary: pd.DataFrame | None = None) -> dict:
    """Bundle everything the presentation webapp needs into one JSON file.
    Uses 'Conditional Demand' on a 'Common' population as the primary summary
    (Tier B Corrections).
    """
    # Skill scores and backward-compatible primary summary table. The rich
    # strategy-aware summaries remain unfiltered in `*_summary_all` below.
    if benchmark_summary is not None:
        benchmark_for_canonical = benchmark_summary
        if "strategy" in benchmark_for_canonical.columns:
            benchmark_for_canonical = benchmark_for_canonical[
                benchmark_for_canonical["strategy"].eq(canonical_strategy)
            ]
        summary = select_primary_summary(benchmark_for_canonical).copy()
    else:
        # Fallback to cv_results (legacy or if summary not provided)
        summary_source = cv_results
        if "regime" in cv_results.columns:
            # Use conditional if possible, else realized
            mask = (cv_results["regime"] == "conditional")
            if mask.any():
                summary_source = cv_results[mask]
            else:
                summary_source = cv_results[cv_results["regime"] == "realized"]

        summary = (summary_source.groupby("model")[["MAE", "RMSE", "WAPE", "Bias", "BiasRatio"]]
                   .mean(numeric_only=True).round(3).reset_index())
    
    summary = order_models(summary)
    summary_idx = summary.set_index("model")
    naive_mae = summary_idx.loc["SeasonalNaive", "MAE"] if "SeasonalNaive" in summary_idx.index else None
    skill_by_model = {}
    if naive_mae:
        for name in summary_idx.index:
            skill_by_model[name] = float(1 - summary_idx.loc[name, "MAE"] / naive_mae)
    skill = skill_by_model.get("NeuralNet")

    history = {}
    for pid in sorted(train_raw["ProductId"].unique()):
        hist = (train_raw[train_raw["ProductId"] == pid]
                .sort_values("DateKey").tail(history_lookback))
        history[str(int(pid))] = {
            "dates": hist["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
            "quantity": hist["Quantity"].astype(float).tolist(),
        }

    test_keys = test_raw[["ProductId", "DateKey"]].reset_index(drop=True)
    forecasts = {}
    for model_name, preds in final_forecasts.items():
        df = test_keys.copy()
        df["Quantity"] = np.asarray(preds, dtype=float)
        per_product = {}
        for pid in sorted(df["ProductId"].unique()):
            sub = df[df["ProductId"] == pid].sort_values("DateKey")
            per_product[str(int(pid))] = {
                "dates": sub["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
                "quantity": sub["Quantity"].astype(float).tolist(),
            }
        forecasts[model_name] = per_product

    available_model_names = set(final_forecasts)
    if forecasts_by_strategy:
        for strategy_forecasts in forecasts_by_strategy.values():
            available_model_names.update(strategy_forecasts)
    models_meta = [
        {
            "key": name,
            "slug": MODEL_SLUGS[name],
            "skill_vs_seasonal_naive": skill_by_model.get(name),
            "strategies": sorted(MODEL_STRATEGY_SUPPORT.get(name, {"direct"})),
            **MODEL_META[name],
        }
        for name in MODEL_ORDER if name in available_model_names
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "forecast_strategy": runtime_options.forecast_strategy.value if runtime_options else "direct",
            "primary_strategy": canonical_strategy,
            "submission_model": runtime_options.submission_model.value if runtime_options else "NeuralNet",
            "selection_metric": runtime_options.selection_metric if runtime_options else "WAPE",
            "selection_protocol": runtime_options.selection_protocol if runtime_options else "global",
            "primary_evaluation_regime": "conditional",
            "primary_comparison_population": "common",
            "primary_aggregation": "global",
            "horizon": cfg.horizon,
            "lag_windows": list(cfg.lag_windows),
            "n_cv_folds": cfg.n_cv_folds,
            "n_dev_origins": len(DEVELOPMENT_ORIGINS),
            "cv_epochs": cfg.cv_epochs,
            "final_epochs": cfg.final_epochs,
            "seeds": list(cfg.seeds),
            "num_products": cfg.num_products,
            "validation_stratum_weights": VALIDATION_STRATUM_WEIGHTS,
            "final_audit_origins": [
                str(pd.Timestamp(origin).date()) for origin in FINAL_AUDIT_ORIGINS
            ],
            "nn_batch_size": cfg.batch_size,
            "nn_reference_batch_size": cfg.reference_batch_size,
            "nn_lr_scaling": cfg.nn_lr_scaling,
            "nn_effective_learning_rate": effective_learning_rate(cfg),
            "nn_training_backend": resolve_training_backend(cfg),
            "training_window_days": cfg.training_window_days,
            "recency_half_life_days": cfg.recency_half_life_days,
            "baseline_variant": cfg.baseline_variant,
            "enable_trend_features": cfg.enable_trend_features,
            "c2_feature_groups": list(cfg.c2_feature_groups),
            "nn_loss": cfg.nn_loss,
            "nn_target_mode": cfg.nn_target_mode,
            "nn_huber_delta": cfg.nn_huber_delta,
            "nn_combined_mse_weight": cfg.nn_combined_mse_weight,
            "tree_target_mode": cfg.tree_target_mode,
            "xgboost_target_mode": cfg.xgboost_target_mode,
            "lightgbm_target_mode": cfg.lightgbm_target_mode,
            "tree_tweedie_variance_power": cfg.tree_tweedie_variance_power,
            "enable_channel_history_features": cfg.enable_channel_history_features,
            "channel_aux_weight": cfg.channel_aux_weight,
            "channel_share_smoothing": cfg.channel_share_smoothing,
            "nn_residual_guard_lower_quantile": cfg.nn_residual_guard_lower_quantile,
            "nn_residual_guard_upper_quantile": cfg.nn_residual_guard_upper_quantile,
            "nn_residual_guard_margin": cfg.nn_residual_guard_margin,
            "recursive_safety_multiplier": cfg.recursive_safety_multiplier,
            "recursive_safety_floor": cfg.recursive_safety_floor,
        },
        "models": models_meta,
        # Canonical compatibility fields used by the original dashboard.
        "cv_results": order_models(cv_results.round(3)).to_dict(orient="records"),
        "cv_summary": summary.to_dict(orient="records"),
        "skill_vs_seasonal_naive": skill,
        # Full strategy-aware fields used by the synchronized dashboard.
        "cv_results_all": (
            order_models(cv_results_all.round(3)).to_dict(orient="records")
            if cv_results_all is not None else
            order_models(cv_results.round(3)).assign(strategy=canonical_strategy).to_dict(orient="records")
        ),
        "benchmark_summary_all": (
            order_models(benchmark_summary.round(6)).to_dict(orient="records")
            if benchmark_summary is not None else []
        ),
        "dev_summary_all": (
            order_models(dev_summary.round(6)).to_dict(orient="records")
            if dev_summary is not None else []
        ),
        # Keep these aliases canonical-only for older consumers.
        "benchmark_summary": (
            order_models(
                benchmark_summary[
                    benchmark_summary["strategy"].eq(canonical_strategy)
                ] if benchmark_summary is not None and "strategy" in benchmark_summary.columns
                else benchmark_summary
            ).round(3).to_dict(orient="records")
            if benchmark_summary is not None else None
        ),
        "dev_summary": (
            order_models(
                dev_summary[
                    dev_summary["strategy"].eq(canonical_strategy)
                ] if dev_summary is not None and "strategy" in dev_summary.columns
                else dev_summary
            ).round(3).to_dict(orient="records")
            if dev_summary is not None else None
        ),
        "submission": submission.assign(
            DateKey=submission["DateKey"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "history": history,
        "forecasts": forecasts,
        "forecasts_by_strategy": forecasts_by_strategy or {canonical_strategy: forecasts},
        "strategy_comparison": (strategy_comparison.round(6).to_dict(orient="records")
                                if strategy_comparison is not None else []),
        "strategy_by_horizon": (strategy_by_horizon.round(6).to_dict(orient="records")
                                if strategy_by_horizon is not None else []),
        "validation_strata_summary": (
            validation_strata_summary.round(6).to_dict(orient="records")
            if validation_strata_summary is not None else []
        ),
        "test_aligned_scores": (
            test_aligned_scores.round(6).to_dict(orient="records")
            if test_aligned_scores is not None else []
        ),
        "prediction_diagnostics": (
            prediction_diagnostics.round(6).to_dict(orient="records")
            if prediction_diagnostics is not None else []
        ),
        "prediction_diagnostics_by_origin": (
            prediction_diagnostics_by_origin.round(6).to_dict(orient="records")
            if prediction_diagnostics_by_origin is not None else []
        ),
        "channel_share_summary": (
            channel_share_summary.round(6).to_dict(orient="records")
            if channel_share_summary is not None else []
        ),
        "selection": {
            "canonical_model": canonical_model,
            "canonical_strategy": canonical_strategy,
            "selected_from": "development",
            "development_winner": canonical_strategy,
            "benchmark_winner": None,
            "recent_benchmark_confirmation": None,
        },
    }

    if benchmark_summary is not None and not benchmark_summary.empty:
        benchmark_candidates = benchmark_summary[
            benchmark_summary["model"].eq(canonical_model)
            & benchmark_summary["evaluation_regime"].eq("conditional")
            & benchmark_summary["comparison_population"].eq("common")
            & benchmark_summary["aggregation"].eq("global")
        ].copy()
        if not benchmark_candidates.empty:
            metric = runtime_options.selection_metric if runtime_options else "WAPE"
            benchmark_winner = str(
                benchmark_candidates.sort_values(metric).iloc[0]["strategy"]
            )
            payload["selection"]["benchmark_winner"] = benchmark_winner
            payload["selection"]["recent_benchmark_confirmation"] = (
                benchmark_winner == canonical_strategy
            )

    payload = _json_safe(payload)
    out_path = path or os.path.join(cfg.output_dir, "results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    os.replace(tmp_path, out_path)
    print(f"Saved: {out_path}")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _choose_canonical_model_strategy(
    options: RuntimeOptions,
    dev_summary: pd.DataFrame,
    test_aligned_scores: pd.DataFrame | None = None,
) -> tuple[str, str]:
    strategies = set(dev_summary["strategy"].unique())
    requested_model = options.submission_model.value

    def choose_from_test_aligned(model: str | None = None) -> tuple[str, str]:
        if test_aligned_scores is None or test_aligned_scores.empty:
            raise RuntimeError(
                "Test-aligned selection requested but no stratum scores exist"
            )
        candidates = test_aligned_scores.copy()
        if model is not None:
            candidates = candidates[candidates["model"].eq(model)]
        if candidates.empty:
            raise RuntimeError(
                f"No test-aligned candidates available for {model or 'auto model selection'}"
            )
        row = candidates.sort_values("test_aligned_score").iloc[0]
        return str(row["model"]), str(row["strategy"])

    if options.submission_model is SubmissionModel.AUTO:
        if options.selection_protocol == "test-aligned":
            return choose_from_test_aligned()
        candidates = dev_summary[
            dev_summary["evaluation_regime"].eq("conditional")
            & dev_summary["comparison_population"].eq("common")
            & dev_summary["aggregation"].eq("global")
            & dev_summary["model"].isin(
                ["NeuralNet", "DynamicRidge", "XGBoost", "LightGBM"]
            )
        ]
        row = candidates.sort_values(options.selection_metric).iloc[0]
        return str(row["model"]), str(row["strategy"])

    if not any(model_supports_strategy(requested_model, s) for s in strategies):
        raise RuntimeError(
            f"{requested_model} does not support requested strategy set {sorted(strategies)}"
        )
    if MODEL_STRATEGY_SUPPORT.get(requested_model) == {"direct"}:
        if strategies == {"recursive"}:
            raise RuntimeError(f"{requested_model} is direct-only")
        return requested_model, "direct"
    if len(strategies) == 1:
        strategy = next(iter(strategies))
        if not model_supports_strategy(requested_model, strategy):
            raise RuntimeError(f"{requested_model} does not support {strategy}")
        return requested_model, strategy
    if options.primary_strategy is PrimaryStrategy.AUTO:
        if options.selection_protocol == "test-aligned":
            return choose_from_test_aligned(requested_model)
        return requested_model, select_primary_strategy(
            dev_summary, model=requested_model, metric=options.selection_metric
        )
    strategy = options.primary_strategy.value
    if not model_supports_strategy(requested_model, strategy):
        raise RuntimeError(f"{requested_model} does not support {strategy}")
    return requested_model, strategy


def _forecast_dict_to_json(test_raw: pd.DataFrame, forecasts: dict) -> dict:
    keys = test_raw[["ProductId", "DateKey"]].reset_index(drop=True)
    result = {}
    for model, preds in forecasts.items():
        frame = keys.copy()
        frame["Quantity"] = np.asarray(preds, dtype=float)
        per_product = {}
        for pid, sub in frame.groupby("ProductId"):
            sub = sub.sort_values("DateKey")
            per_product[str(int(pid))] = {
                "dates": sub["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
                "quantity": sub["Quantity"].tolist(),
            }
        result[model] = per_product
    return result


def main(argv=None) -> None:
    options = parse_args(argv)
    cfg = CFG
    c1_runtime = configure_c1_runtime(cfg, options)
    c2_runtime = configure_c2_runtime(cfg, options)
    c34_runtime = configure_c34_runtime(cfg, options)
    nn_runtime = configure_nn_runtime(cfg, options)
    print(f"Device: {DEVICE}")
    print(f"Forecast strategy: {options.forecast_strategy.value}")
    print(
        f"Selection: {options.selection_protocol} / {options.selection_metric}"
    )
    print(
        "C1 policy: "
        f"window={c1_runtime['training_window_days'] or 'all'}, "
        f"half_life={c1_runtime['recency_half_life_days'] or 'none'}, "
        f"baseline={c1_runtime['baseline_variant']}, "
        f"trend_features={c1_runtime['enable_trend_features']}"
    )
    print(
        "C2 groups: "
        + (", ".join(c2_runtime["c2_feature_groups"]) or "none")
        + f" ({c2_runtime['source']})"
    )
    print(
        "C3/C4 policy: "
        f"loss={c34_runtime['nn_loss']}, "
        f"target={c34_runtime['nn_target_mode']}, "
        f"channel_history={c34_runtime['enable_channel_history_features']}, "
        f"channel_aux={c34_runtime['channel_aux_weight']}, "
        f"tree_target={c34_runtime['tree_target_mode']}, "
        f"xgb_target={c34_runtime['xgboost_target_mode']}, "
        f"lgb_target={c34_runtime['lightgbm_target_mode']}"
    )
    print(
        "NN runtime: "
        f"batch={nn_runtime['batch_size']} "
        f"({nn_runtime['batch_source']}), "
        f"lr={nn_runtime['effective_learning_rate']:.6g} "
        f"[{nn_runtime['lr_scaling']}], "
        f"backend={nn_runtime['training_backend']}"
    )
    if options.reset_checkpoints and os.path.exists(options.checkpoint_dir):
        shutil.rmtree(options.checkpoint_dir)
        print(f"Removed checkpoints: {options.checkpoint_dir}")
    if options.resume:
        print(f"CV resume enabled: {options.checkpoint_dir}")
    run_start = time.perf_counter()
    timings: dict = {
        "cv_folds": [], "nn_runtime": nn_runtime,
        "c1_runtime": c1_runtime, "c2_runtime": c2_runtime,
        "c34_runtime": c34_runtime,
    }

    train_raw, test_raw = load_raw(cfg)
    cfg.num_products = int(max(train_raw["ProductId"].max(), test_raw["ProductId"].max()))
    benchmark_origins = recent_benchmark_origins(train_raw, cfg)
    strategies = resolve_strategies(options.forecast_strategy)

    dev_frames, benchmark_frames = [], []
    for strategy in strategies:
        print(f"\n=== {strategy.value.upper()} development CV ===")
        dev = run_walk_forward_cv(
            train_raw, DEVELOPMENT_ORIGINS, "development", cfg,
            timings=timings["cv_folds"], strategy=strategy,
            checkpoint_dir=options.checkpoint_dir, resume=options.resume,
        )
        dev_frames.append(dev)
        print(f"\n=== {strategy.value.upper()} recent-benchmark CV ===")
        benchmark = run_walk_forward_cv(
            train_raw, benchmark_origins, "recent_benchmark", cfg,
            timings=timings["cv_folds"], strategy=strategy,
            checkpoint_dir=options.checkpoint_dir, resume=options.resume,
        )
        benchmark_frames.append(benchmark)

    dev_oof = pd.concat(dev_frames, ignore_index=True)
    benchmark_oof = pd.concat(benchmark_frames, ignore_index=True)
    oof = pd.concat([dev_oof, benchmark_oof], ignore_index=True)
    dev_summary = summarize_oof_by_strategy(dev_oof)
    benchmark_summary = summarize_oof_by_strategy(benchmark_oof)
    pair_summary = summarize_strategy_pairs(
        dev_oof, evaluation_regime="conditional"
    )
    validation_strata_summary = summarize_validation_strata(oof)
    test_aligned_scores = compute_test_aligned_scores(
        validation_strata_summary, metric=options.selection_metric
    )
    prediction_diagnostics = summarize_prediction_diagnostics(oof)
    prediction_diagnostics_by_origin = summarize_prediction_diagnostics_by_origin(oof)
    channel_share_summary = summarize_channel_share_oof(oof)

    canonical_model, canonical_strategy = _choose_canonical_model_strategy(
        options, dev_summary, test_aligned_scores
    )
    print(f"\nCanonical selection: {canonical_model} / {canonical_strategy}")

    final_by_strategy: dict[str, dict[str, np.ndarray]] = {}
    submissions_by_strategy: dict[str, pd.DataFrame] = {}
    raw_rows = []
    naive_final = run_final_naive_baselines(train_raw, test_raw, cfg)
    for strategy in strategies:
        print(f"\n=== Final {strategy.value} forecasts ===")
        if strategy is ForecastStrategy.DIRECT:
            nn_submission, nn_preds, nn_details = run_final_forecast_direct(
                train_raw, test_raw, cfg, return_diagnostics=True
            )
            structured = run_final_tree_forecast_direct(train_raw, test_raw, cfg)
            paths = {}
        else:
            nn_submission, nn_preds, nn_path = run_final_forecast_recursive(train_raw, test_raw, cfg)
            nn_details = {
                "prediction": nn_preds,
                "app_share": _align_recursive_column(nn_path, test_raw, "app_share"),
                "prediction_app": _align_recursive_column(nn_path, test_raw, "prediction_app"),
                "prediction_web": _align_recursive_column(nn_path, test_raw, "prediction_web"),
            }
            structured, paths = run_final_structured_forecast_recursive(train_raw, test_raw, cfg)
            paths["NeuralNet"] = nn_path
        forecasts = {"NeuralNet": nn_preds, **structured, **naive_final}
        final_by_strategy[strategy.value] = forecasts
        submissions_by_strategy[strategy.value] = nn_submission
        for model, preds in forecasts.items():
            for row_index, ((pid, date), pred) in enumerate(zip(
                test_raw[["ProductId", "DateKey"]].itertuples(index=False, name=None),
                np.asarray(preds, dtype=float),
            )):
                raw_rows.append({
                    "strategy": strategy.value, "model": model,
                    "ProductId": pid, "DateKey": date,
                    "prediction_raw": float(pred),
                    "prediction_submission": int(round(max(float(pred), 0.0))),
                    "fallback_used": False,
                    "nonfinite_raw": False,
                    "catastrophic_guard": False,
                    "residual_guard": False,
                    "residual_nonfinite": False,
                    "residual_raw_min": np.nan,
                    "residual_raw_max": np.nan,
                    "safety_limit": np.nan,
                    "predicted_app_share": (
                        float(nn_details.get("app_share", np.full(len(test_raw), np.nan))[row_index])
                        if model == "NeuralNet" else np.nan
                    ),
                    "prediction_app": (
                        float(nn_details.get("prediction_app", np.full(len(test_raw), np.nan))[row_index])
                        if model == "NeuralNet" else np.nan
                    ),
                    "prediction_web": (
                        float(nn_details.get("prediction_web", np.full(len(test_raw), np.nan))[row_index])
                        if model == "NeuralNet" else np.nan
                    ),
                })
        for model, path in paths.items():
            path_index = path.set_index(["ProductId", "TargetDateKey"])
            fallback_map = path_index["fallback_used"]
            nonfinite_map = path_index.get(
                "nonfinite_raw", pd.Series(False, index=path_index.index)
            )
            catastrophic_map = path_index.get(
                "catastrophic_guard", pd.Series(False, index=path_index.index)
            )
            residual_guard_map = path_index.get(
                "residual_guard", pd.Series(False, index=path_index.index)
            )
            residual_nonfinite_map = path_index.get(
                "residual_nonfinite", pd.Series(False, index=path_index.index)
            )
            residual_min_map = path_index.get(
                "residual_raw_min", pd.Series(np.nan, index=path_index.index)
            )
            residual_max_map = path_index.get(
                "residual_raw_max", pd.Series(np.nan, index=path_index.index)
            )
            safety_limit_map = path_index.get(
                "safety_limit", pd.Series(np.nan, index=path_index.index)
            )
            for row in raw_rows:
                if row["strategy"] == strategy.value and row["model"] == model:
                    key = (row["ProductId"], row["DateKey"])
                    row["fallback_used"] = bool(fallback_map.get(key, False))
                    row["nonfinite_raw"] = bool(nonfinite_map.get(key, False))
                    row["catastrophic_guard"] = bool(catastrophic_map.get(key, False))
                    row["residual_guard"] = bool(residual_guard_map.get(key, False))
                    row["residual_nonfinite"] = bool(
                        residual_nonfinite_map.get(key, False)
                    )
                    row["residual_raw_min"] = float(residual_min_map.get(key, np.nan))
                    row["residual_raw_max"] = float(residual_max_map.get(key, np.nan))
                    row["safety_limit"] = float(safety_limit_map.get(key, np.nan))

    canonical_preds = final_by_strategy[canonical_strategy][canonical_model]
    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(np.clip(canonical_preds, 0, None)).astype(int)

    os.makedirs(cfg.output_dir, exist_ok=True)
    submission.to_parquet(os.path.join(cfg.output_dir, "submission.parquet"), index=False)
    submission.to_csv(os.path.join(cfg.output_dir, "submission.csv"), index=False)
    for strategy, forecasts in final_by_strategy.items():
        strategy_submission = test_raw[["ProductId", "DateKey"]].copy()
        strategy_model = canonical_model if canonical_model in forecasts else "NeuralNet"
        strategy_submission["Quantity"] = np.round(
            np.clip(forecasts[strategy_model], 0, None)
        ).astype(int)
        strategy_submission.to_csv(os.path.join(cfg.output_dir, f"submission_{strategy}.csv"), index=False)
        strategy_submission.to_parquet(os.path.join(cfg.output_dir, f"submission_{strategy}.parquet"), index=False)

    final_forecast_df = pd.DataFrame(raw_rows)
    final_forecast_df.to_parquet(os.path.join(cfg.output_dir, "final_forecasts.parquet"), index=False)
    oof.to_parquet(os.path.join(cfg.output_dir, "oof_predictions.parquet"), index=False)
    dev_summary.to_csv(os.path.join(cfg.output_dir, "dev_summary.csv"), index=False)
    benchmark_summary.to_csv(os.path.join(cfg.output_dir, "benchmark_summary.csv"), index=False)
    pair_summary.to_csv(os.path.join(cfg.output_dir, "strategy_pair_summary.csv"), index=False)
    validation_strata_summary.to_csv(
        os.path.join(cfg.output_dir, "validation_strata_summary.csv"), index=False
    )
    test_aligned_scores.to_csv(
        os.path.join(cfg.output_dir, "test_aligned_scores.csv"), index=False
    )
    prediction_diagnostics.to_csv(
        os.path.join(cfg.output_dir, "prediction_diagnostics.csv"), index=False
    )
    prediction_diagnostics_by_origin.to_csv(
        os.path.join(cfg.output_dir, "prediction_diagnostics_by_origin.csv"),
        index=False,
    )
    channel_share_summary.to_csv(
        os.path.join(cfg.output_dir, "channel_share_summary.csv"), index=False
    )

    by_horizon_frames = []
    # Export both development and recent-benchmark curves.  The explicit
    # origin_type field prevents the presentation layer from implying that
    # benchmark rows participated in development selection.
    for (origin_type, strategy), group in oof.groupby(
        ["origin_type", "strategy"], sort=False
    ):
        columns = prediction_columns_for_strategy(
            OOF_MODEL_COLUMNS, strategy
        )
        for horizon, hgroup in group.groupby("horizon"):
            summary = summarize_oof(hgroup, columns)
            summary["strategy"] = strategy
            summary["horizon"] = horizon
            summary["origin_type"] = origin_type
            by_horizon_frames.append(summary)
    strategy_by_horizon = pd.concat(by_horizon_frames, ignore_index=True)
    strategy_by_horizon.to_csv(os.path.join(cfg.output_dir, "strategy_by_horizon.csv"), index=False)

    cv_results_frames = []
    for strategy_name, strategy_oof in benchmark_oof.groupby("strategy", sort=False):
        strategy_cv = oof_to_legacy_cv_results(strategy_oof)
        strategy_cv["strategy"] = strategy_name
        cv_results_frames.append(strategy_cv)
    cv_results_all = pd.concat(cv_results_frames, ignore_index=True)
    cv_results = cv_results_all[cv_results_all["strategy"].eq(canonical_strategy)].drop(
        columns=["strategy"]
    ).reset_index(drop=True)
    cv_results.to_csv(os.path.join(cfg.output_dir, "cv_results.csv"), index=False)
    cv_results_all.to_csv(os.path.join(cfg.output_dir, "cv_results_all.csv"), index=False)

    forecasts_json = {
        strategy: _forecast_dict_to_json(test_raw, forecasts)
        for strategy, forecasts in final_by_strategy.items()
    }
    export_results_json(
        train_raw, test_raw, submission, final_by_strategy[canonical_strategy], cv_results, cfg,
        dev_summary=dev_summary, benchmark_summary=benchmark_summary,
        runtime_options=options, forecasts_by_strategy=forecasts_json,
        strategy_comparison=pair_summary, canonical_strategy=canonical_strategy,
        canonical_model=canonical_model, cv_results_all=cv_results_all,
        strategy_by_horizon=strategy_by_horizon,
        validation_strata_summary=validation_strata_summary,
        test_aligned_scores=test_aligned_scores,
        prediction_diagnostics=prediction_diagnostics,
        prediction_diagnostics_by_origin=prediction_diagnostics_by_origin,
        channel_share_summary=channel_share_summary,
    )
    try:
        plot_forecast(train_raw, submission, cfg=cfg)
    except Exception as exc:
        print(f"Plot skipped ({exc})")

    timings["total_seconds"] = round(time.perf_counter() - run_start, 2)
    with open(os.path.join(cfg.output_dir, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    print(f"\nSaved canonical submission: {canonical_model}/{canonical_strategy}")
    print(f"Total runtime: {timings['total_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
