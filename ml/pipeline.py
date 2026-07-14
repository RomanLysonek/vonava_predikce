"""
Notino Quantity Prediction - Interview Assignment
====================================================
Forecast total Quantity (QuantityApp + QuantityWeb) for 30 products over the
7 days following the training window. Approach: PyTorch feed-forward network
with product & campaign embeddings (non-tree-based, per the task brief),
benchmarked against XGBoost/LightGBM and two naive baselines.

Pipeline
--------
1. Load train/test parquet files.
2. Engineer calendar, campaign and price features shared by train/test.
3. Compute per-product rolling lag statistics on the chronological series.
4. Walk-forward (rolling-origin) cross-validation over several historical
   7-day windows, benchmarked against XGBoost, LightGBM (the "standard"
   approach the task brief contrasts against) and two naive baselines
   (seasonal-naive, moving-average), to get an honest, leakage-free
   comparison. XGBoost/LightGBM run in a separate subprocess -- see
   `tree_worker.py` for why.
5. Train the final ensemble (multiple seeds) on all available history.
6. Predict all 7 test days directly, in a single pass, from the stacked
   (ForecastOrigin x Horizon x ProductId) panel built by
   `framework.build_direct_panel` -- every horizon's inputs (origin-relative
   rolling lags, target-relative seasonal lags) are already lookups into
   observed data, never a value that would first need to be predicted, so
   there's no recursive feedback loop and no risk of the old one-shot
   shortcut (freezing lags at the last known value) making every day of
   the 7-day horizon look identical to the model.
7. Write submission.csv / submission.parquet (+ cv_results.csv, a plot).

Run:   uv run python ml/pipeline.py   (run from the repo root)
Tests: uv run pytest tests/

This file is the CV/training/export orchestrator. Each model's own
train/predict definition lives under `models/` (`models/neural_net.py`,
`models/xgboost_model.py`, `models/lightgbm_model.py`,
`models/naive_baselines.py`); shared feature engineering, the direct
multi-horizon panel builder and metrics live in `framework.py`.
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
from sklearn.preprocessing import StandardScaler

from framework import (
    CFG,
    MODEL_META,
    MODEL_ORDER,
    MODEL_SLUGS,
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
    order_models,
    prepare_features,
)
from models.naive_baselines import moving_average_predict, seasonal_naive_predict
from models.neural_net import (
    DEVICE, effective_learning_rate, make_tensors, nn_performance_signature,
    predict_direct, residual_log1p_target, resolve_training_backend, train_model,
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
    resume: bool = False
    reset_checkpoints: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    nn_batch_size: str = "auto"
    nn_lr_scaling: str = "auto"
    nn_training_backend: str = "auto"
    nn_benchmark_file: str = "outputs/nn_batch_benchmark.json"


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
    args = parser.parse_args(argv)
    if args.nn_batch_size != "auto":
        try:
            parsed_batch_size = int(args.nn_batch_size)
        except ValueError as exc:
            parser.error("--nn-batch-size must be 'auto' or a positive integer")
        if parsed_batch_size < 2:
            parser.error("--nn-batch-size must be at least 2")
        args.nn_batch_size = str(parsed_batch_size)
    return RuntimeOptions(
        forecast_strategy=ForecastStrategy(args.forecast_strategy),
        primary_strategy=PrimaryStrategy(args.primary_strategy),
        submission_model=SubmissionModel(args.submission_model),
        selection_metric=args.selection_metric,
        resume=args.resume,
        reset_checkpoints=args.reset_checkpoints,
        checkpoint_dir=args.checkpoint_dir,
        nn_batch_size=args.nn_batch_size,
        nn_lr_scaling=args.nn_lr_scaling,
        nn_training_backend=args.nn_training_backend,
        nn_benchmark_file=args.nn_benchmark_file,
    )


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

CHECKPOINT_SCHEMA_VERSION = "direct-recursive-v3-ridge-stability"


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
    if actual == expected:
        return True
    for key in ("schema_version", "strategy", "origin_type", "origin"):
        if actual.get(key) != expected.get(key):
            return False
    actual_cfg = dict(actual.get("cfg") or {})
    expected_cfg = dict(expected.get("cfg") or {})
    # Backward compatibility with stability-v3 checkpoints created before the
    # performance-only fields existed.  Reuse is valid only when every old
    # semantic field still matches and the effective batch/LR policy is the
    # same as the historical fixed-LR setup.
    for key, value in actual_cfg.items():
        if expected_cfg.get(key) != value:
            return False
    if "reference_batch_size" not in actual_cfg:
        old_batch = int(actual_cfg.get("batch_size", 512))
        if int(expected_cfg.get("batch_size", 512)) != old_batch:
            return False
        base_lr = float(actual_cfg.get("lr", 1e-3))
        current_batch = int(expected_cfg.get("batch_size", old_batch))
        reference = int(expected_cfg.get("reference_batch_size", old_batch))
        scaling = expected_cfg.get("nn_lr_scaling", "fixed")
        ratio = current_batch / reference
        factor = {"fixed": 1.0, "sqrt": ratio ** 0.5, "linear": ratio}.get(scaling)
        if factor is None or not np.isclose(base_lr * factor, base_lr):
            return False
    return True


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
    models: tuple = ("XGBoost", "LightGBM", "DynamicRidge"),
    *,
    strategy: str = "direct",
    eval_panel: pd.DataFrame | None = None,
    history_raw: pd.DataFrame | None = None,
    future_covariates: pd.DataFrame | None = None,
    price_ref: pd.Series | None = None,
    first_seen: pd.Series | None = None,
) -> dict:
    """Train and predict structured models in the native-library worker."""
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
        job.update({
            "history_raw": history_raw,
            "future_covariates": sanitize_future_covariates(future_covariates),
            "price_ref": price_ref,
            "first_seen": first_seen,
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
    first_seen = hist_df.groupby("ProductId")["DateKey"].min()  # static historical fact, always in the past
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

        fold_train_feat = prepare_features(fold_train_raw, price_ref, first_seen)
        fold_train_feat = add_train_lags(fold_train_feat, cfg.lag_windows)
        fold_eval_feat = prepare_features(fold_eval_raw, price_ref, first_seen).reset_index(drop=True)

        panel = build_direct_panel(fold_train_feat, horizons, cfg=cfg, future_covariates=fold_eval_feat)
        # Leakage-safe training slice: a training row's own target must
        # already be observable as of `origin` -- an origin close to the
        # fold's own cutoff combined with a large horizon would otherwise
        # land on a target date this fold isn't allowed to have seen yet.
        trainable = panel[panel["TargetDateKey"] <= origin]
        train_available = trainable["TargetProductAvailable"].fillna(False)
        train_panel = (trainable[train_available]
                        .dropna(subset=direct_panel_feature_names(cfg)).reset_index(drop=True))
        eval_panel = panel[panel["OriginDateKey"] == origin].reset_index(drop=True)

        scaler = StandardScaler()
        tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
        y_residual = residual_log1p_target(train_panel)
        nn_start = time.perf_counter()
        seed_models = [train_model(tensors, y_residual, cfg, epochs=cfg.cv_epochs, seed=seed)
                       for seed in cfg.seeds]
        nn_seconds = time.perf_counter() - nn_start

        # Per-seed predictions are independent diagnostics; the ensemble
        # prediction below (averaging all seeds) is the one that matters --
        # it's exactly what `run_final_forecast` submits.
        seed_preds = {seed: predict_direct([model], scaler, eval_panel, cfg)
                      for seed, model in zip(cfg.seeds, seed_models)}
        ensemble_preds = predict_direct(seed_models, scaler, eval_panel, cfg)
        tree_start = time.perf_counter()
        tree_preds = run_tree_baselines(train_panel, eval_panel, cfg)
        tree_seconds = time.perf_counter() - tree_start

        seasonal_pred = seasonal_naive_predict(fold_eval_feat, fold_train_raw, lag_days=cfg.horizon)
        ma_pred = moving_average_predict(fold_eval_feat, fold_train_raw, window=28)
        baseline_pred = compute_baseline(fold_eval_feat, fold_train_raw)

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
        fold_oof["pred_NeuralNet"] = ensemble_preds
        fold_oof["pred_XGBoost"] = tree_preds["XGBoost"]
        fold_oof["pred_LightGBM"] = tree_preds["LightGBM"]
        fold_oof["pred_DynamicRidge"] = tree_preds["DynamicRidge"]
        for seed in cfg.seeds:
            fold_oof[f"pred_NeuralNet_seed{seed}"] = seed_preds[seed]
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
    cfg: Config,
) -> pd.DataFrame:
    panel = build_one_step_panel(fold_train_raw, price_ref, first_seen, cfg)
    cutoff = fold_train_raw["DateKey"].max()
    trainable = panel[panel["TargetDateKey"].le(cutoff)]
    available = trainable["TargetProductAvailable"].fillna(False)
    return trainable[available].dropna(subset=direct_panel_feature_names(cfg)).reset_index(drop=True)


def _recursive_nn_predictions(
    train_panel: pd.DataFrame,
    history_raw: pd.DataFrame,
    future_covariates: pd.DataFrame,
    price_ref: pd.Series,
    first_seen: pd.Series,
    cfg: Config,
    epochs: int,
):
    scaler = StandardScaler()
    tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
    y_residual = residual_log1p_target(train_panel)
    seed_models = [train_model(tensors, y_residual, cfg, epochs=epochs, seed=seed) for seed in cfg.seeds]

    seed_paths = {}
    # Diagnostics: each seed gets its own path. The deployed ensemble path below
    # feeds back the natural-scale ensemble mean, as required.
    for seed, model in zip(cfg.seeds, seed_models):
        seed_paths[seed] = forecast_recursive(
            history_raw, future_covariates,
            lambda panel, model=model: predict_direct([model], scaler, panel, cfg),
            price_ref, first_seen, cfg,
        )
    ensemble_path = forecast_recursive(
        history_raw, future_covariates,
        lambda panel: predict_direct(seed_models, scaler, panel, cfg),
        price_ref, first_seen, cfg,
    )
    return ensemble_path, seed_paths, scaler, seed_models


def run_walk_forward_cv_recursive(
    hist_df: pd.DataFrame, origins, origin_type: str, cfg: Config = CFG,
    timings: list[dict] | None = None, *, checkpoint_dir: str | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """One-step training plus genuine recursive seven-day inference."""
    first_seen = hist_df.groupby("ProductId")["DateKey"].min()
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
        train_panel = _recursive_panel_training_data(fold_train_raw, price_ref, first_seen, cfg)
        future_covariates = sanitize_future_covariates(fold_eval_raw)

        nn_start = time.perf_counter()
        ensemble_path, seed_paths, _, _ = _recursive_nn_predictions(
            train_panel, fold_train_raw, future_covariates, price_ref, first_seen, cfg, cfg.cv_epochs
        )
        nn_seconds = time.perf_counter() - nn_start

        tree_start = time.perf_counter()
        structured = run_structured_models(
            train_panel, cfg, strategy="recursive", history_raw=fold_train_raw,
            future_covariates=future_covariates, price_ref=price_ref, first_seen=first_seen,
        )
        tree_seconds = time.perf_counter() - tree_start

        eval_feat = prepare_features(fold_eval_raw, price_ref, first_seen).reset_index(drop=True)
        naive_df = fold_eval_raw[["ProductId", "DateKey", "Quantity", "ProductAvailable"]].copy()
        naive_df["baseline"] = compute_baseline(eval_feat, fold_train_raw)
        naive_df["pred_SeasonalNaive"] = seasonal_naive_predict(eval_feat, fold_train_raw, lag_days=cfg.horizon)
        naive_df["pred_MovingAvg28"] = moving_average_predict(eval_feat, fold_train_raw, window=28)

        fold_oof = ensemble_path.rename(columns={
            "TargetDateKey": "DateKey", "forecast_horizon": "horizon", "prediction": "pred_NeuralNet"
        })[["ProductId", "DateKey", "horizon", "pred_NeuralNet", "fallback_used"]]
        fold_oof = fold_oof.rename(columns={"fallback_used": "fallback_NeuralNet"})
        fold_oof["origin"] = origin
        fold_oof["origin_type"] = origin_type
        fold_oof["strategy"] = "recursive"
        for seed, path in seed_paths.items():
            seed_col = path[["ProductId", "TargetDateKey", "prediction"]].rename(
                columns={"TargetDateKey": "DateKey", "prediction": f"pred_NeuralNet_seed{seed}"}
            )
            fold_oof = fold_oof.merge(seed_col, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
        for name, payload in structured.items():
            path = pd.DataFrame(payload)
            pred_col = path[["ProductId", "TargetDateKey", "prediction", "fallback_used"]].rename(
                columns={"TargetDateKey": "DateKey", "prediction": f"pred_{name}", "fallback_used": f"fallback_{name}"}
            )
            fold_oof = fold_oof.merge(pred_col, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
        fold_oof = fold_oof.merge(naive_df, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
        fold_oof = fold_oof.rename(columns={"Quantity": "actual"})
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
    pred_cols = list(pred_columns.values())
    
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
        summary = summarize_oof(group, pred_columns)
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


def select_primary_strategy(dev_summary: pd.DataFrame, *, model: str, metric: str) -> str:
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
    pred_cols = [c for c in pred_columns.values() if c in oof.columns]

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
    first_seen = train_raw.groupby("ProductId")["DateKey"].min()

    train_feat = prepare_features(train_raw, price_ref, first_seen)
    train_feat = add_train_lags(train_feat, cfg.lag_windows)
    test_feat = prepare_features(test_raw, price_ref, first_seen).reset_index(drop=True)

    horizons = range(1, cfg.horizon + 1)
    panel = build_direct_panel(train_feat, horizons, cfg=cfg, future_covariates=test_feat)

    last_train_date = train_raw["DateKey"].max()
    trainable = panel[panel["TargetDateKey"] <= last_train_date]
    train_available = trainable["TargetProductAvailable"].fillna(False)
    train_panel = (trainable[train_available]
                    .dropna(subset=direct_panel_feature_names(cfg)).reset_index(drop=True))
    eval_panel = panel[panel["OriginDateKey"] == last_train_date].reset_index(drop=True)
    return train_panel, eval_panel


def run_final_forecast_direct(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
                        cfg: Config = CFG):
    train_panel, eval_panel = _prepare_final_direct_panel(train_raw, test_raw, cfg)

    scaler = StandardScaler()
    tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
    y_residual = residual_log1p_target(train_panel)

    models = []
    for seed in cfg.seeds:
        seed_start = time.perf_counter()
        print(f"    seed {seed}")
        models.append(train_model(tensors, y_residual, cfg, epochs=cfg.final_epochs, seed=seed))
        print(f"      [timing] seed {seed}: {time.perf_counter() - seed_start:.1f}s")

    preds = predict_direct(models, scaler, eval_panel, cfg)
    preds_aligned = _reindex_predictions(eval_panel, preds, "TargetDateKey", test_raw)

    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(preds_aligned).astype(int)
    return submission, preds_aligned


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


def run_final_forecast_recursive(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG):
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = train_raw.groupby("ProductId")["DateKey"].min()
    train_panel = _recursive_panel_training_data(train_raw, price_ref, first_seen, cfg)
    future = sanitize_future_covariates(test_raw)
    path, _, _, _ = _recursive_nn_predictions(
        train_panel, train_raw, future, price_ref, first_seen, cfg, cfg.final_epochs
    )
    preds = _align_recursive_path(path, test_raw)
    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(preds).astype(int)
    return submission, preds, path


def run_final_structured_forecast_recursive(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
                                            cfg: Config = CFG) -> tuple[dict, dict]:
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = train_raw.groupby("ProductId")["DateKey"].min()
    train_panel = _recursive_panel_training_data(train_raw, price_ref, first_seen, cfg)
    payloads = run_structured_models(
        train_panel, cfg, strategy="recursive", history_raw=train_raw,
        future_covariates=test_raw, price_ref=price_ref, first_seen=first_seen,
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
    """Recursively replace non-finite floats (NaN/Inf) with None.

    Calendar-gap reindexing (`reindex_daily_calendar`) and zero-actual
    slices in WAPE/BiasRatio legitimately produce NaN, and `json.dump`
    silently allows non-standard literal NaN/Infinity tokens by default --
    which then breaks the FIRST spec-compliant consumer downstream
    (Starlette's `JSONResponse` in `webapp/server.py`, and any browser's
    `JSON.parse`). Applied once here, at the JSON write boundary, so every
    field is safe regardless of which pipeline stage produced the NaN.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
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
                         strategy_by_horizon: pd.DataFrame | None = None) -> dict:
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

    models_meta = [
        {"key": name, "slug": MODEL_SLUGS[name], "skill_vs_seasonal_naive": skill_by_model.get(name), **MODEL_META[name]}
        for name in MODEL_ORDER if name in final_forecasts
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "forecast_strategy": runtime_options.forecast_strategy.value if runtime_options else "direct",
            "primary_strategy": canonical_strategy,
            "submission_model": runtime_options.submission_model.value if runtime_options else "NeuralNet",
            "selection_metric": runtime_options.selection_metric if runtime_options else "WAPE",
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
            "nn_batch_size": cfg.batch_size,
            "nn_reference_batch_size": cfg.reference_batch_size,
            "nn_lr_scaling": cfg.nn_lr_scaling,
            "nn_effective_learning_rate": effective_learning_rate(cfg),
            "nn_training_backend": resolve_training_backend(cfg),
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
        "selection": {
            "canonical_model": canonical_model,
            "canonical_strategy": canonical_strategy,
            "selected_from": "development",
            "recent_benchmark_confirmation": None,
        },
    }

    payload = _json_safe(payload)
    out_path = path or os.path.join(cfg.output_dir, "results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _choose_canonical_model_strategy(
    options: RuntimeOptions,
    dev_summary: pd.DataFrame,
) -> tuple[str, str]:
    strategies = set(dev_summary["strategy"].unique())
    requested_model = options.submission_model.value
    if options.submission_model is SubmissionModel.AUTO:
        candidates = dev_summary[
            dev_summary["evaluation_regime"].eq("conditional")
            & dev_summary["comparison_population"].eq("common")
            & dev_summary["aggregation"].eq("global")
            & dev_summary["model"].isin(["NeuralNet", "DynamicRidge", "XGBoost", "LightGBM"])
        ]
        row = candidates.sort_values(options.selection_metric).iloc[0]
        return str(row["model"]), str(row["strategy"])
    if len(strategies) == 1:
        return requested_model, next(iter(strategies))
    if options.primary_strategy is PrimaryStrategy.AUTO:
        return requested_model, select_primary_strategy(
            dev_summary, model=requested_model, metric=options.selection_metric
        )
    return requested_model, options.primary_strategy.value


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
    nn_runtime = configure_nn_runtime(cfg, options)
    print(f"Device: {DEVICE}")
    print(f"Forecast strategy: {options.forecast_strategy.value}")
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
    timings: dict = {"cv_folds": [], "nn_runtime": nn_runtime}

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
    pair_summary = summarize_strategy_pairs(dev_oof, evaluation_regime="conditional")

    canonical_model, canonical_strategy = _choose_canonical_model_strategy(options, dev_summary)
    print(f"\nCanonical selection: {canonical_model} / {canonical_strategy}")

    final_by_strategy: dict[str, dict[str, np.ndarray]] = {}
    submissions_by_strategy: dict[str, pd.DataFrame] = {}
    raw_rows = []
    naive_final = run_final_naive_baselines(train_raw, test_raw, cfg)
    for strategy in strategies:
        print(f"\n=== Final {strategy.value} forecasts ===")
        if strategy is ForecastStrategy.DIRECT:
            nn_submission, nn_preds = run_final_forecast_direct(train_raw, test_raw, cfg)
            structured = run_final_tree_forecast_direct(train_raw, test_raw, cfg)
            paths = {}
        else:
            nn_submission, nn_preds, nn_path = run_final_forecast_recursive(train_raw, test_raw, cfg)
            structured, paths = run_final_structured_forecast_recursive(train_raw, test_raw, cfg)
            paths["NeuralNet"] = nn_path
        forecasts = {"NeuralNet": nn_preds, **structured, **naive_final}
        final_by_strategy[strategy.value] = forecasts
        submissions_by_strategy[strategy.value] = nn_submission
        for model, preds in forecasts.items():
            for (pid, date), pred in zip(
                test_raw[["ProductId", "DateKey"]].itertuples(index=False, name=None),
                np.asarray(preds, dtype=float),
            ):
                raw_rows.append({
                    "strategy": strategy.value, "model": model,
                    "ProductId": pid, "DateKey": date,
                    "prediction_raw": float(pred),
                    "prediction_submission": int(round(max(float(pred), 0.0))),
                    "fallback_used": False,
                })
        for model, path in paths.items():
            fallback_map = path.set_index(["ProductId", "TargetDateKey"])["fallback_used"]
            for row in raw_rows:
                if row["strategy"] == strategy.value and row["model"] == model:
                    row["fallback_used"] = bool(fallback_map.get((row["ProductId"], row["DateKey"]), False))

    canonical_preds = final_by_strategy[canonical_strategy][canonical_model]
    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(np.clip(canonical_preds, 0, None)).astype(int)

    os.makedirs(cfg.output_dir, exist_ok=True)
    submission.to_parquet(os.path.join(cfg.output_dir, "submission.parquet"), index=False)
    submission.to_csv(os.path.join(cfg.output_dir, "submission.csv"), index=False)
    for strategy, forecasts in final_by_strategy.items():
        strategy_submission = test_raw[["ProductId", "DateKey"]].copy()
        strategy_submission["Quantity"] = np.round(np.clip(forecasts[canonical_model], 0, None)).astype(int)
        strategy_submission.to_csv(os.path.join(cfg.output_dir, f"submission_{strategy}.csv"), index=False)
        strategy_submission.to_parquet(os.path.join(cfg.output_dir, f"submission_{strategy}.parquet"), index=False)

    final_forecast_df = pd.DataFrame(raw_rows)
    final_forecast_df.to_parquet(os.path.join(cfg.output_dir, "final_forecasts.parquet"), index=False)
    oof.to_parquet(os.path.join(cfg.output_dir, "oof_predictions.parquet"), index=False)
    dev_summary.to_csv(os.path.join(cfg.output_dir, "dev_summary.csv"), index=False)
    benchmark_summary.to_csv(os.path.join(cfg.output_dir, "benchmark_summary.csv"), index=False)
    pair_summary.to_csv(os.path.join(cfg.output_dir, "strategy_pair_summary.csv"), index=False)

    by_horizon_frames = []
    # Strategy horizon curves are a development-OOF diagnostic. Keeping the
    # recent benchmark out prevents the presentation layer from implying it
    # participated in strategy selection.
    for strategy, group in dev_oof.groupby("strategy"):
        for horizon, hgroup in group.groupby("horizon"):
            summary = summarize_oof(hgroup)
            summary["strategy"] = strategy
            summary["horizon"] = horizon
            summary["origin_type"] = "development"
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
