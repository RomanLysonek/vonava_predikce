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

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict
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
    compute_baseline,
    compute_metrics,
    direct_panel_feature_names,
    load_raw,
    order_models,
    prepare_features,
)
from models.naive_baselines import moving_average_predict, seasonal_naive_predict
from models.neural_net import DEVICE, make_tensors, predict_direct, residual_log1p_target, train_model

np.random.seed(CFG.seed)

TREE_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_worker.py")

# Broader, seasonally-scattered origins used to make modeling/feature
# decisions (spring/summer lulls, several Januaries, Black Friday windows,
# pre/post-Christmas, a Valentine's-adjacent week -- relevant for a
# cosmetics retailer). Deliberately disjoint from `recent_holdout_origins`
# below: these are for iteration, the holdout is for a final check only,
# and mixing the two would let repeated tuning quietly overfit to the
# holdout the same way a single reused test set would.
DEVELOPMENT_ORIGINS = pd.to_datetime([
    "2022-02-01", "2022-06-15", "2022-11-20",
    "2023-01-10", "2023-07-01", "2023-11-24", "2023-12-18",
    "2024-02-14", "2024-06-20", "2024-11-29", "2024-12-20",
    "2025-02-10",
])


def recent_holdout_origins(hist_df: pd.DataFrame, cfg: Config = CFG) -> pd.DatetimeIndex:
    """Last `cfg.n_cv_folds` non-overlapping `cfg.horizon`-day origins ending
    at the most recent training data -- the closest pseudo-test periods to
    the actual forecast. Meant as a final model-selection check, not
    something to repeatedly re-tune against."""
    max_date = hist_df["DateKey"].max()
    return pd.DatetimeIndex([max_date - pd.Timedelta(days=(i + 1) * cfg.horizon) for i in range(cfg.n_cv_folds)])


# ---------------------------------------------------------------------------
# XGBoost / LightGBM baselines, run out-of-process (see tree_worker.py)
# ---------------------------------------------------------------------------
def run_tree_baselines(train_panel: pd.DataFrame, eval_panel: pd.DataFrame, cfg: Config = CFG,
                        models: tuple = ("XGBoost", "LightGBM")) -> dict:
    """Train + predict XGBoost/LightGBM directly on the multi-horizon panel
    (see `framework.build_direct_panel`) in a fresh subprocess (never
    imports torch), and return {model_name: predictions}, aligned with
    `eval_panel`'s own row order. Isolating them like this avoids a real
    crash: PyTorch and XGBoost/LightGBM each bundle a different copy of the
    LLVM OpenMP runtime on macOS, and loading both in one process segfaults
    as soon as either runs its native code.
    """
    job = {
        "cfg": asdict(cfg),
        "train_panel": train_panel,
        "eval_panel": eval_panel,
        "models": list(models),
    }
    with tempfile.TemporaryDirectory() as tmp:
        job_path = os.path.join(tmp, "job.pkl")
        out_path = os.path.join(tmp, "out.pkl")
        with open(job_path, "wb") as f:
            pickle.dump(job, f)

        result = subprocess.run(
            [sys.executable, TREE_WORKER_PATH, job_path, out_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tree_worker subprocess failed (exit {result.returncode}):\n{result.stderr}"
            )
        with open(out_path, "rb") as f:
            results = pickle.load(f)
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
def run_walk_forward_cv(hist_df: pd.DataFrame, origins, origin_type: str,
                         cfg: Config = CFG, timings: list[dict] | None = None) -> pd.DataFrame:
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
        fold_oof["pred_NeuralNet"] = ensemble_preds
        fold_oof["pred_XGBoost"] = tree_preds["XGBoost"]
        fold_oof["pred_LightGBM"] = tree_preds["LightGBM"]
        for seed in cfg.seeds:
            fold_oof[f"pred_NeuralNet_seed{seed}"] = seed_preds[seed]
        fold_oof = fold_oof.merge(naive_df, on=["ProductId", "DateKey"], how="left")
        fold_oof = fold_oof.rename(columns={"Quantity": "actual"})
        fold_frames.append(fold_oof)

        fold_seconds = time.perf_counter() - fold_start
        print(f"    [timing] {origin_type} {origin.date()}: NN {nn_seconds:.1f}s | "
              f"trees {tree_seconds:.1f}s | fold total {fold_seconds:.1f}s")
        if timings is not None:
            timings.append({
                "origin_type": origin_type, "origin": str(origin.date()),
                "nn_seconds": round(nn_seconds, 2), "tree_seconds": round(tree_seconds, 2),
                "fold_seconds": round(fold_seconds, 2),
            })

    return pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()


OOF_MODEL_COLUMNS = {
    "NeuralNet": "pred_NeuralNet",
    "XGBoost": "pred_XGBoost",
    "LightGBM": "pred_LightGBM",
    "SeasonalNaive": "pred_SeasonalNaive",
    "MovingAvg28": "pred_MovingAvg28",
}


def summarize_oof(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    """Two labeled aggregations per model, since they are not
    interchangeable: "mean_fold" (macro) averages each origin's own metric
    equally regardless of how many rows it contributed; "global" (micro)
    pools every row across all origins first, then computes the metric
    once. Rows with a NaN prediction for a given model (e.g. a fold where
    that model wasn't run) are dropped for that model only.
    """
    pred_columns = pred_columns or OOF_MODEL_COLUMNS
    rows = []
    for model_name, col in pred_columns.items():
        if col not in oof.columns:
            continue
        valid = oof.dropna(subset=[col])
        if valid.empty:
            continue

        fold_metrics = [compute_metrics(g["actual"], g[col]) for _, g in valid.groupby("origin")]
        mean_fold = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        rows.append({"model": model_name, "aggregation": "mean_fold", "n_folds": len(fold_metrics), **mean_fold})

        global_metrics = compute_metrics(valid["actual"], valid[col])
        rows.append({"model": model_name, "aggregation": "global", "n_folds": len(fold_metrics), **global_metrics})
    return pd.DataFrame(rows)


def oof_to_legacy_cv_results(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    """Reshape row-level OOF predictions back into the older
    fold/model/MAE/RMSE/MAPE(+new metrics) shape the dashboard/export code
    already understands, for a given (single origin_type) OOF slice."""
    pred_columns = pred_columns or OOF_MODEL_COLUMNS
    origins_sorted = sorted(oof["origin"].unique(), reverse=True)  # most recent = fold 0, matching the old numbering
    fold_of_origin = {origin: i for i, origin in enumerate(origins_sorted)}

    rows = []
    for origin, fold_df in oof.groupby("origin"):
        for model_name, col in pred_columns.items():
            if col not in fold_df.columns:
                continue
            valid = fold_df.dropna(subset=[col])
            if valid.empty:
                continue
            rows.append({"fold": fold_of_origin[origin], "model": model_name,
                         **compute_metrics(valid["actual"], valid[col])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final ensemble training + test forecast
# ---------------------------------------------------------------------------
def _prepare_final_panel(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG):
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


def run_final_forecast(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
                        cfg: Config = CFG):
    train_panel, eval_panel = _prepare_final_panel(train_raw, test_raw, cfg)

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


def run_final_tree_forecast(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG) -> dict:
    """XGBoost/LightGBM trained on ALL history and forecast for the actual
    test week -- purely for dashboard comparison parity with the NN's page.
    Not used for the submission file (the task brief asked for a non-tree
    approach as the actual deliverable).
    """
    train_panel, eval_panel = _prepare_final_panel(train_raw, test_raw, cfg)
    tree_preds = run_tree_baselines(train_panel, eval_panel, cfg)
    return {name: _reindex_predictions(eval_panel, preds, "TargetDateKey", test_raw)
            for name, preds in tree_preds.items()}


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


def export_results_json(train_raw: pd.DataFrame, test_raw: pd.DataFrame, submission: pd.DataFrame,
                         final_forecasts: dict, cv_results: pd.DataFrame, cfg: Config = CFG,
                         history_lookback: int = 90, path: str | None = None,
                         dev_summary: pd.DataFrame = None, holdout_summary: pd.DataFrame = None) -> dict:
    """Bundle everything the presentation webapp needs into one JSON file:
    per-fold CV metrics + averages, per-model skill scores, model metadata
    (label/brand color/blurb, for the per-model pages), the full submission
    grid, and shared history + per-model 7-day forecasts for interactive
    charts (one history series per product, one forecast series per
    product per model).
    """
    summary = order_models(cv_results.groupby("model")[["MAE", "RMSE", "MAPE"]]
                            .mean().round(3).reset_index())
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
            "horizon": cfg.horizon,
            "lag_windows": list(cfg.lag_windows),
            "n_cv_folds": cfg.n_cv_folds,
            "n_dev_origins": len(DEVELOPMENT_ORIGINS),
            "cv_epochs": cfg.cv_epochs,
            "final_epochs": cfg.final_epochs,
            "seeds": list(cfg.seeds),
            "num_products": cfg.num_products,
        },
        "models": models_meta,
        "cv_results": order_models(cv_results.round(3)).to_dict(orient="records"),
        "cv_summary": summary.to_dict(orient="records"),
        "skill_vs_seasonal_naive": skill,
        # "recent_holdout": untouched last-N-weeks folds, closest to the real
        # forecast; "development": broader seasonally-scattered folds used to
        # make modeling decisions. Each has both a mean_fold (macro) and a
        # global (micro, pooled) aggregation -- see summarize_oof.
        "holdout_summary": (order_models(holdout_summary.round(3)).to_dict(orient="records")
                             if holdout_summary is not None else None),
        "dev_summary": (order_models(dev_summary.round(3)).to_dict(orient="records")
                         if dev_summary is not None else None),
        "submission": submission.assign(
            DateKey=submission["DateKey"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "history": history,
        "forecasts": forecasts,
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
def main() -> None:
    cfg = CFG
    print(f"Device: {DEVICE}")
    run_start = time.perf_counter()
    # Phase-level + per-fold timings, printed as the run progresses and
    # dumped to outputs/timings.json at the end -- a first step towards
    # accurate wall-clock tracking across runs/machines, not just a single
    # end-to-end number.
    timings: dict = {"cv_folds": []}

    train_raw, test_raw = load_raw(cfg)
    cfg.num_products = int(max(train_raw["ProductId"].max(), test_raw["ProductId"].max()))
    holdout_origins = recent_holdout_origins(train_raw, cfg)

    print(f"\n=== Development CV ({len(DEVELOPMENT_ORIGINS)} scattered origins x {cfg.horizon}d) ===")
    t0 = time.perf_counter()
    dev_oof = run_walk_forward_cv(train_raw, DEVELOPMENT_ORIGINS, "development", cfg, timings=timings["cv_folds"])
    timings["dev_cv_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[timing] development CV total: {timings['dev_cv_seconds']:.1f}s")

    print(f"\n=== Recent-holdout CV ({len(holdout_origins)} folds x {cfg.horizon}d, untouched final check) ===")
    t0 = time.perf_counter()
    holdout_oof = run_walk_forward_cv(train_raw, holdout_origins, "recent_holdout", cfg, timings=timings["cv_folds"])
    timings["holdout_cv_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[timing] recent-holdout CV total: {timings['holdout_cv_seconds']:.1f}s")

    oof = pd.concat([dev_oof, holdout_oof], ignore_index=True)

    print("\nDevelopment summary (mean_fold vs global aggregation, per model):")
    dev_summary = summarize_oof(dev_oof)
    print(order_models(dev_summary.round(3)).to_string(index=False))

    print("\nRecent-holdout summary:")
    holdout_summary = summarize_oof(holdout_oof)
    print(order_models(holdout_summary.round(3)).to_string(index=False))

    # Legacy fold/model shape for the dashboard/export code below -- holdout
    # only, since that's what used to be shown as "cv_results".
    cv_results = oof_to_legacy_cv_results(holdout_oof)
    print("\nRecent-holdout, per fold:")
    print(order_models(cv_results.round(2)).to_string(index=False))

    holdout_global = holdout_summary[holdout_summary["aggregation"] == "global"].set_index("model")
    nn_mae = holdout_global.loc["NeuralNet", "MAE"]
    naive_mae = holdout_global.loc["SeasonalNaive", "MAE"]
    skill = 1 - nn_mae / naive_mae
    print(f"\nSkill vs seasonal-naive baseline (holdout, global MAE): {skill:+.1%} (positive = model beats naive)")

    print("\n=== Training final ensemble on all data (submission model) ===")
    t0 = time.perf_counter()
    submission, nn_preds = run_final_forecast(train_raw, test_raw, cfg)
    timings["final_nn_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[timing] final NN ensemble train+predict: {timings['final_nn_seconds']:.1f}s")

    print("\n=== Training final XGBoost/LightGBM (dashboard comparison only) ===")
    t0 = time.perf_counter()
    tree_final = run_final_tree_forecast(train_raw, test_raw, cfg)
    timings["final_tree_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[timing] final tree train+predict: {timings['final_tree_seconds']:.1f}s")

    t0 = time.perf_counter()
    naive_final = run_final_naive_baselines(train_raw, test_raw, cfg)
    timings["final_naive_seconds"] = round(time.perf_counter() - t0, 2)
    final_forecasts = {
        "NeuralNet": nn_preds,
        "XGBoost": tree_final["XGBoost"],
        "LightGBM": tree_final["LightGBM"],
        "SeasonalNaive": naive_final["SeasonalNaive"],
        "MovingAvg28": naive_final["MovingAvg28"],
    }

    print("\n" + "=" * 60)
    print("SUBMISSION PREVIEW (NeuralNet)")
    print("=" * 60)
    pivot = submission.pivot(index="ProductId", columns="DateKey", values="Quantity")
    pivot.columns = [c.strftime("%Y-%m-%d") for c in pivot.columns]
    print(pivot.to_string())
    print(f"\nTotal rows: {len(submission)} | mean qty: {submission['Quantity'].mean():.1f} "
          f"| min {submission['Quantity'].min()} | max {submission['Quantity'].max()}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    submission.to_parquet(os.path.join(cfg.output_dir, "submission.parquet"), index=False)
    submission.to_csv(os.path.join(cfg.output_dir, "submission.csv"), index=False)
    cv_results.to_csv(os.path.join(cfg.output_dir, "cv_results.csv"), index=False)
    oof.to_parquet(os.path.join(cfg.output_dir, "oof_predictions.parquet"), index=False)
    dev_summary.to_csv(os.path.join(cfg.output_dir, "dev_summary.csv"), index=False)
    holdout_summary.to_csv(os.path.join(cfg.output_dir, "holdout_summary.csv"), index=False)
    print(f"\nSaved: {cfg.output_dir}/submission.{{parquet,csv}}, cv_results.csv, "
          "oof_predictions.parquet, dev_summary.csv, holdout_summary.csv")

    t0 = time.perf_counter()
    try:
        plot_forecast(train_raw, submission, cfg=cfg)
    except Exception as exc:  # pragma: no cover - plotting is a nice-to-have
        print(f"Plot skipped ({exc})")
    timings["plot_seconds"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    export_results_json(train_raw, test_raw, submission, final_forecasts, cv_results, cfg,
                         dev_summary=dev_summary, holdout_summary=holdout_summary)
    timings["export_json_seconds"] = round(time.perf_counter() - t0, 2)

    timings["total_seconds"] = round(time.perf_counter() - run_start, 2)
    print(f"\n[timing] TOTAL pipeline run: {timings['total_seconds']:.1f}s "
          f"({timings['total_seconds'] / 60:.1f} min)")
    with open(os.path.join(cfg.output_dir, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    print(f"Saved: {cfg.output_dir}/timings.json")


if __name__ == "__main__":
    main()
