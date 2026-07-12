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
6. Recursively forecast the 7 test days one day at a time, feeding each
   day's prediction back into the lag features of the next day. This is
   what a genuine multi-step forecast requires; a single frozen snapshot
   of "last known" lags (as in a naive one-shot approach) would make every
   day of the 7-day horizon look identical to the model.
7. Write submission.csv / submission.parquet (+ cv_results.csv, a plot).

Run:   uv run python ml/pipeline.py   (run from the repo root)
Tests: uv run pytest tests/

This file is the CV/training/export orchestrator. Each model's own
train/predict definition lives under `models/` (`models/neural_net.py`,
`models/xgboost_model.py`, `models/lightgbm_model.py`,
`models/naive_baselines.py`); shared feature engineering, the generic
recursive-forecast engine and metrics live in `framework.py`.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
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
    compute_baseline,
    compute_metrics,
    feature_columns,
    init_history,
    load_raw,
    order_models,
    prepare_features,
)
from models.naive_baselines import moving_average_predict, seasonal_naive_predict
from models.neural_net import DEVICE, make_tensors, recursive_forecast, train_model

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
def run_tree_baselines(train_examples: pd.DataFrame, static_eval: pd.DataFrame,
                        history_seed: dict, cfg: Config = CFG,
                        models: tuple = ("XGBoost", "LightGBM")) -> dict:
    """Train + recursively forecast XGBoost/LightGBM in a fresh subprocess
    (never imports torch) and return {model_name: predictions}. Isolating
    them like this avoids a real crash: PyTorch and XGBoost/LightGBM each
    bundle a different copy of the LLVM OpenMP runtime on macOS, and loading
    both in one process segfaults as soon as either runs its native code.
    """
    job = {
        "cfg": asdict(cfg),
        "train_examples": train_examples,
        "static_eval": static_eval,
        "history": history_seed,
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


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------
def run_walk_forward_cv(hist_df: pd.DataFrame, origins, origin_type: str,
                         cfg: Config = CFG) -> pd.DataFrame:
    """Evaluate at each `origin` date (the last training day): trains only
    on data up to and including `origin` (no leakage) and forecasts the
    next `cfg.horizon` days recursively, exactly mirroring the real
    deployment scenario.

    Trains the SAME `cfg.seeds`-sized NN ensemble, fed forward jointly
    (shared history) via `recursive_forecast`, as `run_final_forecast` --
    CV must score the actual estimator being submitted, not a cheaper
    single-seed stand-in (a real mismatch this function used to have: it
    trained one seed while the submission averages three). `cv_epochs` vs
    `final_epochs` remains a deliberate, disclosed compute/accuracy
    trade-off (cheaper proxy training while iterating; the one-time final
    artifact trains longer) -- unlike the seed count, that's not a hidden
    inconsistency, since it's applied identically across every model/fold.

    Returns row-level out-of-fold predictions -- one row per (origin,
    product, date), with per-seed NN columns alongside the ensemble and
    every baseline -- rather than only aggregated metrics, so later
    diagnostics (per-horizon, per-product, paired comparisons, ensemble
    weight fitting) don't require rerunning the CV.
    """
    max_window = max(cfg.lag_windows)
    first_seen = hist_df.groupby("ProductId")["DateKey"].min()  # static historical fact, always in the past
    fold_frames = []

    for origin in origins:
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
        available_mask = fold_train_feat["ProductAvailable"].fillna(False)
        train_examples = (fold_train_feat[available_mask]
                           .dropna(subset=feature_columns(cfg)).reset_index(drop=True))

        scaler = StandardScaler()
        tensors = make_tensors(train_examples, scaler, fit=True, cfg=cfg)
        y_log = np.log1p(train_examples["Quantity"].to_numpy(dtype=np.float32))
        seed_models = [train_model(tensors, y_log, cfg, epochs=cfg.cv_epochs, seed=seed)
                       for seed in cfg.seeds]

        fold_eval_feat = prepare_features(fold_eval_raw, price_ref, first_seen).reset_index(drop=True)

        # Each recursion gets its own fresh copy of history: recursion
        # mutates it in place with that run's own predictions, so sharing
        # one dict across runs would leak one run's forecasts into
        # another's lags. Per-seed recursions are independent (own history
        # each) and kept only as diagnostics; the ensemble recursion below
        # is the one that matters -- it jointly feeds the *ensemble's* daily
        # prediction forward, exactly like `run_final_forecast`'s submission.
        seed_preds = {
            seed: recursive_forecast([model], scaler, fold_eval_feat,
                                      init_history(fold_train_feat, max_window), cfg)
            for seed, model in zip(cfg.seeds, seed_models)
        }
        ensemble_preds = recursive_forecast(seed_models, scaler, fold_eval_feat,
                                             init_history(fold_train_feat, max_window), cfg)
        tree_preds = run_tree_baselines(train_examples, fold_eval_feat,
                                         init_history(fold_train_feat, max_window), cfg)

        seasonal_pred = seasonal_naive_predict(fold_eval_feat, fold_train_raw, lag_days=cfg.horizon)
        ma_pred = moving_average_predict(fold_eval_feat, fold_train_raw, window=28)
        baseline_pred = compute_baseline(fold_eval_feat, fold_train_raw)
        actual = fold_eval_feat["Quantity"].to_numpy(dtype=float)

        fold_oof = pd.DataFrame({
            "origin": origin,
            "origin_type": origin_type,
            "horizon": (fold_eval_feat["DateKey"] - origin).dt.days.to_numpy(),
            "ProductId": fold_eval_feat["ProductId"].to_numpy(),
            "DateKey": fold_eval_feat["DateKey"].to_numpy(),
            "ProductAvailable": fold_eval_feat["ProductAvailable"].to_numpy(),
            "actual": actual,
            "baseline": baseline_pred,
            "pred_NeuralNet": ensemble_preds,
            "pred_XGBoost": tree_preds["XGBoost"],
            "pred_LightGBM": tree_preds["LightGBM"],
            "pred_SeasonalNaive": seasonal_pred,
            "pred_MovingAvg28": ma_pred,
        })
        for seed in cfg.seeds:
            fold_oof[f"pred_NeuralNet_seed{seed}"] = seed_preds[seed]
        fold_frames.append(fold_oof)

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
def run_final_forecast(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
                        cfg: Config = CFG):
    max_window = max(cfg.lag_windows)
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = train_raw.groupby("ProductId")["DateKey"].min()

    train_feat = prepare_features(train_raw, price_ref, first_seen)
    train_feat = add_train_lags(train_feat, cfg.lag_windows)
    available_mask = train_feat["ProductAvailable"].fillna(False)
    train_examples = (train_feat[available_mask]
                       .dropna(subset=feature_columns(cfg)).reset_index(drop=True))

    scaler = StandardScaler()
    tensors = make_tensors(train_examples, scaler, fit=True, cfg=cfg)
    y_log = np.log1p(train_examples["Quantity"].to_numpy(dtype=np.float32))

    models = []
    for seed in cfg.seeds:
        print(f"    seed {seed}")
        models.append(train_model(tensors, y_log, cfg, epochs=cfg.final_epochs, seed=seed))

    history = init_history(train_feat, max_window)
    test_feat = prepare_features(test_raw, price_ref, first_seen).reset_index(drop=True)
    preds = recursive_forecast(models, scaler, test_feat, history, cfg)

    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(preds).astype(int)
    return submission, preds


def run_final_tree_forecast(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG) -> dict:
    """XGBoost/LightGBM trained on ALL history and forecast for the actual
    test week -- purely for dashboard comparison parity with the NN's page.
    Not used for the submission file (the task brief asked for a non-tree
    approach as the actual deliverable).
    """
    max_window = max(cfg.lag_windows)
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = train_raw.groupby("ProductId")["DateKey"].min()

    train_feat = prepare_features(train_raw, price_ref, first_seen)
    train_feat = add_train_lags(train_feat, cfg.lag_windows)
    available_mask = train_feat["ProductAvailable"].fillna(False)
    train_examples = (train_feat[available_mask]
                       .dropna(subset=feature_columns(cfg)).reset_index(drop=True))

    test_feat = prepare_features(test_raw, price_ref, first_seen).reset_index(drop=True)
    history = init_history(train_feat, max_window)
    return run_tree_baselines(train_examples, test_feat, history, cfg)


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

    train_raw, test_raw = load_raw(cfg)
    cfg.num_products = int(max(train_raw["ProductId"].max(), test_raw["ProductId"].max()))
    holdout_origins = recent_holdout_origins(train_raw, cfg)

    print(f"\n=== Development CV ({len(DEVELOPMENT_ORIGINS)} scattered origins x {cfg.horizon}d) ===")
    dev_oof = run_walk_forward_cv(train_raw, DEVELOPMENT_ORIGINS, "development", cfg)

    print(f"\n=== Recent-holdout CV ({len(holdout_origins)} folds x {cfg.horizon}d, untouched final check) ===")
    holdout_oof = run_walk_forward_cv(train_raw, holdout_origins, "recent_holdout", cfg)

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
    submission, nn_preds = run_final_forecast(train_raw, test_raw, cfg)

    print("\n=== Training final XGBoost/LightGBM (dashboard comparison only) ===")
    tree_final = run_final_tree_forecast(train_raw, test_raw, cfg)
    naive_final = run_final_naive_baselines(train_raw, test_raw, cfg)
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

    try:
        plot_forecast(train_raw, submission, cfg=cfg)
    except Exception as exc:  # pragma: no cover - plotting is a nice-to-have
        print(f"Plot skipped ({exc})")

    export_results_json(train_raw, test_raw, submission, final_forecasts, cv_results, cfg,
                         dev_summary=dev_summary, holdout_summary=holdout_summary)


if __name__ == "__main__":
    main()
