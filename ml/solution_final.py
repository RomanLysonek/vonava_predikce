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

Run:   uv run python ml/solution_final.py   (run from the repo root)
Tests: uv run pytest tests/
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from features import (
    CFG,
    MODEL_META,
    MODEL_ORDER,
    MODEL_SLUGS,
    NUM_CAMPAIGN_CATS,
    Config,
    add_train_lags,
    compute_metrics,
    feature_columns,
    init_history,
    load_raw,
    moving_average_predict,
    order_models,
    prepare_features,
    recursive_forecast_generic,
    seasonal_naive_predict,
)

np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TREE_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_worker.py")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class QuantityNet(nn.Module):
    def __init__(self, num_numeric: int, cfg: Config):
        super().__init__()
        self.product_emb = nn.Embedding(cfg.num_products, cfg.embed_dim_product)
        self.campaign_emb_web = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.campaign_emb_app = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)

        input_dim = num_numeric + cfg.embed_dim_product + 2 * cfg.embed_dim_campaign
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden, p in zip(cfg.hidden_dims, cfg.dropout):
            layers += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_prod, x_camp_web, x_camp_app):
        emb = torch.cat([
            self.product_emb(x_prod),
            self.campaign_emb_web(x_camp_web),
            self.campaign_emb_app(x_camp_app),
        ], dim=1)
        x = torch.cat([x_num, emb], dim=1)
        return self.net(x).squeeze(-1)


def make_tensors(df: pd.DataFrame, scaler: StandardScaler, fit: bool, cfg: Config = CFG) -> dict[str, torch.Tensor]:
    num = df[feature_columns(cfg)].to_numpy(dtype=np.float32)
    num = scaler.fit_transform(num) if fit else scaler.transform(num)
    return {
        "num": torch.tensor(num, dtype=torch.float32),
        "prod": torch.tensor(df["product_idx"].to_numpy(dtype=np.int64)),
        "cw": torch.tensor(df["campaign_idx_web"].to_numpy(dtype=np.int64)),
        "ca": torch.tensor(df["campaign_idx_app"].to_numpy(dtype=np.int64)),
    }


def train_model(tensors: dict, y_log: np.ndarray, cfg: Config, epochs: int, seed: int) -> QuantityNet:
    """Fixed-epoch training (no early stopping): every fold/final run gets a
    comparable, leakage-free training budget instead of peeking at the
    evaluation window to pick the "best" epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = QuantityNet(len(feature_columns(cfg)), cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.lr * 0.01)
    crit = nn.HuberLoss(delta=1.0)

    ds = TensorDataset(tensors["num"], tensors["prod"], tensors["cw"], tensors["ca"],
                       torch.tensor(y_log, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xn, xp, xcw, xca, yt in dl:
            xn, xp, xcw, xca, yt = (xn.to(DEVICE), xp.to(DEVICE), xcw.to(DEVICE),
                                     xca.to(DEVICE), yt.to(DEVICE))
            pred = model(xn, xp, xcw, xca)
            loss = crit(pred, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xn.size(0)
        sched.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"      epoch {epoch:3d}/{epochs} | train loss {total / len(ds):.4f}")
    model.eval()
    return model


def predict_ensemble(models: list, tensors: dict) -> np.ndarray:
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(tensors["num"].to(DEVICE), tensors["prod"].to(DEVICE),
                   tensors["cw"].to(DEVICE), tensors["ca"].to(DEVICE)).cpu().numpy()
        preds.append(np.expm1(p))
    return np.clip(np.mean(preds, axis=0), 0, None)


def recursive_forecast(models: list, scaler: StandardScaler,
                        static_df: pd.DataFrame, history: dict,
                        cfg: Config = CFG) -> np.ndarray:
    """Neural-net ensemble convenience wrapper around `recursive_forecast_generic`."""
    def predict_fn(day_df: pd.DataFrame) -> np.ndarray:
        tensors = make_tensors(day_df, scaler, fit=False, cfg=cfg)
        return predict_ensemble(models, tensors)

    return recursive_forecast_generic(predict_fn, static_df, history, cfg)


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
def run_walk_forward_cv(hist_df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Evaluate on the last `n_cv_folds` non-overlapping 7-day blocks. Each
    fold trains only on data strictly before the eval block (no leakage) and
    forecasts recursively, exactly mirroring the real deployment scenario.
    """
    max_window = max(cfg.lag_windows)
    max_date = hist_df["DateKey"].max()
    first_seen = hist_df.groupby("ProductId")["DateKey"].min()  # static historical fact, always in the past
    rows = []

    for i in range(cfg.n_cv_folds):
        eval_end = max_date - pd.Timedelta(days=i * cfg.horizon)
        eval_start = eval_end - pd.Timedelta(days=cfg.horizon - 1)
        fold_train_raw = hist_df[hist_df["DateKey"] < eval_start].copy()
        fold_eval_raw = hist_df[(hist_df["DateKey"] >= eval_start) & (hist_df["DateKey"] <= eval_end)].copy()
        if fold_train_raw.empty or fold_eval_raw.empty:
            continue

        print(f"  Fold {i}: train until {(eval_start - pd.Timedelta(days=1)).date()}, "
              f"eval {eval_start.date()}..{eval_end.date()}")

        price_ref = fold_train_raw.groupby("ProductId")["PriceLocalVat"].median()

        fold_train_feat = prepare_features(fold_train_raw, price_ref, first_seen)
        fold_train_feat = add_train_lags(fold_train_feat, cfg.lag_windows)
        train_examples = (fold_train_feat[fold_train_feat["ProductAvailable"]]
                           .dropna(subset=feature_columns(cfg)).reset_index(drop=True))

        scaler = StandardScaler()
        tensors = make_tensors(train_examples, scaler, fit=True, cfg=cfg)
        y_log = np.log1p(train_examples["Quantity"].to_numpy(dtype=np.float32))
        model = train_model(tensors, y_log, cfg, epochs=cfg.cv_epochs, seed=cfg.seed)

        fold_eval_feat = prepare_features(fold_eval_raw, price_ref, first_seen).reset_index(drop=True)

        # Each model gets its own fresh copy of history: recursion mutates it
        # in place with that model's own predictions, so sharing one dict
        # across models would leak one model's forecasts into another's lags.
        nn_preds = recursive_forecast([model], scaler, fold_eval_feat,
                                       init_history(fold_train_feat, max_window), cfg)
        tree_preds = run_tree_baselines(train_examples, fold_eval_feat,
                                         init_history(fold_train_feat, max_window), cfg)

        seasonal_pred = seasonal_naive_predict(fold_eval_feat, fold_train_raw, lag_days=cfg.horizon)
        ma_pred = moving_average_predict(fold_eval_feat, fold_train_raw, window=28)
        actual = fold_eval_feat["Quantity"].to_numpy(dtype=float)

        preds_by_model = [
            ("NeuralNet", nn_preds),
            ("XGBoost", tree_preds["XGBoost"]),
            ("LightGBM", tree_preds["LightGBM"]),
            ("SeasonalNaive", seasonal_pred),
            ("MovingAvg28", ma_pred),
        ]
        for name, pred in preds_by_model:
            rows.append({"fold": i, "model": name, **compute_metrics(actual, pred)})

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
    train_examples = (train_feat[train_feat["ProductAvailable"]]
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
    train_examples = (train_feat[train_feat["ProductAvailable"]]
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


def export_results_json(train_raw: pd.DataFrame, test_raw: pd.DataFrame, submission: pd.DataFrame,
                         final_forecasts: dict, cv_results: pd.DataFrame, cfg: Config = CFG,
                         history_lookback: int = 90, path: str | None = None) -> dict:
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
            "cv_epochs": cfg.cv_epochs,
            "final_epochs": cfg.final_epochs,
            "seeds": list(cfg.seeds),
            "num_products": cfg.num_products,
        },
        "models": models_meta,
        "cv_results": order_models(cv_results.round(3)).to_dict(orient="records"),
        "cv_summary": summary.to_dict(orient="records"),
        "skill_vs_seasonal_naive": skill,
        "submission": submission.assign(
            DateKey=submission["DateKey"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "history": history,
        "forecasts": forecasts,
    }

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

    print(f"\n=== Walk-forward validation ({cfg.n_cv_folds} folds x {cfg.horizon}d) ===")
    cv_results = run_walk_forward_cv(train_raw, cfg)
    print()
    print(order_models(cv_results.round(2)).to_string(index=False))
    summary = order_models(cv_results.groupby("model")[["MAE", "RMSE", "MAPE"]]
                            .mean().round(2).reset_index()).set_index("model")
    print("\nAverage over folds:")
    print(summary.to_string())

    nn_mae = summary.loc["NeuralNet", "MAE"]
    naive_mae = summary.loc["SeasonalNaive", "MAE"]
    skill = 1 - nn_mae / naive_mae
    print(f"\nSkill vs seasonal-naive baseline: {skill:+.1%} (positive = model beats naive)")

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
    print(f"\nSaved: {cfg.output_dir}/submission.parquet, submission.csv, cv_results.csv")

    try:
        plot_forecast(train_raw, submission, cfg=cfg)
    except Exception as exc:  # pragma: no cover - plotting is a nice-to-have
        print(f"Plot skipped ({exc})")

    export_results_json(train_raw, test_raw, submission, final_forecasts, cv_results, cfg)


if __name__ == "__main__":
    main()
