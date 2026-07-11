"""Unit tests for solution_final.py.

These target the specific correctness properties that matter for a
multi-step demand forecast: no target leakage in lag features, recursive
lag updates across the forecast horizon (the bug this rewrite fixes),
and the baseline/metric helpers used for walk-forward validation.

Run with: uv run pytest tests/
"""

import pickle
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from features import (
    CAMPAIGN_TO_IDX,
    TREE_CATEGORICAL_COLUMNS,
    Config,
    add_calendar_features,
    add_train_lags,
    compute_metrics,
    feature_columns,
    moving_average_predict,
    order_models,
    prepare_features,
    seasonal_naive_predict,
    tree_feature_frame,
)
from solution_final import recursive_forecast

TREE_WORKER_PATH = Path(__file__).resolve().parents[1] / "ml" / "tree_worker.py"


def test_add_calendar_features_bounds_and_weekend():
    df = pd.DataFrame({"DateKey": pd.date_range("2026-01-12", periods=14, freq="D")})
    out = add_calendar_features(df.copy())

    cyclic_cols = [c for c in out.columns if c.endswith(("_sin", "_cos"))]
    assert cyclic_cols, "expected cyclic columns to be created"
    for col in cyclic_cols:
        assert out[col].between(-1.0001, 1.0001).all()

    weekend_dates = out.loc[out["is_weekend"] == 1, "DateKey"]
    assert set(weekend_dates.dt.day_name()) == {"Saturday", "Sunday"}


def test_add_train_lags_no_self_leakage():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "ProductId": [1] * 5,
        "DateKey": dates,
        "Quantity": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    out = add_train_lags(df, windows=(2,))

    assert pd.isna(out.loc[0, "qty_roll_mean_2"])          # no history yet
    assert out.loc[1, "qty_roll_mean_2"] == 10.0            # only prior value
    assert out.loc[2, "qty_roll_mean_2"] == 15.0            # mean(10, 20)
    assert out.loc[4, "qty_roll_mean_2"] == 35.0            # mean(30, 40); never sees its own 50


def test_prepare_features_price_rel_days_since_launch_and_unseen_campaign():
    df = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2026-01-10", "2026-01-11"]),
        "CampaignSubTypeWeb": [-1, 16],
        "CampaignSubTypeApp": [-1, 999],  # unseen category id -> should fall back safely
        "DiscountValueWebRelative": [0.0, 10.0],
        "DiscountValueAppRelative": [0.0, 0.0],
        "IsSaleOrPromo": [False, True],
        "PriceLocalVat": [100.0, 100.0],
    })
    price_ref = pd.Series({1: 100.0})
    first_seen = pd.Series({1: pd.Timestamp("2026-01-01")})

    out = prepare_features(df, price_ref, first_seen)

    assert np.allclose(out["price_rel"], [1.0, 1.0])
    assert list(out["days_since_launch"]) == [9, 10]
    assert out.loc[1, "campaign_idx_web"] == CAMPAIGN_TO_IDX[16]
    assert out.loc[1, "campaign_idx_app"] == 0  # fallback index for an unseen code


def test_seasonal_naive_predict_looks_up_correct_lag():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    train_df = pd.DataFrame({
        "ProductId": [1] * 10,
        "DateKey": dates,
        "Quantity": np.arange(10, dtype=float),
    })
    eval_df = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": [dates[9] + pd.Timedelta(days=1), dates[9] + pd.Timedelta(days=2)],
    })
    preds = seasonal_naive_predict(eval_df, train_df, lag_days=7)
    assert np.allclose(preds, [3.0, 4.0])


def test_moving_average_predict_uses_window_tail():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    train_df = pd.DataFrame({
        "ProductId": [1] * 5,
        "DateKey": dates,
        "Quantity": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    eval_df = pd.DataFrame({"ProductId": [1]})
    preds = moving_average_predict(eval_df, train_df, window=3)
    assert np.allclose(preds, [4.0])  # mean(3, 4, 5)


def test_compute_metrics_matches_manual_calculation():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 3
    assert np.isclose(m["MAE"], np.mean([2, 2, 3]))
    assert np.isclose(m["RMSE"], np.sqrt(np.mean([4.0, 4.0, 9.0])))


class _EchoLagMeanModel(nn.Module):
    """Stand-in model whose prediction is a direct function of the rolling
    lag-mean input feature, so we can detect whether that feature is being
    recomputed each forecast day (correct) or frozen (the bug being fixed).
    """

    def __init__(self, col_idx: int, scale: float = 0.01):
        super().__init__()
        self.col_idx = col_idx
        self.scale = scale

    def forward(self, x_num, x_prod, x_camp_web, x_camp_app):
        return self.scale * x_num[:, self.col_idx]


def test_recursive_forecast_updates_lag_features_each_day():
    cfg = Config(lag_windows=(2,), num_products=2)
    cols = feature_columns(cfg)
    col_idx = cols.index("qty_roll_mean_2")

    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    product_ids = [1, 2]
    rows = []
    for d in dates:
        for pid in product_ids:
            row = {feat: 0.0 for feat in cols}
            row.update({
                "ProductId": pid, "DateKey": d, "product_idx": pid - 1,
                "campaign_idx_web": 0, "campaign_idx_app": 0,
            })
            rows.append(row)
    static_df = pd.DataFrame(rows)

    scaler = StandardScaler().fit(np.zeros((4, len(cols))))
    history = {1: [5.0, 5.0], 2: [10.0, 10.0]}
    model = _EchoLagMeanModel(col_idx)

    preds = recursive_forecast([model], scaler, static_df, history, cfg)

    assert len(preds) == len(static_df)
    assert len(history[1]) == 2 + len(dates)
    assert len(history[2]) == 2 + len(dates)

    product1_preds = preds[static_df["ProductId"].to_numpy() == 1]
    # If lag features were frozen (the original bug), all 3 days would be identical.
    assert len(set(np.round(product1_preds, 8))) == len(dates)


def test_order_models_ml_first_then_naive_then_unknown_alphabetical():
    df = pd.DataFrame({
        "model": ["MovingAvg28", "Zeta", "SeasonalNaive", "LightGBM", "XGBoost", "NeuralNet"],
        "MAE": [1, 2, 3, 4, 5, 6],
    })
    ordered = order_models(df)
    assert list(ordered["model"]) == [
        "NeuralNet", "XGBoost", "LightGBM", "SeasonalNaive", "MovingAvg28", "Zeta",
    ]


def _make_synthetic_raw(n_days: int = 40, product_ids=(1, 2, 3), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for pid in product_ids:
        base = 10 * pid
        for i, d in enumerate(dates):
            rows.append({
                "ProductId": pid,
                "DateKey": d,
                "ProductAvailable": True,
                "CampaignSubTypeWeb": 0 if i % 5 == 0 else -1,
                "CampaignSubTypeApp": -1,
                "DiscountValueWebRelative": 0.0,
                "DiscountValueAppRelative": 0.0,
                "IsSaleOrPromo": False,
                "PriceLocalVat": 100.0 + pid,
                "Quantity": float(base + rng.integers(0, 5)),
            })
    return pd.DataFrame(rows)


def test_tree_feature_frame_casts_categoricals():
    cfg = Config(lag_windows=(3, 7))
    raw = _make_synthetic_raw()
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = raw.groupby("ProductId")["DateKey"].min()

    feat = prepare_features(raw, price_ref, first_seen)
    feat = add_train_lags(feat, cfg.lag_windows)
    train_examples = feat.dropna(subset=feature_columns(cfg)).reset_index(drop=True)

    X = tree_feature_frame(train_examples, cfg)
    for col in TREE_CATEGORICAL_COLUMNS:
        assert str(X[col].dtype) == "category"
    for col in feature_columns(cfg):
        assert str(X[col].dtype) != "category"


def test_tree_worker_subprocess_smoke(tmp_path):
    """Integration smoke test for the XGBoost/LightGBM subprocess worker,
    invoked as a real subprocess -- exactly how `solution_final.py`'s
    `run_tree_baselines` uses it. NOT called in-process here: this test
    module also imports `solution_final` (torch), and running XGBoost's or
    LightGBM's native training code in the same process as an already-loaded
    torch segfaults on macOS (each bundles its own, different, copy of the
    LLVM OpenMP runtime). The subprocess is what keeps them apart for real.
    """
    cfg = Config(lag_windows=(3, 7))
    raw = _make_synthetic_raw()
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = raw.groupby("ProductId")["DateKey"].min()

    feat = prepare_features(raw, price_ref, first_seen)
    feat = add_train_lags(feat, cfg.lag_windows)
    train_examples = feat.dropna(subset=feature_columns(cfg)).reset_index(drop=True)

    future_dates = pd.date_range(raw["DateKey"].max() + pd.Timedelta(days=1), periods=6, freq="D")
    future_rows = [
        {
            "ProductId": pid, "DateKey": d, "CampaignSubTypeWeb": -1, "CampaignSubTypeApp": -1,
            "DiscountValueWebRelative": 0.0, "DiscountValueAppRelative": 0.0,
            "IsSaleOrPromo": False, "PriceLocalVat": 100.0 + pid,
        }
        for pid in raw["ProductId"].unique()
        for d in future_dates
    ]
    future = prepare_features(pd.DataFrame(future_rows), price_ref, first_seen).reset_index(drop=True)
    max_window = max(cfg.lag_windows)
    history = {
        int(pid): list(sub.sort_values("DateKey")["Quantity"].to_numpy()[-max_window:])
        for pid, sub in feat.groupby("ProductId")
    }

    job = {
        "cfg": asdict(cfg),
        "train_examples": train_examples,
        "static_eval": future,
        "history": history,
        "models": ["XGBoost", "LightGBM"],
    }
    job_path = tmp_path / "job.pkl"
    out_path = tmp_path / "out.pkl"
    with open(job_path, "wb") as f:
        pickle.dump(job, f)

    result = subprocess.run(
        [sys.executable, str(TREE_WORKER_PATH), str(job_path), str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    with open(out_path, "rb") as f:
        results = pickle.load(f)

    assert set(results) == {"XGBoost", "LightGBM"}
    for preds in results.values():
        preds = np.asarray(preds)
        assert len(preds) == len(future)
        assert (preds >= 0).all()
