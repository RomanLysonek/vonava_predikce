"""Unit tests for the forecasting pipeline (framework.py + models/ + pipeline.py).

These target the specific correctness properties that matter for a
multi-step demand forecast: no target leakage in lag features, leakage-safe
horizon offsets in the direct multi-horizon panel (build_direct_panel), and
the baseline/metric helpers used for walk-forward validation.

Run with: uv run pytest tests/
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from framework import (
    CAMPAIGN_TO_IDX,
    TREE_CATEGORICAL_COLUMNS,
    Config,
    add_calendar_features,
    add_train_lags,
    build_direct_panel,
    compute_baseline,
    compute_metrics,
    direct_panel_feature_names,
    direct_panel_tree_frame,
    order_models,
    prepare_features,
    reindex_daily_calendar,
)
from models.naive_baselines import moving_average_predict, seasonal_naive_predict
from models.neural_net import (
    make_tensors,
    numeric_feature_columns,
    predict_direct,
    predict_ensemble,
    residual_log1p_target,
)
from pipeline import _json_safe

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
        "ProductAvailable": [True] * 5,
        "Quantity": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    out = add_train_lags(df, windows=(2,))

    assert pd.isna(out.loc[0, "qty_roll_mean_2"])          # no history yet
    assert out.loc[1, "qty_roll_mean_2"] == 10.0            # only prior value
    assert out.loc[2, "qty_roll_mean_2"] == 15.0            # mean(10, 20)
    assert out.loc[4, "qty_roll_mean_2"] == 35.0            # mean(30, 40); never sees its own 50


def test_add_train_lags_excludes_stockout_from_rolling_stats():
    """A ProductAvailable=False day's Quantity is censored, not a real
    zero -- it must not drag the rolling mean/count down."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "ProductId": [1] * 5,
        "DateKey": dates,
        "ProductAvailable": [True, True, False, True, True],
        "Quantity": [10.0, 20.0, 0.0, 40.0, 50.0],
    })
    out = add_train_lags(df, windows=(2,))

    # Row 3 (index 3): prior 2 rows are day 2 (stockout, excluded) and day 1
    # (20.0) -- rolling mean should be 20.0, not mean(0, 20) = 10.0, and the
    # available-count for that window should be 1, not 2.
    assert out.loc[3, "qty_roll_mean_2"] == 20.0
    assert out.loc[3, "qty_available_count_2"] == 1
    assert out.loc[3, "stockout_rate_2"] == 0.5


def test_reindex_daily_calendar_fills_gaps_as_nan_not_zero():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    df = pd.DataFrame({
        "ProductId": [1, 1, 1],
        "DateKey": dates,
        "ProductAvailable": [True, True, True],
        "Quantity": [10.0, 20.0, 50.0],
    })
    out = reindex_daily_calendar(df)

    assert len(out) == 5  # Jan 1..5 inclusive
    gap_rows = out[out["is_gap_filled"]]
    assert set(gap_rows["DateKey"].dt.day) == {3, 4}
    assert gap_rows["Quantity"].isna().all()
    assert gap_rows["ProductAvailable"].isna().all()


def test_compute_baseline_renormalizes_over_available_lags():
    """Weighted 4:3:2:1 same-weekday baseline (lags 7/14/21/28) should skip
    an unavailable lag and renormalize the remaining weights, not propagate
    NaN into the whole baseline."""
    dates = pd.date_range("2025-11-01", periods=29, freq="D")
    quantity = [0.0] * 29
    available = [True] * 29
    # lag-7, lag-14, lag-21, lag-28 relative to day 28 are days 21, 14, 7, 0.
    for day_idx, value in [(21, 10.0), (14, 20.0), (7, 30.0), (0, 40.0)]:
        quantity[day_idx] = value
    available[21] = False  # the lag-7 observation is a stockout -> excluded
    df = pd.DataFrame({
        "ProductId": [1] * 29,
        "DateKey": dates,
        "ProductAvailable": available,
        "Quantity": quantity,
    })
    target = df.iloc[[28]]
    baseline = compute_baseline(target, df)

    # Renormalized over lag-14/21/28 weights (3, 2, 1) only.
    expected = (3 * 20.0 + 2 * 30.0 + 1 * 40.0) / (3 + 2 + 1)
    assert np.isclose(baseline[0], expected)


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
        "ProductAvailable": [True] * 10,
        "Quantity": np.arange(10, dtype=float),
    })
    eval_df = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": [dates[9] + pd.Timedelta(days=1), dates[9] + pd.Timedelta(days=2)],
    })
    preds = seasonal_naive_predict(eval_df, train_df, lag_days=7)
    assert np.allclose(preds, [3.0, 4.0])


def test_seasonal_naive_predict_falls_back_to_baseline_when_lag_is_stockout():
    # 40 days of history so the lag-14/21/28 baseline components (relative
    # to the eval date) fall within range even though the exact lag-7 day
    # is a stockout.
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    available = [True] * 40
    available[33] = False  # exactly eval_date - 7 days
    train_df = pd.DataFrame({
        "ProductId": [1] * 40,
        "DateKey": dates,
        "ProductAvailable": available,
        "Quantity": np.arange(40, dtype=float),
    })
    eval_df = pd.DataFrame({"ProductId": [1], "DateKey": [dates[39] + pd.Timedelta(days=1)]})
    preds = seasonal_naive_predict(eval_df, train_df, lag_days=7)
    assert not np.isnan(preds[0])  # fell back to compute_baseline instead of NaN


def test_moving_average_predict_uses_window_tail():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    train_df = pd.DataFrame({
        "ProductId": [1] * 5,
        "DateKey": dates,
        "ProductAvailable": [True] * 5,
        "Quantity": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    eval_df = pd.DataFrame({"ProductId": [1]})
    preds = moving_average_predict(eval_df, train_df, window=3)
    assert np.allclose(preds, [4.0])  # mean(3, 4, 5)


def test_moving_average_predict_excludes_stockout_days():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    train_df = pd.DataFrame({
        "ProductId": [1] * 5,
        "DateKey": dates,
        "ProductAvailable": [True, True, False, True, True],
        "Quantity": [1.0, 2.0, 0.0, 4.0, 5.0],
    })
    eval_df = pd.DataFrame({"ProductId": [1]})
    preds = moving_average_predict(eval_df, train_df, window=3)
    assert np.allclose(preds, [4.5])  # mean(4, 5); day 3's censored 0 is excluded, not averaged in


def test_compute_metrics_matches_manual_calculation():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 3
    assert np.isclose(m["MAE"], np.mean([2, 2, 3]))
    assert np.isclose(m["RMSE"], np.sqrt(np.mean([4.0, 4.0, 9.0])))
    # WAPE = sum|error| / sum|actual| = 7 / 60
    assert np.isclose(m["WAPE"], 7.0 / 60.0)
    # Bias = mean(pred - actual) = mean(2, -2, 3)
    assert np.isclose(m["Bias"], np.mean([2.0, -2.0, 3.0]))
    assert np.isclose(m["BiasRatio"], np.sum([2.0, -2.0, 3.0]) / 60.0)
    for key in ("sMAPE", "RMSLE"):
        assert key in m and not np.isnan(m[key])


class _EchoHorizonModel(nn.Module):
    """Stand-in model whose prediction is a direct function of the horizon
    embedding index, so we can confirm `make_tensors`/`predict_direct`
    correctly wire a distinct `horizon` value through to the model for
    every row -- the direct-panel replacement for the old
    recursion-freezing check (build_direct_panel's own tests cover the
    lag/offset correctness the old recursive test used to exercise)."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x_num, x_prod, x_camp_web, x_camp_app, x_horizon):
        return self.scale * x_horizon.float()


def test_predict_direct_varies_by_horizon_via_embedding():
    cfg = Config(lag_windows=(7, 14, 28), num_products=2)
    cols = numeric_feature_columns(cfg)

    n = cfg.horizon
    df = pd.DataFrame({col: 0.0 for col in cols}, index=range(n))
    df["ProductId"] = 1
    df["product_idx"] = 0
    df["campaign_idx_web"] = 0
    df["campaign_idx_app"] = 0
    df["horizon"] = range(1, n + 1)

    scaler = StandardScaler().fit(np.zeros((4, len(cols))))
    model = _EchoHorizonModel()

    preds = predict_direct([model], scaler, df, cfg)

    assert len(preds) == n
    # Every horizon must produce a distinct prediction -- if make_tensors
    # dropped or mis-indexed the horizon tensor, every row would collapse
    # to the same value regardless of its own horizon.
    assert len(set(np.round(preds, 8))) == n


def test_residual_log1p_target_matches_manual_calculation():
    df = pd.DataFrame({
        "target": [5.0, 0.0, 100.0],
        "target_baseline": [4.0, 1.0, 50.0],
    })
    residual = residual_log1p_target(df)
    expected = np.log1p([5.0, 0.0, 100.0]) - np.log1p([4.0, 1.0, 50.0])
    assert np.allclose(residual, expected)


class _ConstantModel(nn.Module):
    """Stand-in model that ignores its inputs and always outputs a fixed,
    pre-supplied per-row tensor -- lets `predict_ensemble`'s baseline
    skip-connection add-back be checked in isolation from any actual
    training."""

    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, x_num, x_prod, x_camp_web, x_camp_app, x_horizon):
        return self.output


def _make_baseline_tensors(baseline: np.ndarray, cfg: Config) -> dict:
    cols = numeric_feature_columns(cfg)
    df = pd.DataFrame({col: 0.0 for col in cols}, index=range(len(baseline)))
    df["target_baseline"] = baseline
    df["product_idx"] = 0
    df["campaign_idx_web"] = 0
    df["campaign_idx_app"] = 0
    df["horizon"] = 1
    scaler = StandardScaler().fit(np.zeros((len(baseline), len(cols))))
    return make_tensors(df, scaler, fit=False, cfg=cfg)


def test_predict_ensemble_skip_connection_reconstructs_exact_target_from_perfect_residual():
    """Tier B2's design rests on this round-trip: if the model predicts
    exactly `residual_log1p_target`, `predict_ensemble`'s baseline add-back
    must exactly reconstruct the original target, not just produce
    "some" finite number."""
    cfg = Config(lag_windows=(7, 14, 28), num_products=2)
    target = np.array([5.0, 0.0, 100.0, 12.5], dtype=np.float32)
    baseline = np.array([4.0, 1.0, 50.0, 12.5], dtype=np.float32)

    tensors = _make_baseline_tensors(baseline, cfg)
    perfect_residual = torch.tensor(
        residual_log1p_target(pd.DataFrame({"target": target, "target_baseline": baseline})),
        dtype=torch.float32,
    )

    preds = predict_ensemble([_ConstantModel(perfect_residual)], tensors)
    assert np.allclose(preds, target, atol=1e-4)


def test_predict_ensemble_zero_residual_falls_back_to_baseline():
    cfg = Config(lag_windows=(7, 14, 28), num_products=2)
    baseline = np.array([10.0, 0.0, 42.0], dtype=np.float32)
    tensors = _make_baseline_tensors(baseline, cfg)

    preds = predict_ensemble([_ConstantModel(torch.zeros(3))], tensors)
    assert np.allclose(preds, baseline, atol=1e-4)


def test_predict_ensemble_averages_reconstructed_quantities_not_residuals():
    """Ensembling must average each seed's already-reconstructed Quantity
    (post baseline add-back, post expm1) -- averaging the raw residuals
    first and reconstructing once at the end would give a different,
    wrong number since expm1 is nonlinear."""
    cfg = Config(lag_windows=(7, 14, 28), num_products=2)
    baseline = np.array([10.0, 10.0], dtype=np.float32)
    tensors = _make_baseline_tensors(baseline, cfg)

    residual_a = torch.zeros(2, dtype=torch.float32)
    residual_b = torch.full((2,), float(np.log(2.0)), dtype=torch.float32)

    preds = predict_ensemble([_ConstantModel(residual_a), _ConstantModel(residual_b)], tensors)

    reconstructed_a = np.expm1(residual_a.numpy() + np.log1p(baseline))
    reconstructed_b = np.expm1(residual_b.numpy() + np.log1p(baseline))
    expected_correct_order = (reconstructed_a + reconstructed_b) / 2.0
    expected_wrong_order = np.expm1((residual_a.numpy() + residual_b.numpy()) / 2.0 + np.log1p(baseline))

    assert not np.allclose(expected_correct_order, expected_wrong_order)  # the two orders differ here
    assert np.allclose(preds, expected_correct_order, atol=1e-4)


def test_order_models_ml_first_then_naive_then_unknown_alphabetical():
    df = pd.DataFrame({
        "model": ["MovingAvg28", "Zeta", "SeasonalNaive", "LightGBM", "XGBoost", "Ensemble", "NeuralNet"],
        "MAE": [1, 2, 3, 4, 5, 6, 7],
    })
    ordered = order_models(df)
    assert list(ordered["model"]) == [
        "NeuralNet", "Ensemble", "XGBoost", "LightGBM", "SeasonalNaive", "MovingAvg28", "Zeta",
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


def test_direct_panel_tree_frame_casts_categoricals():
    cfg = Config(lag_windows=(3, 7))
    raw = _make_synthetic_raw()
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = raw.groupby("ProductId")["DateKey"].min()

    feat = add_train_lags(prepare_features(raw, price_ref, first_seen), cfg.lag_windows)
    panel = build_direct_panel(feat, range(1, cfg.horizon + 1), cfg=cfg)

    X = direct_panel_tree_frame(panel, cfg)
    for col in TREE_CATEGORICAL_COLUMNS:
        assert str(X[col].dtype) == "category"
    for col in direct_panel_feature_names(cfg):
        assert str(X[col].dtype) != "category"



def _make_deterministic_raw(n_days: int = 40, product_id: int = 1, start: str = "2026-01-01") -> pd.DataFrame:
    """Single-product series with Quantity == 1-indexed day number, so
    target/seasonal-lag lookups can be checked against an exact expected
    value instead of just "not NaN"."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({
        "ProductId": product_id, "DateKey": dates,
        "Quantity": np.arange(1, n_days + 1, dtype=float),
        "ProductAvailable": True,
        "CampaignSubTypeWeb": -1, "CampaignSubTypeApp": -1,
        "DiscountValueWebRelative": 0.0, "DiscountValueAppRelative": 0.0,
        "IsSaleOrPromo": False, "PriceLocalVat": 100.0,
    })


def _build_panel(raw: pd.DataFrame, horizons=range(1, 8), future_raw: pd.DataFrame = None,
                  lag_windows=(7, 14, 28)):
    cfg = Config(lag_windows=lag_windows)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = raw.groupby("ProductId")["DateKey"].min()
    feat = add_train_lags(prepare_features(raw, price_ref, first_seen), cfg.lag_windows)
    future_feat = None
    if future_raw is not None:
        future_feat = prepare_features(future_raw, price_ref, first_seen)
    return build_direct_panel(feat, horizons, cfg=cfg, future_covariates=future_feat), cfg


def test_build_direct_panel_target_date_matches_origin_plus_horizon():
    raw = _make_deterministic_raw()
    panel, _ = _build_panel(raw)

    has_target_date = panel["TargetDateKey"].notna()
    delta_days = (panel.loc[has_target_date, "TargetDateKey"] - panel.loc[has_target_date, "OriginDateKey"]).dt.days
    assert (delta_days == panel.loc[has_target_date, "horizon"]).all()

    date_to_qty = dict(zip(raw["DateKey"], raw["Quantity"]))
    valid = panel.dropna(subset=["target"])
    expected = valid["TargetDateKey"].map(date_to_qty).to_numpy(dtype=float)
    assert np.allclose(valid["target"].to_numpy(dtype=float), expected)


def test_build_direct_panel_seasonal_lag_is_leakage_safe_offset():
    """seasonal_lag_{L} at a given (origin, horizon) must equal the actual
    Quantity exactly L days before the TARGET date -- i.e. an origin-
    relative shift of (L - horizon) days, always >= 0 (a lookup into data
    at or before the origin, never a value from another horizon's own
    prediction)."""
    raw = _make_deterministic_raw(n_days=40)
    panel, _ = _build_panel(raw)
    date_to_qty = dict(zip(raw["DateKey"], raw["Quantity"]))

    origin_date = raw["DateKey"].iloc[34]  # Quantity 35, enough lookback for lag=28
    row = panel[(panel["OriginDateKey"] == origin_date) & (panel["horizon"] == 3)].iloc[0]

    target_date = origin_date + pd.Timedelta(days=3)
    for lag in (7, 14, 21, 28):
        expected = date_to_qty.get(target_date - pd.Timedelta(days=lag), np.nan)
        assert row[f"seasonal_lag_{lag}"] == expected, f"lag={lag}"
    # sanity: the offset really is (lag - horizon) relative to the origin
    assert row["seasonal_lag_7"] == date_to_qty[origin_date - pd.Timedelta(days=4)]


def test_build_direct_panel_origin_relative_features_are_horizon_independent():
    raw = _make_deterministic_raw(n_days=40)
    panel, _ = _build_panel(raw)

    origin_date = raw["DateKey"].iloc[34]
    same_origin = panel[panel["OriginDateKey"] == origin_date]
    assert same_origin["horizon"].nunique() == 7  # one row per horizon, same origin
    assert same_origin["qty_lag_1"].nunique() == 1
    assert same_origin["qty_roll_mean_7"].nunique() == 1


def test_build_direct_panel_future_covariates_supply_tail_targets_without_becoming_origins():
    raw = _make_deterministic_raw(n_days=40)
    future_dates = pd.date_range(raw["DateKey"].max() + pd.Timedelta(days=1), periods=7, freq="D")
    future_raw = pd.DataFrame({
        "ProductId": 1, "DateKey": future_dates,
        "Quantity": np.arange(41, 48, dtype=float), "ProductAvailable": True,
        "CampaignSubTypeWeb": -1, "CampaignSubTypeApp": -1,
        "DiscountValueWebRelative": 0.0, "DiscountValueAppRelative": 0.0,
        "IsSaleOrPromo": False, "PriceLocalVat": 100.0,
    })

    panel_no_future, _ = _build_panel(raw)
    last_origin = raw["DateKey"].max()
    assert panel_no_future.loc[panel_no_future["OriginDateKey"] == last_origin, "target"].isna().all()

    panel_with_future, _ = _build_panel(raw, future_raw=future_raw)
    tail = (panel_with_future[panel_with_future["OriginDateKey"] == last_origin]
            .sort_values("horizon"))
    assert np.allclose(tail["target"].to_numpy(dtype=float), np.arange(41, 48, dtype=float))

    future_index = pd.MultiIndex.from_frame(future_raw[["ProductId", "DateKey"]].rename(columns={"DateKey": "OriginDateKey"}))
    panel_index = pd.MultiIndex.from_frame(panel_with_future[["ProductId", "OriginDateKey"]])
    assert not panel_index.isin(future_index).any()


def test_direct_panel_feature_names_matches_panel_columns():
    raw = _make_deterministic_raw(n_days=40)
    panel, cfg = _build_panel(raw)
    for col in direct_panel_feature_names(cfg):
        assert col in panel.columns, f"missing column {col}"


def test_direct_panel_feature_names_includes_target_baseline():
    """Tier B2: the weighted same-weekday baseline must be part of the
    shared numeric feature schema so both the NN (input + skip-connection
    reference) and the trees (input only) pick it up automatically."""
    assert "target_baseline" in direct_panel_feature_names(Config())


def test_build_direct_panel_target_baseline_matches_compute_baseline():
    """`build_direct_panel`'s `target_baseline` (computed off the panel's
    own already-shifted seasonal_lag_{7,14,21,28} columns) must agree with
    the standalone `compute_baseline` hist_df lookup for the exact same
    target dates -- the two code paths must not silently diverge."""
    raw = _make_synthetic_raw(n_days=90, product_ids=(1, 2))
    panel, cfg = _build_panel(raw, horizons=range(1, 8), lag_windows=(7, 14, 28))

    # Origins in the last `h` rows per product have TargetDateKey == NaT
    # (row-position shift(-h) can't reach a real row without future_covariates)
    # even though target_baseline itself stays valid there -- exclude those so
    # compute_baseline isn't asked to look up a NaT date.
    valid = panel[
        panel["TargetDateKey"].notna()
        & panel["target_baseline_missing"].eq(0.0)
    ]
    assert len(valid) > 0
    target_rows = pd.DataFrame({
        "ProductId": valid["ProductId"].to_numpy(),
        "DateKey": valid["TargetDateKey"].to_numpy(),
    })
    expected = compute_baseline(target_rows, raw)
    assert np.allclose(valid["target_baseline"].to_numpy(dtype=float), expected, equal_nan=True)


def test_dynamic_ridge_train_predict():
    from models.dynamic_ridge import train_dynamic_ridge, predict_dynamic_ridge
    
    cfg = Config(lag_windows=(3, 7))
    raw = _make_synthetic_raw(n_days=60)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = raw.groupby("ProductId")["DateKey"].min()
    
    feat = prepare_features(raw, price_ref, first_seen)
    feat = add_train_lags(feat, cfg.lag_windows)
    
    horizons = range(1, cfg.horizon + 1)
    panel = build_direct_panel(feat, horizons, cfg=cfg)
    
    # Train only on rows that have a target and baseline
    train_panel = panel.dropna(subset=["target", "target_baseline"]).head(100)
    
    model = train_dynamic_ridge(train_panel, cfg)
    preds = predict_dynamic_ridge(model, train_panel, cfg)
    
    assert len(preds) == len(train_panel)
    assert np.all(preds >= 0)
    assert not np.isnan(preds).any()


def test_json_safe_replaces_nan_and_inf_but_keeps_other_values():
    """Regression test: `export_results_json`'s `history` block passes NaN
    calendar-gap Quantity values (from `reindex_daily_calendar`) straight
    into the payload. `json.dump` writes those as non-standard literal NaN
    tokens by default, which crashed `webapp/server.py`'s `/api/results`
    (Starlette's `JSONResponse` uses `allow_nan=False`) with a 500 and an
    empty body. `_json_safe` must be applied before writing so the file
    stays standard-compliant for every downstream consumer."""
    payload = {
        "history": {"30": {"dates": ["2026-01-01", "2026-01-02"], "quantity": [1.0, float("nan")]}},
        "cv_summary": [{"model": "NeuralNet", "WAPE": float("inf")}],
        "skill_vs_seasonal_naive": None,
        "n": 3,
        "label": "ok",
    }
    safe = _json_safe(payload)

    assert safe["history"]["30"]["quantity"] == [1.0, None]
    assert safe["cv_summary"][0]["WAPE"] is None
    assert safe["skill_vs_seasonal_naive"] is None
    assert safe["n"] == 3
    assert safe["label"] == "ok"

    import json
    json.dumps(safe)  # must not raise -- standard-compliant, no NaN/Infinity tokens
