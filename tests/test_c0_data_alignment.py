import numpy as np
import pandas as pd

from ml.framework import (
    ANNUAL_LAG_DAYS,
    Config,
    add_train_lags,
    build_direct_panel,
    prepare_features,
    product_reference_dates,
    select_trainable_panel_rows,
)
from ml.models.neural_net import (
    make_numeric_preprocessor,
    make_tensors,
    numeric_feature_columns,
    predict_direct,
    residual_log1p_target,
    train_model,
)
from ml.pipeline import (
    DEVELOPMENT_ORIGINS,
    FINAL_AUDIT_ORIGINS,
    classify_validation_stratum,
    compute_test_aligned_scores,
    summarize_prediction_diagnostics,
    summarize_validation_strata,
)


def _raw(periods=100, *, product_id=1):
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    quantity = 10.0 + np.arange(periods) % 7
    return pd.DataFrame({
        "ProductId": product_id,
        "DateKey": dates,
        "Quantity": quantity,
        "QuantityApp": quantity,
        "QuantityWeb": 0.0,
        "ProductAvailable": pd.Series([True] * periods, dtype="boolean"),
        "CampaignSubTypeWeb": -1,
        "CampaignSubTypeApp": -1,
        "DiscountValueWebRelative": 0.0,
        "DiscountValueAppRelative": 0.0,
        "IsSaleOrPromo": False,
        "PriceLocalVat": 100.0,
        "is_gap_filled": False,
    })


def test_calendar_gap_and_unavailability_are_separate_states():
    raw = _raw(periods=4)
    raw.loc[1, "Quantity"] = np.nan
    raw.loc[1, "ProductAvailable"] = pd.NA
    raw.loc[1, "is_gap_filled"] = True
    raw.loc[2, "Quantity"] = 0.0
    raw.loc[2, "ProductAvailable"] = False

    first_seen, first_available = product_reference_dates(raw)
    feat = prepare_features(
        raw,
        raw.groupby("ProductId")["PriceLocalVat"].median(),
        first_seen,
        first_available,
    )
    lagged = add_train_lags(feat, windows=(3,))
    row = lagged.iloc[3]

    assert row["calendar_gap_count_3"] == 1
    assert row["unavailable_count_3"] == 1
    assert row["observed_count_3"] == 2
    assert row["qty_available_count_3"] == 1
    assert row["calendar_gap_rate_3"] == 1 / 3
    assert row["unavailable_rate_3"] == 1 / 3
    assert row["stockout_rate_3"] == row["unavailable_rate_3"]


def test_lifecycle_clocks_distinguish_first_row_from_first_available():
    raw = _raw(periods=5)
    raw.loc[:1, "ProductAvailable"] = False
    first_seen, first_available = product_reference_dates(raw)
    feat = prepare_features(
        raw,
        raw.groupby("ProductId")["PriceLocalVat"].median(),
        first_seen,
        first_available,
    )

    assert first_seen.iloc[0] == raw.loc[0, "DateKey"]
    assert first_available.iloc[0] == raw.loc[2, "DateKey"]
    assert feat.loc[0, "days_since_first_row"] == 0
    assert feat.loc[0, "days_since_first_available"] == -2
    assert feat.loc[0, "is_pre_first_available"] == 1
    assert feat.loc[2, "is_pre_first_available"] == 0


def test_trainable_panel_keeps_rows_with_missing_annual_lags():
    raw = _raw(periods=120)
    cfg = Config(num_products=1, horizon=7)
    first_seen, first_available = product_reference_dates(raw)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    feat = add_train_lags(
        prepare_features(raw, price_ref, first_seen, first_available),
        cfg.lag_windows,
    )
    panel = build_direct_panel(feat, range(1, 8), cfg)
    selected = select_trainable_panel_rows(
        panel, cutoff=raw["DateKey"].max(), available_only=True
    )

    assert not selected.empty
    assert selected[[f"seasonal_lag_{lag}" for lag in ANNUAL_LAG_DAYS]].isna().any(axis=1).any()
    assert selected[[f"seasonal_lag_{lag}_missing" for lag in ANNUAL_LAG_DAYS]].max().max() == 1.0
    assert np.isfinite(selected["target_baseline"]).all()


def test_nn_numeric_preprocessor_imputes_and_adds_indicators():
    cfg = Config(num_products=1, horizon=1, lag_windows=(7,))
    raw = _raw(periods=30)
    first_seen, first_available = product_reference_dates(raw)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    feat = add_train_lags(
        prepare_features(raw, price_ref, first_seen, first_available),
        cfg.lag_windows,
    )
    panel = build_direct_panel(feat, [1], cfg)
    panel = select_trainable_panel_rows(
        panel, cutoff=raw["DateKey"].max(), available_only=True
    )
    preprocessor = make_numeric_preprocessor()
    tensors = make_tensors(panel, preprocessor, fit=True, cfg=cfg)

    assert np.isfinite(tensors["num"].numpy()).all()
    # Missing annual lags create at least one indicator beyond raw numeric width.
    assert tensors["num"].shape[1] > len(numeric_feature_columns(cfg))



def test_nn_model_accepts_imputer_expanded_numeric_width():
    cfg = Config(
        num_products=1, horizon=1, lag_windows=(7,), hidden_dims=(8,),
        dropout=(0.0,), batch_size=16, cv_epochs=1, final_epochs=1,
    )
    raw = _raw(periods=40)
    first_seen, first_available = product_reference_dates(raw)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    feat = add_train_lags(
        prepare_features(raw, price_ref, first_seen, first_available),
        cfg.lag_windows,
    )
    panel = select_trainable_panel_rows(
        build_direct_panel(feat, [1], cfg),
        cutoff=raw["DateKey"].max(),
        available_only=True,
    )
    preprocessor = make_numeric_preprocessor()
    tensors = make_tensors(panel, preprocessor, fit=True, cfg=cfg)
    assert tensors["num"].shape[1] > len(numeric_feature_columns(cfg))

    model = train_model(
        tensors, residual_log1p_target(panel), cfg, epochs=1, seed=42
    )
    prediction = predict_direct([model], preprocessor, panel.iloc[:5], cfg)
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()


def test_validation_strata_and_test_aligned_score():
    assert classify_validation_stratum(pd.Timestamp("2023-01-10")) == "winter_test_like"
    assert classify_validation_stratum(pd.Timestamp("2023-11-24")) == "holiday_event"
    assert classify_validation_stratum(pd.Timestamp("2023-07-01")) == "regular"

    rows = []
    for origin, stratum, actual, direct, recursive in [
        (pd.Timestamp("2023-01-10"), "winter_test_like", 10.0, 9.0, 8.0),
        (pd.Timestamp("2023-07-01"), "regular", 10.0, 10.0, 12.0),
        (pd.Timestamp("2023-11-24"), "holiday_event", 10.0, 12.0, 10.0),
    ]:
        for strategy, pred in [("direct", direct), ("recursive", recursive)]:
            rows.append({
                "origin_type": "development",
                "origin": origin,
                "strategy": strategy,
                "validation_stratum": stratum,
                "ProductId": 1,
                "DateKey": origin + pd.Timedelta(days=1),
                "horizon": 1,
                "actual": actual,
                "ProductAvailable": True,
                "pred_NeuralNet": pred,
                "pred_XGBoost": pred,
                "pred_LightGBM": pred,
                "pred_SeasonalNaive": pred,
                "pred_MovingAvg28": pred,
            })
    oof = pd.DataFrame(rows)
    summary = summarize_validation_strata(oof)
    scores = compute_test_aligned_scores(summary, metric="WAPE")

    assert set(summary["validation_stratum"]) == {
        "winter_test_like", "regular", "holiday_event"
    }
    assert set(scores["strategy"]) == {"direct", "recursive"}
    assert scores["weight_sum"].eq(1.0).all()


def test_prediction_diagnostics_expose_fallbacks_and_extremes():
    oof = pd.DataFrame({
        "origin_type": ["development"] * 3,
        "strategy": ["recursive"] * 3,
        "actual": [10.0, 20.0, 30.0],
        "pred_NeuralNet": [11.0, 22.0, 100.0],
        "fallback_NeuralNet": [False, True, False],
        "nonfinite_NeuralNet": [False, True, False],
        "catastrophic_NeuralNet": [False, False, True],
    })
    diagnostics = summarize_prediction_diagnostics(
        oof, {"NeuralNet": "pred_NeuralNet"}
    )
    row = diagnostics.iloc[0]

    assert row["fallback_count"] == 1
    assert row["nonfinite_raw_count"] == 1
    assert row["catastrophic_guard_count"] == 1
    assert row["prediction_max"] == 100.0
    assert row["prediction_to_observed_max_ratio"] == 100.0 / 30.0


def test_final_audit_origins_are_frozen_and_disjoint():
    assert len(FINAL_AUDIT_ORIGINS) == 3
    assert set(FINAL_AUDIT_ORIGINS).isdisjoint(set(DEVELOPMENT_ORIGINS))

    audit_targets = {
        origin + pd.Timedelta(days=step)
        for origin in FINAL_AUDIT_ORIGINS
        for step in range(1, 8)
    }
    development_targets = {
        origin + pd.Timedelta(days=step)
        for origin in DEVELOPMENT_ORIGINS
        for step in range(1, 8)
    }
    assert audit_targets.isdisjoint(development_targets)
