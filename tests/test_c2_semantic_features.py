import json

import numpy as np
import pandas as pd

from ml.framework import (
    C2_FEATURE_GROUPS,
    Config,
    add_train_lags,
    build_direct_panel,
    direct_panel_feature_names,
    normalize_c2_feature_groups,
    prepare_features,
    product_reference_dates,
)
from ml.pipeline import RuntimeOptions, configure_c2_runtime
from ml.run_c2_screening import _best_eligible, _group_key


def _raw(periods=50, products=(1, 2)):
    dates = pd.date_range("2025-10-01", periods=periods, freq="D")
    rows = []
    for pid in products:
        for i, date in enumerate(dates):
            qty = float(10 * pid + i % 7)
            rows.append({
                "ProductId": pid,
                "DateKey": date,
                "Quantity": qty,
                "QuantityApp": qty,
                "QuantityWeb": 0.0,
                "ProductAvailable": True,
                "CampaignSubTypeWeb": -1,
                "CampaignSubTypeApp": -1,
                "DiscountValueWebRelative": 0.0,
                "DiscountValueAppRelative": 0.0,
                "IsSaleOrPromo": False,
                "PriceLocalVat": 100.0 + 10.0 * pid,
                "is_gap_filled": False,
            })
    return pd.DataFrame(rows)


def _features(raw, cfg):
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)
    return add_train_lags(
        prepare_features(raw, price_ref, first_seen, first_available, cfg),
        cfg.lag_windows,
        baseline_variant=cfg.baseline_variant,
    )


def test_c2_group_normalization_is_stable_and_validated():
    assert normalize_c2_feature_groups("none") == ()
    assert normalize_c2_feature_groups("all") == C2_FEATURE_GROUPS
    assert normalize_c2_feature_groups("market,price,market") == (
        "price", "market"
    )
    try:
        normalize_c2_feature_groups("price,unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown C2 group should fail")


def test_campaign_anomaly_and_app_only_features_preserve_numeric_discount_truth():
    raw = _raw(periods=2, products=(1,))
    raw.loc[0, "DiscountValueWebRelative"] = 15.0
    raw.loc[0, "DiscountValueAppRelative"] = 20.0
    raw.loc[1, "CampaignSubTypeApp"] = 16
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)
    cfg = Config(num_products=1, c2_feature_groups=("campaign",))
    feat = prepare_features(raw, price_ref, first_seen, first_available, cfg)

    assert feat.loc[0, "discount_without_campaign_web"] == 1.0
    assert feat.loc[0, "discount_without_campaign_app"] == 1.0
    assert feat.loc[0, "app_discount_advantage"] == 5.0
    assert feat.loc[1, "app_only_campaign"] == 1.0
    assert feat.loc[1, "campaign_subtypes_match"] == 0.0


def test_event_features_identify_black_friday_and_new_year_windows():
    raw = _raw(periods=1, products=(1,))
    raw.loc[0, "DateKey"] = pd.Timestamp("2025-11-28")
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)
    cfg = Config(num_products=1, c2_feature_groups=("event",))
    feat = prepare_features(raw, price_ref, first_seen, first_available, cfg)
    assert feat.loc[0, "days_from_black_friday"] == 0.0
    assert feat.loc[0, "black_friday_proximity_14"] == 1.0
    assert feat.loc[0, "is_black_friday_window"] == 1.0

    raw.loc[0, "DateKey"] = pd.Timestamp("2026-01-03")
    first_seen, first_available = product_reference_dates(raw)
    feat = prepare_features(raw, price_ref, first_seen, first_available, cfg)
    assert feat.loc[0, "is_new_year_window"] == 1.0


def test_price_group_uses_target_offer_against_observed_origin_state():
    raw = _raw(periods=45, products=(1,))
    raw.loc[raw.index[-1], "PriceLocalVat"] = 132.0
    cfg = Config(num_products=1, c2_feature_groups=("price",))
    panel = build_direct_panel(_features(raw, cfg), [1], cfg)
    row = panel[
        panel["OriginDateKey"].eq(raw["DateKey"].iloc[-2])
    ].iloc[0]
    expected = np.log1p(132.0) - np.log1p(110.0)
    assert np.isclose(row["price_log_ratio_vs_origin"], expected)
    assert np.isclose(row["price_log_ratio_vs_median28"], expected)
    assert set(["price_log_ratio_vs_origin"]).issubset(
        direct_panel_feature_names(cfg)
    )


def test_market_origin_state_does_not_use_target_quantity():
    raw = _raw(periods=45)
    cfg = Config(num_products=2, c2_feature_groups=("market",))
    panel_a = build_direct_panel(_features(raw, cfg), [1], cfg)

    changed = raw.copy()
    target_date = changed["DateKey"].max()
    changed.loc[changed["DateKey"].eq(target_date), "Quantity"] = 100_000.0
    changed.loc[changed["DateKey"].eq(target_date), "QuantityApp"] = 100_000.0
    panel_b = build_direct_panel(_features(changed, cfg), [1], cfg)

    origin_date = target_date - pd.Timedelta(days=1)
    cols = [
        "market_total_qty_lag0",
        "market_roll_mean_7",
        "market_roll_mean_28",
        "market_total_excl_product_lag0",
    ]
    a = panel_a[panel_a["OriginDateKey"].eq(origin_date)].sort_values("ProductId")
    b = panel_b[panel_b["OriginDateKey"].eq(origin_date)].sort_values("ProductId")
    np.testing.assert_allclose(a[cols], b[cols], equal_nan=True)


def test_market_target_campaign_intensity_is_cross_sectional_and_known_future():
    raw = _raw(periods=3)
    date = raw["DateKey"].max()
    raw.loc[(raw["DateKey"].eq(date)) & (raw["ProductId"].eq(1)), "CampaignSubTypeApp"] = 16
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)
    cfg = Config(num_products=2, c2_feature_groups=("market",))
    feat = prepare_features(raw, price_ref, first_seen, first_available, cfg)
    target = feat[feat["DateKey"].eq(date)]
    np.testing.assert_allclose(target["market_app_only_campaign_rate"], 0.5)
    np.testing.assert_allclose(target["market_campaign_app_rate"], 0.5)


def test_lifecycle_group_separates_unavailable_streak_from_calendar_gap():
    raw = _raw(periods=40, products=(1,))
    raw.loc[raw.index[-3:], "ProductAvailable"] = False
    raw.loc[raw.index[-3:], "Quantity"] = 0.0
    cfg = Config(num_products=1, c2_feature_groups=("lifecycle",))
    panel = build_direct_panel(_features(raw, cfg), [1], cfg)
    row = panel[panel["OriginDateKey"].eq(raw["DateKey"].iloc[-2])].iloc[0]
    assert row["consecutive_unavailable_days"] == 2.0
    assert row["current_is_calendar_gap"] == 0.0
    assert row["current_is_available"] == 0.0


def test_default_c1_schema_does_not_silently_enable_c2():
    base = Config()
    all_features = Config(c2_feature_groups=C2_FEATURE_GROUPS)
    base_names = set(direct_panel_feature_names(base))
    all_names = set(direct_panel_feature_names(all_features))
    assert "app_only_campaign" not in base_names
    assert "app_only_campaign" in all_names
    assert "market_total_qty_lag0" in all_names
    assert "days_from_black_friday" in all_names


def test_pipeline_loads_c2_recommendation_and_cli_override(tmp_path):
    path = tmp_path / "c2.json"
    path.write_text(json.dumps({
        "recommendation": {"config": {"c2_feature_groups": ["market", "campaign"]}}
    }))
    cfg = Config()
    runtime = configure_c2_runtime(cfg, RuntimeOptions(c2_config=str(path)))
    assert cfg.c2_feature_groups == ("campaign", "market")
    assert runtime["source"] == str(path)

    cfg = Config()
    configure_c2_runtime(
        cfg,
        RuntimeOptions(c2_config=str(path), c2_feature_groups="price,event"),
    )
    assert cfg.c2_feature_groups == ("price", "event")


def test_c2_screening_selection_respects_broad_quality_guard():
    rows = []
    for candidate, broad, aligned, bias, groups in [
        ("control", 0.30, 0.29, 0.04, ""),
        ("fast_bad", 0.40, 0.20, 0.00, "market"),
        ("eligible", 0.305, 0.27, 0.02, "campaign"),
    ]:
        rows.append({
            "candidate": candidate,
            "model": "NeuralNet",
            "WAPE": broad,
            "test_aligned_WAPE": aligned,
            "BiasRatio": bias,
            "Coverage": 1.0,
            "n_feature_groups": 0 if not groups else 1,
        })
    assert _best_eligible(
        rows, ["control", "fast_bad", "eligible"], "control", 0.03
    ) == "eligible"
    assert _group_key(("market", "price")) == "price+market"
