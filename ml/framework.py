"""Shared framework: config, feature engineering, tree-model feature framing,
the model-agnostic recursive forecasting engine, the model registry/metadata,
and metrics. Every actual model's train/predict definition lives under
`models/` instead (`models/neural_net.py`, `models/xgboost_model.py`,
`models/lightgbm_model.py`, `models/naive_baselines.py`); this module is
everything those model definitions -- and `pipeline.py`'s orchestration --
share.

Deliberately has NO dependency on torch, xgboost, or lightgbm. This lets
`tree_worker.py` (which needs xgboost/lightgbm) and `pipeline.py` (which
needs torch) both import this module without ever importing each other's
heavy native-code dependency into the same process -- see `tree_worker.py`'s
docstring for why that matters on macOS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    train_path: str = "data/train_data.parquet"
    test_path: str = "data/test_data.parquet"
    output_dir: str = "outputs"
    horizon: int = 7                      # forecast horizon in days
    lag_windows: tuple = (7, 14, 28)
    num_products: int = 30                # overwritten from data in main()
    embed_dim_product: int = 12
    embed_dim_campaign: int = 4
    embed_dim_horizon: int = 4
    hidden_dims: tuple = (256, 128, 64)
    dropout: tuple = (0.20, 0.15, 0.10)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    cv_epochs: int = 30                   # per fold, no early stopping (avoids peeking at eval fold)
    final_epochs: int = 60                # for the submission ensemble
    seeds: tuple = (42, 123, 777)
    n_cv_folds: int = 4
    seed: int = 42
    ridge_alpha: float = 10.0
    ridge_prediction_cap: float | None = None
    # Numerical guard only: recursive feedback values above this data-scaled
    # threshold are replaced by the baseline rather than fed back. This is
    # deliberately much looser than any model cap considered in Tier C3.
    recursive_safety_multiplier: float = 1000.0
    recursive_safety_floor: float = 1_000_000.0


CFG = Config()
np.random.seed(CFG.seed)

# Campaign sub-type ids are categorical codes, not an ordinal scale -> embed
# them (NN) / mark them as pandas 'category' dtype (trees) instead of
# feeding the raw integer in as a numeric feature.
CAMPAIGN_CATEGORIES = [-1, 0, 1, 2, 3, 4, 5, 16, 18, 19]
CAMPAIGN_TO_IDX = {v: i for i, v in enumerate(CAMPAIGN_CATEGORIES)}
NUM_CAMPAIGN_CATS = len(CAMPAIGN_CATEGORIES)

STATIC_NUMERIC_FEATURES = [
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "week_of_year_sin", "week_of_year_cos",
    "day_of_month", "is_weekend",
    "discount_web", "discount_app", "discount_max",
    "effective_price_web", "effective_price_app",
    "is_sale", "price", "price_rel", "days_since_launch",
]


def lag_feature_names(lag_windows) -> list[str]:
    names = []
    for w in lag_windows:
        names += [f"qty_roll_mean_{w}", f"qty_roll_std_{w}", f"qty_roll_median_{w}",
                  f"qty_available_count_{w}", f"stockout_rate_{w}"]
    return names


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def reindex_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in any missing calendar days per product so every later
    `shift(1)` means "yesterday", not "whatever row happened to be
    previous". Two products in this dataset (1 and 30) have gaps sitting in
    the middle of otherwise-continuous, available history -- a data glitch,
    not a real absence from the catalog -- so a gap day's Quantity /
    ProductAvailable are unknown, not zero: they're filled as NaN / <NA>,
    which the availability-aware rolling stats in `add_train_lags` then
    treat exactly like a stockout day. `is_gap_filled` records provenance.
    """
    frames = []
    for pid, sub in df.groupby("ProductId", sort=True):
        sub = sub.sort_values("DateKey")
        full_idx = pd.date_range(sub["DateKey"].min(), sub["DateKey"].max(), freq="D")
        original_dates = set(sub["DateKey"])
        reindexed = sub.set_index("DateKey").reindex(full_idx)
        reindexed.index.name = "DateKey"
        reindexed["is_gap_filled"] = ~reindexed.index.isin(original_dates)
        reindexed["ProductId"] = pid
        frames.append(reindexed.reset_index())

    out = pd.concat(frames, ignore_index=True).sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    out["ProductAvailable"] = out["ProductAvailable"].astype("boolean")  # nullable -> NaN for gap rows
    out["Quantity"] = out["Quantity"].astype(float)                     # NaN for gap rows

    carry_forward = ["CampaignSubTypeWeb", "CampaignSubTypeApp", "DiscountValueWebRelative",
                      "DiscountValueAppRelative", "IsSaleOrPromo", "PriceLocalVat"]
    for col in carry_forward:
        if col in out.columns:
            out[col] = out.groupby("ProductId")[col].transform(lambda s: s.ffill().bfill())
    return out


def load_raw(cfg: Config = CFG) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(cfg.train_path)
    test = pd.read_parquet(cfg.test_path)
    train["Quantity"] = (train["QuantityApp"].fillna(0) + train["QuantityWeb"].fillna(0)).astype(float)

    ids = sorted(train["ProductId"].unique())
    assert ids == list(range(1, len(ids) + 1)), "ProductId is expected to be contiguous 1..N"

    train = reindex_daily_calendar(train)
    return train, test


# ---------------------------------------------------------------------------
# Feature engineering (static features: no leakage, safe for train/eval/test)
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["DateKey"]
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    for col, period in [("day_of_week", 7), ("month", 12), ("day_of_year", 365), ("week_of_year", 52)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


def prepare_features(df: pd.DataFrame, price_ref: pd.Series, first_seen: pd.Series) -> pd.DataFrame:
    """Add all features that do NOT depend on the target's own recent history.

    `price_ref` (per-product median price) and `first_seen` (per-product first
    observed date) must be computed from training-only data by the caller, so
    the same reference values are reused consistently across train/eval/test.
    """
    df = df.copy()
    df = add_calendar_features(df)

    df["campaign_idx_web"] = df["CampaignSubTypeWeb"].map(CAMPAIGN_TO_IDX).fillna(0).astype(int)
    df["campaign_idx_app"] = df["CampaignSubTypeApp"].map(CAMPAIGN_TO_IDX).fillna(0).astype(int)
    df["discount_web"] = df["DiscountValueWebRelative"].fillna(0).astype(float)
    df["discount_app"] = df["DiscountValueAppRelative"].fillna(0).astype(float)
    df["discount_max"] = np.maximum(df["discount_web"], df["discount_app"])
    df["is_sale"] = df["IsSaleOrPromo"].astype(int)
    df["price"] = df["PriceLocalVat"].fillna(0).astype(float)
    # Two channel-specific discount percentages don't sum to a meaningful
    # "total discount" (a 10% web cut + 10% app cut is not a 20% market
    # discount) -- effective per-channel price is the economically sound
    # combination instead.
    df["effective_price_web"] = df["price"] * (1.0 - df["discount_web"] / 100.0)
    df["effective_price_app"] = df["price"] * (1.0 - df["discount_app"] / 100.0)

    ref = df["ProductId"].map(price_ref).replace(0, np.nan)
    df["price_rel"] = (df["price"] / ref).fillna(1.0)
    df["days_since_launch"] = (df["DateKey"] - df["ProductId"].map(first_seen)).dt.days

    df["product_idx"] = df["ProductId"] - 1
    return df


# Weighted same-weekday baseline: a 4:3:2:1 weighted average of Quantity at
# lags 7/14/21/28 days. Shared by `compute_baseline` below (a `hist_df`
# lookup, used for the naive-baseline diagnostic column and as a
# seasonal-naive fallback) and `build_direct_panel`'s `target_baseline`
# feature (Tier B2), which reuses the exact same weights/renormalization
# vectorized straight off the panel's own already-computed
# `seasonal_lag_{7,14,21,28}` columns instead of a second hist_df lookup.
BASELINE_LAGS = (7, 14, 21, 28)
BASELINE_WEIGHTS = np.array([4.0, 3.0, 2.0, 1.0])


def _weighted_baseline(lag_matrix: np.ndarray) -> np.ndarray:
    """Row-wise NaN-aware weighted average over `BASELINE_LAGS`/
    `BASELINE_WEIGHTS` (columns of `lag_matrix` must be in that same lag
    order). Weights renormalize over whichever lags are actually observed
    in a row -- a stockout/unknown-gap/insufficient-history lag drops out
    of the average instead of forcing the whole row to NaN."""
    observed = np.isfinite(lag_matrix)
    numerator = np.nansum(lag_matrix * BASELINE_WEIGHTS, axis=1)
    denominator = (observed * BASELINE_WEIGHTS).sum(axis=1)
    return np.divide(numerator, denominator,
                      out=np.full(len(lag_matrix), np.nan, dtype=float),
                      where=denominator > 0)


def compute_baseline(target_df: pd.DataFrame, hist_df: pd.DataFrame) -> np.ndarray:
    """Availability-aware weighted same-weekday baseline (see
    `BASELINE_LAGS`/`BASELINE_WEIGHTS`), using only observed-and-available
    demand from `hist_df`. `hist_df` is the lookup source (e.g. training
    history); `target_df` is whatever rows need a baseline value (can be
    the same frame, for training rows, or later out-of-sample eval/test
    rows looking back into `hist_df`)."""
    available = hist_df["ProductAvailable"].fillna(False)
    qty_available = hist_df["Quantity"].where(available)
    lookup = pd.Series(qty_available.to_numpy(),
                        index=pd.MultiIndex.from_frame(hist_df[["ProductId", "DateKey"]]))

    lag_matrix = np.full((len(target_df), len(BASELINE_LAGS)), np.nan)
    for j, lag in enumerate(BASELINE_LAGS):
        keys = list(zip(target_df["ProductId"], target_df["DateKey"] - pd.Timedelta(days=lag)))
        lag_matrix[:, j] = [lookup.get(k, np.nan) for k in keys]

    return _weighted_baseline(lag_matrix)


def add_train_lags(df: pd.DataFrame, windows: tuple = CFG.lag_windows) -> pd.DataFrame:
    """Rolling lag statistics computed strictly from the past (`shift(1)`),
    grouped per product, using only observed-and-available demand. A
    stockout day (ProductAvailable=False) or a reindexed calendar gap has
    unknown true demand, so it's excluded (NaN) from `qty_available` before
    rolling -- pandas' rolling mean/std/median already skip NaN internally
    (subject to min_periods), so a stockout no longer silently drags the
    average toward zero. `qty_available_count_{w}` / `stockout_rate_{w}`
    expose how much real signal backs each window. Only usable on rows
    where the target is known."""
    df = df.sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    available = df["ProductAvailable"].fillna(False)
    df["qty_available"] = df["Quantity"].where(available)

    g = df.groupby("ProductId")["qty_available"]
    row_num = df.groupby("ProductId").cumcount()
    for w in windows:
        df[f"qty_roll_mean_{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"qty_roll_std_{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).std().fillna(0))
        df[f"qty_roll_median_{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).median())
        count = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).count())
        df[f"qty_available_count_{w}"] = count
        window_days = np.minimum(row_num, w).clip(lower=1)
        df[f"stockout_rate_{w}"] = 1.0 - count / window_days

    df["baseline"] = compute_baseline(df, df)
    return df


# ---------------------------------------------------------------------------
# Direct multi-horizon panel (Tier B1): eliminates recursion entirely for
# NN/XGBoost/LightGBM -- see build_direct_panel's docstring for why every
# horizon's inputs are always a lookup into already-observed data, never a
# value that would first need to be predicted.
# ---------------------------------------------------------------------------
RECENT_POINT_LAGS = (0, 1, 2, 6, 7)
# BASELINE_LAGS first, so `target_baseline` below can always read
# seasonal_lag_{7,14,21,28} straight off the columns this computes for
# every horizon -- weekly-seasonal lags plus 3 yearly-seasonal lags.
SEASONAL_LAG_DAYS = BASELINE_LAGS + (364, 365, 371)

# Columns whose VALUE must be shifted forward from the target row (the two
# campaign category codes included, so the panel reflects whatever
# campaign is active ON the target date) -- used only by `build_direct_panel`
# itself. "Future-known" because the task's own test_data.parquet already
# supplies these for the real forecast week -- an assumption this panel
# inherits, not one it introduces.
TARGET_COVARIATE_COLUMNS = STATIC_NUMERIC_FEATURES + ["campaign_idx_web", "campaign_idx_app"]


def direct_panel_feature_names(cfg: Config = CFG) -> list[str]:
    """Full numeric feature schema for `build_direct_panel`'s output:
    target-date covariates + origin-relative rolling stats (from
    `add_train_lags`, just relative to whichever row is the origin here) +
    origin-relative point lags + target-relative seasonal lags + horizon
    itself. Deliberately uses `STATIC_NUMERIC_FEATURES`, not the wider
    `TARGET_COVARIATE_COLUMNS` -- the two campaign category codes get
    separate categorical (`TREE_CATEGORICAL_COLUMNS`) / embedding
    treatment instead of being counted as plain numeric features (mirrors
    how product/campaign indices were always excluded from the old
    recursive pipeline's `feature_columns`); including them here too would
    hand tree models the same column twice under two different roles.
    `target_baseline` (Tier B2) is the weighted same-weekday baseline for
    the target date itself -- see `build_direct_panel`."""
    return (STATIC_NUMERIC_FEATURES + lag_feature_names(cfg.lag_windows)
            + [f"qty_lag_{lag}" for lag in RECENT_POINT_LAGS]
            + [f"seasonal_lag_{lag}" for lag in SEASONAL_LAG_DAYS]
            + ["target_baseline", "horizon"])


def build_origin_state_features(feature_df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Build features known at the end of each origin day.

    Unlike target-row lag features, these include the origin day's observed
    available demand itself (`qty_lag_0`) and rolling windows ending at the
    origin. This gives direct and recursive forecasting the same information
    cutoff: all observations through and including ForecastOrigin are usable.
    """
    df = feature_df.sort_values(["ProductId", "DateKey"]).reset_index(drop=True).copy()
    if "qty_available" not in df.columns:
        available = df["ProductAvailable"].fillna(False)
        df["qty_available"] = df["Quantity"].where(available)
    g = df.groupby("ProductId")["qty_available"]
    out = df[["ProductId", "DateKey"]].copy()
    for lag in RECENT_POINT_LAGS:
        out[f"qty_lag_{lag}"] = g.shift(lag)
    row_num = df.groupby("ProductId").cumcount() + 1
    for w in cfg.lag_windows:
        out[f"qty_roll_mean_{w}"] = g.transform(lambda x: x.rolling(w, min_periods=1).mean())
        out[f"qty_roll_std_{w}"] = g.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        out[f"qty_roll_median_{w}"] = g.transform(lambda x: x.rolling(w, min_periods=1).median())
        count = g.transform(lambda x: x.rolling(w, min_periods=1).count())
        out[f"qty_available_count_{w}"] = count
        window_days = np.minimum(row_num, w).clip(lower=1)
        out[f"stockout_rate_{w}"] = 1.0 - count / window_days
    return out


def build_direct_panel(train_feat: pd.DataFrame, horizons, cfg: Config = CFG,
                        future_covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack (ForecastOrigin x Horizon x ProductId) into a direct panel.

    Origin-state features use observations through the origin itself. Target
    covariates and seasonal lags are aligned to each target date. The horizon
    guard guarantees every target-relative seasonal lookup remains at or
    before the origin.
    """
    horizons = tuple(int(h) for h in horizons)
    if not horizons:
        raise ValueError("At least one forecast horizon is required")
    if min(horizons) < 1:
        raise ValueError("Forecast horizons must be positive")
    if max(horizons) > min(SEASONAL_LAG_DAYS):
        raise ValueError("Target-relative seasonal lags would require future observations")
    if max(horizons) > cfg.horizon:
        raise ValueError("Requested horizon exceeds Config.horizon and the NN horizon embedding domain")
    for name, frame in (("train_feat", train_feat), ("future_covariates", future_covariates)):
        if frame is not None and frame.duplicated(["ProductId", "DateKey"]).any():
            raise ValueError(f"{name} contains duplicate ProductId/DateKey keys")

    train_feat = train_feat.sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    origin_index = pd.MultiIndex.from_frame(train_feat[["ProductId", "DateKey"]])
    combined = train_feat.copy()
    if future_covariates is not None:
        future_covariates = future_covariates.copy()
        for col in ("Quantity", "ProductAvailable"):
            if col not in future_covariates.columns:
                future_covariates[col] = np.nan
        keep = ["ProductId", "DateKey", "Quantity", "ProductAvailable"] + TARGET_COVARIATE_COLUMNS
        combined = pd.concat([train_feat, future_covariates[keep]], ignore_index=True, sort=False)
    combined = combined.sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    if "qty_available" not in combined.columns:
        combined["qty_available"] = combined["Quantity"].where(combined["ProductAvailable"].fillna(False))
    else:
        # Future rows arrive without lag engineering; derive their value safely.
        missing = combined["qty_available"].isna()
        combined.loc[missing, "qty_available"] = combined.loc[missing, "Quantity"].where(
            combined.loc[missing, "ProductAvailable"].fillna(False))
    g = combined.groupby("ProductId")
    origin = build_origin_state_features(combined, cfg)

    frames = []
    for h in horizons:
        panel_h = origin.copy()
        panel_h["horizon"] = h
        target_cols = ["DateKey", "Quantity", "ProductAvailable"] + TARGET_COVARIATE_COLUMNS
        target = g[target_cols].shift(-h)
        panel_h["TargetDateKey"] = target["DateKey"]
        panel_h["target"] = target["Quantity"]
        panel_h["TargetProductAvailable"] = target["ProductAvailable"]
        for col in TARGET_COVARIATE_COLUMNS:
            panel_h[col] = target[col]
        for lag in SEASONAL_LAG_DAYS:
            panel_h[f"seasonal_lag_{lag}"] = g["qty_available"].shift(lag - h)
        lag_matrix = np.column_stack([
            panel_h[f"seasonal_lag_{lag}"].to_numpy(dtype=float)
            for lag in BASELINE_LAGS
        ])
        panel_h["target_baseline"] = _weighted_baseline(lag_matrix)
        frames.append(panel_h)

    panel = pd.concat(frames, ignore_index=True).rename(columns={"DateKey": "OriginDateKey"})
    panel["product_idx"] = panel["ProductId"] - 1
    panel_index = pd.MultiIndex.from_arrays([panel["ProductId"], panel["OriginDateKey"]])
    return panel[panel_index.isin(origin_index)].reset_index(drop=True)


def build_one_step_panel(raw_df: pd.DataFrame, price_ref: pd.Series,
                         first_seen: pd.Series, cfg: Config = CFG) -> pd.DataFrame:
    """Build one-step-ahead training rows for recursive models."""
    feat = prepare_features(raw_df, price_ref, first_seen)
    feat = add_train_lags(feat, cfg.lag_windows)
    return build_direct_panel(feat, [1], cfg=cfg)


KNOWN_FUTURE_RAW_COLUMNS = [
    "ProductId", "DateKey", "CampaignSubTypeWeb", "CampaignSubTypeApp",
    "DiscountValueWebRelative", "DiscountValueAppRelative", "IsSaleOrPromo",
    "PriceLocalVat",
]


def sanitize_future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Return only features legitimately known before future demand occurs."""
    missing = [c for c in KNOWN_FUTURE_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Future covariates are missing required columns: {missing}")
    out = df[KNOWN_FUTURE_RAW_COLUMNS].copy()
    if out.duplicated(["ProductId", "DateKey"]).any():
        raise ValueError("future_covariates contains duplicate ProductId/DateKey keys")
    return out


def build_recursive_step_panel(history_raw: pd.DataFrame, target_covariates: pd.DataFrame,
                               price_ref: pd.Series, first_seen: pd.Series,
                               cfg: Config = CFG) -> pd.DataFrame:
    """Build the one-step panel for the next target day from current history."""
    future = sanitize_future_covariates(target_covariates)
    future["Quantity"] = np.nan
    future["ProductAvailable"] = pd.Series([pd.NA] * len(future), dtype="boolean")
    history_feat = prepare_features(history_raw, price_ref, first_seen)
    history_feat = add_train_lags(history_feat, cfg.lag_windows)
    future_feat = prepare_features(future, price_ref, first_seen)
    panel = build_direct_panel(history_feat, [1], cfg=cfg, future_covariates=future_feat)
    origin = history_raw["DateKey"].max()
    step = panel[panel["OriginDateKey"].eq(origin)].reset_index(drop=True)
    step["horizon"] = 1
    return step


def forecast_recursive(history_raw: pd.DataFrame, future_covariates: pd.DataFrame,
                       predict_step, price_ref: pd.Series, first_seen: pd.Series,
                       cfg: Config = CFG) -> pd.DataFrame:
    """Forecast future dates sequentially, feeding predictions into history."""
    history = history_raw.copy().sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    future = sanitize_future_covariates(future_covariates)
    dates = sorted(pd.to_datetime(future["DateKey"].drop_duplicates()))
    if len(dates) != cfg.horizon:
        raise ValueError(f"Expected {cfg.horizon} future dates, got {len(dates)}")
    results = []
    for forecast_horizon, target_date in enumerate(dates, start=1):
        current = future[future["DateKey"].eq(target_date)].copy()
        step_panel = build_recursive_step_panel(history, current, price_ref, first_seen, cfg)
        if not step_panel["horizon"].eq(1).all():
            raise AssertionError("Recursive model input horizon must always equal 1")
        prediction = np.asarray(predict_step(step_panel), dtype=float)
        if len(prediction) != len(step_panel):
            raise ValueError("predict_step returned a prediction vector with the wrong length")
        baseline = step_panel["target_baseline"].to_numpy(dtype=float)

        # A recursive model can turn one extreme but finite extrapolation into
        # progressively larger lag features.  Treat only catastrophic values
        # as numerical failures; this guard is intentionally orders of
        # magnitude looser than any prediction cap considered in Tier C3.
        history_quantity = pd.to_numeric(history["Quantity"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        history_max = history.assign(_finite_quantity=history_quantity).groupby(
            "ProductId"
        )["_finite_quantity"].max()
        observed_scale = step_panel["ProductId"].map(history_max).to_numpy(dtype=float)
        lag0 = step_panel.get(
            "qty_lag_0", pd.Series(np.nan, index=step_panel.index)
        ).to_numpy(dtype=float)
        reference_scale = np.nanmax(
            np.column_stack([
                np.where(np.isfinite(observed_scale), observed_scale, 0.0),
                np.where(np.isfinite(baseline), baseline, 0.0),
                np.where(np.isfinite(lag0), lag0, 0.0),
                np.ones(len(step_panel), dtype=float),
            ]),
            axis=1,
        )
        safety_limit = np.maximum(
            cfg.recursive_safety_floor,
            cfg.recursive_safety_multiplier * reference_scale,
        )
        catastrophic = np.isfinite(prediction) & (prediction > safety_limit)
        fallback = ~np.isfinite(prediction) | catastrophic

        fallback_value = np.where(
            np.isfinite(baseline) & (baseline >= 0.0),
            baseline,
            np.where(
                np.isfinite(lag0) & (lag0 >= 0.0),
                lag0,
                0.0,
            ),
        )
        prediction = np.where(fallback, fallback_value, prediction)
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        prediction = np.clip(prediction, 0.0, None)
        result = step_panel[["ProductId", "TargetDateKey"]].copy()
        result["forecast_horizon"] = forecast_horizon
        result["prediction"] = prediction
        result["fallback_used"] = fallback
        results.append(result)

        generated = current.merge(
            result[["ProductId", "prediction"]], on="ProductId", how="left", validate="one_to_one"
        )
        generated["Quantity"] = generated.pop("prediction")
        generated["ProductAvailable"] = True
        # Keep raw-schema compatibility for downstream data preparation.
        generated["QuantityApp"] = generated["Quantity"]
        generated["QuantityWeb"] = 0.0
        history = pd.concat([history, generated], ignore_index=True, sort=False)
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Tree-model feature framing (shared shape used by tree_worker.py)
# ---------------------------------------------------------------------------
TREE_CATEGORICAL_COLUMNS = ["product_idx", "campaign_idx_web", "campaign_idx_app"]


def direct_panel_tree_frame(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Numeric features + native pandas 'category' dtype columns for
    `build_direct_panel`'s output, understood directly by both XGBoost
    (`enable_categorical=True`) and LightGBM (auto-detected). `horizon` is
    left as a plain numeric/ordinal column (not forced into a 'category'
    dtype like product/campaign) since it has a genuine order and small
    trees split on it naturally either way.
    """
    cols = direct_panel_feature_names(cfg) + TREE_CATEGORICAL_COLUMNS
    X = df[cols].copy()
    # Fixed, cfg-derived category domains -- NOT a bare `.astype("category")`,
    # which would infer each column's categories from whatever values
    # happen to be present in THIS specific DataFrame. train_panel and
    # eval_panel are built independently (different origins/rows) and
    # routinely disagree on which product/campaign ids are actually
    # present; XGBoost hard-errors the moment eval contains a category
    # train's slice didn't happen to include, and LightGBM would silently
    # misalign category codes instead of erroring. Every product_idx in
    # `0..cfg.num_products-1` / campaign_idx in `0..NUM_CAMPAIGN_CATS-1` is
    # declared upfront so train and eval always share identical categories.
    category_domains = {
        "product_idx": range(cfg.num_products),
        "campaign_idx_web": range(NUM_CAMPAIGN_CATS),
        "campaign_idx_app": range(NUM_CAMPAIGN_CATS),
    }
    for c in TREE_CATEGORICAL_COLUMNS:
        # campaign_idx_web/app come out of build_direct_panel's shift(-h)
        # against `train_feat`'s own int columns -- shifting past the end
        # of available data introduces NaN, which upcasts the whole column
        # to float64 (an int dtype can't hold NaN). Restore int (same
        # fillna(0) sentinel `prepare_features` already uses for an
        # unmapped/missing campaign) before the category cast: this
        # xgboost version hard-rejects category codes with a
        # floating-point dtype ("consider using strings or integers
        # instead").
        codes = X[c].fillna(0).astype(int)
        X[c] = pd.Categorical(codes, categories=category_domains[c])
    return X


# ---------------------------------------------------------------------------
# Model registry/metadata & metrics
# ---------------------------------------------------------------------------
MODEL_ORDER = ["NeuralNet", "XGBoost", "LightGBM", "DynamicRidge", "SeasonalNaive", "MovingAvg28"]

# Colors match each model's own project branding, so the dashboard visually
# echoes the tool it's describing: PyTorch's site/logo orange for the NN
# (this submission is a PyTorch model), XGBoost's brandfetch.com/xgboost.ai
# brand purple, and the LLVM/"Read the Docs" theme blue that
# lightgbm.readthedocs.io itself is built on. The two naive baselines have no
# such brand, so they get neutral slate tones.
MODEL_META = {
    "NeuralNet": {
        "label": "Neural Net",
        "short": "PyTorch",
        "color": "#EE4C2C",
        "kind": "primary",
        "source_url": "https://pytorch.org",
        "blurb": ("Feed-forward network with product & campaign embeddings. "
                  "The task brief's requested non-tree approach -- this is the actual submission."),
    },
    "XGBoost": {
        "label": "XGBoost",
        "short": "xgboost.ai",
        "color": "#7A43B6",
        "kind": "baseline",
        "source_url": "https://xgboost.ai",
        "blurb": ("Gradient-boosted trees (dmlc/xgboost). The task brief's own standard-approach "
                  "baseline -- evaluated for an honest comparison, not used for the final submission."),
    },
    "LightGBM": {
        "label": "LightGBM",
        "short": "readthedocs",
        "color": "#2980B9",
        "kind": "baseline",
        "source_url": "https://lightgbm.readthedocs.io/en/stable/",
        "blurb": ("Gradient-boosted trees with leaf-wise growth (Microsoft). Same role as "
                  "XGBoost: a standard-approach baseline, not the submission."),
    },
    "DynamicRidge": {
        "label": "Dynamic Ridge",
        "short": "sklearn-ridge",
        "color": "#10B981",
        "kind": "baseline",
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
        "blurb": "Linear model with L2 regularization, trained on the stacked panel. Represents a 'structured statistical' baseline.",
    },
    "SeasonalNaive": {
        "label": "Seasonal Naive",
        "short": "lag-7 baseline",
        "color": "#64748B",
        "kind": "naive",
        "source_url": None,
        "blurb": "Predicts each day using the actual value from exactly 7 days earlier. The sanity-check floor any real model should beat.",
    },
    "MovingAvg28": {
        "label": "Moving Average",
        "short": "28-day baseline",
        "color": "#94A3B8",
        "kind": "naive",
        "source_url": None,
        "blurb": "Predicts a flat value: the mean of the last 28 days. An even simpler floor baseline.",
    },
}


def model_slug(name: str) -> str:
    """URL-friendly key, e.g. "SeasonalNaive" -> "seasonalnaive"."""
    return name.lower().replace(" ", "")


MODEL_SLUGS = {name: model_slug(name) for name in MODEL_ORDER}
SLUG_TO_MODEL = {slug: name for name, slug in MODEL_SLUGS.items()}


def order_models(df: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    """Sort rows so the ML models come first (NN, then the two tree-based
    "standard approach" baselines), followed by the naive baselines. Any
    unlisted model name is appended alphabetically at the end."""
    present = set(df[column].unique())
    order = [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))
    original_columns = list(df.columns)
    result = df.set_index(column).loc[order].reset_index()
    return result[original_columns]


def compute_metrics(y_true, y_pred) -> dict:
    """MAE/RMSE stay scale-dependent; MAPE is kept only as a supplementary
    number since clipping its denominator at 1 makes it unstable near-zero.
    WAPE (sum|error|/sum|actual|) is scale-aware and the primary metric for
    comparing models across products of very different volume. sMAPE/RMSLE
    add robustness/percentage views; Bias/BiasRatio expose systematic over-
    or under-forecasting that MAE/RMSE hide (two models can share an MAE
    while one is unbiased and the other consistently over-forecasts).

    Calling this once per (fold, model) gives a "mean-fold" (macro) metric
    when averaged across folds by the caller; computing it once over all
    folds' pooled rows instead gives a "global" (micro) metric -- the two
    are not interchangeable and callers should label whichever they use.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    error = y_pred - y_true
    abs_error = np.abs(error)
    sum_abs_actual = float(np.sum(np.abs(y_true)))

    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mape = float(np.mean(abs_error / np.clip(y_true, 1, None)) * 100)
    wape = float(np.sum(abs_error) / sum_abs_actual) if sum_abs_actual > 0 else float("nan")
    smape = float(np.mean(2.0 * abs_error / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))
    rmsle = float(np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(np.clip(y_true, 0, None))) ** 2)))
    bias = float(np.mean(error))
    bias_ratio = float(np.sum(error) / sum_abs_actual) if sum_abs_actual > 0 else float("nan")

    return {
        "MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "sMAPE": smape,
        "RMSLE": rmsle, "Bias": bias, "BiasRatio": bias_ratio, "n": int(mask.sum()),
    }
