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
    reference_batch_size: int = 512
    nn_lr_scaling: str = "fixed"        # fixed | sqrt | linear
    nn_training_backend: str = "auto"   # auto | device_resident | dataloader
    cv_epochs: int = 30                   # no early stopping (avoids peeking at eval folds)
    final_epochs: int = 30                # must match the validated estimator policy
    seeds: tuple = (42, 123, 777)
    n_cv_folds: int = 4
    seed: int = 42
    ridge_alpha: float = 10.0
    ridge_prediction_cap: float | None = None

    # Tier C0.1: recursive numerical stability.  The neural network is
    # trained in baseline-relative log-residual space; robust support bounds
    # prevent a finite but unsupported residual from becoming a six-figure
    # natural-scale feedback value.  The generic guard remains a deliberately
    # broad last resort, not a normal retail prediction cap.
    nn_residual_guard_lower_quantile: float = 0.001
    nn_residual_guard_upper_quantile: float = 0.999
    nn_residual_guard_margin: float = 1.0
    recursive_safety_multiplier: float = 50.0
    recursive_safety_floor: float = 10_000.0

    # Tier C1: nonstationarity controls.  Defaults exactly preserve the C0
    # estimator.  A history window removes older supervised targets, while
    # half-life weighting keeps them but discounts their loss contribution.
    training_window_days: int | None = None
    recency_half_life_days: float | None = None
    baseline_variant: str = "weighted_4321"
    enable_trend_features: bool = False

    # Tier C2: semantic feature groups. The empty tuple preserves the C1
    # estimator exactly; groups are enabled explicitly by the screening runner
    # or CLI. Keeping groups named and atomic makes ablations reproducible.
    c2_feature_groups: tuple[str, ...] = ()

    # Tier C3: objective and target formulation. Defaults preserve the
    # confirmed C2 estimator. ``combined`` mixes Huber and MSE per row;
    # ``log1p`` predicts the raw log-count instead of a baseline residual.
    nn_loss: str = "huber"              # huber | mse | combined | logcosh
    nn_target_mode: str = "residual"    # residual | log1p
    nn_huber_delta: float = 1.0
    nn_combined_mse_weight: float = 0.25
    tree_target_mode: str = "log1p"     # shared fallback
    xgboost_target_mode: str | None = None
    lightgbm_target_mode: str | None = None
    tree_tweedie_variance_power: float = 1.5

    # Tier C4: channel-composition state and auxiliary task. Channel-history
    # features are opt-in so the confirmed C2 estimator remains reproducible.
    # Positive auxiliary weight trains an app-share head through the shared
    # representation while the submitted target stays total quantity.
    enable_channel_history_features: bool = False
    channel_aux_weight: float = 0.0
    channel_share_smoothing: float = 0.5

    # Tier C5: convex OOF ensemble. These controls do not change any member
    # estimator and are therefore excluded from fold-checkpoint signatures.
    enable_ensemble: bool = False
    ensemble_models: tuple[str, ...] = (
        "NeuralNet", "XGBoost", "LightGBM",
    )
    ensemble_grid_step: float = 0.01
    ensemble_min_relative_improvement: float = 0.002
    ensemble_benchmark_max_relative_regression: float = 0.02


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
    "is_sale", "price", "price_rel",
    # Keep the historical name for compatibility, but distinguish the two
    # lifecycle clocks explicitly.  Some products have rows long before they
    # first become available, while others simply have no pre-launch rows.
    "days_since_launch", "days_since_first_row",
    "days_since_first_available", "is_pre_first_available",
]

# C1 features are opt-in so the C0 baseline remains reproducible.  The
# calendar-time feature is target-date known; ratio features are origin/target
# history summaries and are computed in ``build_direct_panel``.
TREND_TARGET_FEATURES = ["calendar_time_years"]
TREND_ORIGIN_FEATURES = [
    "trend_log_ratio_mean_7_28",
    "trend_log_ratio_mean_14_28",
    "trend_log_ratio_lag0_28",
    "trend_log_slope_7",
    "trend_log_slope_28",
]
TREND_SEASONAL_FEATURES = [
    "annual_reference",
    "annual_reference_missing",
    "trend_log_ratio_baseline_annual",
]

BASELINE_VARIANTS = {
    "weighted_4321",
    "weighted_8421",
    "lag7",
    "weekday_median",
}

# Tier C2 semantic groups. These are deliberately grouped around business
# mechanisms rather than individual columns so the ablation answers useful
# questions without a combinatorial feature search.
C2_FEATURE_GROUPS = ("price", "campaign", "lifecycle", "market", "event")

NN_LOSSES = ("huber", "mse", "combined", "logcosh")
NN_TARGET_MODES = ("residual", "log1p")
TREE_TARGET_MODES = ("log1p", "residual", "tweedie")

CHANNEL_HISTORY_FEATURES = [
    "app_share_lag_0",
    "app_share_lag_7",
    "app_share_roll_7",
    "app_share_roll_28",
    "app_share_recent_long_delta",
    "app_share_observed_count_28",
    "app_qty_roll_mean_7",
    "app_qty_roll_mean_28",
    "web_qty_roll_mean_7",
    "web_qty_roll_mean_28",
]

PRICE_TARGET_FEATURES = [
    "app_effective_price_log_advantage",
]
PRICE_PANEL_FEATURES = [
    "price_log_ratio_vs_origin",
    "price_log_ratio_vs_lag7",
    "price_log_ratio_vs_median28",
    "effective_price_web_log_ratio_vs_median28",
    "effective_price_app_log_ratio_vs_median28",
]
CAMPAIGN_SEMANTIC_FEATURES = [
    "campaign_web_active",
    "campaign_app_active",
    "campaign_any_active",
    "app_only_campaign",
    "campaign_subtypes_match",
    "discount_without_campaign_web",
    "discount_without_campaign_app",
    "app_discount_advantage",
]
LIFECYCLE_ORIGIN_FEATURES = [
    "current_is_available",
    "current_is_calendar_gap",
    "consecutive_unavailable_days",
    "days_since_last_observed",
    "history_observed_days",
    "history_available_days",
    "recently_reavailable",
]
MARKET_TARGET_FEATURES = [
    "market_campaign_web_rate",
    "market_campaign_app_rate",
    "market_app_only_campaign_rate",
    "market_mean_discount_web",
    "market_mean_discount_app",
    "market_mean_app_discount_advantage",
]
MARKET_ORIGIN_FEATURES = [
    "market_total_qty_lag0",
    "market_total_qty_lag1",
    "market_total_qty_lag7",
    "market_roll_mean_7",
    "market_roll_mean_28",
    "market_recent_long_log_ratio",
    "market_mean_qty_per_available_lag0",
    "market_available_product_count_lag0",
    "market_total_excl_product_lag0",
]
EVENT_TARGET_FEATURES = [
    "days_from_black_friday",
    "black_friday_proximity_14",
    "days_from_christmas",
    "christmas_proximity_14",
    "days_from_valentine",
    "valentine_proximity_14",
    "days_from_mothers_day",
    "mothers_day_proximity_14",
    "is_black_friday_window",
    "is_christmas_window",
    "is_new_year_window",
]


def normalize_c2_feature_groups(value) -> tuple[str, ...]:
    """Canonicalise a C2 group specification.

    Accepts an iterable or a comma-separated string. ``all`` expands to every
    group and ``none``/empty disables C2, which preserves the confirmed C1
    estimator. The canonical order is stable for checkpoint signatures.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none"}:
            return ()
        if text == "all":
            return C2_FEATURE_GROUPS
        tokens = [token.strip().lower() for token in text.split(",") if token.strip()]
    else:
        tokens = [str(token).strip().lower() for token in value if str(token).strip()]
    unknown = sorted(set(tokens) - set(C2_FEATURE_GROUPS))
    if unknown:
        raise ValueError(
            f"Unknown C2 feature groups {unknown}; expected {list(C2_FEATURE_GROUPS)}"
        )
    token_set = set(tokens)
    return tuple(group for group in C2_FEATURE_GROUPS if group in token_set)


def c2_group_enabled(cfg: Config, group: str) -> bool:
    return group in normalize_c2_feature_groups(cfg.c2_feature_groups)


def lag_feature_names(lag_windows) -> list[str]:
    """Origin-history feature names.

    ``stockout_rate`` is retained as a backward-compatible alias for the
    observed-unavailable rate.  Calendar gaps are tracked separately rather
    than silently being counted as stockouts.
    """
    names = []
    for w in lag_windows:
        names += [
            f"qty_roll_mean_{w}", f"qty_roll_std_{w}",
            f"qty_roll_median_{w}", f"qty_available_count_{w}",
            f"observed_count_{w}", f"unavailable_count_{w}",
            f"calendar_gap_count_{w}", f"available_observation_rate_{w}",
            f"observed_rate_{w}", f"unavailable_rate_{w}",
            f"calendar_gap_rate_{w}", f"stockout_rate_{w}",
        ]
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
    ProductAvailable are unknown, not zero: they're filled as NaN / <NA>.
    Availability-aware rolling features keep these calendar gaps separate
    from observed stockouts. `is_gap_filled` records provenance.
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


def product_reference_dates(
    raw_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Training-only first-row and first-confirmed-available dates.

    Products in this dataset encode lifecycle differently: some have no rows
    before launch, while others have long observed-but-unavailable prefixes.
    Keeping both clocks prevents those states from being conflated.
    """
    first_seen = raw_df.groupby("ProductId")["DateKey"].min()
    gap = raw_df.get(
        "is_gap_filled", pd.Series(False, index=raw_df.index)
    ).astype("boolean").fillna(False).astype(bool)
    available = raw_df["ProductAvailable"].fillna(False).astype(bool) & ~gap
    first_available = (
        raw_df.loc[available]
        .groupby("ProductId")["DateKey"]
        .min()
        .reindex(first_seen.index)
        .fillna(first_seen)
    )
    return first_seen, first_available


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


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _black_friday(year: int) -> pd.Timestamp:
    return _nth_weekday_of_month(year, 11, 4, 4)


def _mothers_day(year: int) -> pd.Timestamp:
    # Czech and Slovak retail calendars use the second Sunday in May.
    return _nth_weekday_of_month(year, 5, 6, 2)


def _nearest_annual_event_distance(
    dates: pd.Series, event_factory, *, clip_days: int = 60
) -> np.ndarray:
    """Signed days from the nearest annual occurrence, clipped for stability."""
    values = pd.to_datetime(dates).reset_index(drop=True)
    years = values.dt.year
    required_years = range(int(years.min()) - 1, int(years.max()) + 2)
    event_by_year = {year: event_factory(year) for year in required_years}
    distances = []
    for offset in (-1, 0, 1):
        events = (years + offset).map(event_by_year)
        distances.append((values - events).dt.days.to_numpy(dtype=float))
    matrix = np.column_stack(distances)
    nearest = matrix[np.arange(len(matrix)), np.abs(matrix).argmin(axis=1)]
    return np.clip(nearest, -clip_days, clip_days).astype(float)


def add_retail_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Known-in-advance retail-event distance and window features."""
    dates = pd.to_datetime(df["DateKey"])
    event_specs = {
        "black_friday": _black_friday,
        "christmas": lambda year: pd.Timestamp(year=year, month=12, day=24),
        "valentine": lambda year: pd.Timestamp(year=year, month=2, day=14),
        "mothers_day": _mothers_day,
    }
    for name, factory in event_specs.items():
        distance = _nearest_annual_event_distance(dates, factory)
        df[f"days_from_{name}"] = distance
        df[f"{name}_proximity_14"] = np.exp(-np.abs(distance) / 14.0)
    df["is_black_friday_window"] = (
        np.abs(df["days_from_black_friday"]) <= 4
    ).astype(float)
    df["is_christmas_window"] = (
        np.abs(df["days_from_christmas"]) <= 10
    ).astype(float)
    # The post-Christmas/New-Year demand regime spans the turn of the year.
    month_day = dates.dt.strftime("%m-%d")
    df["is_new_year_window"] = (
        month_day.ge("12-27") | month_day.le("01-07")
    ).astype(float)
    return df


def prepare_features(
    df: pd.DataFrame,
    price_ref: pd.Series,
    first_seen: pd.Series,
    first_available: pd.Series | None = None,
    cfg: Config = CFG,
) -> pd.DataFrame:
    """Add features that do not depend on the target's own recent history.

    ``price_ref``, ``first_seen`` and ``first_available`` must be computed from
    training-only data by the caller.  ``first_available`` is optional for
    backward compatibility; when absent it falls back to ``first_seen``.
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

    # C2 semantics are computed only when their group is active. This keeps
    # the confirmed C1 control fast and ensures local experiment Config copies
    # (not only the module-global CFG) determine the actual feature contract.
    need_campaign_semantics = (
        c2_group_enabled(cfg, "campaign") or c2_group_enabled(cfg, "market")
    )
    if need_campaign_semantics:
        web_subtype = pd.to_numeric(
            df["CampaignSubTypeWeb"], errors="coerce"
        ).fillna(-1).astype(int)
        app_subtype = pd.to_numeric(
            df["CampaignSubTypeApp"], errors="coerce"
        ).fillna(-1).astype(int)
        df["campaign_web_active"] = (web_subtype != -1).astype(float)
        df["campaign_app_active"] = (app_subtype != -1).astype(float)
        df["campaign_any_active"] = (
            (df["campaign_web_active"] > 0) | (df["campaign_app_active"] > 0)
        ).astype(float)
        df["app_only_campaign"] = (
            (df["campaign_app_active"] > 0) & (df["campaign_web_active"] == 0)
        ).astype(float)
        df["campaign_subtypes_match"] = (web_subtype == app_subtype).astype(float)
        df["discount_without_campaign_web"] = (
            (web_subtype == -1) & (df["discount_web"] > 0)
        ).astype(float)
        df["discount_without_campaign_app"] = (
            (app_subtype == -1) & (df["discount_app"] > 0)
        ).astype(float)
        df["app_discount_advantage"] = df["discount_app"] - df["discount_web"]

    if c2_group_enabled(cfg, "price"):
        df["app_effective_price_log_advantage"] = (
            np.log1p(np.clip(df["effective_price_web"].to_numpy(dtype=float), 0.0, None))
            - np.log1p(np.clip(df["effective_price_app"].to_numpy(dtype=float), 0.0, None))
        )

    if c2_group_enabled(cfg, "market"):
        # Target-date market promotion intensity is known for the supplied
        # future panel. It contains no quantity information.
        by_date = df.groupby("DateKey", sort=False)
        df["market_campaign_web_rate"] = by_date["campaign_web_active"].transform("mean")
        df["market_campaign_app_rate"] = by_date["campaign_app_active"].transform("mean")
        df["market_app_only_campaign_rate"] = by_date["app_only_campaign"].transform("mean")
        df["market_mean_discount_web"] = by_date["discount_web"].transform("mean")
        df["market_mean_discount_app"] = by_date["discount_app"].transform("mean")
        df["market_mean_app_discount_advantage"] = by_date["app_discount_advantage"].transform("mean")

    if c2_group_enabled(cfg, "event"):
        df = add_retail_event_features(df)

    ref = df["ProductId"].map(price_ref).replace(0, np.nan)
    df["price_rel"] = (df["price"] / ref).fillna(1.0)
    first_row_date = df["ProductId"].map(first_seen)
    if first_available is None:
        first_available = first_seen
    first_available_date = df["ProductId"].map(first_available).fillna(first_row_date)
    df["days_since_first_row"] = (df["DateKey"] - first_row_date).dt.days
    # Historical compatibility: the old feature was actually days since the
    # first row, not necessarily since launch/availability.
    df["days_since_launch"] = df["days_since_first_row"]
    df["days_since_first_available"] = (
        df["DateKey"] - first_available_date
    ).dt.days
    df["is_pre_first_available"] = (
        df["DateKey"] < first_available_date
    ).astype(int)
    # Absolute calendar time lets a pooled model represent market-wide level
    # drift (especially the 2024-2026 web decline) without treating ProductId
    # lifecycle age as a proxy for the global regime.  It is only included in
    # the model schema when ``enable_trend_features`` is true.
    df["calendar_time_years"] = (
        df["DateKey"] - pd.Timestamp("2021-01-01")
    ).dt.days.astype(float) / 365.25

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


def _baseline_weights(variant: str) -> np.ndarray | None:
    if variant not in BASELINE_VARIANTS:
        raise ValueError(
            f"Unknown baseline_variant={variant!r}; expected one of "
            f"{sorted(BASELINE_VARIANTS)}"
        )
    if variant == "weighted_4321":
        return np.array([4.0, 3.0, 2.0, 1.0], dtype=float)
    if variant == "weighted_8421":
        return np.array([8.0, 4.0, 2.0, 1.0], dtype=float)
    if variant == "lag7":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return None


def _weighted_baseline(
    lag_matrix: np.ndarray,
    variant: str = "weighted_4321",
) -> np.ndarray:
    """Row-wise NaN-aware same-weekday baseline.

    ``weighted_4321`` is the C0 default. C1 can compare a pure lag-7
    baseline, a more recent-heavy 8:4:2:1 average, and a robust weekday
    median. Missing lags never force an otherwise usable row to be dropped.
    """
    matrix = np.asarray(lag_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(BASELINE_LAGS):
        raise ValueError(
            f"lag_matrix must have {len(BASELINE_LAGS)} columns in BASELINE_LAGS order"
        )
    observed = np.isfinite(matrix)
    if variant == "weekday_median":
        result = np.full(len(matrix), np.nan, dtype=float)
        valid = observed.any(axis=1)
        if valid.any():
            result[valid] = np.nanmedian(matrix[valid], axis=1)
        return result

    weights = _baseline_weights(variant)
    numerator = np.nansum(matrix * weights, axis=1)
    denominator = (observed * weights).sum(axis=1)
    return np.divide(
        numerator, denominator,
        out=np.full(len(matrix), np.nan, dtype=float),
        where=denominator > 0,
    )


def compute_baseline(
    target_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    baseline_variant: str = "weighted_4321",
) -> np.ndarray:
    """Availability-aware same-weekday baseline using observed history only."""
    available = hist_df["ProductAvailable"].fillna(False)
    qty_available = hist_df["Quantity"].where(available)
    lookup = pd.Series(
        qty_available.to_numpy(),
        index=pd.MultiIndex.from_frame(hist_df[["ProductId", "DateKey"]]),
    )

    lag_matrix = np.full((len(target_df), len(BASELINE_LAGS)), np.nan)
    for j, lag in enumerate(BASELINE_LAGS):
        keys = list(zip(
            target_df["ProductId"],
            target_df["DateKey"] - pd.Timedelta(days=lag),
        ))
        lag_matrix[:, j] = [lookup.get(k, np.nan) for k in keys]

    return _weighted_baseline(lag_matrix, baseline_variant)


def _availability_state(df: pd.DataFrame) -> pd.DataFrame:
    """Return explicit observed/available/unavailable/gap state columns."""
    state = pd.DataFrame(index=df.index)
    if "is_gap_filled" in df.columns:
        gap = df["is_gap_filled"].astype("boolean").fillna(False).astype(bool)
    else:
        gap = pd.Series(False, index=df.index)
    product_available = df["ProductAvailable"].astype("boolean")
    observed = ~gap
    available = observed & product_available.fillna(False).astype(bool)
    unavailable = observed & product_available.eq(False).fillna(False).astype(bool)
    state["_is_gap"] = gap.astype(float)
    state["_is_observed"] = observed.astype(float)
    state["_is_available"] = available.astype(float)
    state["_is_unavailable"] = unavailable.astype(float)
    state["qty_available"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    ).where(available)
    return state


def _window_state_features(
    df: pd.DataFrame,
    windows: tuple,
    *,
    include_current: bool,
) -> pd.DataFrame:
    """Compute availability-aware rolling features with explicit states."""
    state = _availability_state(df)
    work = pd.concat([df[["ProductId", "DateKey"]].reset_index(drop=True),
                      state.reset_index(drop=True)], axis=1)
    qty_group = work.groupby("ProductId")["qty_available"]
    observed_group = work.groupby("ProductId")["_is_observed"]
    unavailable_group = work.groupby("ProductId")["_is_unavailable"]
    gap_group = work.groupby("ProductId")["_is_gap"]
    offset = 0 if include_current else 1
    row_num = work.groupby("ProductId").cumcount() + (1 if include_current else 0)
    out = work[["ProductId", "DateKey"]].copy()
    out["qty_available"] = work["qty_available"]

    def rolled(group, window, method, *, fill_std=False):
        def apply(series):
            base = series if offset == 0 else series.shift(offset)
            result = getattr(base.rolling(window, min_periods=1), method)()
            return result.fillna(0.0) if fill_std else result
        return group.transform(apply)

    for w in windows:
        out[f"qty_roll_mean_{w}"] = rolled(qty_group, w, "mean")
        out[f"qty_roll_std_{w}"] = rolled(qty_group, w, "std", fill_std=True)
        out[f"qty_roll_median_{w}"] = rolled(qty_group, w, "median")
        qty_count = rolled(qty_group, w, "count")
        observed_count = rolled(observed_group, w, "sum")
        unavailable_count = rolled(unavailable_group, w, "sum")
        gap_count = rolled(gap_group, w, "sum")
        denominator = np.minimum(row_num, w).clip(lower=1).astype(float)
        out[f"qty_available_count_{w}"] = qty_count
        out[f"observed_count_{w}"] = observed_count
        out[f"unavailable_count_{w}"] = unavailable_count
        out[f"calendar_gap_count_{w}"] = gap_count
        out[f"available_observation_rate_{w}"] = qty_count / denominator
        out[f"observed_rate_{w}"] = observed_count / denominator
        out[f"unavailable_rate_{w}"] = unavailable_count / denominator
        out[f"calendar_gap_rate_{w}"] = gap_count / denominator
        out[f"stockout_rate_{w}"] = out[f"unavailable_rate_{w}"]
    return out


def add_train_lags(
    df: pd.DataFrame,
    windows: tuple = CFG.lag_windows,
    *,
    baseline_variant: str = "weighted_4321",
) -> pd.DataFrame:
    """Target-row history features computed strictly from prior days.

    Calendar gaps and observed-unavailable rows are now represented
    separately.  Only observed-and-available quantities enter demand rolling
    statistics; ``stockout_rate`` is an alias of the explicit unavailable
    rate rather than a mixture of stockouts and unknown calendar gaps.
    """
    df = df.sort_values(["ProductId", "DateKey"]).reset_index(drop=True).copy()
    history = _window_state_features(df, windows, include_current=False)
    for col in history.columns:
        if col not in {"ProductId", "DateKey"}:
            df[col] = history[col].to_numpy()
    df["baseline"] = compute_baseline(
        df, df, baseline_variant=baseline_variant
    )
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
ANNUAL_LAG_DAYS = (364, 365, 371)
SEASONAL_LAG_DAYS = BASELINE_LAGS + ANNUAL_LAG_DAYS
ANNUAL_LAG_MISSING_FEATURES = [
    f"seasonal_lag_{lag}_missing" for lag in ANNUAL_LAG_DAYS
]
ORIGIN_LIFECYCLE_FEATURES = ["days_since_last_available", "ever_available_before"]

# Columns whose VALUE must be shifted forward from the target row (the two
# campaign category codes included, so the panel reflects whatever
# campaign is active ON the target date) -- used only by `build_direct_panel`
# itself. "Future-known" because the task's own test_data.parquet already
# supplies these for the real forecast week -- an assumption this panel
# inherits, not one it introduces.
TARGET_COVARIATE_COLUMNS = STATIC_NUMERIC_FEATURES + ["campaign_idx_web", "campaign_idx_app"]


def target_numeric_feature_names(cfg: Config = CFG) -> list[str]:
    features = list(STATIC_NUMERIC_FEATURES)
    if cfg.enable_trend_features:
        features += TREND_TARGET_FEATURES
    if c2_group_enabled(cfg, "price"):
        features += PRICE_TARGET_FEATURES
    if c2_group_enabled(cfg, "campaign"):
        features += CAMPAIGN_SEMANTIC_FEATURES
    if c2_group_enabled(cfg, "market"):
        features += MARKET_TARGET_FEATURES
    if c2_group_enabled(cfg, "event"):
        features += EVENT_TARGET_FEATURES
    return features


def target_covariate_columns(cfg: Config = CFG) -> list[str]:
    return target_numeric_feature_names(cfg) + [
        "campaign_idx_web", "campaign_idx_app"
    ]


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
    trend_origin = TREND_ORIGIN_FEATURES if cfg.enable_trend_features else []
    trend_seasonal = TREND_SEASONAL_FEATURES if cfg.enable_trend_features else []
    c2_origin: list[str] = []
    c2_panel: list[str] = []
    if c2_group_enabled(cfg, "price"):
        c2_panel += PRICE_PANEL_FEATURES
    if c2_group_enabled(cfg, "lifecycle"):
        c2_origin += LIFECYCLE_ORIGIN_FEATURES
    if c2_group_enabled(cfg, "market"):
        c2_origin += MARKET_ORIGIN_FEATURES
    return (target_numeric_feature_names(cfg) + lag_feature_names(cfg.lag_windows)
            + ORIGIN_LIFECYCLE_FEATURES
            + [f"qty_lag_{lag}" for lag in RECENT_POINT_LAGS]
            + trend_origin + c2_origin
            + (CHANNEL_HISTORY_FEATURES if cfg.enable_channel_history_features else [])
            + [f"seasonal_lag_{lag}" for lag in SEASONAL_LAG_DAYS]
            + ANNUAL_LAG_MISSING_FEATURES
            + trend_seasonal + c2_panel
            + ["target_baseline_missing", "target_baseline", "horizon"])


def _rolling_log_slope(values: np.ndarray) -> float:
    """Least-squares slope of log1p demand over observed positions only."""
    arr = np.asarray(values, dtype=float)
    observed = np.isfinite(arr) & (arr >= 0.0)
    if observed.sum() < 2:
        return np.nan
    x = np.arange(len(arr), dtype=float)[observed]
    y = np.log1p(arr[observed])
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return np.nan
    return float(np.dot(x_centered, y - y.mean()) / denominator)


def _safe_log_ratio_values(left, right) -> np.ndarray:
    left_arr = pd.to_numeric(pd.Series(left), errors="coerce").to_numpy(dtype=float)
    right_arr = pd.to_numeric(pd.Series(right), errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(left_arr) & np.isfinite(right_arr)
        & (left_arr >= 0.0) & (right_arr >= 0.0)
    )
    result = np.full(len(left_arr), np.nan, dtype=float)
    result[valid] = np.log1p(left_arr[valid]) - np.log1p(right_arr[valid])
    return result


def build_origin_state_features(feature_df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Build features known at the end of each origin day."""
    df = feature_df.sort_values(["ProductId", "DateKey"]).reset_index(drop=True).copy()
    state = _window_state_features(df, cfg.lag_windows, include_current=True)
    out = state.drop(columns=["qty_available"]).copy()
    qty_group = state.groupby("ProductId")["qty_available"]
    for lag in RECENT_POINT_LAGS:
        out[f"qty_lag_{lag}"] = qty_group.shift(lag)

    if cfg.enable_channel_history_features:
        # Channel-state features use only observations available by the origin.
        # Unavailable/gap rows are excluded consistently with total-demand
        # rolling features. Recursive synthetic rows are marked available and
        # carry the model-predicted split, so the same contract remains valid
        # beyond horizon one.
        availability = _availability_state(df)
        valid = availability["_is_available"].astype(bool)
        app = pd.to_numeric(
            df.get("QuantityApp", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        ).where(valid)
        web = pd.to_numeric(
            df.get("QuantityWeb", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        ).where(valid)
        total = app + web
        share = pd.Series(
            np.divide(
                app.to_numpy(dtype=float),
                total.to_numpy(dtype=float),
                out=np.full(len(df), np.nan, dtype=float),
                where=np.isfinite(total.to_numpy(dtype=float))
                & (total.to_numpy(dtype=float) > 0.0),
            ),
            index=df.index,
        )
        product = df["ProductId"]
        share_group = share.groupby(product, sort=False)
        app_group = app.groupby(product, sort=False)
        web_group = web.groupby(product, sort=False)
        total_group = total.groupby(product, sort=False)
        out["app_share_lag_0"] = share
        out["app_share_lag_7"] = share_group.shift(7)

        share_roll = {}
        for window in (7, 28):
            app_sum = app_group.transform(
                lambda series, w=window: series.rolling(w, min_periods=1).sum()
            )
            total_sum = total_group.transform(
                lambda series, w=window: series.rolling(w, min_periods=1).sum()
            )
            share_roll[window] = np.divide(
                app_sum.to_numpy(dtype=float),
                total_sum.to_numpy(dtype=float),
                out=np.full(len(df), np.nan, dtype=float),
                where=np.isfinite(total_sum.to_numpy(dtype=float))
                & (total_sum.to_numpy(dtype=float) > 0.0),
            )
            out[f"app_share_roll_{window}"] = share_roll[window]
            out[f"app_qty_roll_mean_{window}"] = app_group.transform(
                lambda series, w=window: series.rolling(w, min_periods=1).mean()
            )
            out[f"web_qty_roll_mean_{window}"] = web_group.transform(
                lambda series, w=window: series.rolling(w, min_periods=1).mean()
            )
        out["app_share_recent_long_delta"] = share_roll[7] - share_roll[28]
        out["app_share_observed_count_28"] = share_group.transform(
            lambda series: series.rolling(28, min_periods=1).count()
        )

    if c2_group_enabled(cfg, "price"):
        price_group = df.groupby("ProductId", sort=False)
        out["_origin_price_lag0"] = pd.to_numeric(df["price"], errors="coerce")
        out["_origin_price_lag7"] = price_group["price"].shift(7)
        out["_origin_price_median28"] = price_group["price"].transform(
            lambda series: series.rolling(28, min_periods=1).median()
        )
        out["_origin_effective_price_web_median28"] = price_group[
            "effective_price_web"
        ].transform(lambda series: series.rolling(28, min_periods=1).median())
        out["_origin_effective_price_app_median28"] = price_group[
            "effective_price_app"
        ].transform(lambda series: series.rolling(28, min_periods=1).median())

    if cfg.enable_trend_features:
        def log_ratio(left: pd.Series, right: pd.Series) -> np.ndarray:
            left_arr = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
            right_arr = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
            valid = (
                np.isfinite(left_arr) & np.isfinite(right_arr)
                & (left_arr >= 0.0) & (right_arr >= 0.0)
            )
            result = np.full(len(out), np.nan, dtype=float)
            result[valid] = np.log1p(left_arr[valid]) - np.log1p(right_arr[valid])
            return result

        if 7 in cfg.lag_windows and 28 in cfg.lag_windows:
            out["trend_log_ratio_mean_7_28"] = log_ratio(
                out["qty_roll_mean_7"], out["qty_roll_mean_28"]
            )
            out["trend_log_ratio_lag0_28"] = log_ratio(
                out["qty_lag_0"], out["qty_roll_mean_28"]
            )
        else:
            out["trend_log_ratio_mean_7_28"] = np.nan
            out["trend_log_ratio_lag0_28"] = np.nan
        if 14 in cfg.lag_windows and 28 in cfg.lag_windows:
            out["trend_log_ratio_mean_14_28"] = log_ratio(
                out["qty_roll_mean_14"], out["qty_roll_mean_28"]
            )
        else:
            out["trend_log_ratio_mean_14_28"] = np.nan

        # Short and medium log-demand slopes expose direction of travel, not
        # merely the recent/long level ratio. Missing calendar/availability
        # states are ignored while their original positions remain in x.
        for window in (7, 28):
            out[f"trend_log_slope_{window}"] = qty_group.transform(
                lambda series, w=window: series.rolling(
                    w, min_periods=2
                ).apply(_rolling_log_slope, raw=True)
            )

    availability = _availability_state(df)
    available_bool = availability["_is_available"].astype(bool)
    observed_bool = availability["_is_observed"].astype(bool)
    unavailable_bool = availability["_is_unavailable"].astype(bool)
    gap_bool = availability["_is_gap"].astype(bool)

    available_date = df["DateKey"].where(available_bool)
    last_available = available_date.groupby(df["ProductId"]).ffill()
    out["days_since_last_available"] = (
        df["DateKey"] - last_available
    ).dt.days.astype(float)
    out["ever_available_before"] = last_available.notna().astype(float)

    if c2_group_enabled(cfg, "lifecycle"):
        observed_date = df["DateKey"].where(observed_bool)
        last_observed = observed_date.groupby(df["ProductId"]).ffill()
        out["current_is_available"] = available_bool.astype(float)
        out["current_is_calendar_gap"] = gap_bool.astype(float)
        out["days_since_last_observed"] = (
            df["DateKey"] - last_observed
        ).dt.days.astype(float)
        out["history_observed_days"] = observed_bool.astype(float).groupby(
            df["ProductId"]
        ).cumsum()
        out["history_available_days"] = available_bool.astype(float).groupby(
            df["ProductId"]
        ).cumsum()
        out["consecutive_unavailable_days"] = unavailable_bool.astype(float).groupby(
            [df["ProductId"], (~unavailable_bool).groupby(df["ProductId"]).cumsum()]
        ).cumsum()
        previous_unavailable = (
            unavailable_bool.groupby(df["ProductId"]).shift(1)
            .astype("boolean").fillna(False).astype(bool)
        )
        out["recently_reavailable"] = (
            available_bool & previous_unavailable.astype(bool)
        ).astype(float)

    if c2_group_enabled(cfg, "market"):
        market_work = pd.DataFrame({
            "DateKey": df["DateKey"],
            "qty_available": state["qty_available"],
            "is_available": availability["_is_available"],
        })
        market = market_work.groupby("DateKey", sort=True).agg(
            market_total_qty=("qty_available", lambda x: x.sum(min_count=1)),
            market_available_product_count=("is_available", "sum"),
        ).sort_index()
        market["market_mean_qty_per_available"] = np.divide(
            market["market_total_qty"],
            market["market_available_product_count"],
        )
        market["market_total_qty_lag0"] = market["market_total_qty"]
        market["market_total_qty_lag1"] = market["market_total_qty"].shift(1)
        market["market_total_qty_lag7"] = market["market_total_qty"].shift(7)
        market["market_roll_mean_7"] = market["market_total_qty"].rolling(
            7, min_periods=1
        ).mean()
        market["market_roll_mean_28"] = market["market_total_qty"].rolling(
            28, min_periods=1
        ).mean()
        market["market_recent_long_log_ratio"] = _safe_log_ratio_values(
            market["market_roll_mean_7"], market["market_roll_mean_28"]
        )
        market["market_mean_qty_per_available_lag0"] = market[
            "market_mean_qty_per_available"
        ]
        market["market_available_product_count_lag0"] = market[
            "market_available_product_count"
        ]
        for column in MARKET_ORIGIN_FEATURES:
            if column == "market_total_excl_product_lag0":
                continue
            out[column] = df["DateKey"].map(market[column])
        own_qty = state["qty_available"].fillna(0.0).to_numpy(dtype=float)
        out["market_total_excl_product_lag0"] = (
            out["market_total_qty_lag0"].to_numpy(dtype=float) - own_qty
        )

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

    train_feat = train_feat.copy()
    if "QuantityApp" not in train_feat.columns:
        train_feat["QuantityApp"] = train_feat.get("Quantity", np.nan)
    if "QuantityWeb" not in train_feat.columns:
        train_feat["QuantityWeb"] = 0.0
    train_feat = train_feat.sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    origin_index = pd.MultiIndex.from_frame(train_feat[["ProductId", "DateKey"]])
    combined = train_feat.copy()
    if future_covariates is not None:
        future_covariates = future_covariates.copy()
        for col in ("Quantity", "QuantityApp", "QuantityWeb", "ProductAvailable"):
            if col not in future_covariates.columns:
                future_covariates[col] = np.nan
        covariate_columns = target_covariate_columns(cfg)
        keep = [
            "ProductId", "DateKey", "Quantity", "QuantityApp", "QuantityWeb",
            "ProductAvailable",
        ] + covariate_columns
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
        covariate_columns = target_covariate_columns(cfg)
        target_cols = [
            "DateKey", "Quantity", "QuantityApp", "QuantityWeb",
            "ProductAvailable",
        ] + covariate_columns
        target = g[target_cols].shift(-h)
        panel_h["TargetDateKey"] = target["DateKey"]
        panel_h["target"] = target["Quantity"]
        panel_h["target_app"] = target["QuantityApp"]
        panel_h["target_web"] = target["QuantityWeb"]
        panel_h["TargetProductAvailable"] = target["ProductAvailable"]
        for col in covariate_columns:
            panel_h[col] = target[col]

        if c2_group_enabled(cfg, "price"):
            panel_h["price_log_ratio_vs_origin"] = _safe_log_ratio_values(
                panel_h["price"], panel_h["_origin_price_lag0"]
            )
            panel_h["price_log_ratio_vs_lag7"] = _safe_log_ratio_values(
                panel_h["price"], panel_h["_origin_price_lag7"]
            )
            panel_h["price_log_ratio_vs_median28"] = _safe_log_ratio_values(
                panel_h["price"], panel_h["_origin_price_median28"]
            )
            panel_h["effective_price_web_log_ratio_vs_median28"] = (
                _safe_log_ratio_values(
                    panel_h["effective_price_web"],
                    panel_h["_origin_effective_price_web_median28"],
                )
            )
            panel_h["effective_price_app_log_ratio_vs_median28"] = (
                _safe_log_ratio_values(
                    panel_h["effective_price_app"],
                    panel_h["_origin_effective_price_app_median28"],
                )
            )

        for lag in SEASONAL_LAG_DAYS:
            panel_h[f"seasonal_lag_{lag}"] = g["qty_available"].shift(lag - h)
        for lag in ANNUAL_LAG_DAYS:
            panel_h[f"seasonal_lag_{lag}_missing"] = (
                ~np.isfinite(panel_h[f"seasonal_lag_{lag}"].to_numpy(dtype=float))
            ).astype(float)
        lag_matrix = np.column_stack([
            panel_h[f"seasonal_lag_{lag}"].to_numpy(dtype=float)
            for lag in BASELINE_LAGS
        ])
        raw_baseline = _weighted_baseline(lag_matrix, cfg.baseline_variant)
        panel_h["target_baseline_missing"] = (~np.isfinite(raw_baseline)).astype(float)
        fallback = panel_h[f"qty_roll_median_{cfg.lag_windows[0]}"].to_numpy(dtype=float)
        fallback = np.where(
            np.isfinite(fallback), fallback,
            panel_h[f"qty_roll_mean_{cfg.lag_windows[0]}"].to_numpy(dtype=float),
        )
        fallback = np.where(
            np.isfinite(fallback), fallback,
            panel_h["qty_lag_0"].to_numpy(dtype=float),
        )
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        panel_h["target_baseline"] = np.where(
            np.isfinite(raw_baseline), raw_baseline, fallback
        )
        if cfg.enable_trend_features:
            annual_matrix = np.column_stack([
                panel_h[f"seasonal_lag_{lag}"].to_numpy(dtype=float)
                for lag in ANNUAL_LAG_DAYS
            ])
            annual_observed = np.isfinite(annual_matrix).any(axis=1)
            annual_reference = np.full(len(panel_h), np.nan, dtype=float)
            if annual_observed.any():
                annual_reference[annual_observed] = np.nanmedian(
                    annual_matrix[annual_observed], axis=1
                )
            panel_h["annual_reference"] = annual_reference
            panel_h["annual_reference_missing"] = (~annual_observed).astype(float)
            baseline_values = panel_h["target_baseline"].to_numpy(dtype=float)
            valid_ratio = (
                np.isfinite(baseline_values) & np.isfinite(annual_reference)
                & (baseline_values >= 0.0) & (annual_reference >= 0.0)
            )
            ratio = np.full(len(panel_h), np.nan, dtype=float)
            ratio[valid_ratio] = (
                np.log1p(baseline_values[valid_ratio])
                - np.log1p(annual_reference[valid_ratio])
            )
            panel_h["trend_log_ratio_baseline_annual"] = ratio
        frames.append(panel_h)

    panel = pd.concat(frames, ignore_index=True).rename(columns={"DateKey": "OriginDateKey"})
    panel["product_idx"] = panel["ProductId"] - 1
    panel_index = pd.MultiIndex.from_arrays([panel["ProductId"], panel["OriginDateKey"]])
    return panel[panel_index.isin(origin_index)].reset_index(drop=True)


def recency_sample_weights(
    target_dates: pd.Series,
    cutoff: pd.Timestamp,
    half_life_days: float | None,
) -> np.ndarray:
    """Return mean-one exponential time-decay weights.

    Normalising to mean one preserves the overall loss scale and therefore
    avoids silently changing the effective learning rate when C1 enables
    recency weighting.
    """
    dates = pd.to_datetime(target_dates)
    age_days = (pd.Timestamp(cutoff) - dates).dt.days.to_numpy(dtype=float)
    age_days = np.clip(age_days, 0.0, None)
    if half_life_days is None:
        return np.ones(len(dates), dtype=float)
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("recency_half_life_days must be positive or None")
    weights = np.exp2(-age_days / float(half_life_days))
    mean_weight = float(np.mean(weights)) if len(weights) else 1.0
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Recency weighting produced an invalid mean weight")
    return weights / mean_weight


def select_trainable_panel_rows(
    panel: pd.DataFrame,
    *,
    cutoff: pd.Timestamp | None = None,
    available_only: bool = True,
    cfg: Config = CFG,
) -> pd.DataFrame:
    """Select supervised rows without requiring every feature to be present.

    Numeric feature missingness is handled by each model's fitted
    preprocessing/native missing-value support.  This prevents annual lags
    from silently deleting young-product and early-history observations.
    """
    mask = panel["target"].notna() & np.isfinite(
        pd.to_numeric(panel["target"], errors="coerce")
    )
    mask &= panel["target_baseline"].notna() & np.isfinite(
        pd.to_numeric(panel["target_baseline"], errors="coerce")
    )
    effective_cutoff = (
        pd.Timestamp(cutoff)
        if cutoff is not None
        else pd.to_datetime(panel.loc[mask, "TargetDateKey"]).max()
    )
    if pd.isna(effective_cutoff):
        selected = panel.loc[mask].reset_index(drop=True).copy()
        selected["sample_weight"] = np.ones(len(selected), dtype=float)
        return selected
    if cutoff is not None:
        mask &= panel["TargetDateKey"].le(effective_cutoff)
    if cfg.training_window_days is not None:
        if cfg.training_window_days <= 0:
            raise ValueError("training_window_days must be positive or None")
        earliest = effective_cutoff - pd.Timedelta(
            days=int(cfg.training_window_days) - 1
        )
        mask &= panel["TargetDateKey"].ge(earliest)
    if available_only:
        mask &= (
            panel["TargetProductAvailable"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    selected = panel.loc[mask].reset_index(drop=True).copy()
    selected["sample_weight"] = recency_sample_weights(
        selected["TargetDateKey"],
        effective_cutoff,
        cfg.recency_half_life_days,
    )
    return selected


def build_one_step_panel(
    raw_df: pd.DataFrame,
    price_ref: pd.Series,
    first_seen: pd.Series,
    cfg: Config = CFG,
    first_available: pd.Series | None = None,
) -> pd.DataFrame:
    """Build one-step-ahead training rows for recursive models."""
    feat = prepare_features(raw_df, price_ref, first_seen, first_available, cfg)
    feat = add_train_lags(
        feat, cfg.lag_windows, baseline_variant=cfg.baseline_variant
    )
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


def build_recursive_step_panel(
    history_raw: pd.DataFrame,
    target_covariates: pd.DataFrame,
    price_ref: pd.Series,
    first_seen: pd.Series,
    cfg: Config = CFG,
    first_available: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the one-step panel for the next target day from current history."""
    future = sanitize_future_covariates(target_covariates)
    future["Quantity"] = np.nan
    future["QuantityApp"] = np.nan
    future["QuantityWeb"] = np.nan
    future["ProductAvailable"] = pd.Series([pd.NA] * len(future), dtype="boolean")
    history_feat = prepare_features(
        history_raw, price_ref, first_seen, first_available, cfg
    )
    history_feat = add_train_lags(
        history_feat, cfg.lag_windows, baseline_variant=cfg.baseline_variant
    )
    future_feat = prepare_features(
        future, price_ref, first_seen, first_available, cfg
    )
    panel = build_direct_panel(history_feat, [1], cfg=cfg, future_covariates=future_feat)
    origin = history_raw["DateKey"].max()
    step = panel[panel["OriginDateKey"].eq(origin)].reset_index(drop=True)
    step["horizon"] = 1
    return step


def forecast_recursive(
    history_raw: pd.DataFrame,
    future_covariates: pd.DataFrame,
    predict_step,
    price_ref: pd.Series,
    first_seen: pd.Series,
    cfg: Config = CFG,
    first_available: pd.Series | None = None,
) -> pd.DataFrame:
    """Forecast future dates sequentially, feeding predictions into history."""
    history = history_raw.copy().sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    future = sanitize_future_covariates(future_covariates)
    dates = sorted(pd.to_datetime(future["DateKey"].drop_duplicates()))
    if len(dates) != cfg.horizon:
        raise ValueError(f"Expected {cfg.horizon} future dates, got {len(dates)}")
    # Freeze the numerical reference scale from genuinely observed history.
    # Synthetic recursive rows must never raise their own future safety limit.
    initial_quantity = pd.to_numeric(
        history["Quantity"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    initial_history_max = history.assign(
        _finite_quantity=initial_quantity
    ).groupby("ProductId")['_finite_quantity'].max()

    results = []
    for forecast_horizon, target_date in enumerate(dates, start=1):
        current = future[future["DateKey"].eq(target_date)].copy()
        step_panel = build_recursive_step_panel(
            history, current, price_ref, first_seen, cfg, first_available
        )
        if not step_panel["horizon"].eq(1).all():
            raise AssertionError("Recursive model input horizon must always equal 1")
        step_output = predict_step(step_panel)
        if isinstance(step_output, dict):
            if "prediction" not in step_output:
                raise ValueError("predict_step diagnostic payload is missing 'prediction'")
            prediction = np.asarray(step_output["prediction"], dtype=float)
            residual_guard = np.asarray(
                step_output.get("residual_guard", np.zeros(len(step_panel), dtype=bool)),
                dtype=bool,
            )
            residual_nonfinite = np.asarray(
                step_output.get(
                    "residual_nonfinite", np.zeros(len(step_panel), dtype=bool)
                ),
                dtype=bool,
            )
            residual_raw_min = np.asarray(
                step_output.get("residual_raw_min", np.full(len(step_panel), np.nan)),
                dtype=float,
            )
            residual_raw_max = np.asarray(
                step_output.get("residual_raw_max", np.full(len(step_panel), np.nan)),
                dtype=float,
            )
            app_share = np.asarray(
                step_output.get("app_share", np.full(len(step_panel), np.nan)),
                dtype=float,
            )
        else:
            prediction = np.asarray(step_output, dtype=float)
            residual_guard = np.zeros(len(step_panel), dtype=bool)
            residual_nonfinite = np.zeros(len(step_panel), dtype=bool)
            residual_raw_min = np.full(len(step_panel), np.nan, dtype=float)
            residual_raw_max = np.full(len(step_panel), np.nan, dtype=float)
            app_share = np.full(len(step_panel), np.nan, dtype=float)
        if len(prediction) != len(step_panel):
            raise ValueError("predict_step returned a prediction vector with the wrong length")
        for name, values in (
            ("residual_guard", residual_guard),
            ("residual_nonfinite", residual_nonfinite),
            ("residual_raw_min", residual_raw_min),
            ("residual_raw_max", residual_raw_max),
            ("app_share", app_share),
        ):
            if len(values) != len(step_panel):
                raise ValueError(f"predict_step returned {name} with the wrong length")
        baseline = step_panel["target_baseline"].to_numpy(dtype=float)

        # A recursive model can turn one extreme but finite extrapolation into
        # progressively larger lag features.  Treat only catastrophic values
        # as numerical failures; this guard is intentionally orders of
        # magnitude looser than any prediction cap considered in Tier C3.
        observed_scale = step_panel["ProductId"].map(
            initial_history_max
        ).to_numpy(dtype=float)
        lag0 = step_panel.get(
            "qty_lag_0", pd.Series(np.nan, index=step_panel.index)
        ).to_numpy(dtype=float)
        reference_scale = np.nanmax(
            np.column_stack([
                np.where(np.isfinite(observed_scale), observed_scale, 0.0),
                # The seven-day seasonal baseline is anchored in observed
                # pre-origin history.  Do not include recursively generated
                # lag-0 values, otherwise one extreme prediction can inflate
                # the safety threshold for the next step.
                np.where(np.isfinite(baseline), baseline, 0.0),
                np.ones(len(step_panel), dtype=float),
            ]),
            axis=1,
        )
        safety_limit = np.maximum(
            cfg.recursive_safety_floor,
            cfg.recursive_safety_multiplier * reference_scale,
        )
        nonfinite_raw = ~np.isfinite(prediction)
        catastrophic = np.isfinite(prediction) & (prediction > safety_limit)
        fallback = nonfinite_raw | catastrophic

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
        result["nonfinite_raw"] = nonfinite_raw
        result["catastrophic_guard"] = catastrophic
        result["residual_guard"] = residual_guard
        result["residual_nonfinite"] = residual_nonfinite
        result["residual_raw_min"] = residual_raw_min
        result["residual_raw_max"] = residual_raw_max
        result["safety_limit"] = safety_limit
        result["app_share"] = np.where(
            np.isfinite(app_share), np.clip(app_share, 0.0, 1.0), np.nan
        )
        # Recursive channel-history features also need a semantically valid
        # split when the predictor has no auxiliary share head (for example,
        # a history-only NN candidate or a tree model). Fall back to the
        # observed 28-day product share, then lag-0 share, rather than placing
        # the entire synthetic total into one channel.
        feedback_share = result["app_share"].to_numpy(dtype=float)
        if cfg.enable_channel_history_features:
            historical_share = step_panel.get(
                "app_share_roll_28", pd.Series(np.nan, index=step_panel.index)
            ).to_numpy(dtype=float)
            lag0_share = step_panel.get(
                "app_share_lag_0", pd.Series(np.nan, index=step_panel.index)
            ).to_numpy(dtype=float)
            historical_share = np.where(
                np.isfinite(historical_share), historical_share, lag0_share
            )
            feedback_share = np.where(
                np.isfinite(feedback_share), feedback_share, historical_share
            )
        feedback_share = np.where(
            np.isfinite(feedback_share), np.clip(feedback_share, 0.0, 1.0), np.nan
        )
        result["feedback_app_share"] = feedback_share
        result["prediction_app"] = result["prediction"] * result["app_share"]
        result["prediction_web"] = result["prediction"] * (1.0 - result["app_share"])
        results.append(result)

        generated = current.merge(
            result[["ProductId", "prediction", "feedback_app_share"]],
            on="ProductId", how="left", validate="one_to_one"
        )
        generated["Quantity"] = generated.pop("prediction")
        generated["ProductAvailable"] = True
        share = generated.pop("feedback_app_share").to_numpy(dtype=float)
        valid_share = np.isfinite(share)
        generated["QuantityApp"] = np.where(
            valid_share, generated["Quantity"].to_numpy(dtype=float) * share,
            generated["Quantity"].to_numpy(dtype=float),
        )
        generated["QuantityWeb"] = np.where(
            valid_share,
            generated["Quantity"].to_numpy(dtype=float) * (1.0 - share),
            0.0,
        )
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
    numeric_cols = [c for c in direct_panel_feature_names(cfg) if c in X.columns]
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
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
MODEL_ORDER = [
    "NeuralNet", "Ensemble", "XGBoost", "LightGBM", "DynamicRidge",
    "SeasonalNaive", "MovingAvg28",
]
MODEL_STRATEGY_SUPPORT = {
    "NeuralNet": {"direct", "recursive"},
    "Ensemble": {"direct", "recursive"},
    "XGBoost": {"direct", "recursive"},
    "LightGBM": {"direct", "recursive"},
    # Recursive Ridge is empirically unstable on this panel even after
    # numerical overflow guards.  Keep it as a useful direct structured
    # baseline rather than presenting a pathological recursive variant.
    "DynamicRidge": {"direct"},
    "SeasonalNaive": {"direct", "recursive"},
    "MovingAvg28": {"direct", "recursive"},
}


def model_supports_strategy(model: str, strategy: str) -> bool:
    return strategy in MODEL_STRATEGY_SUPPORT.get(model, set())


def prediction_columns_for_strategy(
    pred_columns: dict[str, str], strategy: str
) -> dict[str, str]:
    return {
        model: column
        for model, column in pred_columns.items()
        if model_supports_strategy(model, strategy)
    }

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
    "Ensemble": {
        "label": "OOF Ensemble",
        "short": "Convex blend",
        "color": "#D81B60",
        "kind": "ensemble",
        "source_url": None,
        "blurb": (
            "Non-negative, sum-to-one blend fitted only on development OOF "
            "predictions. Weights are frozen before the recent benchmark and "
            "final forecast are evaluated."
        ),
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
