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


def feature_columns(cfg: Config = CFG) -> list[str]:
    """Full numeric feature schema for a given config. Kept as a function of
    `cfg.lag_windows` (rather than a hardcoded list) so that changing the lag
    windows can never silently desync from the columns actually produced by
    `add_train_lags` / `recursive_forecast_generic`."""
    return STATIC_NUMERIC_FEATURES + lag_feature_names(cfg.lag_windows)


NUMERIC_FEATURES = feature_columns(CFG)  # default schema, used wherever cfg == CFG


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


def compute_baseline(target_df: pd.DataFrame, hist_df: pd.DataFrame) -> np.ndarray:
    """Availability-aware weighted same-weekday baseline: a 4:3:2:1 weighted
    average of Quantity at lags 7/14/21/28 days, using only observed-and-
    available demand from `hist_df`. Weights renormalize over whichever
    lags are actually observed -- a stockout/unknown-gap lag drops out of
    the average instead of forcing the whole baseline to NaN. `hist_df` is
    the lookup source (e.g. training history); `target_df` is whatever rows
    need a baseline value (can be the same frame, for training rows, or
    later out-of-sample eval/test rows looking back into `hist_df`)."""
    available = hist_df["ProductAvailable"].fillna(False)
    qty_available = hist_df["Quantity"].where(available)
    lookup = pd.Series(qty_available.to_numpy(),
                        index=pd.MultiIndex.from_frame(hist_df[["ProductId", "DateKey"]]))

    lags = (7, 14, 21, 28)
    weights = np.array([4.0, 3.0, 2.0, 1.0])
    lag_matrix = np.full((len(target_df), len(lags)), np.nan)
    for j, lag in enumerate(lags):
        keys = list(zip(target_df["ProductId"], target_df["DateKey"] - pd.Timedelta(days=lag)))
        lag_matrix[:, j] = [lookup.get(k, np.nan) for k in keys]

    observed = np.isfinite(lag_matrix)
    numerator = np.nansum(lag_matrix * weights, axis=1)
    denominator = (observed * weights).sum(axis=1)
    return np.divide(numerator, denominator,
                      out=np.full(len(target_df), np.nan, dtype=float),
                      where=denominator > 0)


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


def _nan_safe_window_stats(values: np.ndarray) -> tuple[float, float, float, int]:
    """mean/std/median/count over the non-NaN entries of `values`. Falls
    back to (0, 0, 0, 0) if the entire window is unavailable/unknown --
    plain `.mean()`/`.std()` on an all-NaN slice would otherwise poison the
    recursive forecast with NaN for every subsequent day."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0
    return float(valid.mean()), float(valid.std()), float(np.median(valid)), int(len(valid))


def init_history(df: pd.DataFrame, max_window: int) -> dict[int, list[float]]:
    """Per-product tail of availability-aware quantities (NaN for a
    stockout or unknown-calendar-gap day), used as the starting point for
    recursive forecasting. NaN is preserved -- not zero-filled -- so
    `recursive_forecast_generic` can skip it via `_nan_safe_window_stats`
    instead of letting a recent stockout drag the rolling stats toward
    zero."""
    hist: dict[int, list[float]] = {}
    col = "qty_available" if "qty_available" in df.columns else "Quantity"
    for pid, sub in df.sort_values("DateKey").groupby("ProductId"):
        vals = sub[col].to_numpy(dtype=float)
        hist[int(pid)] = list(vals[-max_window:]) if len(vals) else [np.nan]
    return hist


# ---------------------------------------------------------------------------
# Model-agnostic recursive forecasting
# ---------------------------------------------------------------------------
def recursive_forecast_generic(predict_fn, static_df: pd.DataFrame, history: dict,
                                cfg: Config = CFG) -> np.ndarray:
    """Walk the horizon forward one day at a time. Each day's lag features are
    computed from `history`, which is then extended with that day's own
    prediction (via `predict_fn(day_df) -> np.ndarray`) before moving to the
    next day. This is what makes the forecast genuinely multi-step instead of
    7 identical copies of a single-step prediction, for ANY model (neural
    net, XGBoost, LightGBM, ...).

    `history` is mutated in place -- pass a fresh dict per model/run.
    """
    static_df = static_df.reset_index(drop=True)
    dates = sorted(static_df["DateKey"].unique())
    out = np.zeros(len(static_df), dtype=np.float32)

    for d in dates:
        mask = (static_df["DateKey"] == d).to_numpy()
        day_df = static_df.loc[mask].copy()
        for w in cfg.lag_windows:
            means, stds, meds, counts = [], [], [], []
            for pid in day_df["ProductId"]:
                arr = np.asarray(history[int(pid)][-w:], dtype=float)
                m, s, md, c = _nan_safe_window_stats(arr)
                means.append(m); stds.append(s); meds.append(md); counts.append(c)
            day_df[f"qty_roll_mean_{w}"] = means
            day_df[f"qty_roll_std_{w}"] = stds
            day_df[f"qty_roll_median_{w}"] = meds
            day_df[f"qty_available_count_{w}"] = counts
            window_days = [min(len(history[int(pid)][-w:]), w) for pid in day_df["ProductId"]]
            day_df[f"stockout_rate_{w}"] = 1.0 - np.asarray(counts) / np.maximum(window_days, 1)

        day_pred = predict_fn(day_df)
        out[mask] = day_pred
        for pid, q in zip(day_df["ProductId"], day_pred):
            history[int(pid)].append(float(q))

    return out


# ---------------------------------------------------------------------------
# Tree-model feature framing (shared shape used by tree_worker.py)
# ---------------------------------------------------------------------------
TREE_CATEGORICAL_COLUMNS = ["product_idx", "campaign_idx_web", "campaign_idx_app"]


def tree_feature_frame(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Numeric features + native pandas 'category' dtype columns, understood
    directly by both XGBoost (`enable_categorical=True`) and LightGBM
    (auto-detected), which avoids imposing a false ordinal scale on
    product/campaign ids the way a plain integer column would.
    """
    cols = feature_columns(cfg) + TREE_CATEGORICAL_COLUMNS
    X = df[cols].copy()
    for c in TREE_CATEGORICAL_COLUMNS:
        X[c] = X[c].astype("category")
    return X


# ---------------------------------------------------------------------------
# Model registry/metadata & metrics
# ---------------------------------------------------------------------------
MODEL_ORDER = ["NeuralNet", "XGBoost", "LightGBM", "SeasonalNaive", "MovingAvg28"]

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
