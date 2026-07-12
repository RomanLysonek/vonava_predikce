"""Two floor baselines any real model should beat -- grouped in one file
since each is a single availability-aware lookup/mean, not a trained model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework import compute_baseline


def seasonal_naive_predict(eval_df: pd.DataFrame, train_df: pd.DataFrame, lag_days: int) -> np.ndarray:
    """Predict Quantity(date) := Quantity(date - lag_days) for the same
    product, using only observed-and-available demand. If the exact lag day
    was itself a stockout/unknown-calendar-gap (NaN), falls back to
    `compute_baseline` instead of silently predicting a censored zero."""
    available = train_df["ProductAvailable"].fillna(False)
    qty_available = train_df["Quantity"].where(available)
    lookup = pd.Series(qty_available.to_numpy(),
                        index=pd.MultiIndex.from_frame(train_df[["ProductId", "DateKey"]]))
    keys = list(zip(eval_df["ProductId"], eval_df["DateKey"] - pd.Timedelta(days=lag_days)))
    preds = np.array([lookup.get(k, np.nan) for k in keys], dtype=float)

    missing = np.isnan(preds)
    if missing.any():
        preds[missing] = compute_baseline(eval_df.loc[missing], train_df)
    return preds


def moving_average_predict(eval_df: pd.DataFrame, train_df: pd.DataFrame, window: int = 28) -> np.ndarray:
    """Predict a flat value: the availability-aware mean of the last
    `window` calendar days per product (stockout/unknown-gap days excluded,
    not counted as zero demand)."""
    available = train_df["ProductAvailable"].fillna(False)
    qty_available = train_df["Quantity"].where(available)
    means = (train_df.assign(_qty_available=qty_available)
             .sort_values("DateKey").groupby("ProductId")["_qty_available"]
             .apply(lambda s: s.tail(window).mean()))
    return eval_df["ProductId"].map(means).to_numpy(dtype=float)
