"""Dynamic Ridge: a direct-only linear structured baseline.

The model uses the same stacked direct panel and baseline-relative log-residual
target as the neural network, with one-hot encoded categorical features and L2
regularisation. Recursive Ridge was removed from the competitive model set in
Tier C0 because recursive feedback remained statistically unstable even after
numerical overflow protection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from framework import CFG, Config, TREE_CATEGORICAL_COLUMNS, direct_panel_feature_names


def _finite_feature_frame(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Convert numeric infinities to missing values for fitted imputers.

    ``SimpleImputer`` accepts NaN but correctly rejects +/-inf. This
    defensive conversion keeps the direct baseline robust to malformed or
    numerically extreme feature rows.
    """
    out = panel.copy()
    numeric = direct_panel_feature_names(cfg)
    for column in numeric:
        if column not in out.columns:
            out[column] = np.nan
    for column in TREE_CATEGORICAL_COLUMNS:
        if column not in out.columns:
            out[column] = 0
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


def train_dynamic_ridge(train_panel: pd.DataFrame, cfg: Config = CFG):
    """Fit a direct baseline-residual Ridge model."""
    numeric_features = direct_panel_feature_names(cfg)
    categorical_features = TREE_CATEGORICAL_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("ridge", Ridge(alpha=cfg.ridge_alpha)),
    ])

    target = train_panel["target"].to_numpy(dtype=float)
    baseline = train_panel["target_baseline"].to_numpy(dtype=float)
    y = np.log1p(target) - np.log1p(baseline)
    mask = np.isfinite(y)
    if not mask.any():
        raise ValueError("Dynamic Ridge received no finite training targets")

    X = _finite_feature_frame(train_panel.loc[mask], cfg)
    y_fit = y[mask]
    sample_weight = train_panel.loc[mask].get(
        "sample_weight", pd.Series(1.0, index=train_panel.loc[mask].index)
    ).to_numpy(dtype=float)
    model.fit(X, y_fit, ridge__sample_weight=sample_weight)

    return model


def predict_dynamic_ridge(
    model,
    panel: pd.DataFrame,
    cfg: Config = CFG,
) -> np.ndarray:
    """Predict direct natural-scale quantity with safe reconstruction."""
    safe_panel = _finite_feature_frame(panel, cfg)
    residual = np.asarray(model.predict(safe_panel), dtype=float)

    baseline = panel["target_baseline"].to_numpy(dtype=float)
    baseline = np.where(
        np.isfinite(baseline) & (baseline >= 0.0), baseline, 0.0
    )
    log_prediction = residual + np.log1p(baseline)

    # Do not let np.expm1 emit an overflow warning. Direct Ridge requires
    # complete output coverage, so unsafe extrapolations fall back to the
    # same-weekday baseline and are visible later through OOF diagnostics.
    max_log = np.log(np.finfo(np.float64).max) - 2.0
    safe = np.isfinite(log_prediction) & (log_prediction <= max_log)
    preds = np.full(log_prediction.shape, np.nan, dtype=float)
    preds[safe] = np.expm1(log_prediction[safe])
    preds = np.where(
        np.isfinite(preds), np.clip(preds, 0.0, None), baseline
    )

    if cfg.ridge_prediction_cap is not None:
        preds = np.minimum(preds, cfg.ridge_prediction_cap)
    return preds
