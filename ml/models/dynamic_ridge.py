"""Dynamic Ridge model: linear regression with L2 regularization.

Trained on the same panel as the other structured models, with one-hot
encoding for categorical features.  The target is the same baseline-relative
log residual used by the neural network.

Recursive inference needs one additional numerical safeguard: an unconstrained
linear model can extrapolate far outside the residual support seen during
training, and those values would then be fed back as lag features.  The model
therefore remembers the finite residual range observed during fitting and may
clip to that range *only* when called for recursive inference.  Direct
predictions remain uncapped unless ``Config.ridge_prediction_cap`` is set.
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

    ``SimpleImputer`` accepts NaN but correctly rejects +/-inf.  Recursive
    models should normally never create infinities, but this defensive
    conversion lets the pipeline recover through the trained median imputer
    instead of crashing an hour-long CV run.
    """
    out = panel.copy()
    numeric = [c for c in direct_panel_feature_names(cfg) if c in out.columns]
    if numeric:
        out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


def train_dynamic_ridge(train_panel: pd.DataFrame, cfg: Config = CFG):
    """Fit baseline-residual Ridge and retain its observed target support."""
    numeric_features = direct_panel_feature_names(cfg)
    categorical_features = TREE_CATEGORICAL_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
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
    model.fit(X, y_fit)

    # Store fitted target support on the sklearn Pipeline.  Clipping to this
    # range during recursive inference is a target-support constraint, not the
    # optional quantity cap evaluated later in Tier C3.
    model.residual_bounds_ = (float(np.min(y_fit)), float(np.max(y_fit)))
    return model


def predict_dynamic_ridge(
    model,
    panel: pd.DataFrame,
    cfg: Config = CFG,
    *,
    recursive: bool = False,
) -> np.ndarray:
    """Predict natural-scale quantity with overflow-safe reconstruction.

    Parameters
    ----------
    recursive:
        When true, constrain the predicted residual to the finite residual
        support observed during fitting.  This prevents unstable linear
        extrapolation from recursively contaminating later lag features.
    """
    safe_panel = _finite_feature_frame(panel, cfg)
    residual = np.asarray(model.predict(safe_panel), dtype=float)

    if recursive:
        lower, upper = getattr(model, "residual_bounds_", (-np.inf, np.inf))
        residual = np.clip(residual, lower, upper)

    baseline = panel["target_baseline"].to_numpy(dtype=float)
    baseline = np.where(np.isfinite(baseline) & (baseline >= 0.0), baseline, np.nan)
    log_prediction = residual + np.log1p(baseline)

    # Do not let np.expm1 emit an overflow warning and an uncontrolled value.
    # Returning NaN for an unsafe row deliberately delegates to
    # framework.forecast_recursive's recorded baseline fallback.
    max_log = np.log(np.finfo(np.float64).max) - 2.0
    safe = np.isfinite(log_prediction) & (log_prediction <= max_log)
    preds = np.full(log_prediction.shape, np.nan, dtype=float)
    preds[safe] = np.expm1(log_prediction[safe])
    preds = np.where(np.isfinite(preds), np.clip(preds, 0.0, None), np.nan)

    # Direct callers do not have the recursive engine's recorded fallback, so
    # preserve complete coverage by falling back locally for unsafe rows.
    if not recursive:
        preds = np.where(np.isfinite(preds), preds, baseline)

    if cfg.ridge_prediction_cap is not None:
        preds = np.minimum(preds, cfg.ridge_prediction_cap)
    return preds
