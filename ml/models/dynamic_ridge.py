"""Dynamic Ridge model: linear regression with L2 regularization.

Trained on the same stacked panel as the trees, but using one-hot encoding
for categorical features. Represents a 'structured statistical' baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from framework import CFG, Config, direct_panel_feature_names, TREE_CATEGORICAL_COLUMNS


def train_dynamic_ridge(train_panel: pd.DataFrame, cfg: Config = CFG):
    """Ridge regression with one-hot encoding for categories and scaling
    for numeric features.
    """
    numeric_features = direct_panel_feature_names(cfg)
    categorical_features = TREE_CATEGORICAL_COLUMNS
    
    # Preprocessor: scale numeric features and one-hot encode categories.
    # Ridge does not handle NaNs natively, so we impute with median for numeric
    # and most_frequent for categorical.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ]
    )
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("ridge", Ridge(alpha=cfg.ridge_alpha, random_state=cfg.seed))
    ])
    
    X = train_panel
    # Ridge trained on the same baseline-relative log residual used by NN (Tier B Corrections)
    y = (
        np.log1p(train_panel["target"].to_numpy(dtype=float))
        - np.log1p(train_panel["target_baseline"].to_numpy(dtype=float))
    )
    
    # Filter out any rows with NaN target (should already be handled by caller)
    mask = ~np.isnan(y)
    model.fit(X[mask], y[mask])
    
    return model


def predict_dynamic_ridge(model, panel: pd.DataFrame, cfg: Config = CFG) -> np.ndarray:
    """Predict and apply a safety cap (Tier B3)."""
    residual = model.predict(panel)
    
    # Reconstruct from baseline-relative log residual
    preds = np.expm1(
        residual
        + np.log1p(panel["target_baseline"].to_numpy(dtype=float))
    )
    
    # Tier B3 "Dynamic Ridge's cap": linear models can occasionally 
    # extrapolate to extreme values. 
    preds = np.clip(preds, 0.0, None)
    
    if cfg.ridge_prediction_cap is not None:
        preds = np.minimum(preds, cfg.ridge_prediction_cap)
        
    return preds
