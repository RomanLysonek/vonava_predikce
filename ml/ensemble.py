"""Tier C5: development-OOF convex ensemble fitting.

Weights are estimated only on development OOF rows, constrained to be
non-negative and sum to one, and applied unchanged to the recent benchmark
and final test forecasts.  The objective is the same test-aligned WAPE used
elsewhere in the project: stratum-level WAPEs are combined with frozen
winter/regular/event weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ENSEMBLE_SCHEMA_VERSION = "c5-convex-ensemble-v1"
DEFAULT_ENSEMBLE_MODELS = ("NeuralNet", "XGBoost", "LightGBM")


@dataclass(frozen=True)
class EnsembleFit:
    strategy: str
    models: tuple[str, ...]
    weights: dict[str, float]
    best_single_model: str
    best_single_test_aligned_wape: float
    ensemble_test_aligned_wape: float
    equal_weight_test_aligned_wape: float
    broad_wape: float
    best_single_broad_wape: float
    relative_improvement: float
    accepted_on_development: bool
    n_rows: int
    n_origins: int
    grid_step: float
    strata_present: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "models": list(self.models),
            "weights": self.weights,
            "best_single_model": self.best_single_model,
            "best_single_test_aligned_wape": self.best_single_test_aligned_wape,
            "ensemble_test_aligned_wape": self.ensemble_test_aligned_wape,
            "equal_weight_test_aligned_wape": self.equal_weight_test_aligned_wape,
            "broad_wape": self.broad_wape,
            "best_single_broad_wape": self.best_single_broad_wape,
            "relative_improvement": self.relative_improvement,
            "accepted_on_development": self.accepted_on_development,
            "n_rows": self.n_rows,
            "n_origins": self.n_origins,
            "grid_step": self.grid_step,
            "strata_present": list(self.strata_present),
        }


def parse_model_list(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_ENSEMBLE_MODELS
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    else:
        items = [str(part).strip() for part in value if str(part).strip()]
    deduplicated = tuple(dict.fromkeys(items))
    if len(deduplicated) < 2:
        raise ValueError("The ensemble requires at least two distinct models")
    return deduplicated


def _prediction_column(model: str) -> str:
    return f"pred_{model}"


def common_conditional_frame(
    oof: pd.DataFrame,
    *,
    strategy: str,
    models: Iterable[str],
) -> pd.DataFrame:
    models = tuple(models)
    columns = [_prediction_column(model) for model in models]
    missing = [column for column in columns if column not in oof.columns]
    if missing:
        raise ValueError(f"OOF is missing ensemble prediction columns: {missing}")
    required = {"actual", "ProductAvailable", "origin", "strategy"}
    missing_required = sorted(required - set(oof.columns))
    if missing_required:
        raise ValueError(f"OOF is missing required columns: {missing_required}")

    frame = oof[oof["strategy"].astype(str).eq(strategy)].copy()
    key_columns = [
        column for column in ("origin_type", "origin", "ProductId", "DateKey", "horizon")
        if column in frame.columns
    ]
    if key_columns and frame.duplicated(key_columns).any():
        examples = frame.loc[frame.duplicated(key_columns, keep=False), key_columns].head(5)
        raise ValueError(
            "Ensemble OOF contains duplicate forecast keys: "
            + examples.to_dict(orient="records").__repr__()
        )
    mask = (
        frame["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["actual"], errors="coerce"))
    )
    for column in columns:
        mask &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    frame = frame.loc[mask].copy()
    if frame.empty:
        raise ValueError(f"No common conditional OOF rows for strategy={strategy!r}")
    if "validation_stratum" not in frame.columns:
        frame["validation_stratum"] = "regular"
    return frame


def _normalized_stratum_weights(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
) -> dict[str, float]:
    present = [
        stratum for stratum in frame["validation_stratum"].dropna().unique()
        if stratum in weights and float(weights[stratum]) > 0
    ]
    if not present:
        return {"regular": 1.0}
    total = float(sum(float(weights[stratum]) for stratum in present))
    return {stratum: float(weights[stratum]) / total for stratum in present}


def test_aligned_error_coefficients(
    frame: pd.DataFrame,
    stratum_weights: Mapping[str, float],
) -> np.ndarray:
    """Return per-row coefficients whose weighted absolute-error sum equals
    the frozen weighted average of stratum WAPEs.

    For stratum ``s`` the coefficient is ``w_s / sum(actual_s)``. Therefore
    summing ``coefficient_i * abs(error_i)`` is exactly
    ``sum_s w_s * WAPE_s`` rather than a row-count-weighted approximation.
    """
    normalized = _normalized_stratum_weights(frame, stratum_weights)
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    strata = frame["validation_stratum"].astype(str).to_numpy()
    coefficients = np.zeros(len(frame), dtype=float)
    for stratum, weight in normalized.items():
        mask = strata == stratum
        denominator = float(np.abs(actual[mask]).sum())
        if denominator <= 0:
            denominator = float(mask.sum()) or 1.0
        coefficients[mask] = weight / denominator
    if not np.isfinite(coefficients).all() or coefficients.sum() <= 0:
        denominator = float(np.abs(actual).sum()) or float(len(actual)) or 1.0
        coefficients = np.full(len(actual), 1.0 / denominator, dtype=float)
    return coefficients


def broad_wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = float(np.abs(actual).sum())
    if denominator <= 0:
        return float(np.mean(np.abs(actual - prediction)))
    return float(np.abs(actual - prediction).sum() / denominator)


def bias_ratio(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = float(np.abs(actual).sum())
    if denominator <= 0:
        return float(np.mean(prediction - actual))
    return float((prediction - actual).sum() / denominator)


def test_aligned_wape(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    stratum_weights: Mapping[str, float],
) -> float:
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    coefficients = test_aligned_error_coefficients(frame, stratum_weights)
    return float(np.sum(coefficients * np.abs(actual - np.asarray(prediction, dtype=float))))


def simplex_weights(n_models: int, step: float) -> np.ndarray:
    """Enumerate an exact simplex grid.

    ``step=0.01`` yields 5,151 candidates for three models, small enough for
    deterministic exhaustive search and easier to audit than a black-box
    optimizer. The grid includes every one-hot model and equal weights when
    representable.
    """
    if n_models < 2:
        raise ValueError("n_models must be at least 2")
    if not np.isfinite(step) or step <= 0 or step > 0.5:
        raise ValueError("step must be finite and in (0, 0.5]")
    units = int(round(1.0 / float(step)))
    if not np.isclose(units * float(step), 1.0, atol=1e-9):
        raise ValueError("step must divide 1.0 exactly, e.g. 0.1, 0.05, 0.02, 0.01")

    rows: list[list[int]] = []

    def recurse(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            recurse(prefix + [value], remaining - value, slots - 1)

    recurse([], units, n_models)
    return np.asarray(rows, dtype=float) / units


def fit_convex_ensemble(
    dev_oof: pd.DataFrame,
    *,
    strategy: str,
    models: Iterable[str] = DEFAULT_ENSEMBLE_MODELS,
    stratum_weights: Mapping[str, float],
    grid_step: float = 0.01,
    min_relative_improvement: float = 0.002,
) -> EnsembleFit:
    models = tuple(models)
    frame = common_conditional_frame(dev_oof, strategy=strategy, models=models)
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    matrix = np.column_stack([
        pd.to_numeric(frame[_prediction_column(model)], errors="coerce").to_numpy(dtype=float)
        for model in models
    ])
    coefficients = test_aligned_error_coefficients(frame, stratum_weights)

    single_scores = np.sum(coefficients[:, None] * np.abs(actual[:, None] - matrix), axis=0)
    best_single_index = int(np.argmin(single_scores))
    best_single_model = models[best_single_index]
    best_single_score = float(single_scores[best_single_index])
    best_single_broad = broad_wape(actual, matrix[:, best_single_index])

    equal = np.full(len(models), 1.0 / len(models), dtype=float)
    equal_prediction = matrix @ equal
    equal_score = float(np.sum(coefficients * np.abs(actual - equal_prediction)))

    grid = simplex_weights(len(models), grid_step)
    best_score = float("inf")
    best_weights = None
    best_concentration = float("inf")
    # Chunking avoids a large n_rows × n_candidates temporary matrix when
    # callers choose a very fine grid.
    for start in range(0, len(grid), 512):
        block = grid[start:start + 512]
        predictions = matrix @ block.T
        scores = np.sum(coefficients[:, None] * np.abs(actual[:, None] - predictions), axis=0)
        local_score = float(np.min(scores))
        local_candidates = np.flatnonzero(np.isclose(scores, local_score, atol=1e-12, rtol=0.0))
        for local_index in local_candidates:
            weights = block[int(local_index)]
            concentration = float(np.sum(np.square(weights)))
            if (
                local_score < best_score - 1e-12
                or (np.isclose(local_score, best_score, atol=1e-12) and concentration < best_concentration)
            ):
                best_score = local_score
                best_weights = weights.copy()
                best_concentration = concentration

    if best_weights is None:
        raise RuntimeError("No ensemble candidate was evaluated")
    prediction = matrix @ best_weights
    relative_improvement = (
        (best_single_score - best_score) / best_single_score
        if best_single_score > 0 else 0.0
    )
    return EnsembleFit(
        strategy=strategy,
        models=models,
        weights={model: float(weight) for model, weight in zip(models, best_weights)},
        best_single_model=best_single_model,
        best_single_test_aligned_wape=best_single_score,
        ensemble_test_aligned_wape=best_score,
        equal_weight_test_aligned_wape=equal_score,
        broad_wape=broad_wape(actual, prediction),
        best_single_broad_wape=best_single_broad,
        relative_improvement=float(relative_improvement),
        accepted_on_development=bool(relative_improvement >= min_relative_improvement),
        n_rows=int(len(frame)),
        n_origins=int(frame["origin"].nunique()),
        grid_step=float(grid_step),
        strata_present=tuple(sorted(frame["validation_stratum"].astype(str).unique())),
    )


def apply_ensemble_prediction(
    oof: pd.DataFrame,
    fits: Mapping[str, EnsembleFit | Mapping[str, float]],
    *,
    output_column: str = "pred_Ensemble",
) -> pd.DataFrame:
    result = oof.copy()
    result[output_column] = np.nan
    for strategy, fit in fits.items():
        weights = fit.weights if isinstance(fit, EnsembleFit) else dict(fit)
        mask = result["strategy"].astype(str).eq(str(strategy))
        if not mask.any():
            continue
        columns = [_prediction_column(model) for model in weights]
        missing = [column for column in columns if column not in result.columns]
        if missing:
            raise ValueError(f"Cannot apply ensemble; missing columns: {missing}")
        matrix = result.loc[mask, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(matrix).all(axis=1)
        predictions = np.full(matrix.shape[0], np.nan, dtype=float)
        weight_vector = np.asarray([weights[model] for model in weights], dtype=float)
        predictions[finite] = matrix[finite] @ weight_vector
        result.loc[mask, output_column] = predictions
        for prefix in (
            "fallback", "nonfinite", "catastrophic", "residual_guard", "residual_nonfinite"
        ):
            result.loc[mask, f"{prefix}_Ensemble"] = False
        result.loc[mask, "residual_raw_min_Ensemble"] = np.nan
        result.loc[mask, "residual_raw_max_Ensemble"] = np.nan
        result.loc[mask, "safety_limit_Ensemble"] = np.nan
    return result


def evaluate_fit(
    oof: pd.DataFrame,
    fit: EnsembleFit,
    *,
    stratum_weights: Mapping[str, float],
) -> dict:
    frame = common_conditional_frame(oof, strategy=fit.strategy, models=fit.models)
    matrix = np.column_stack([
        pd.to_numeric(frame[_prediction_column(model)], errors="coerce").to_numpy(dtype=float)
        for model in fit.models
    ])
    weights = np.asarray([fit.weights[model] for model in fit.models], dtype=float)
    prediction = matrix @ weights
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    single_index = fit.models.index(fit.best_single_model)
    best_single_prediction = matrix[:, single_index]
    ensemble_score = test_aligned_wape(frame, prediction, stratum_weights)
    single_score = test_aligned_wape(frame, best_single_prediction, stratum_weights)
    ensemble_broad = broad_wape(actual, prediction)
    single_broad = broad_wape(actual, best_single_prediction)
    return {
        "strategy": fit.strategy,
        "n_rows": int(len(frame)),
        "n_origins": int(frame["origin"].nunique()),
        "ensemble_test_aligned_wape": ensemble_score,
        "best_single_test_aligned_wape": single_score,
        "ensemble_broad_wape": ensemble_broad,
        "best_single_broad_wape": single_broad,
        "ensemble_bias_ratio": bias_ratio(actual, prediction),
        "best_single_bias_ratio": bias_ratio(actual, best_single_prediction),
        "relative_test_aligned_change": (
            (ensemble_score - single_score) / single_score if single_score > 0 else np.nan
        ),
        "relative_broad_change": (
            (ensemble_broad - single_broad) / single_broad if single_broad > 0 else np.nan
        ),
    }


def combine_forecasts(
    forecasts: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    missing = [model for model in weights if model not in forecasts]
    if missing:
        raise ValueError(f"Final forecasts missing ensemble members: {missing}")
    arrays = [np.asarray(forecasts[model], dtype=float) for model in weights]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("Ensemble member forecasts have inconsistent lengths")
    matrix = np.column_stack(arrays)
    result = matrix @ np.asarray([weights[model] for model in weights], dtype=float)
    return np.clip(result, 0.0, None)
