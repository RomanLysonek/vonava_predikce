"""Tier C6 dashboard diagnostics and static-site publication helpers."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from framework import compute_metrics, model_supports_strategy, prediction_columns_for_strategy


DASHBOARD_SCHEMA_VERSION = "c6-dashboard-v1"
GENERATED_README = (
    "# Generated GitHub Pages site\n\n"
    "Do not edit this directory manually. It is generated from "
    "`webapp/static`, `outputs/results.json`, and "
    "`outputs/run_manifest.json` by `ml/publish_dashboard.py`.\n"
)


def summarize_per_product_oof(
    oof: pd.DataFrame,
    pred_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Conditional/common product-level error and bias diagnostics."""
    if oof.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (origin_type, strategy), split in oof.groupby(["origin_type", "strategy"], sort=False):
        columns = prediction_columns_for_strategy(dict(pred_columns), str(strategy))
        columns = {model: col for model, col in columns.items() if col in split.columns}
        if not columns:
            continue
        common = (
            split["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(split["actual"], errors="coerce"))
            & split[list(columns.values())].apply(np.isfinite).all(axis=1)
        )
        work = split.loc[common].copy()
        for product_id, product in work.groupby("ProductId", sort=True):
            actual = pd.to_numeric(product["actual"], errors="coerce").to_numpy(dtype=float)
            for model, column in columns.items():
                prediction = pd.to_numeric(product[column], errors="coerce").to_numpy(dtype=float)
                metrics = compute_metrics(actual, prediction)
                rows.append({
                    "origin_type": str(origin_type),
                    "strategy": str(strategy),
                    "ProductId": int(product_id),
                    "model": model,
                    "n": int(len(product)),
                    "actual_total": float(np.sum(actual)),
                    "prediction_total": float(np.sum(prediction)),
                    **metrics,
                })
    return pd.DataFrame(rows)


def summarize_origin_dispersion(
    oof: pd.DataFrame,
    pred_columns: Mapping[str, str],
    *,
    origin_type: str = "development",
) -> pd.DataFrame:
    """Summarize origin-level conditional/common WAPE dispersion."""
    if oof.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    selected = oof[oof["origin_type"].astype(str).eq(origin_type)]
    for strategy, split in selected.groupby("strategy", sort=False):
        columns = prediction_columns_for_strategy(
            dict(pred_columns), str(strategy)
        )
        columns = {
            model: column
            for model, column in columns.items()
            if column in split.columns
        }
        if not columns:
            continue
        common = (
            split["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(split["actual"], errors="coerce"))
            & split[list(columns.values())].apply(np.isfinite).all(axis=1)
        )
        work = split.loc[common]
        origin_scores: dict[str, list[float]] = {
            model: [] for model in columns
        }
        for _, origin in work.groupby("origin", sort=True):
            actual = pd.to_numeric(
                origin["actual"], errors="coerce"
            ).to_numpy(dtype=float)
            for model, column in columns.items():
                prediction = pd.to_numeric(
                    origin[column], errors="coerce"
                ).to_numpy(dtype=float)
                origin_scores[model].append(
                    float(compute_metrics(actual, prediction)["WAPE"])
                )
        for model, values in origin_scores.items():
            if not values:
                continue
            scores = np.asarray(values, dtype=float)
            rows.append({
                "origin_type": origin_type,
                "strategy": str(strategy),
                "model": model,
                "metric": "WAPE",
                "n_origins": int(len(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.quantile(scores, 0.25)),
                "q75": float(np.quantile(scores, 0.75)),
                "minimum": float(np.min(scores)),
                "maximum": float(np.max(scores)),
            })
    return pd.DataFrame(rows)


def summarize_top_deciles(
    oof: pd.DataFrame,
    pred_columns: Mapping[str, str],
    *,
    quantile: float = 0.90,
    max_error_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top-demand metrics and the largest row-level errors.

    The high-demand population is defined from actual demand once per
    split/strategy and therefore remains comparable across models.  Largest
    error rows are model-specific explanatory diagnostics, not a selection
    population.
    """
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be in [0.5, 1.0)")
    summary_rows: list[dict] = []
    error_rows: list[pd.DataFrame] = []
    for (origin_type, strategy), split in oof.groupby(["origin_type", "strategy"], sort=False):
        columns = prediction_columns_for_strategy(dict(pred_columns), str(strategy))
        columns = {model: col for model, col in columns.items() if col in split.columns}
        if not columns:
            continue
        common = (
            split["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(split["actual"], errors="coerce"))
            & split[list(columns.values())].apply(np.isfinite).all(axis=1)
        )
        work = split.loc[common].copy()
        if work.empty:
            continue
        actual = pd.to_numeric(work["actual"], errors="coerce")
        threshold = float(actual.quantile(quantile))
        high = work.loc[actual.ge(threshold)]
        for model, column in columns.items():
            metrics = compute_metrics(high["actual"], high[column])
            summary_rows.append({
                "origin_type": str(origin_type),
                "strategy": str(strategy),
                "model": model,
                "quantile": float(quantile),
                "actual_threshold": threshold,
                "n": int(len(high)),
                **metrics,
            })
            detail_columns = [
                col for col in (
                    "origin_type", "strategy", "origin", "validation_stratum",
                    "ProductId", "DateKey", "horizon", "actual"
                ) if col in work.columns
            ]
            detail = work[detail_columns].copy()
            detail["model"] = model
            detail["prediction"] = pd.to_numeric(work[column], errors="coerce")
            detail["absolute_error"] = np.abs(detail["prediction"] - detail["actual"])
            detail["signed_error"] = detail["prediction"] - detail["actual"]
            detail = detail.nlargest(max_error_rows, "absolute_error")
            error_rows.append(detail)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(error_rows, ignore_index=True) if error_rows else pd.DataFrame(),
    )


def _safe_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_strategy_development(output_dir: str | Path) -> dict:
    """Expose the retained recursive development check without live controls."""
    payload = _safe_json(
        Path(output_dir)
        / "c01_recursive_check"
        / "c01_recursive_check.json"
    )
    if not payload:
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    finite_wape = [
        float(row["WAPE"])
        for row in results
        if isinstance(row, dict)
        and np.isfinite(pd.to_numeric(row.get("WAPE"), errors="coerce"))
    ]
    return {
        "schema_version": payload.get("schema_version"),
        "role": "development_history",
        "strategy": "recursive",
        "passed_stability_check": bool(payload.get("passed")),
        "n_origins": len(results),
        "epochs": (payload.get("config") or {}).get("epochs"),
        "seed": (payload.get("config") or {}).get("seed"),
        "wape_min": min(finite_wape) if finite_wape else None,
        "wape_max": max(finite_wape) if finite_wape else None,
        "published_final_forecast": False,
    }


def collect_ablation_showcase(output_dir: str | Path) -> pd.DataFrame:
    """Normalize the persisted C1/C2/C3/C4 screens for the dashboard."""
    output_dir = Path(output_dir)
    specifications = [
        (
            "C1",
            output_dir / "c1_screening" / "c1_screening_results.csv",
            output_dir / "c1_screening" / "recommendation.json",
        ),
        (
            "C2",
            output_dir / "c2_screening" / "c2_screening_results.csv",
            output_dir / "c2_screening" / "recommendation.json",
        ),
        (
            "C3/C4",
            output_dir / "c34_screening" / "c34_screening_results.csv",
            output_dir / "c34_screening" / "recommendation.json",
        ),
    ]
    frames: list[pd.DataFrame] = []
    for tier, csv_path, recommendation_path in specifications:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty or "candidate" not in frame.columns:
            continue
        recommendation = _safe_json(recommendation_path)
        selected_names: set[str] = set()
        rec = recommendation.get("recommendation", {})
        if isinstance(rec, dict) and rec.get("candidate"):
            selected_names.add(str(rec["candidate"]))
        winners = recommendation.get("stage_winners", {})
        if isinstance(winners, dict):
            selected_names.update(str(value) for value in winners.values() if value)
        group_winner = recommendation.get("group_winner", {})
        if isinstance(group_winner, dict) and group_winner.get("candidate"):
            selected_names.add(str(group_winner["candidate"]))

        normalized = pd.DataFrame({
            "tier": tier,
            "stage": frame.get("stage", pd.Series(tier, index=frame.index)).astype(str),
            "candidate": frame["candidate"].astype(str),
            "model": frame.get("model", pd.Series("NeuralNet", index=frame.index)).astype(str),
            "WAPE": pd.to_numeric(frame.get("WAPE"), errors="coerce"),
            "test_aligned_WAPE": pd.to_numeric(
                frame.get("test_aligned_WAPE"), errors="coerce"
            ),
            "BiasRatio": pd.to_numeric(frame.get("BiasRatio"), errors="coerce"),
            "Coverage": pd.to_numeric(frame.get("Coverage"), errors="coerce"),
            "selected": frame["candidate"].astype(str).isin(selected_names),
        })
        normalized["description"] = ""
        if tier == "C1":
            normalized["description"] = (
                "window=" + frame.get("training_window_days", pd.Series("all", index=frame.index)).astype(str)
                + ", half-life=" + frame.get("recency_half_life_days", pd.Series("none", index=frame.index)).astype(str)
                + ", baseline=" + frame.get("baseline_variant", pd.Series("", index=frame.index)).astype(str)
            )
        elif tier == "C2":
            normalized["description"] = (
                "groups=" + frame.get("c2_feature_groups", pd.Series("none", index=frame.index)).fillna("none").astype(str)
                + ", half-life=" + frame.get("recency_half_life_days", pd.Series("none", index=frame.index)).astype(str)
            )
        else:
            normalized["description"] = (
                "loss=" + frame.get("nn_loss", pd.Series("", index=frame.index)).astype(str)
                + ", target=" + frame.get("nn_target_mode", pd.Series("", index=frame.index)).astype(str)
                + ", channel_aux=" + frame.get("channel_aux_weight", pd.Series(0.0, index=frame.index)).astype(str)
                + ", tree=" + frame.get("tree_target_mode", pd.Series("", index=frame.index)).astype(str)
            )
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=[
            "tier", "stage", "candidate", "model", "WAPE",
            "test_aligned_WAPE", "BiasRatio", "Coverage", "selected",
            "description",
        ])
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["tier", "stage", "selected", "test_aligned_WAPE"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _static_html(source: str) -> str:
    result = source.replace('/static/', './')
    result = result.replace('href="/dataset"', 'href="./dataset.html"')
    result = result.replace('href="/evaluation"', 'href="./evaluation.html"')
    result = result.replace('href="/whole-story"', 'href="./whole-story.html"')
    result = result.replace('href="/"', 'href="./index.html"')
    marker = '<script src="./common.js'
    if marker in result and "window.STATIC_DASHBOARD" not in result:
        result = result.replace(
            marker,
            '<script>window.STATIC_DASHBOARD = true;</script>\n  ' + marker,
            1,
        )
    return result


def _dashboard_manifest(root: Path, static_dir: Path, docs_dir: Path) -> dict:
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "runtime_results": str((static_dir / "results.json").relative_to(root)),
        "run_manifest": str(
            (static_dir / "run_manifest.json").relative_to(root)
        ),
        "static_site": str(docs_dir.relative_to(root)),
        "entrypoint": "docs/index.html",
    }


def check_static_dashboard(repository_root: str | Path) -> list[str]:
    """Return deterministic publication drift errors without modifying files."""
    root = Path(repository_root)
    static_dir = root / "webapp" / "static"
    docs_dir = root / "docs"
    outputs_dir = root / "outputs"
    errors: list[str] = []
    generated_sources = {
        "results.json": outputs_dir / "results.json",
        "run_manifest.json": outputs_dir / "run_manifest.json",
    }
    for name, source in generated_sources.items():
        static_copy = static_dir / name
        if not source.exists():
            errors.append(f"missing canonical artifact: {source.relative_to(root)}")
        elif not static_copy.exists() or static_copy.read_bytes() != source.read_bytes():
            errors.append(f"stale generated copy: {static_copy.relative_to(root)}")

    expected: dict[str, bytes] = {}
    for source in static_dir.iterdir():
        if not source.is_file():
            continue
        canonical = generated_sources.get(source.name, source)
        if not canonical.exists():
            continue
        if source.suffix.lower() == ".html":
            expected[source.name] = _static_html(
                canonical.read_text(encoding="utf-8")
            ).encode("utf-8")
        else:
            expected[source.name] = canonical.read_bytes()
    expected[".nojekyll"] = b""
    expected["README.md"] = GENERATED_README.encode("utf-8")

    actual_names = {
        str(path.relative_to(docs_dir))
        for path in docs_dir.rglob("*")
        if path.is_file()
    } if docs_dir.exists() else set()
    expected_names = set(expected)
    for name in sorted(expected_names - actual_names):
        errors.append(f"missing generated file: docs/{name}")
    for name in sorted(actual_names - expected_names):
        errors.append(f"unexpected generated file: docs/{name}")
    for name in sorted(expected_names & actual_names):
        if (docs_dir / name).read_bytes() != expected[name]:
            errors.append(f"stale generated file: docs/{name}")

    manifest_path = outputs_dir / "dashboard_manifest.json"
    expected_manifest = _dashboard_manifest(root, static_dir, docs_dir)
    if not manifest_path.exists():
        errors.append("missing canonical artifact: outputs/dashboard_manifest.json")
    else:
        try:
            actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("invalid JSON: outputs/dashboard_manifest.json")
        else:
            if actual_manifest != expected_manifest:
                errors.append("stale generated file: outputs/dashboard_manifest.json")
    return errors


def publish_static_dashboard(
    repository_root: str | Path,
    results_path: str | Path,
    run_manifest_path: str | Path | None = None,
) -> dict:
    """Copy the strict results payload beside runtime assets and build docs/.

    ``webapp/static/results.json`` is a local API fallback. ``docs/`` is a
    self-contained GitHub Pages site with relative asset and navigation URLs.
    """
    root = Path(repository_root)
    results_path = Path(results_path)
    run_manifest_path = Path(
        run_manifest_path or root / "outputs" / "run_manifest.json"
    )
    static_dir = root / "webapp" / "static"
    docs_dir = root / "docs"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not run_manifest_path.exists():
        raise FileNotFoundError(run_manifest_path)
    static_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_path, static_dir / "results.json")
    shutil.copy2(run_manifest_path, static_dir / "run_manifest.json")

    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True)
    for source in static_dir.iterdir():
        if not source.is_file():
            continue
        destination = docs_dir / source.name
        if source.suffix.lower() == ".html":
            destination.write_text(
                _static_html(source.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, destination)
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
    (docs_dir / "README.md").write_text(GENERATED_README, encoding="utf-8")
    manifest = _dashboard_manifest(root, static_dir, docs_dir)
    manifest_path = root / "outputs" / "dashboard_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
