"""Execute the frozen Tier-C final audit exactly once.

The normal pipeline deliberately excludes ``FINAL_AUDIT_ORIGINS``. This
script evaluates the already-frozen C1-C5 configuration on those origins,
applies the development-fitted ensemble weights without refitting, writes
separate audit artifacts, and refreshes the dashboard payload through the
artifact-only exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ensemble import apply_ensemble_prediction
from pipeline import (
    CFG,
    FINAL_AUDIT_ORIGINS,
    ForecastStrategy,
    OOF_MODEL_COLUMNS,
    compute_test_aligned_scores,
    configure_c1_runtime,
    configure_c2_runtime,
    configure_c34_runtime,
    configure_c5_runtime,
    configure_nn_runtime,
    load_raw,
    parse_args,
    run_walk_forward_cv,
    summarize_oof_by_strategy,
    summarize_prediction_diagnostics,
    summarize_prediction_diagnostics_by_origin,
    summarize_validation_strata,
)


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse(argv=None):
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--force", action="store_true")
    gate.add_argument("--no-refresh-dashboard", action="store_true")
    known, remaining = gate.parse_known_args(argv)
    options = parse_args(remaining)
    return known, remaining, options


def _validate_configuration(results: dict, cfg) -> None:
    stored = results.get("config") or {}
    expected = {
        "training_window_days": cfg.training_window_days,
        "recency_half_life_days": cfg.recency_half_life_days,
        "baseline_variant": cfg.baseline_variant,
        "enable_trend_features": cfg.enable_trend_features,
        "c2_feature_groups": list(cfg.c2_feature_groups),
        "nn_loss": cfg.nn_loss,
        "nn_target_mode": cfg.nn_target_mode,
        "nn_combined_mse_weight": cfg.nn_combined_mse_weight,
        "xgboost_target_mode": cfg.xgboost_target_mode,
        "lightgbm_target_mode": cfg.lightgbm_target_mode,
        "enable_channel_history_features": cfg.enable_channel_history_features,
        "channel_aux_weight": cfg.channel_aux_weight,
        "channel_share_smoothing": cfg.channel_share_smoothing,
        "nn_batch_size": cfg.batch_size,
        "nn_lr_scaling": cfg.nn_lr_scaling,
    }
    mismatches = {
        key: {"stored": stored.get(key), "requested": value}
        for key, value in expected.items()
        if stored.get(key) != value
    }
    if mismatches:
        lines = "\n".join(
            f"  {key}: stored={value['stored']!r}, requested={value['requested']!r}"
            for key, value in mismatches.items()
        )
        raise RuntimeError(
            "Final-audit configuration differs from the frozen full run. "
            "Use the same C1/C2/C3/C4/NN arguments.\n" + lines
        )


def main(argv=None) -> None:
    gate, pipeline_args, options = _parse(argv)
    if options.forecast_strategy is not ForecastStrategy.DIRECT:
        raise RuntimeError("The frozen final audit is direct-only; pass --forecast-strategy direct")
    if options.ensemble != "on":
        raise RuntimeError("Pass --ensemble on so frozen C5 weights are audited")

    cfg = CFG
    configure_c1_runtime(cfg, options)
    configure_c2_runtime(cfg, options)
    configure_c34_runtime(cfg, options)
    configure_c5_runtime(cfg, options)
    configure_nn_runtime(cfg, options)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "final_audit_manifest.json"
    if manifest_path.exists() and not gate.force:
        raise RuntimeError(
            f"{manifest_path} already exists. The audit is intentionally one-shot. "
            "Use --force only when the earlier audit artifact is invalid."
        )

    weights_path = output_dir / "ensemble_weights.json"
    results_path = output_dir / "results.json"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}; complete the full C3/C4+C5 run first"
        )
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    ensemble_payload = _json(weights_path)
    results = _json(results_path)
    _validate_configuration(results, cfg)
    direct = (ensemble_payload.get("strategies") or {}).get("direct")
    if not direct or not direct.get("weights"):
        raise RuntimeError("No frozen direct ensemble weights in ensemble_weights.json")
    weights = {str(model): float(weight) for model, weight in direct["weights"].items()}

    train_raw, _ = load_raw(cfg)
    checkpoint_dir = output_dir / "final_audit_checkpoints"
    if gate.force and checkpoint_dir.exists() and not options.resume:
        import shutil
        shutil.rmtree(checkpoint_dir)

    print("=== Frozen direct final audit ===")
    print("Origins:", ", ".join(str(pd.Timestamp(x).date()) for x in FINAL_AUDIT_ORIGINS))
    print("Ensemble weights:", ", ".join(f"{k}={v:.3f}" for k, v in weights.items()))
    oof = run_walk_forward_cv(
        train_raw,
        FINAL_AUDIT_ORIGINS,
        "final_audit",
        cfg,
        timings=[],
        strategy="direct",
        checkpoint_dir=str(checkpoint_dir),
        resume=options.resume,
    )
    oof = apply_ensemble_prediction(oof, {"direct": weights})

    summary = summarize_oof_by_strategy(oof, OOF_MODEL_COLUMNS)
    strata = summarize_validation_strata(oof, OOF_MODEL_COLUMNS)
    aligned = compute_test_aligned_scores(
        strata, metric=options.selection_metric, origin_type="final_audit"
    )
    diagnostics = summarize_prediction_diagnostics(oof)
    diagnostics_by_origin = summarize_prediction_diagnostics_by_origin(oof)

    oof.to_parquet(output_dir / "final_audit_oof.parquet", index=False)
    summary.to_csv(output_dir / "final_audit_summary.csv", index=False)
    strata.to_csv(output_dir / "final_audit_validation_strata.csv", index=False)
    aligned.to_csv(output_dir / "final_audit_test_aligned_scores.csv", index=False)
    diagnostics.to_csv(output_dir / "final_audit_prediction_diagnostics.csv", index=False)
    diagnostics_by_origin.to_csv(
        output_dir / "final_audit_prediction_diagnostics_by_origin.csv", index=False
    )

    manifest = {
        "schema_version": "c5-final-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "origins": [str(pd.Timestamp(origin).date()) for origin in FINAL_AUDIT_ORIGINS],
        "strategy": "direct",
        "selection_metric": options.selection_metric,
        "ensemble_weights_sha256": _sha256(weights_path),
        "ensemble_weights": weights,
        "results_sha256_before_refresh": _sha256(results_path),
        "configuration": results.get("config", {}),
        "force_used": bool(gate.force),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)

    primary = summary[
        summary["evaluation_regime"].eq("conditional")
        & summary["comparison_population"].eq("common")
        & summary["aggregation"].eq("global")
    ].sort_values(options.selection_metric)
    print("\nFinal-audit conditional/common/global:")
    print(primary[["model", options.selection_metric, "BiasRatio", "coverage"]].to_string(index=False))

    if not gate.no_refresh_dashboard:
        exporter = Path(__file__).with_name("export_results.py")
        refresh_args = [arg for arg in pipeline_args if arg not in {"--reset-checkpoints", "--resume"}]
        subprocess.run([sys.executable, str(exporter), *refresh_args], check=True)
    print("Saved final-audit artifacts. No ensemble weights were refitted.")


if __name__ == "__main__":
    main()
