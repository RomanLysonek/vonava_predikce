import json
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from framework import Config
from models.neural_net import (
    _batch_ranges,
    effective_learning_rate,
    make_tensors,
    nn_performance_signature,
    numeric_feature_columns,
    train_model,
)
from benchmark_nn_batch_size import choose_recommendation
from pipeline import RuntimeOptions, configure_nn_runtime


def test_effective_learning_rate_scaling():
    base = Config(batch_size=2048, reference_batch_size=512, lr=1e-3)
    assert effective_learning_rate(replace(base, nn_lr_scaling="fixed")) == 1e-3
    assert np.isclose(
        effective_learning_rate(replace(base, nn_lr_scaling="sqrt")), 2e-3
    )
    assert np.isclose(
        effective_learning_rate(replace(base, nn_lr_scaling="linear")), 4e-3
    )


def test_batch_ranges_never_emit_singleton():
    ranges = list(_batch_ranges(1025, 512))
    assert ranges == [(0, 512), (512, 1023), (1023, 1025)]
    assert all(end - start >= 2 for start, end in ranges)


def test_device_resident_training_reports_stats():
    cfg = Config(
        num_products=2,
        hidden_dims=(8,),
        dropout=(0.0,),
        batch_size=4,
        nn_training_backend="device_resident",
        nn_lr_scaling="fixed",
    )
    n = 8
    frame = pd.DataFrame({col: np.zeros(n) for col in numeric_feature_columns(cfg)})
    frame["product_idx"] = np.arange(n) % 2
    frame["campaign_idx_web"] = 0
    frame["campaign_idx_app"] = 0
    frame["horizon"] = 1
    frame["target_baseline"] = 10.0
    tensors = make_tensors(frame, StandardScaler(), fit=True, cfg=cfg)
    stats = {}
    model = train_model(
        tensors,
        np.zeros(n, dtype=np.float32),
        cfg,
        epochs=1,
        seed=42,
        stats_out=stats,
    )
    assert model is not None
    assert stats["optimizer_steps"] == 2
    assert stats["backend"] == "device_resident"
    assert np.isfinite(stats["final_train_loss"])


def test_quality_aware_batch_recommendation():
    results = [
        {"batch_size": 512, "lr_scaling": "fixed", "WAPE": 0.20, "elapsed_seconds": 100.0},
        {"batch_size": 2048, "lr_scaling": "sqrt", "WAPE": 0.203, "elapsed_seconds": 45.0,
         "effective_learning_rate": 0.002},
        {"batch_size": 4096, "lr_scaling": "sqrt", "WAPE": 0.22, "elapsed_seconds": 30.0,
         "effective_learning_rate": 0.002828},
    ]
    results[0]["effective_learning_rate"] = 0.001
    rec = choose_recommendation(results, tolerance=0.02)
    assert rec["batch_size"] == 2048
    assert np.isclose(rec["speedup_vs_reference"], 100 / 45)


def test_auto_runtime_uses_same_device_benchmark(tmp_path, monkeypatch):
    from pipeline import DEVICE

    benchmark = tmp_path / "benchmark.json"
    cfg = Config()
    benchmark.write_text(json.dumps({
        "schema_version": "nn-batch-v1",
        "environment": {"device": DEVICE.type},
        "model_signature": nn_performance_signature(cfg),
        "recommendation": {"batch_size": 2048, "lr_scaling": "sqrt"},
    }))
    options = RuntimeOptions(nn_benchmark_file=str(benchmark))
    runtime = configure_nn_runtime(cfg, options)
    assert runtime["batch_size"] == 2048
    assert runtime["lr_scaling"] == "sqrt"
    assert np.isclose(runtime["effective_learning_rate"], 0.002)
