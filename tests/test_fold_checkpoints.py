import pandas as pd

from ml.framework import Config
from ml.pipeline import _load_fold_checkpoint, _save_fold_checkpoint


def test_fold_checkpoint_roundtrip_and_signature_guard(tmp_path):
    cfg = Config(num_products=2)
    origin = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({"ProductId": [1], "prediction": [2.0]})
    timing = {"strategy": "recursive", "fold_seconds": 1.5}

    _save_fold_checkpoint(
        str(tmp_path), "recursive", "development", origin, cfg, frame, timing
    )
    loaded = _load_fold_checkpoint(
        str(tmp_path), "recursive", "development", origin, cfg
    )
    pd.testing.assert_frame_equal(loaded["oof"], frame)
    assert loaded["timing"] == timing

    incompatible = Config(num_products=3)
    assert _load_fold_checkpoint(
        str(tmp_path), "recursive", "development", origin, incompatible
    ) is None


def test_pre_performance_checkpoint_is_reusable_for_512_fixed(tmp_path):
    import os
    import pickle
    from dataclasses import asdict
    from ml.pipeline import CHECKPOINT_SCHEMA_VERSION, _fold_checkpoint_path

    cfg = Config(num_products=2)
    origin = pd.Timestamp("2024-01-01")
    path = _fold_checkpoint_path(
        str(tmp_path), "direct", "development", origin
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old_cfg = asdict(cfg)
    old_cfg.pop("reference_batch_size")
    old_cfg.pop("nn_lr_scaling")
    old_cfg.pop("nn_training_backend")
    payload = {
        "signature": {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "strategy": "direct",
            "origin_type": "development",
            "origin": origin.isoformat(),
            "cfg": old_cfg,
        },
        "oof": pd.DataFrame({"ProductId": [1], "prediction": [2.0]}),
        "timing": {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    assert _load_fold_checkpoint(
        str(tmp_path), "direct", "development", origin, cfg
    ) is not None

    faster_cfg = Config(
        num_products=2,
        batch_size=2048,
        nn_lr_scaling="sqrt",
    )
    assert _load_fold_checkpoint(
        str(tmp_path), "direct", "development", origin, faster_cfg
    ) is None
