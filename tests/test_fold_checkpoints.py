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
