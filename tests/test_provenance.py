import json

from provenance import (
    build_run_manifest,
    refresh_legacy_audit_manifest,
    sha256_file,
    write_run_manifest,
)


def test_run_manifest_hashes_inputs_lock_config_and_outputs(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "data" / "train_data.parquet").write_bytes(b"train")
    (tmp_path / "data" / "test_data.parquet").write_bytes(b"test")
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")
    result = tmp_path / "outputs" / "results.json"
    result.write_text('{"ok": true}', encoding="utf-8")
    config = {"horizon": 7}

    manifest = build_run_manifest(
        tmp_path,
        command=["python", "ml/export_results.py"],
        config=config,
        output_paths=[result],
        device=None,
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert manifest["generated_at"] == "2026-01-01T00:00:00+00:00"
    assert manifest["inputs"]["train"]["sha256"] == sha256_file(
        tmp_path / "data" / "train_data.parquet"
    )
    assert manifest["lock"]["sha256"] == sha256_file(tmp_path / "uv.lock")
    assert manifest["outputs"]["outputs/results.json"] == sha256_file(result)
    assert manifest["configuration"]["values"] == config
    assert manifest["runtime"]["device"] is None


def test_run_manifest_is_written_atomically_without_self_hash(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "data" / "train_data.parquet").write_bytes(b"train")
    (tmp_path / "data" / "test_data.parquet").write_bytes(b"test")
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")
    (tmp_path / "outputs" / "results.json").write_text("{}", encoding="utf-8")

    write_run_manifest(
        tmp_path,
        command=["export"],
        config={"epochs": 30},
        device={"type": "cpu"},
    )
    payload = json.loads(
        (tmp_path / "outputs" / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert "outputs/run_manifest.json" not in payload["outputs"]
    assert payload["runtime"]["device"] == {"type": "cpu"}


def test_legacy_audit_manifest_marks_unknown_history_instead_of_guessing(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "results.json").write_text('{"published": true}')
    (outputs / "final_audit_summary.csv").write_text("model,WAPE\nNN,0.2\n")
    (outputs / "final_audit_manifest.json").write_text(json.dumps({
        "schema_version": "c5-final-audit-v1",
        "results_sha256_before_refresh": "old",
    }))

    payload = refresh_legacy_audit_manifest(tmp_path)

    assert payload["provenance_status"] == "partial_historical"
    assert payload["source"] is None
    assert payload["inputs"] is None
    assert payload["legacy_results_sha256_before_refresh"] == "old"
    assert payload["published_results_sha256_after_refresh"] == sha256_file(
        outputs / "results.json"
    )
    assert payload["audit_output_hashes"]["outputs/final_audit_summary.csv"]

    refreshed = refresh_legacy_audit_manifest(tmp_path)
    assert refreshed["legacy_results_sha256_before_refresh"] == "old"
