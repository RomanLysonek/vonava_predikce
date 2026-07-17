"""Immutable provenance manifests for published forecasting artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA_VERSION = "forecast-run-v1"
PACKAGE_NAMES = (
    "fastapi",
    "lightgbm",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "torch",
    "uvicorn",
    "xgboost",
)
GENERATED_EXCLUSIONS = {
    "outputs/dashboard_manifest.json",
    "outputs/run_manifest.json",
    "webapp/static/results.json",
    "webapp/static/run_manifest.json",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_metadata(root: Path) -> dict:
    revision = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    diff = _git(
        root,
        "diff",
        "--binary",
        "HEAD",
        "--",
        ".",
        ":(exclude)outputs",
        ":(exclude)docs",
        ":(exclude)webapp/static/results.json",
        ":(exclude)webapp/static/run_manifest.json",
    )
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    untracked_source = []
    for value in (untracked or "").splitlines():
        if (
            value.startswith(("outputs/", "docs/"))
            or value in GENERATED_EXCLUSIONS
        ):
            continue
        path = root / value
        if path.is_file():
            untracked_source.append(
                {"path": value, "sha256": sha256_file(path)}
            )
    dirty_material = {
        "tracked_diff": diff or "",
        "untracked_source": untracked_source,
    }
    return {
        "revision": revision,
        "tree": tree,
        "dirty": bool(status),
        "working_tree_source_sha256": (
            sha256_json(dirty_material) if status else None
        ),
    }


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def discover_output_paths(root: str | Path) -> list[Path]:
    output_dir = Path(root) / "outputs"
    return sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "run_manifest.json"
        and "checkpoints" not in path.parts
    )


def build_run_manifest(
    repository_root: str | Path,
    *,
    command: Iterable[str],
    config: Mapping,
    output_paths: Iterable[str | Path] | None = None,
    device: Mapping | None = None,
    generated_at: str | None = None,
) -> dict:
    root = Path(repository_root).resolve()
    train_path = root / str(config.get("train_path", "data/train_data.parquet"))
    test_path = root / str(config.get("test_path", "data/test_data.parquet"))
    lock_path = root / "uv.lock"
    inputs = {}
    for label, path in (("train", train_path), ("test", test_path)):
        inputs[label] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    outputs = {}
    for value in output_paths or discover_output_paths(root):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(path)
        relative = str(path.relative_to(root))
        if relative in GENERATED_EXCLUSIONS:
            continue
        outputs[relative] = sha256_file(path)
    command_parts = [str(part) for part in command]
    safe_config = _json_safe(dict(config))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "source": _source_metadata(root),
        "inputs": inputs,
        "configuration": {
            "sha256": sha256_json(safe_config),
            "values": safe_config,
        },
        "lock": {
            "path": "uv.lock",
            "sha256": sha256_file(lock_path) if lock_path.exists() else None,
        },
        "command": {
            "argv": command_parts,
            "display": shlex.join(command_parts),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
            "device": dict(device) if device else None,
        },
        "outputs": outputs,
    }


def write_run_manifest(
    repository_root: str | Path,
    *,
    command: Iterable[str],
    config: Mapping,
    output_paths: Iterable[str | Path] | None = None,
    device: Mapping | None = None,
    manifest_path: str | Path = "outputs/run_manifest.json",
) -> dict:
    root = Path(repository_root).resolve()
    destination = Path(manifest_path)
    if not destination.is_absolute():
        destination = root / destination
    manifest = build_run_manifest(
        root,
        command=command,
        config=config,
        output_paths=output_paths,
        device=device,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return manifest


def refresh_legacy_audit_manifest(repository_root: str | Path) -> dict | None:
    """Link a legacy audit record to current outputs without inventing history."""
    root = Path(repository_root).resolve()
    path = root / "outputs" / "final_audit_manifest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "c5-final-audit-v1":
        return payload
    results_path = root / "outputs" / "results.json"
    audit_names = (
        "final_audit_oof.parquet",
        "final_audit_summary.csv",
        "final_audit_validation_strata.csv",
        "final_audit_test_aligned_scores.csv",
        "final_audit_prediction_diagnostics.csv",
        "final_audit_prediction_diagnostics_by_origin.csv",
    )
    payload["provenance_status"] = "partial_historical"
    payload["provenance_note"] = (
        "The original one-shot run did not record source, input, lock, command, "
        "or runtime metadata. Those fields remain null rather than being guessed; "
        "outputs/run_manifest.json describes the current published export. The "
        "audit evaluates the cv_epochs fold estimator; historical final_epochs "
        "describes a separate final-forecast fit and is not audit policy."
    )
    payload["source"] = None
    payload["inputs"] = None
    payload["lock"] = None
    payload["command"] = None
    payload["runtime"] = None
    payload["legacy_results_sha256_before_refresh"] = (
        payload.get("legacy_results_sha256_before_refresh")
        or payload.pop("results_sha256_before_refresh", None)
    )
    payload["published_results_sha256_after_refresh"] = (
        sha256_file(results_path) if results_path.exists() else None
    )
    payload["evaluated_estimator"] = payload.get("evaluated_estimator") or {
        "epochs": (payload.get("configuration") or {}).get("cv_epochs"),
        "seeds": (payload.get("configuration") or {}).get("seeds"),
        "strategy": payload.get("strategy"),
    }
    if not payload.get("audit_output_hashes"):
        payload["audit_output_hashes"] = {
            f"outputs/{name}": sha256_file(root / "outputs" / name)
            for name in audit_names
            if (root / "outputs" / name).exists()
        }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload
