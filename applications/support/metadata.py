"""Manuscript workflow helpers, extracted from code/utopia_compound/metadata.py."""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def canonical_json(metadata: Mapping[str, Any]) -> str:
    """Return deterministic JSON used for artifact identity hashing."""
    return json.dumps(
        _normalize_for_json(metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def metadata_hash(metadata: Mapping[str, Any], length: int = 12) -> str:
    digest = hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()
    return digest[:length]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(
    path: str | Path, hash_max_bytes: int = 32 * 1024 * 1024
) -> dict[str, Any]:
    """Fingerprint a preprocessing artifact without forcing hashes of huge arrays."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    fingerprint: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= hash_max_bytes:
        fingerprint["sha256"] = file_sha256(path)
    else:
        fingerprint["sha256"] = None
        fingerprint["sha256_skipped_reason"] = f"larger than {hash_max_bytes} bytes"
    return fingerprint


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_normalize_for_json(payload), indent=2, sort_keys=True) + "\n"
    )
