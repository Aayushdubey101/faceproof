"""Canonicalisation and SHA-256.

Equivalent evidence must always produce the same fingerprint, so the document
is serialised with sorted keys and no incidental whitespace before hashing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

DIGEST_BYTES = 32


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(document: Dict[str, Any]) -> bytes:
    """Byte-stable serialisation - key order and spacing must never drift."""
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(evidence: Dict[str, Any]) -> str:
    """SHA-256 of the canonical evidence document."""
    return sha256_bytes(canonical_json(evidence))
