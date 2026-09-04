"""Evidence document construction.

The manifest holds everything a later verifier needs: what was scanned, what
was discovered, how the match was verified, and how the search itself can be
re-checked. Only this structure is hashed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from faceproof.evidence.hashing import canonical_json, fingerprint, sha256_bytes
from faceproof.matching.verifier import Match

SCHEMA = "faceproof/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evidence_id(probe_digest: str) -> str:
    """Stable, sortable, filesystem-safe identifier for one pipeline run."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    return f"{stamp}-{probe_digest[:8]}"


def build_evidence(
    probe_digest: str,
    probe_scan: Dict[str, Any],
    match: Match,
    search_engine: str,
    probe_image_url: str,
    search_response: Dict[str, Any],
    candidates_returned: int,
    candidates_checked: int,
) -> Dict[str, Any]:
    """Assemble the hashed region of the evidence document."""
    return {
        "schema": SCHEMA,
        "discovered_at": _now(),
        "probe": {
            "sha256": probe_digest,
            "faces_detected": probe_scan["faces_detected"],
            "embedding_model": probe_scan["model"],
            "detector_backend": probe_scan["detector"],
            "embedding_dimensions": probe_scan["embedding_dimensions"],
        },
        "post": {
            "page_url": match.candidate.page_url,
            "image_url": match.candidate.image_url,
            "image_sha256": sha256_bytes(match.image_bytes),
            "title": match.candidate.title,
            "source": match.candidate.source,
            "is_social": match.candidate.is_social,
        },
        "match": {
            "model": match.model,
            "distance_metric": match.metric,
            "distance": match.distance,
            "threshold": match.threshold,
            "verified": True,
        },
        "search": {
            "engine": search_engine,
            "probe_image_url": probe_image_url,
            # binds the bulky search proof without embedding it in the hashed region
            "response_sha256": sha256_bytes(canonical_json(search_response)),
            "candidates_returned": candidates_returned,
            "candidates_checked": candidates_checked,
        },
    }


def build_document(
    evidence: Dict[str, Any],
    anchor: Dict[str, Any],
    match_image_name: str,
    search_response: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap the hashed evidence with its fingerprint, anchor and artifacts."""
    return {
        "evidence": evidence,
        "fingerprint": fingerprint(evidence),
        "anchor": anchor,
        "artifacts": {"match_image": match_image_name},
        "search_response": search_response,
    }
