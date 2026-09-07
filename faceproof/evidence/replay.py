"""Cached investigation replay for FaceProof.

A pure read-only projection layer over existing evidence documents in `evidence/<id>.json`.
Cached replay reconstructs the recorded investigation from persisted evidence without repeating
web discovery, candidate image downloads, face-engine inference (RetinaFace/ArcFace), or
blockchain anchoring transactions.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from faceproof.discovery.candidates import Candidate, parse_visual_matches
from faceproof.evidence import hashing, provenance
from faceproof.matching import (
    distance_scale_percent,
    explain_candidate,
)
from faceproof.matching.verifier import MATCH, NO_MATCH

# Permitted investigation ID pattern: safe filesystem chars, no path separators
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.:]+$")

# Matched image artifact states
ARTIFACT_VERIFIED = "VERIFIED"
ARTIFACT_INTEGRITY_FAILED = "INTEGRITY_FAILED"
ARTIFACT_MISSING = "MISSING"


class InvestigationNotFoundError(FileNotFoundError):
    """Raised when the requested investigation cannot be found in the evidence archive."""

    def __init__(self, investigation_id: str) -> None:
        super().__init__(f"Investigation not found: {investigation_id}")
        self.investigation_id = investigation_id


class InvestigationSecurityError(ValueError):
    """Raised when an investigation ID contains disallowed characters or path traversal."""

    def __init__(self, investigation_id: str) -> None:
        super().__init__(f"Invalid investigation identifier: {investigation_id}")
        self.investigation_id = investigation_id


def _domain(url: Any) -> str:
    return urlsplit(str(url)).netloc or str(url)


def sanitize_investigation_id(investigation_id: str) -> str:
    """Validate and sanitize an investigation identifier.
    
    Rejects path traversal, directory separators, and invalid characters.
    Strips trailing .json if supplied.
    """
    if not investigation_id or not isinstance(investigation_id, str):
        raise InvestigationSecurityError(str(investigation_id))
    
    clean_id = investigation_id.strip()
    if clean_id.endswith(".json"):
        clean_id = clean_id[:-5]

    if not clean_id or not SAFE_ID_PATTERN.match(clean_id):
        raise InvestigationSecurityError(investigation_id)

    if ".." in clean_id or "/" in clean_id or "\\" in clean_id:
        raise InvestigationSecurityError(investigation_id)

    return clean_id


@dataclass(frozen=True)
class InvestigationReplay:
    """Read-only projection representing a reconstructed historical investigation."""

    investigation_id: str
    discovered_at: str
    document: Dict[str, Any]
    evidence: Dict[str, Any]
    fingerprint: str
    anchor: Dict[str, Any]
    post: Dict[str, Any]
    probe: Dict[str, Any]
    search: Dict[str, Any]
    match: Dict[str, Any]
    artifacts: Dict[str, Any]
    manifest_ok: bool
    search_response_ok: Optional[bool]
    match_image_state: str
    overall_integrity_ok: bool
    best_candidate: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    events: List[provenance.InvestigationEvent]
    summary: Dict[str, Any]
    replay_mode: str = "cached"

    @property
    def is_match(self) -> bool:
        return bool(self.match.get("verified"))

    def as_dict(self) -> Dict[str, Any]:
        """Serializable dictionary for API consumption without leaking secrets."""
        return {
            "investigation_id": self.investigation_id,
            "discovered_at": self.discovered_at,
            "replay_mode": self.replay_mode,
            "fingerprint": self.fingerprint,
            "manifest_ok": self.manifest_ok,
            "search_response_ok": self.search_response_ok,
            "match_image_state": self.match_image_state,
            "overall_integrity_ok": self.overall_integrity_ok,
            "is_match": self.is_match,
            "probe": dict(self.probe),
            "search": {
                "engine": self.search.get("engine"),
                "candidates_returned": self.search.get("candidates_returned"),
                "candidates_checked": self.search.get("candidates_checked"),
                "response_sha256": self.search.get("response_sha256"),
            },
            "post": dict(self.post),
            "match": dict(self.match),
            "anchor": {
                "network": self.anchor.get("network"),
                "chain_id": self.anchor.get("chain_id"),
                "block_number": self.anchor.get("block_number"),
                "gas_used": self.anchor.get("gas_used"),
                "tx_hash": self.anchor.get("tx_hash"),
            },
            "best_candidate": self.best_candidate,
            "candidate_count": len(self.candidates),
            "candidates": self.candidates,
            "summary": self.summary,
        }


def _build_candidate_card_dict(
    index: int,
    candidate: Candidate,
    is_winning_match: bool,
    winning_match: Dict[str, Any],
    artifact_image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a dictionary suitable for candidate card and best match templates."""
    status = MATCH if is_winning_match else NO_MATCH
    distance = winning_match.get("distance") if is_winning_match else None
    threshold = winning_match.get("threshold") if is_winning_match else None
    model = winning_match.get("model") if is_winning_match else "ArcFace"

    # Derive the presentation explanation without running any ML models
    if is_winning_match and distance is not None:
        explanation = explain_candidate(
            status=MATCH,
            distance=float(distance),
            threshold=float(threshold) if threshold is not None else 0.68,
            model=model,
            metric=winning_match.get("distance_metric") or "cosine",
        )
    else:
        explanation = explain_candidate(
            status=status,
            distance=None,
            threshold=None,
            model=model,
        )

    # Use local artifact if verified match and available, otherwise candidate remote image
    image = artifact_image_url if (is_winning_match and artifact_image_url) else candidate.image_url

    return {
        "number": index + 1,
        "rank": candidate.position,
        "is_social": candidate.is_social,
        "label": "VERIFIED" if is_winning_match else "RECORDED",
        "tone": "ok" if is_winning_match else "neutral",
        "source": candidate.source or _domain(candidate.page_url),
        "page_url": candidate.page_url,
        "image": image,
        "distance": distance,
        "threshold": threshold,
        "reason": "Verified match from recorded investigation" if is_winning_match else "Discovered candidate in recorded search",
        "status": status,
        "margin": explanation.margin,
        "margin_display": explanation.margin_display,
        "decision_rule": explanation.decision_rule,
        "comparison": explanation.comparison,
        "reasons": explanation.reasons,
        "detailed_reasons": explanation.detailed_reasons,
        "trace": explanation.trace,
        "distance_percent": distance_scale_percent(distance),
        "threshold_percent": distance_scale_percent(threshold) if threshold is not None else 68.0,
    }


def load_replay(investigation_id: str, evidence_dir: str = "evidence") -> InvestigationReplay:
    """Load an existing investigation and construct a read-only replay model.
    
    Does NOT invoke:
      - Google Lens or SerpApi
      - Candidate image downloads
      - RetinaFace or ArcFace
      - Candidate ranking on live web data
      - Base Sepolia blockchain anchoring
      
    Replay validates local integrity and accurately represents recorded investigation state.
    """
    clean_id = sanitize_investigation_id(investigation_id)
    filename = f"{clean_id}.json"
    evidence_path = os.path.join(evidence_dir, filename)

    if not os.path.isfile(evidence_path):
        raise InvestigationNotFoundError(clean_id)

    with open(evidence_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    if not isinstance(document, dict) or "evidence" not in document or "fingerprint" not in document:
        raise ValueError(f"Corrupted or invalid evidence format in {filename}")

    evidence = document["evidence"]
    stored_fingerprint = str(document["fingerprint"])
    recomputed_fingerprint = hashing.fingerprint(evidence)
    manifest_ok = (recomputed_fingerprint == stored_fingerprint)

    # 1. Search response integrity check
    search_response_ok: Optional[bool] = None
    if "search_response" in document and isinstance(document["search_response"], dict):
        search_digest = hashing.sha256_bytes(hashing.canonical_json(document["search_response"]))
        expected_search_digest = evidence.get("search", {}).get("response_sha256")
        search_response_ok = (search_digest == expected_search_digest)
    elif evidence.get("search", {}).get("response_sha256"):
        search_response_ok = False

    # 2. Matched image artifact integrity check
    match_image_state = ARTIFACT_MISSING
    artifact_name = document.get("artifacts", {}).get("match_image")
    artifact_image_url: Optional[str] = None
    if artifact_name:
        artifact_path = os.path.join(evidence_dir, artifact_name)
        if os.path.isfile(artifact_path):
            expected_image_hash = evidence.get("post", {}).get("image_sha256")
            actual_image_hash = hashing.sha256_file(artifact_path)
            if actual_image_hash == expected_image_hash:
                match_image_state = ARTIFACT_VERIFIED
                artifact_image_url = f"/evidence/{artifact_name}"
            else:
                match_image_state = ARTIFACT_INTEGRITY_FAILED
        else:
            match_image_state = ARTIFACT_MISSING

    # Overall local integrity: manifest intact and no explicit integrity failures
    overall_integrity_ok = (
        manifest_ok
        and (search_response_ok is not False)
        and (match_image_state != ARTIFACT_INTEGRITY_FAILED)
    )

    post = evidence.get("post", {})
    match = evidence.get("match", {})
    probe = evidence.get("probe", {})
    search = evidence.get("search", {})
    anchor = document.get("anchor", {})
    artifacts = document.get("artifacts", {})

    # 3. Candidate reconstruction:
    # Priority: inspect stored search_response for visual matches
    candidates: List[Dict[str, Any]] = []
    matched_candidate_in_list = False
    
    if "search_response" in document and isinstance(document["search_response"], dict):
        try:
            parsed_cands = parse_visual_matches(document["search_response"])
            post_page = post.get("page_url", "")
            post_image = post.get("image_url", "")

            for idx, c in enumerate(parsed_cands):
                is_win = (c.page_url == post_page) or (c.image_url == post_image)
                if is_win:
                    matched_candidate_in_list = True
                cand_dict = _build_candidate_card_dict(
                    index=idx,
                    candidate=c,
                    is_winning_match=is_win,
                    winning_match=match,
                    artifact_image_url=artifact_image_url,
                )
                candidates.append(cand_dict)
        except Exception:
            candidates = []

    # If the winning match wasn't in the parsed list or search response had no matches,
    # construct candidate directly from recorded post metadata.
    winning_cand_obj = Candidate(
        title=post.get("title", ""),
        page_url=post.get("page_url", ""),
        image_url=post.get("image_url", ""),
        source=post.get("source", ""),
        position=1,
        is_social=bool(post.get("is_social")),
    )

    best_candidate = _build_candidate_card_dict(
        index=0,
        candidate=winning_cand_obj,
        is_winning_match=bool(match.get("verified")),
        winning_match=match,
        artifact_image_url=artifact_image_url,
    )

    if not candidates or not matched_candidate_in_list:
        candidates.insert(0, best_candidate)

    # Provenance timeline and summary
    events = provenance.timeline(document)
    summary = provenance.summary(document)

    return InvestigationReplay(
        investigation_id=clean_id,
        discovered_at=evidence.get("discovered_at", ""),
        document=document,
        evidence=evidence,
        fingerprint=stored_fingerprint,
        anchor=anchor,
        post=post,
        probe=probe,
        search=search,
        match=match,
        artifacts=artifacts,
        manifest_ok=manifest_ok,
        search_response_ok=search_response_ok,
        match_image_state=match_image_state,
        overall_integrity_ok=overall_integrity_ok,
        best_candidate=best_candidate,
        candidates=candidates,
        events=events,
        summary=summary,
        replay_mode="cached",
    )
