"""The investigation chain of a finished evidence document.

A read model, and only that: every value below is read out of the evidence
document that was already built and hashed, or out of an `Audit` that already
ran. Nothing here recomputes a distance, re-reads the chain, or decides whether
something passed - it arranges facts that already exist into the order they
happened in, so a reviewer can follow face to anchor without reading a log.

The one exception is the local fingerprint check, which is the same one-line
recomputation the archive already does: hash the manifest, compare it with the
fingerprint stored beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from faceproof.evidence import hashing

FACE_SCAN = "FACE_SCAN"
WEB_DISCOVERY = "WEB_DISCOVERY"
CANDIDATE_RANKING = "CANDIDATE_RANKING"
FACE_VERIFICATION = "FACE_VERIFICATION"
EVIDENCE_CREATED = "EVIDENCE_CREATED"
HASH_GENERATED = "HASH_GENERATED"
BLOCKCHAIN_ANCHORED = "BLOCKCHAIN_ANCHORED"
INDEPENDENT_AUDIT = "INDEPENDENT_AUDIT"

STAGES = (
    FACE_SCAN,
    WEB_DISCOVERY,
    CANDIDATE_RANKING,
    FACE_VERIFICATION,
    EVIDENCE_CREATED,
    HASH_GENERATED,
    BLOCKCHAIN_ANCHORED,
    INDEPENDENT_AUDIT,
)

COMPLETE = "complete"
FAILED = "failed"
PENDING = "pending"
VERIFIED = "verified"
TAMPERED = "tampered"


class AuditReport(Protocol):
    """What `pipeline.audit` returns.

    Declared structurally rather than imported: evidence sits below the
    pipeline, and a read model has no business reaching back up to it.
    """

    # read-only members, so a frozen dataclass satisfies them
    @property
    def local_fingerprint(self) -> str: ...

    @property
    def onchain_fingerprint(self) -> str: ...

    @property
    def tx_hash(self) -> str: ...

    @property
    def search_ok(self) -> Optional[bool]: ...

    @property
    def image_ok(self) -> Optional[bool]: ...

    @property
    def local_ok(self) -> bool: ...

    @property
    def onchain_ok(self) -> bool: ...

    @property
    def passed(self) -> bool: ...


@dataclass(frozen=True)
class InvestigationEvent:
    """One step of the chain that produced a piece of evidence.

    `timestamp` is None wherever the document does not carry one. The manifest
    records a single `discovered_at` for the run that built it, so that is the
    only stage given a time - inventing one per stage would be precision the
    evidence never had.
    """

    stage: str
    status: str
    title: str
    description: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (COMPLETE, VERIFIED)

    @property
    def bad(self) -> bool:
        return self.status in (FAILED, TAMPERED)


def _drop_empty(values: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields the document actually carried."""
    return {key: value for key, value in values.items() if value not in (None, "", [])}


def _face_scan(probe: Dict[str, Any]) -> InvestigationEvent:
    faces = probe.get("faces_detected")
    dimensions = probe.get("embedding_dimensions")
    return InvestigationEvent(
        stage=FACE_SCAN,
        status=COMPLETE if probe.get("sha256") else FAILED,
        title="Face scan",
        description=(
            f"{faces} face(s) detected and encoded to a {dimensions}-dimension vector"
            if faces and dimensions
            else "Probe face detected and encoded"
        ),
        metadata=_drop_empty(
            {
                "Detector": probe.get("detector_backend"),
                "Model": probe.get("embedding_model"),
                "Embedding": f"{dimensions}-d" if dimensions else None,
                "Faces detected": faces,
                "Probe SHA-256": probe.get("sha256"),
            }
        ),
    )


def _web_discovery(search: Dict[str, Any]) -> InvestigationEvent:
    returned = search.get("candidates_returned")
    return InvestigationEvent(
        stage=WEB_DISCOVERY,
        status=COMPLETE if returned else FAILED,
        title="Web discovery",
        description=(
            f"Reverse image search returned {returned} candidate page(s)"
            if returned
            else "Reverse image search returned no candidates"
        ),
        metadata=_drop_empty(
            {
                "Provider": search.get("engine"),
                "Candidates discovered": returned,
                "Search response SHA-256": search.get("response_sha256"),
            }
        ),
    )


def _candidate_ranking(search: Dict[str, Any]) -> InvestigationEvent:
    returned, checked = search.get("candidates_returned"), search.get("candidates_checked")
    skipped = (
        (returned - checked) if isinstance(returned, int) and isinstance(checked, int) else None
    )
    return InvestigationEvent(
        stage=CANDIDATE_RANKING,
        status=COMPLETE if checked else FAILED,
        title="Candidate ranking",
        description=(
            f"{checked} of {returned} candidates entered face verification; the rest "
            f"ranked below the budget and were never investigated"
            if skipped
            else f"{checked} candidate(s) entered face verification"
        ),
        metadata=_drop_empty(
            {
                "Discovered": returned,
                "Investigated": checked,
                "Not investigated": skipped,
            }
        ),
    )


def _face_verification(match: Dict[str, Any], post: Dict[str, Any]) -> InvestigationEvent:
    distance, threshold = match.get("distance"), match.get("threshold")
    verified = bool(match.get("verified"))
    return InvestigationEvent(
        stage=FACE_VERIFICATION,
        status=COMPLETE if verified else FAILED,
        title="Face verification",
        description=(
            f"Closest candidate scored {distance:.4f}, inside the {threshold} threshold"
            if verified and isinstance(distance, (int, float))
            else "No candidate passed the face-match threshold"
        ),
        metadata=_drop_empty(
            {
                "Source": post.get("source"),
                "Matched page": post.get("page_url"),
                "Distance": f"{distance:.4f}" if isinstance(distance, (int, float)) else None,
                "Threshold": threshold,
                "Metric": match.get("distance_metric"),
                "Model": match.get("model"),
            }
        ),
    )


def _evidence_created(evidence: Dict[str, Any], post: Dict[str, Any]) -> InvestigationEvent:
    return InvestigationEvent(
        stage=EVIDENCE_CREATED,
        status=COMPLETE,
        title="Evidence created",
        description="Manifest assembled from the probe, the matched post and the search proof",
        # the only timestamp the document carries, and it belongs to this stage
        timestamp=evidence.get("discovered_at"),
        metadata=_drop_empty(
            {
                "Schema": evidence.get("schema"),
                "Created": evidence.get("discovered_at"),
                "Matched image SHA-256": post.get("image_sha256"),
            }
        ),
    )


def _hash_generated(evidence: Dict[str, Any], fingerprint: str) -> InvestigationEvent:
    intact = hashing.fingerprint(evidence) == fingerprint
    return InvestigationEvent(
        stage=HASH_GENERATED,
        status=COMPLETE if intact else FAILED,
        title="SHA-256 fingerprint",
        description=(
            "Canonical JSON hashed to a 64-character digest"
            if intact
            else "The manifest no longer hashes to the fingerprint stored beside it"
        ),
        metadata={"Fingerprint": fingerprint},
    )


def _blockchain_anchored(anchor: Dict[str, Any]) -> InvestigationEvent:
    tx_hash = anchor.get("tx_hash")
    network = anchor.get("network")
    return InvestigationEvent(
        stage=BLOCKCHAIN_ANCHORED,
        status=COMPLETE if tx_hash else FAILED,
        title="Blockchain anchor",
        description=(
            f"Digest written to {network} in block {anchor.get('block_number')}"
            if tx_hash
            else "This evidence carries no anchor transaction"
        ),
        metadata=_drop_empty(
            {
                "Network": network,
                "Chain ID": anchor.get("chain_id"),
                "Block": anchor.get("block_number"),
                "Gas": anchor.get("gas_used"),
                "Transaction": tx_hash,
            }
        ),
    )


def _independent_audit(audit: Optional[AuditReport]) -> InvestigationEvent:
    """The audit stage, and only what an audit actually established.

    With no audit yet the stage is pending, never "verified": the run that
    created the evidence cannot audit itself, and saying otherwise would be the
    exact fake this project exists to rule out.
    """
    if audit is None:
        return InvestigationEvent(
            stage=INDEPENDENT_AUDIT,
            status=PENDING,
            title="Independent audit",
            description=(
                "Not yet audited. Re-verify this record to recompute the fingerprint "
                "and read the anchored one back off the chain."
            ),
        )

    passed = audit.passed
    return InvestigationEvent(
        stage=INDEPENDENT_AUDIT,
        status=VERIFIED if passed else TAMPERED,
        title="Independent audit",
        description=(
            "The recomputed fingerprint matches the digest anchored on chain"
            if passed
            else "The recomputed fingerprint does not match the digest anchored on chain"
        ),
        metadata=_drop_empty(
            {
                "Local fingerprint": audit.local_fingerprint,
                "On-chain digest": audit.onchain_fingerprint,
                "Transaction": audit.tx_hash,
            }
        ),
    )


def timeline(
    document: Dict[str, Any], audit: Optional[AuditReport] = None
) -> List[InvestigationEvent]:
    """The chain that produced this evidence, face scan to independent audit.

    A stage reports `failed` where the document itself says so - no candidates,
    no verified match, a manifest that no longer hashes to its fingerprint, a
    missing anchor. None of those is turned into a success.
    """
    evidence = document.get("evidence", {})
    anchor = document.get("anchor", {})
    search = evidence.get("search", {})
    return [
        _face_scan(evidence.get("probe", {})),
        _web_discovery(search),
        _candidate_ranking(search),
        _face_verification(evidence.get("match", {}), evidence.get("post", {})),
        _evidence_created(evidence, evidence.get("post", {})),
        _hash_generated(evidence, str(document.get("fingerprint", ""))),
        _blockchain_anchored(anchor if isinstance(anchor, dict) else {}),
        _independent_audit(audit),
    ]


def summary(document: Dict[str, Any], audit: Optional[AuditReport] = None) -> Dict[str, Any]:
    """Everything a reviewer needs about one evidence file, on one screen.

    Fields the document does not carry are simply absent; nothing is inferred
    and nothing is filled in with a placeholder that could read as a fact.
    """
    evidence = document.get("evidence", {})
    raw_anchor = document.get("anchor")
    anchor = raw_anchor if isinstance(raw_anchor, dict) else {}
    probe = evidence.get("probe", {})
    post, match = evidence.get("post", {}), evidence.get("match", {})
    search = evidence.get("search", {})

    audited: Dict[str, Any] = {}
    if audit is not None:
        audited = {
            "local_fingerprint": audit.local_fingerprint,
            "onchain_digest": audit.onchain_fingerprint,
            "audit_result": VERIFIED if audit.passed else TAMPERED,
            "search_response_ok": audit.search_ok,
            "matched_image_ok": audit.image_ok,
        }

    return _drop_empty(
        {
            "fingerprint": document.get("fingerprint"),
            "created": evidence.get("discovered_at"),
            "matched_url": post.get("page_url"),
            "matched_source": post.get("source"),
            "matched_image_sha256": post.get("image_sha256"),
            "distance": match.get("distance"),
            "threshold": match.get("threshold"),
            "model": match.get("model"),
            "metric": match.get("distance_metric"),
            "probe_sha256": probe.get("sha256"),
            "detector": probe.get("detector_backend"),
            "embedding_dimensions": probe.get("embedding_dimensions"),
            "search_provider": search.get("engine"),
            "candidates_discovered": search.get("candidates_returned"),
            "candidates_investigated": search.get("candidates_checked"),
            "search_response_sha256": search.get("response_sha256"),
            "network": anchor.get("network"),
            "chain_id": anchor.get("chain_id"),
            "block_number": anchor.get("block_number"),
            "tx_hash": anchor.get("tx_hash"),
            **audited,
        }
    )
