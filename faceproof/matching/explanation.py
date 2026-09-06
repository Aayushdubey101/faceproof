"""Explainable match scoring for FaceProof.

A pure, read-only explanation layer that interprets the existing face verification
results (ArcFace cosine distance vs authoritative threshold) without modifying
underlying models, verification objects, candidate ranking, or evidence hashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from faceproof.face import adapter
from faceproof.matching.verifier import DOWNLOAD_FAILED, MATCH, NO_FACE, NO_MATCH, CandidateResult, Match


@dataclass(frozen=True)
class MatchExplanation:
    """Read-only forensic explanation of a face-matching verification decision.

    Derived strictly from the existing verification result; does not introduce
    a secondary scoring or statistical confidence model.
    """

    result: str  # MATCH, NO_MATCH, NO_FACE, or DOWNLOAD_FAILED
    distance: Optional[float]
    threshold: Optional[float]
    margin: Optional[float]
    margin_display: str
    comparison: Optional[str]
    decision_rule: Optional[str]
    reasons: List[str]
    detailed_reasons: List[Tuple[bool, str]]
    trace: List[Tuple[str, str, bool]]

    @property
    def passed(self) -> bool:
        """Derived property: true only when the candidate verified as a MATCH."""
        return self.result == MATCH

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary for presentation and API responses."""
        return {
            "result": self.result,
            "passed": self.passed,
            "distance": self.distance,
            "threshold": self.threshold,
            "margin": self.margin,
            "margin_display": self.margin_display,
            "comparison": self.comparison,
            "decision_rule": self.decision_rule,
            "reasons": list(self.reasons),
            "detailed_reasons": list(self.detailed_reasons),
            "trace": list(self.trace),
        }


def distance_scale_percent(distance: Optional[float], max_scale: float = 1.0) -> Optional[float]:
    """Calculate the percentage along a [0.0, max_scale] visual scale.

    Returns None if distance is None. Clamped to [0.0, 100.0]%.
    """
    if distance is None or max_scale <= 0:
        return None
    return max(0.0, min(100.0, round((distance / max_scale) * 100.0, 2)))


def explain_candidate(
    status: str,
    distance: Optional[float] = None,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    metric: Optional[str] = None,
) -> MatchExplanation:
    """Generate a deterministic MatchExplanation from raw candidate verification outputs.

    Does not modify any inputs or verification state.
    """
    model_name = model or adapter.DEFAULT_MODEL
    metric_name = metric or adapter.DEFAULT_METRIC

    # If a distance is present but threshold was omitted, query the single authoritative
    # engine threshold via the adapter rather than hardcoding a second value.
    effective_threshold: Optional[float] = threshold
    if distance is not None and effective_threshold is None:
        effective_threshold = adapter.verification_threshold(model_name, metric_name)

    margin: Optional[float] = None
    margin_display: str = "—"
    comparison: Optional[str] = None
    decision_rule: Optional[str] = None

    if distance is not None and effective_threshold is not None:
        margin = round(effective_threshold - distance, 4)
        margin_display = f"{margin:+.2f}"
        if status == MATCH:
            comparison = "distance <= threshold"
            decision_rule = f"{distance:.2f} \u2264 {effective_threshold:.2f}"
        else:
            comparison = "distance > threshold"
            decision_rule = f"{distance:.2f} > {effective_threshold:.2f}"

    reasons: List[str] = []
    detailed_reasons: List[Tuple[bool, str]] = []
    trace: List[Tuple[str, str, bool]] = []

    if status == MATCH:
        reasons = [
            "Face detected in candidate",
            f"{model_name} embedding generated",
            "Cosine distance satisfies threshold",
            "Verification completed",
        ]
        dist_str = f"{distance:.4f}" if distance is not None else "—"
        thr_str = f"{effective_threshold:.4f}" if effective_threshold is not None else "—"
        detailed_reasons = [
            (True, "Face detected in candidate"),
            (True, f"{model_name} embedding generated"),
            (True, f"Cosine distance below threshold ({dist_str} \u2264 {thr_str})"),
            (True, "Verification threshold satisfied"),
        ]
        trace = [
            ("01", "Candidate image retrieved", True),
            ("02", "Face detected", True),
            ("03", f"{model_name} embedding generated", True),
            ("04", "Cosine distance calculated", True),
            ("05", "Distance compared with threshold", True),
            ("06", f"Threshold condition satisfied ({decision_rule or 'pass'})", True),
            ("07", "Decision: MATCH", True),
        ]

    elif status == NO_MATCH:
        reasons = [
            "Face detected in candidate",
            f"{model_name} embedding generated",
            "Cosine distance exceeds threshold",
        ]
        dist_str = f"{distance:.4f}" if distance is not None else "—"
        thr_str = f"{effective_threshold:.4f}" if effective_threshold is not None else "—"
        detailed_reasons = [
            (True, "Face detected in candidate"),
            (True, f"{model_name} embedding generated"),
            (False, f"Cosine distance exceeds threshold ({dist_str} > {thr_str})"),
        ]
        trace = [
            ("01", "Candidate image retrieved", True),
            ("02", "Face detected", True),
            ("03", f"{model_name} embedding generated", True),
            ("04", "Cosine distance calculated", True),
            ("05", "Distance compared with threshold", True),
            ("06", f"Threshold condition not satisfied ({decision_rule or 'fail'})", False),
            ("07", "Decision: NO MATCH", False),
        ]

    elif status == NO_FACE:
        reasons = ["No usable face detected in candidate"]
        detailed_reasons = [(False, "No usable face detected in candidate")]
        trace = [
            ("01", "Candidate image retrieved", True),
            ("02", "Face detection attempted", True),
            ("03", "No usable face detected in candidate", False),
            ("04", "Decision: NO FACE (verification unavailable)", False),
        ]

    elif status == DOWNLOAD_FAILED:
        reasons = ["Candidate image could not be downloaded"]
        detailed_reasons = [(False, "Candidate image could not be downloaded")]
        trace = [
            ("01", "Candidate image retrieval attempted", True),
            ("02", "Candidate image could not be downloaded", False),
            ("03", "Decision: DOWNLOAD FAILED (verification not performed)", False),
        ]

    else:
        reasons = [f"Unknown status: {status}"]
        detailed_reasons = [(False, f"Unknown status: {status}")]
        trace = [("01", f"Unknown verification outcome: {status}", False)]

    return MatchExplanation(
        result=status,
        distance=distance,
        threshold=effective_threshold,
        margin=margin,
        margin_display=margin_display,
        comparison=comparison,
        decision_rule=decision_rule,
        reasons=reasons,
        detailed_reasons=detailed_reasons,
        trace=trace,
    )


def explain_match(match: Match) -> MatchExplanation:
    """Generate an explanation from a verified Match object without mutating it."""
    return explain_candidate(
        status=MATCH,
        distance=match.distance,
        threshold=match.threshold,
        model=match.model,
        metric=match.metric,
    )


def explain_result(
    result: CandidateResult,
    model: Optional[str] = None,
    metric: Optional[str] = None,
) -> MatchExplanation:
    """Generate an explanation from a CandidateResult object without mutating it."""
    return explain_candidate(
        status=result.status,
        distance=result.distance,
        threshold=result.threshold,
        model=model,
        metric=metric,
    )


def explain_dict(data: Dict[str, Any]) -> MatchExplanation:
    """Generate an explanation from a dictionary (e.g. from an evidence manifest or summary)."""
    verified = bool(data.get("verified"))
    status = data.get("status") or (MATCH if verified else NO_MATCH)
    distance = data.get("distance")
    if distance is not None:
        distance = float(distance)
    threshold = data.get("threshold")
    if threshold is not None:
        threshold = float(threshold)

    return explain_candidate(
        status=status,
        distance=distance,
        threshold=threshold,
        model=data.get("model"),
        metric=data.get("metric") or data.get("distance_metric"),
    )
