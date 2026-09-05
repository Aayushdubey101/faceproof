"""Candidate face verification.

Discovery produces visually related candidates. This module decides, using the
same face-recognition foundation that encoded the probe, whether a candidate
actually shows the same face.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from faceproof.face import adapter
from faceproof.discovery.candidates import Candidate
from faceproof.discovery.retrieval import download_image

MAX_CANDIDATES = 25
DEMO_CANDIDATES = 10  # what one investigation costs interactively; the cap above is the ceiling

# per-candidate cost lands here, not on stdout: the demo output stays readable
LOG = logging.getLogger(__name__)

MATCH = "MATCH"
NO_MATCH = "NO_MATCH"
NO_FACE = "NO_FACE"
DOWNLOAD_FAILED = "DOWNLOAD_FAILED"

REASONS = {
    MATCH: "Face distance within the configured threshold",
    NO_MATCH: "Below face-match threshold",
    NO_FACE: "No detectable face",
    DOWNLOAD_FAILED: "Candidate image unavailable",
}


@dataclass(frozen=True)
class Match:
    """A candidate whose face was independently verified against the probe."""

    candidate: Candidate
    image_path: str
    image_bytes: bytes
    distance: float
    threshold: float
    model: str
    metric: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "post": self.candidate.as_dict(),
            "model": self.model,
            "distance_metric": self.metric,
            "distance": self.distance,
            "threshold": self.threshold,
            "verified": True,
        }


@dataclass(frozen=True)
class CandidateResult:
    """What verification concluded about one candidate.

    `distance` and `threshold` stay None for a candidate that was never
    compared - an unreachable image or one without a face has no distance, and
    inventing one would misrepresent the evidence.
    """

    candidate: Candidate
    status: str
    distance: Optional[float] = None
    threshold: Optional[float] = None
    image_bytes: Optional[bytes] = None

    @property
    def verified(self) -> bool:
        return self.status == MATCH

    @property
    def reason(self) -> str:
        return REASONS[self.status]


@dataclass
class Investigation:
    """Collector a caller can pass in to watch what verification actually did.

    It is filled in place, so a caller still holds every candidate outcome even
    when the run later fails with no verified match at all.
    """

    discovered: int = 0
    results: List[CandidateResult] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.results)

    @property
    def verified(self) -> List[CandidateResult]:
        return [result for result in self.results if result.verified]


def _record(investigation: Optional[Investigation], result: CandidateResult) -> None:
    if investigation is not None:
        investigation.results.append(result)


def _log_candidate(
    index: int, download_seconds: float, compare_started: float, candidate_started: float, note: str
) -> None:
    """Developer timing for one candidate - download, comparison, total."""
    now = time.perf_counter()
    LOG.debug(
        "candidate %02d: download %.2fs compare %.2fs total %.2fs (%s)",
        index,
        download_seconds,
        now - compare_started,
        now - candidate_started,
        note,
    )


def verify_candidates(
    probe_path: str,
    candidates: Iterable[Candidate],
    work_dir: str,
    model_name: str = adapter.DEFAULT_MODEL,
    detector_backend: str = adapter.DEFAULT_DETECTOR,
    distance_metric: str = adapter.DEFAULT_METRIC,
    max_candidates: int = MAX_CANDIDATES,
    verbose: bool = True,
    investigation: Optional[Investigation] = None,
    probe_embedding: Optional[Sequence[float]] = None,
) -> Optional[Match]:
    """Verify candidates against the probe and return the closest confirmed one.

    A candidate whose image is unavailable, or contains no detectable face, is
    skipped rather than failed - a logo thumbnail or a crowd shot is simply not
    evidence. Every outcome is appended to `investigation` when one is given.

    The probe is encoded once, here or by the caller: comparing an image path
    against every candidate would re-detect and re-encode the same probe face on
    every iteration, which costs about as much as the candidate itself.

    ponytail: candidates are downloaded and embedded one at a time. Batch the
    embedding pass if the candidate budget grows well past a few dozen.
    """
    os.makedirs(work_dir, exist_ok=True)
    matches: List[Match] = []

    started = time.perf_counter()
    if probe_embedding is None:
        probe_embedding = adapter.scan_face(probe_path, model_name, detector_backend)["embedding"]
    # None means the probe holds several faces - only the image carries them all
    probe: Union[str, Sequence[float]] = probe_path if probe_embedding is None else probe_embedding
    LOG.debug(
        "probe ready in %.2fs (pre-encoded=%s)",
        time.perf_counter() - started,
        not isinstance(probe, str),
    )

    checked = 0
    for index, candidate in enumerate(list(candidates)[:max_candidates]):
        checked += 1
        candidate_started = time.perf_counter()
        image_path = download_image(candidate.image_url, work_dir, f"candidate_{index:02d}.jpg")
        download_seconds = time.perf_counter() - candidate_started
        if image_path is None:
            LOG.debug("candidate %02d: download failed after %.2fs", index, download_seconds)
            _record(investigation, CandidateResult(candidate, DOWNLOAD_FAILED))
            continue

        with open(image_path, "rb") as handle:
            image_bytes = handle.read()  # a thumbnail; kept so the UI can show what was checked

        compare_started = time.perf_counter()
        try:
            result = adapter.compare_faces(
                probe=probe,
                candidate_path=image_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
            )
        except ValueError:  # no detectable face in the candidate image
            _log_candidate(index, download_seconds, compare_started, candidate_started, "no face")
            _record(investigation, CandidateResult(candidate, NO_FACE, image_bytes=image_bytes))
            continue

        _log_candidate(index, download_seconds, compare_started, candidate_started, "compared")

        if verbose:
            flag = "MATCH" if result["verified"] else "no"
            print(
                f"  [{index:02d}] {flag:>5}  d={result['distance']:.4f} "
                f"(thr {result['threshold']}) {candidate.page_url[:70]}"
            )

        _record(
            investigation,
            CandidateResult(
                candidate=candidate,
                status=MATCH if result["verified"] else NO_MATCH,
                distance=result["distance"],
                threshold=result["threshold"],
                image_bytes=image_bytes,
            ),
        )

        if not result["verified"]:
            continue

        matches.append(
            Match(
                candidate=candidate,
                image_path=image_path,
                image_bytes=image_bytes,
                distance=result["distance"],
                threshold=result["threshold"],
                model=model_name,
                metric=distance_metric,
            )
        )

    LOG.info(
        "verified %d of %d candidates in %.1fs",
        len(matches),
        checked,
        time.perf_counter() - started,
    )
    if not matches:
        return None
    return min(matches, key=lambda match: match.distance)
