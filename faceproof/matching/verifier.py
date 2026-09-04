"""Candidate face verification.

Discovery produces visually related candidates. This module decides, using the
same face-recognition foundation that encoded the probe, whether a candidate
actually shows the same face.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from faceproof.face import adapter
from faceproof.discovery.candidates import Candidate
from faceproof.discovery.retrieval import download_image

MAX_CANDIDATES = 25


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


def verify_candidates(
    probe_path: str,
    candidates: Iterable[Candidate],
    work_dir: str,
    model_name: str = adapter.DEFAULT_MODEL,
    detector_backend: str = adapter.DEFAULT_DETECTOR,
    distance_metric: str = adapter.DEFAULT_METRIC,
    max_candidates: int = MAX_CANDIDATES,
    verbose: bool = True,
) -> Optional[Match]:
    """Verify candidates against the probe and return the closest confirmed one.

    A candidate whose image is unavailable, or contains no detectable face, is
    skipped rather than failed - a logo thumbnail or a crowd shot is simply not
    evidence.

    ponytail: candidates are downloaded and embedded one at a time. Batch the
    embedding pass if the candidate budget grows well past a few dozen.
    """
    os.makedirs(work_dir, exist_ok=True)
    matches: List[Match] = []

    for index, candidate in enumerate(list(candidates)[:max_candidates]):
        image_path = download_image(candidate.image_url, work_dir, f"candidate_{index:02d}.jpg")
        if image_path is None:
            continue

        try:
            result = adapter.compare_faces(
                probe_path=probe_path,
                candidate_path=image_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
            )
        except ValueError:
            continue  # no detectable face in the candidate image

        if verbose:
            flag = "MATCH" if result["verified"] else "no"
            print(
                f"  [{index:02d}] {flag:>5}  d={result['distance']:.4f} "
                f"(thr {result['threshold']}) {candidate.page_url[:70]}"
            )

        if not result["verified"]:
            continue

        with open(image_path, "rb") as handle:
            matches.append(
                Match(
                    candidate=candidate,
                    image_path=image_path,
                    image_bytes=handle.read(),
                    distance=result["distance"],
                    threshold=result["threshold"],
                    model=model_name,
                    metric=distance_metric,
                )
            )

    if not matches:
        return None
    return min(matches, key=lambda match: match.distance)
