"""Candidate face verification.

Discovery produces visually related candidates. This module decides, using the
same face-recognition foundation that encoded the probe, whether a candidate
actually shows the same face.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from faceproof.face import adapter
from faceproof.discovery.candidates import Candidate
from faceproof.discovery.retrieval import download_image

MAX_CANDIDATES = 25
DEMO_CANDIDATES = 10  # what one investigation costs interactively; the cap above is the ceiling

DOWNLOAD_WORKERS_ENV = "FACEPROOF_DOWNLOAD_WORKERS"
DEFAULT_DOWNLOAD_WORKERS = 4
MAX_DOWNLOAD_WORKERS = 8  # a candidate thumbnail is a few KB; more sockets buy nothing

COMPARE_WORKERS_ENV = "FACEPROOF_COMPARE_WORKERS"
# 2, not more: RetinaFace already spreads one image across every core, so a
# second comparison fills the gaps while a third mostly buys memory. Measured
# over 25 candidates: 26.8s / 23.3s / 22.5s / 22.1s at 1 / 2 / 3 / 4 workers,
# for 1.9 / 2.5 / 2.8 / 3.0 GiB peak. See bench/FACE_VERIFICATION_PERFORMANCE.md.
DEFAULT_COMPARE_WORKERS = 2
MAX_COMPARE_WORKERS = 4

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

    `discovered` counts everything the search returned, `unique` what survived
    deduplication, and `results` only what was actually compared. Candidates
    the budget did not reach are kept in `not_investigated` and given no
    status: not investigated is not the same as rejected, and reporting one as
    the other would be a claim the pipeline never tested.
    """

    discovered: int = 0
    unique: int = 0
    results: List[CandidateResult] = field(default_factory=list)
    not_investigated: List[Candidate] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.results)

    @property
    def verified(self) -> List[CandidateResult]:
        return [result for result in self.results if result.verified]


def _record(investigation: Optional[Investigation], result: CandidateResult) -> None:
    if investigation is not None:
        investigation.results.append(result)


def _log_candidate(index: int, download_seconds: float, compare_seconds: float, note: str) -> None:
    """Developer timing for one candidate - download and comparison."""
    LOG.debug(
        "candidate %02d: download %.2fs compare %.2fs (%s)",
        index,
        download_seconds,
        compare_seconds,
        note,
    )


def _workers(name: str, default: int, maximum: int) -> int:
    """A worker count from the environment, clamped to something sane."""
    raw = os.environ.get(name, "")
    try:
        workers = int(raw) if raw else default
    except ValueError:
        LOG.warning("%s=%r is not a number", name, raw)
        workers = default
    return max(1, min(workers, maximum))


def download_workers() -> int:
    """How many candidate downloads may be in flight at once."""
    return _workers(DOWNLOAD_WORKERS_ENV, DEFAULT_DOWNLOAD_WORKERS, MAX_DOWNLOAD_WORKERS)


def compare_workers() -> int:
    """How many candidate face comparisons may run at once.

    Set to 1 to make verification strictly sequential; the verdicts are the
    same either way, it only costs time.
    """
    return _workers(COMPARE_WORKERS_ENV, DEFAULT_COMPARE_WORKERS, MAX_COMPARE_WORKERS)


@dataclass(frozen=True)
class _Fetched:
    """One downloaded candidate image. `path` is None when it never arrived."""

    path: Optional[str]
    image_bytes: bytes
    seconds: float


def _prefetch(candidates: Sequence[Candidate], work_dir: str) -> Iterator[_Fetched]:
    """Download candidate images ahead of the comparison loop, in candidate order.

    Downloads are latency-bound - a thumbnail is a few kilobytes, the round trip
    is the cost - while comparison is CPU-bound, so fetching the next images
    while the current one is being compared takes the network almost entirely
    off the critical path.

    `ThreadPoolExecutor.map` keeps both properties the rest of the pipeline
    depends on: at most `download_workers()` sockets are open at once, and
    results come back in submission order, so candidate identity, ranking and
    failure status are exactly what the sequential loop produced.
    """
    # Two candidates count as duplicates only when their image URL matches
    # exactly; the same thumbnail is then fetched once and reused, which never
    # crosses an investigation boundary because the pool is built per call.
    origins: Dict[str, int] = {}
    plan = [origins.setdefault(c.image_url, index) for index, c in enumerate(candidates)]

    def fetch(index: int) -> _Fetched:
        started = time.perf_counter()
        payload = b""
        try:
            path = download_image(
                candidates[index].image_url, work_dir, f"candidate_{index:02d}.jpg"
            )
            if path is not None:
                with open(path, "rb") as handle:
                    payload = handle.read()  # a thumbnail; the UI shows what was checked
        except OSError:  # a worker must fail like an unreachable image, never crash the run
            LOG.debug("candidate %02d: could not be stored", index, exc_info=True)
            path = None
        return _Fetched(path, payload, time.perf_counter() - started)

    firsts = [index for index, origin in enumerate(plan) if origin == index]
    fetched: Dict[int, _Fetched] = {}
    with ThreadPoolExecutor(
        max_workers=download_workers(), thread_name_prefix="fp-download"
    ) as pool:
        stream = zip(firsts, pool.map(fetch, firsts))
        for index, origin in enumerate(plan):
            if origin == index:
                key, result = next(stream)
                fetched[key] = result
            yield fetched[origin]


@dataclass(frozen=True)
class _Compared:
    """What the model concluded about one fetched candidate.

    `result` is None both when the image never arrived and when it held no
    detectable face; `fetched.path` tells those two apart, and the caller turns
    them into DOWNLOAD_FAILED or NO_FACE.
    """

    fetched: _Fetched
    result: Optional[Dict[str, Any]]
    seconds: float


def _compared(
    stream: Iterator[_Fetched], compare: Callable[[_Fetched], _Compared]
) -> Iterator[_Compared]:
    """Run `compare` over `stream`, up to `compare_workers()` at a time, in order.

    RetinaFace already spreads one image across every core, so a second
    comparison only fills the gaps it leaves: measured over 25 candidates,
    1 -> 2 workers is 13% faster and 2 -> 4 buys a further 5% for another
    0.5 GiB. Ordering is `pool.map`'s, which is submission order, so the
    verdicts and the ranking are what one thread would have produced - the
    A/B in bench/FACE_VERIFICATION_PERFORMANCE.md shows no difference.

    One worker skips the pool entirely rather than paying for a thread that
    can never overlap with anything.
    """
    workers = compare_workers()
    if workers == 1:
        yield from map(compare, stream)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fp-compare") as pool:
        yield from pool.map(compare, stream)


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

    Candidates are taken in the order given, which the discovery layer has
    already deduplicated and ranked by investigation priority; the first
    `max_candidates` of them are compared and the rest are recorded as
    uninvestigated. That ordering is a retrieval heuristic and decides only
    what gets looked at - identity is decided here and only here, by the
    distance against the threshold.

    A candidate whose image is unavailable, or contains no detectable face, is
    skipped rather than failed - a logo thumbnail or a crowd shot is simply not
    evidence. Every outcome is appended to `investigation` when one is given.

    The probe is encoded once, here or by the caller: comparing an image path
    against every candidate would re-detect and re-encode the same probe face on
    every iteration, which costs about as much as the candidate itself.

    Candidate images are prefetched concurrently (see `_prefetch`) and compared
    on a small bounded pool (see `_compared`). Both preserve order, so what this
    function returns does not depend on how many workers it used.

    ponytail: candidates are embedded one at a time. Batch the embedding pass if
    the candidate budget grows well past a few dozen.
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

    def compare(fetched: _Fetched) -> _Compared:
        """The model call for one candidate, and nothing else.

        This is the only part that runs off the main thread, so it touches no
        shared state: every verdict is still recorded, printed and ranked in
        candidate order by the loop below.
        """
        if fetched.path is None:
            return _Compared(fetched, None, 0.0)
        started_at = time.perf_counter()
        try:
            result = adapter.compare_faces(
                probe=probe,
                candidate_path=fetched.path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
            )
        except ValueError:  # no detectable face in the candidate image
            result = None
        return _Compared(fetched, result, time.perf_counter() - started_at)

    checked = 0
    # candidates arrive already deduplicated and ranked, so the budget is spent
    # on the most promising ones; the rest are recorded as uninvestigated
    ordered = list(candidates)
    budget, beyond = ordered[:max_candidates], ordered[max_candidates:]
    if investigation is not None:
        investigation.not_investigated.extend(beyond)

    stream = _compared(_prefetch(budget, work_dir), compare)
    for index, (candidate, compared) in enumerate(zip(budget, stream)):
        checked += 1
        fetched, result = compared.fetched, compared.result
        if fetched.path is None:
            LOG.debug("candidate %02d: download failed after %.2fs", index, fetched.seconds)
            _record(investigation, CandidateResult(candidate, DOWNLOAD_FAILED))
            continue

        image_bytes = fetched.image_bytes
        if result is None:
            _log_candidate(index, fetched.seconds, compared.seconds, "no face")
            _record(investigation, CandidateResult(candidate, NO_FACE, image_bytes=image_bytes))
            continue

        _log_candidate(index, fetched.seconds, compared.seconds, "compared")

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
                image_path=fetched.path,
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
