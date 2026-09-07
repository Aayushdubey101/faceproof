"""Where does one candidate comparison actually spend its time?

Download prefetching removed the download from the critical path and candidate
ranking decided which candidates are worth comparing, which leaves face
comparison as ~99% of a run.
This script takes that number apart, per candidate and per stage, before
anything is optimized.

    uv run python bench/profile_verify.py --mirror bench/mirror --repeats 3

Instrumentation is applied by wrapping functions **in this process only** -
`face_engine/` is never edited. The seams are the natural boundaries of one
comparison:

    load_image                 decode the candidate JPEG
    RetinaFaceClient.detect    RetinaFace inference alone
    detect_faces               that, plus the 50% black border and per-face crop
    extract_face               alignment: rotate about the eyes, crop, resize
    represent                  ArcFace embedding
    find_distance              cosine distance

Results are written to bench/results/profile.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark
from faceproof.face import adapter

RESULTS_DIR = benchmark.RESULTS_DIR
# "cosine", not "distance": a row also carries the candidate's own distance
STAGES = ("decode", "retinaface", "detect", "align", "embed", "cosine", "total")


def rss_mb() -> float:
    """Resident set size of this process, in MiB. 0.0 where unavailable."""
    try:  # Windows: psutil is not a project dependency, and one counter is not
        import ctypes  # worth adding one
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        # argtypes matter: without them the 64-bit handle is truncated and the
        # call quietly fails
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return 0.0
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:  # any platform without that API: report nothing, invent nothing
        return 0.0


def cpu_seconds() -> float:
    """CPU time this process has used, user + system, across all its threads.

    Divided by wall time it gives the number of cores kept busy, which is what
    says whether another worker thread has an idle core to run on.
    """
    return time.process_time()


class Profile:
    """Per-candidate stage timings, collected by wrapping the engine's seams."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.current: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self.shape: Dict[str, int] = {}
        self.model_builds: List[str] = []
        self.model_ids: Dict[str, set] = {}
        self.peak_mb = 0.0
        self.started = 0.0

    def _timed(self, stage: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return real(*args, **kwargs)
            finally:
                self.current[stage] = self.current.get(stage, 0.0) + (time.perf_counter() - started)
                self.calls[stage] = self.calls.get(stage, 0) + 1

        return wrapper

    def install(self) -> None:
        from face_engine.commons import image_utils
        from face_engine.models.face_detection import RetinaFace as retina
        from face_engine.modules import detection, modeling, representation, verification

        real_load = image_utils.load_image
        real_build = modeling.build_model

        def load_image(img_path: Any) -> Any:
            started = time.perf_counter()
            img, name = real_load(img_path)
            self.current["decode"] = self.current.get("decode", 0.0) + (
                time.perf_counter() - started
            )
            # the first load of a comparison is the candidate itself; later ones
            # are the cropped face being re-loaded by represent(detector=skip)
            if img is not None and getattr(img, "ndim", 0) == 3 and not self.shape:
                self.shape = {"image_height": int(img.shape[0]), "image_width": int(img.shape[1])}
            return img, name

        def build_model(task: str, model_name: str) -> Any:
            model = real_build(task=task, model_name=model_name)
            # identity, not call count: build_model is *called* per candidate but
            # must hand back the same object every time
            self.model_ids.setdefault(f"{task}/{model_name}", set()).add(id(model))
            self.model_builds.append(f"{task}/{model_name}")
            return model

        image_utils.load_image = load_image
        detection.image_utils.load_image = load_image
        modeling.build_model = build_model
        detection.modeling.build_model = build_model
        detection.detect_faces = self._timed("detect", detection.detect_faces)
        detection.extract_face = self._timed("align", detection.extract_face)
        representation.represent = self._timed("embed", representation.represent)
        verification.find_distance = self._timed("cosine", verification.find_distance)
        retina.RetinaFaceClient.detect_faces = self._timed(
            "retinaface", retina.RetinaFaceClient.detect_faces
        )

    def start_candidate(self) -> None:
        self.current = {}
        self.shape = {}
        self.calls = {}
        self.started = time.perf_counter()

    def finish_candidate(self, page_url: str, status: str, distance: Optional[float]) -> None:
        # the wall time of the whole comparison, so the stages can be checked
        # against it: if they sum to more, the instrumentation is double-counting
        self.current["total"] = time.perf_counter() - self.started
        row: Dict[str, Any] = {stage: round(self.current.get(stage, 0.0), 4) for stage in STAGES}
        row.update(self.shape)
        row.update({"page_url": page_url, "status": status, "distance": distance})
        row["calls"] = dict(self.calls)
        self.rows.append(row)
        self.peak_mb = max(self.peak_mb, rss_mb())

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for stage in STAGES:
            values = sorted(row[stage] for row in self.rows)
            if not values:
                continue
            index = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
            summary[stage] = {
                "mean": round(statistics.fmean(values), 4),
                "median": round(statistics.median(values), 4),
                "p95": round(values[index], 4),
                "total": round(sum(values), 3),
            }
        return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", help="serve candidate images from this local mirror")
    parser.add_argument("--candidates", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    corpus, probe = benchmark.newest_corpus()
    candidates = benchmark.load_candidates(corpus)[: args.candidates]
    start_mb = rss_mb()

    profile = Profile()
    benchmark.Stopwatch().install(args.mirror)
    profile.install()

    scan = adapter.scan_face(probe)  # cold: builds both models
    after_models_mb = rss_mb()
    probe_embedding = scan["embedding"]

    # candidate by candidate, so every row belongs to exactly one URL
    work_dir = tempfile.mkdtemp(prefix="profile-")
    try:
        for _ in range(args.repeats):
            for index, candidate in enumerate(candidates):
                path = benchmark.retrieval.download_image(
                    candidate.image_url, work_dir, f"c{index:02d}.jpg"
                )
                if path is None:
                    continue
                profile.start_candidate()
                try:
                    result = adapter.compare_faces(probe=probe_embedding, candidate_path=path)
                except ValueError:
                    profile.finish_candidate(candidate.page_url, "NO_FACE", None)
                    continue
                profile.finish_candidate(
                    candidate.page_url,
                    "MATCH" if result["verified"] else "NO_MATCH",
                    round(result["distance"], 6),
                )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    summary = profile.aggregate()
    payload = {
        "corpus": corpus,
        "probe": probe,
        "mirror": args.mirror,
        "candidates": len(candidates),
        "repeats": args.repeats,
        "model_builds": profile.model_builds,
        "memory_mb": {
            "start": round(start_mb, 1),
            "after_models": round(after_models_mb, 1),
            "peak": round(profile.peak_mb, 1),
        },
        "rows": profile.rows,
        "aggregate": summary,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "profile.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"corpus      {corpus}")
    print(f"candidates  {len(candidates)} x {args.repeats} repeats = {len(profile.rows)} rows")
    print(
        f"models      {len(profile.model_builds)} build_model calls, "
        f"distinct instances: "
        + ", ".join(f"{name} x{len(ids)}" for name, ids in sorted(profile.model_ids.items()))
    )
    print(
        f"memory      start {start_mb:.0f} MiB, after models {after_models_mb:.0f} MiB, "
        f"peak {profile.peak_mb:.0f} MiB"
    )
    print()
    print(f"{'stage':<12} {'mean':>8} {'median':>8} {'p95':>8} {'total':>9}   share")
    whole = summary.get("total", {}).get("total", 0.0)
    for stage in STAGES:
        if stage not in summary:
            continue
        one = summary[stage]
        share = one["total"] / whole * 100 if whole else 0.0
        note = " (inside detect)" if stage in ("decode", "retinaface", "align") else ""
        note = " (wall clock)" if stage == "total" else note
        print(
            f"{stage:<12} {one['mean']:>8.4f} {one['median']:>8.4f} {one['p95']:>8.4f} "
            f"{one['total']:>8.2f}s  {share:>5.1f}%{note}"
        )
    sizes = sorted(row.get("image_width", 0) * row.get("image_height", 0) for row in profile.rows)
    if sizes:
        print()
        print(
            f"candidate pixels: median {statistics.median(sizes):,.0f}  "
            f"min {sizes[0]:,}  max {sizes[-1]:,}"
        )
    print(f"\nwritten     {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
