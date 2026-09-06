"""Latency benchmark for the discovery -> verification -> evidence path.

A stored evidence file already carries the full reverse-search response, so the
candidate set can be replayed exactly. That makes the benchmark reproducible
without spending a search credit and, more importantly, makes two runs
comparable: the only variable left is the code under test.

    uv run python bench/benchmark.py --label baseline
    uv run python bench/benchmark.py --label optimized

    # deterministic bytes for the correctness A/B - no network in the loop
    uv run python bench/benchmark.py --label baseline --mirror bench/mirror

Results are written to bench/results/<label>.json; compare.py diffs two of them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceproof.discovery import retrieval
from faceproof.discovery.candidates import Candidate, parse_visual_matches
from faceproof.evidence import hashing, manifest
from faceproof.face import adapter
from faceproof.matching import verifier as face_verifier

DEFAULT_COUNTS = (5, 10, 25)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def newest_corpus() -> Tuple[str, str]:
    """Pick the most recent evidence file that still has its matched image."""
    for path in sorted(glob.glob(os.path.join("evidence", "*.json")), reverse=True):
        image = path[: -len(".json")] + ".match.jpg"
        if os.path.isfile(image):
            return path, image
    raise SystemExit("no evidence/<id>.json with a matching .match.jpg to benchmark against")


def load_candidates(evidence_path: str) -> List[Candidate]:
    with open(evidence_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    response = document.get("search_response")
    if not response:
        raise SystemExit(f"{evidence_path} carries no search_response to replay")
    return parse_visual_matches(response)


def _mirror_name(image_url: str) -> str:
    return hashing.sha256_bytes(image_url.encode("utf-8"))[:16] + ".jpg"


# --------------------------------------------------------------- instrumentation


class Stopwatch:
    """Records how long every download and every comparison took.

    Downloads may run on worker threads once the pipeline is parallel, so the
    wall time of the stage and the sum of its parts are both kept: their ratio
    is the concurrency actually achieved.
    """

    def __init__(self) -> None:
        self.downloads: List[float] = []
        self.download_bytes: List[int] = []
        self.compares: List[float] = []

    def install(self, mirror: Optional[str]) -> None:
        real_download = retrieval.download_image
        real_compare = adapter.compare_faces

        def timed_download(url: str, dest_dir: str, name: str) -> Optional[str]:
            started = time.perf_counter()
            path = (
                _mirror_copy(mirror, url, dest_dir, name)
                if mirror
                else real_download(url, dest_dir, name)
            )
            self.downloads.append(time.perf_counter() - started)  # append is atomic
            self.download_bytes.append(os.path.getsize(path) if path else 0)
            return path

        def timed_compare(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            started = time.perf_counter()
            try:
                return real_compare(*args, **kwargs)
            finally:
                self.compares.append(time.perf_counter() - started)

        _patch_everywhere("download_image", timed_download)
        _patch_everywhere("compare_faces", timed_compare)

    def totals(self) -> Dict[str, Any]:
        compares = self.compares or [0.0]
        return {
            "downloads": len(self.downloads),
            "download_seconds_sum": round(sum(self.downloads), 3),
            "download_seconds_max": round(max(self.downloads, default=0.0), 3),
            "download_bytes": sum(self.download_bytes),
            "compares": len(self.compares),
            "compare_seconds_sum": round(sum(self.compares), 3),
            "compare_seconds_mean": round(sum(compares) / len(compares), 3),
        }


def _patch_everywhere(name: str, replacement: Callable[..., Any]) -> None:
    """Rebind a function on every module that imported it by name."""
    for module in (retrieval, adapter, face_verifier):
        if hasattr(module, name):
            setattr(module, name, replacement)


def _mirror_copy(mirror: str, url: str, dest_dir: str, name: str) -> Optional[str]:
    """Serve a candidate image from the local mirror instead of the network."""
    source = os.path.join(mirror, _mirror_name(url))
    if not os.path.isfile(source):
        return None  # mirrored as unreachable, exactly as the real download was
    path = os.path.join(dest_dir, name)
    shutil.copyfile(source, path)
    return path


def build_mirror(candidates: Sequence[Candidate], mirror: str) -> Dict[str, int]:
    """Download every candidate once so later runs compare identical bytes."""
    os.makedirs(mirror, exist_ok=True)
    fetched = failed = cached = 0
    for candidate in candidates:
        target = os.path.join(mirror, _mirror_name(candidate.image_url))
        if os.path.isfile(target):
            cached += 1
            continue
        with tempfile.TemporaryDirectory() as scratch:
            path = retrieval.download_image(candidate.image_url, scratch, "image.jpg")
            if path is None:
                failed += 1
                continue
            shutil.copyfile(path, target)
            fetched += 1
    return {"fetched": fetched, "cached": cached, "unreachable": failed}


# ------------------------------------------------------------------- the run


def measure(
    probe_path: str,
    candidates: Sequence[Candidate],
    count: int,
    mirror: Optional[str],
) -> Dict[str, Any]:
    """One investigation at `count` candidates, stage by stage."""
    watch = Stopwatch()
    watch.install(mirror)

    started = time.perf_counter()
    probe_scan = adapter.scan_face(probe_path)
    probe_digest = hashing.sha256_file(probe_path)
    scan_seconds = time.perf_counter() - started

    investigation = face_verifier.Investigation(discovered=len(candidates))
    work_dir = tempfile.mkdtemp(prefix="bench-")
    verify_started = time.perf_counter()
    evidence_seconds = 0.0
    try:
        match = face_verifier.verify_candidates(
            probe_path=probe_path,
            candidates=candidates,
            work_dir=work_dir,
            max_candidates=count,
            verbose=False,
            investigation=investigation,
            probe_embedding=probe_scan["embedding"],
        )
        verify_seconds = time.perf_counter() - verify_started

        evidence_started = time.perf_counter()
        if match is not None:
            evidence = manifest.build_evidence(
                probe_digest=probe_digest,
                probe_scan=probe_scan,
                match=match,
                search_engine="serpapi/google_lens",
                probe_image_url="https://example.invalid/probe.jpg",
                search_response={},
                candidates_returned=len(candidates),
                candidates_checked=min(count, len(candidates)),
            )
            hashing.fingerprint(evidence)
            document = manifest.build_document(evidence, {"tx_hash": "0x" + "0" * 64}, "m.jpg", {})
            json.dumps(document, indent=2, sort_keys=True)
            evidence_seconds = time.perf_counter() - evidence_started
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    statuses = [result.status for result in investigation.results]
    return {
        "candidates": count,
        "total_seconds": round(time.perf_counter() - started, 3),
        "probe_scan_seconds": round(scan_seconds, 3),
        "verification_seconds": round(verify_seconds, 3),
        "evidence_seconds": round(evidence_seconds, 3),
        "stages": watch.totals(),
        # discovered counts every parsed result, including any later merged as a
        # duplicate; getattr keeps one harness runnable against both revisions
        "discovered": sum(1 + len(getattr(one, "duplicates", ())) for one in candidates),
        "unique": len(candidates),
        "investigated": len(investigation.results),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "best_distance": round(match.distance, 6) if match else None,
        "best_page_url": match.candidate.page_url if match else None,
        "best_source": match.candidate.source if match else None,
        "statuses": statuses,
        "results": [
            {
                "page_url": result.candidate.page_url,
                "image_url": result.candidate.image_url,
                "status": result.status,
                "distance": None if result.distance is None else round(result.distance, 6),
            }
            for result in investigation.results
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="name of this run, e.g. baseline")
    parser.add_argument("--evidence", help="evidence/<id>.json to replay (default: newest)")
    parser.add_argument("--probe", help="probe image (default: the matched image next to it)")
    parser.add_argument("--counts", default=",".join(str(n) for n in DEFAULT_COUNTS))
    parser.add_argument("--mirror", help="serve candidate images from this local mirror")
    parser.add_argument("--build-mirror", action="store_true", help="populate the mirror and exit")
    args = parser.parse_args(argv)

    corpus_path, corpus_probe = newest_corpus()
    evidence_path = args.evidence or corpus_path
    probe_path = args.probe or (corpus_probe if evidence_path == corpus_path else None)
    if probe_path is None:
        raise SystemExit("--evidence given without --probe")
    parse_started = time.perf_counter()
    candidates = load_candidates(evidence_path)  # parse, normalize, deduplicate, rank
    preprocess_seconds = time.perf_counter() - parse_started
    counts = [int(value) for value in args.counts.split(",") if value.strip()]

    if args.build_mirror:
        if not args.mirror:
            raise SystemExit("--build-mirror needs --mirror DIR")
        print(f"mirror {args.mirror}: {build_mirror(candidates[: max(counts)], args.mirror)}")
        return 0

    print(f"corpus     {evidence_path} ({len(candidates)} candidates)")
    print(f"probe      {probe_path}")
    print(f"mode       {'mirror ' + args.mirror if args.mirror else 'live network'}")
    print(f"workers    downloads={os.environ.get('FACEPROOF_DOWNLOAD_WORKERS', 'unset')}")

    cold_started = time.perf_counter()
    adapter.scan_face(probe_path)
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    adapter.scan_face(probe_path)
    warm_seconds = time.perf_counter() - warm_started
    print(f"model init cold={cold_seconds:.2f}s warm={warm_seconds:.2f}s")

    runs = []
    for count in counts:
        run = measure(probe_path, candidates, count, args.mirror)
        runs.append(run)
        stages = run["stages"]
        print(
            f"n={count:<3} total={run['total_seconds']:>7.2f}s  "
            f"verify={run['verification_seconds']:>7.2f}s  "
            f"downloads={stages['download_seconds_sum']:>6.2f}s "
            f"compares={stages['compare_seconds_sum']:>7.2f}s  "
            f"evidence={run['evidence_seconds']:.2f}s"
        )

    payload = {
        "label": args.label,
        "corpus": evidence_path,
        "probe": probe_path,
        "mirror": args.mirror,
        "download_workers": os.environ.get("FACEPROOF_DOWNLOAD_WORKERS"),
        "preprocess_seconds": round(preprocess_seconds, 6),
        "model_init": {
            "cold_seconds": round(cold_seconds, 3),
            "warm_seconds": round(warm_seconds, 3),
        },
        "runs": runs,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{args.label}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"written    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
