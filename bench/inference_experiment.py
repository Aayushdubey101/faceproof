"""Does candidate inference get faster with threads, or with fewer TF threads?

Two experiments, one harness, because both move the same number: the wall time
of N candidate comparisons.

    # inference concurrency (section 8)
    uv run python bench/inference_experiment.py --label w1 --workers 1
    uv run python bench/inference_experiment.py --label w2 --workers 2
    uv run python bench/inference_experiment.py --label w4 --workers 4

    # CPU threading (section 9) - TF reads these at import, so they have to be
    # set before the process starts; the harness records whatever it was given
    TF_NUM_INTRAOP_THREADS=4 uv run python bench/inference_experiment.py --label intra4

Every run records its own per-candidate verdict, so a configuration that is
faster but disagrees with the sequential baseline shows up as a correctness
failure rather than as a speedup:

    uv run python bench/inference_experiment.py --compare w1 w2 w4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark
from profile_verify import cpu_seconds, rss_mb
from faceproof.face import adapter

RESULTS_DIR = benchmark.RESULTS_DIR
THREAD_VARS = (
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "OMP_NUM_THREADS",
    "TF_ENABLE_ONEDNN_OPTS",
)


# how many candidate faces one comparison scored the probe against; thread-local
# because a worker must not count another worker's faces
_counter = threading.local()


def install_face_counter() -> None:
    """Count the probe-to-candidate-face comparisons the engine actually makes.

    That count *is* the face-selection behaviour: verify() scores the probe
    against every face it detected and keeps the closest, so if a change altered
    how many faces are found or which one wins, this number moves.
    """
    from face_engine.modules import verification

    real = verification.find_distance

    def counted(*args: Any, **kwargs: Any) -> Any:
        _counter.faces = getattr(_counter, "faces", 0) + 1
        return real(*args, **kwargs)

    verification.find_distance = counted


def compare_one(probe: Any, path: str) -> Tuple[str, Optional[float], int]:
    """One candidate, exactly as the verifier does it."""
    _counter.faces = 0
    try:
        result = adapter.compare_faces(probe=probe, candidate_path=path)
    except ValueError:
        return "NO_FACE", None, _counter.faces
    return (
        ("MATCH" if result["verified"] else "NO_MATCH"),
        round(result["distance"], 6),
        _counter.faces,
    )


def run_pass(
    probe: Any, paths: List[str], workers: int, peak: List[float], cpu_used: List[float]
) -> Tuple[float, List[Tuple[str, Optional[float], int]]]:
    """Compare every candidate once, with `workers` comparisons in flight."""
    done = threading.Event()

    def sample() -> None:
        while not done.wait(0.25):
            peak.append(rss_mb())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    cpu_before = cpu_seconds()
    started = time.perf_counter()
    try:
        if workers == 1:
            results = [compare_one(probe, path) for path in paths]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fp-compare") as pool:
                # map keeps submission order, so a result still belongs to its candidate
                results = list(pool.map(lambda path: compare_one(probe, path), paths))
        seconds = time.perf_counter() - started
        cpu_used.append(cpu_seconds() - cpu_before)
    finally:
        done.set()
        sampler.join(timeout=1.0)
    peak.append(rss_mb())
    return seconds, results


def measure(args: argparse.Namespace) -> Dict[str, Any]:
    corpus, probe_path = benchmark.newest_corpus()
    candidates = benchmark.load_candidates(corpus)[: args.candidates]
    benchmark.Stopwatch().install(args.mirror)

    scan = adapter.scan_face(probe_path)  # also the first DeepFace import
    install_face_counter()
    probe = scan["embedding"]
    peak: List[float] = [rss_mb()]
    cpu_used: List[float] = []
    # what TF actually applied, not what the environment asked for: an env var
    # TF ignored must not be reported as a configuration that was tested
    import tensorflow as tf

    applied = {
        "intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
    }

    work_dir = tempfile.mkdtemp(prefix="inference-")
    try:
        paths: List[str] = []
        urls: List[str] = []
        for index, candidate in enumerate(candidates):
            path = benchmark.retrieval.download_image(
                candidate.image_url, work_dir, f"c{index:02d}.jpg"
            )
            if path is not None:
                paths.append(path)
                urls.append(candidate.page_url)

        runs: List[float] = []
        passes: List[List[Tuple[str, Optional[float], int]]] = []
        for _ in range(args.repeats):
            seconds, results = run_pass(probe, paths, args.workers, peak, cpu_used)
            runs.append(round(seconds, 3))
            passes.append(results)
        # every pass of one configuration must agree with every other, or the
        # configuration is nondeterministic and cannot be shipped at any speed
        stable = all(one == passes[0] for one in passes)
        results = passes[0]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "label": args.label,
        "corpus": corpus,
        "workers": args.workers,
        "candidates": len(paths),
        "repeats": args.repeats,
        "cpu_count": os.cpu_count(),
        "threads": {name: os.environ.get(name) for name in THREAD_VARS},
        "threads_applied": applied,
        "runs": runs,
        "median": round(statistics.median(runs), 3),
        "best": min(runs),
        "worst": max(runs),
        "peak_memory_mb": round(max(peak), 1),
        "cpu_seconds": [round(one, 1) for one in cpu_used],
        # cores kept busy on average: cpu time / wall time
        "cpu_cores_busy": (
            round(statistics.median(cpu_used) / statistics.median(runs), 2) if cpu_used else 0.0
        ),
        "results_stable": stable,
        "results": [
            {"page_url": url, "status": status, "distance": distance, "faces": faces}
            for url, (status, distance, faces) in zip(urls, results)
        ],
    }


def load(label: str) -> Dict[str, Any]:
    path = os.path.join(RESULTS_DIR, f"inference-{label}.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare(labels: List[str]) -> int:
    """Rank configurations by median time, and refuse to rank a wrong one."""
    runs = [load(label) for label in labels]
    reference = runs[0]
    print(
        f"reference  {reference['label']} ({reference['candidates']} candidates, "
        f"{reference['repeats']} repeats, {reference['cpu_count']} CPUs)"
    )
    print()
    print(
        f"{'label':<10} {'workers':>7} {'median':>8} {'best':>8} {'worst':>8} "
        f"{'peak MiB':>9} {'cores':>6}  {'vs ref':>7}  correctness"
    )
    failures = 0
    for one in runs:
        gain = (reference["median"] - one["median"]) / reference["median"] * 100
        differences = [
            f"{a['page_url']}: {a['status']}/{a['distance']}/{a.get('faces')} faces "
            f"-> {b['status']}/{b['distance']}/{b.get('faces')} faces"
            for a, b in zip(reference["results"], one["results"])
            if a["status"] != b["status"]
            or a["distance"] != b["distance"]
            or a.get("faces") != b.get("faces")
        ]
        failures += len(differences)
        verdict = "identical" if not differences else f"{len(differences)} DIFFERENCES"
        if not one.get("results_stable", True):
            verdict += " + UNSTABLE ACROSS REPEATS"
            failures += 1
        print(
            f"{one['label']:<10} {one['workers']:>7} {one['median']:>7.2f}s {one['best']:>7.2f}s "
            f"{one['worst']:>7.2f}s {one['peak_memory_mb']:>9.0f} "
            f"{one.get('cpu_cores_busy', 0):>6.2f}  {gain:>6.1f}%  {verdict}"
        )
        for line in differences[:5]:
            print(f"    ! {line}")
    print()
    print("correctness:", "no differences" if not failures else f"{failures} DIFFERENCES")
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="name of this configuration, e.g. w2")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mirror", default="bench/mirror")
    parser.add_argument("--compare", nargs="+", help="rank previously measured labels")
    args = parser.parse_args(argv)

    if args.compare:
        return compare(args.compare)
    if not args.label:
        raise SystemExit("--label is required unless --compare is given")

    payload = measure(args)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"inference-{args.label}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(
        f"{payload['label']}: workers={payload['workers']} runs={payload['runs']} "
        f"median={payload['median']}s peak={payload['peak_memory_mb']:.0f} MiB "
        f"cores_busy={payload['cpu_cores_busy']} of {payload['cpu_count']} "
        f"stable={payload['results_stable']} "
        f"applied={payload['threads_applied']} env={payload['threads']}"
    )
    print(f"written    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
