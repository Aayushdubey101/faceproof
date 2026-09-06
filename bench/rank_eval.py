"""Does candidate ranking put the useful evidence inside the budget?

The experiment needs a corpus where the answer for *every* candidate is known,
so the first step verifies the whole discovered set once and caches the
verdicts. Nothing after that runs a model to produce a number: an ordering is
scored by looking up verdicts that were measured, never by predicting them.

    uv run python bench/rank_eval.py --mirror bench/mirror            # score orderings
    uv run python bench/rank_eval.py --mirror bench/mirror --refresh  # recompute verdicts
    uv run python bench/rank_eval.py --mirror bench/mirror --verify   # A/B the orderings

`--mirror` serves candidate images from disk (see benchmark.py --build-mirror),
which removes the network as a source of difference between two orderings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark
from faceproof.discovery.candidates import Candidate
from faceproof.face import adapter
from faceproof.matching import verifier as face_verifier

RESULTS_DIR = benchmark.RESULTS_DIR
TRUTH_PATH = os.path.join(RESULTS_DIR, "groundtruth.json")
BUDGETS = (5, 10, 25)


def provider_order(candidates: Sequence[Candidate]) -> List[Candidate]:
    """The order the search provider returned."""
    return sorted(candidates, key=lambda one: one.position)


def legacy_order(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Ordering before this task: social first, then the provider's position."""
    return sorted(candidates, key=lambda one: (not one.is_social, one.position))


def ranked_order(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Ordering now: investigation priority, as parse_visual_matches returns it."""
    return list(candidates)


ORDERINGS = {
    "provider": provider_order,
    "legacy": legacy_order,
    "ranked": ranked_order,
}


# ------------------------------------------------------------- ground truth


def _verify(
    probe: str,
    candidates: Sequence[Candidate],
    budget: int,
    mirror: Optional[str],
) -> face_verifier.Investigation:
    """One real verification pass, exactly as the pipeline runs it."""
    benchmark.Stopwatch().install(mirror)
    scan = adapter.scan_face(probe)
    investigation = face_verifier.Investigation(discovered=len(candidates), unique=len(candidates))
    work_dir = tempfile.mkdtemp(prefix="rank-eval-")
    try:
        face_verifier.verify_candidates(
            probe_path=probe,
            candidates=candidates,
            work_dir=work_dir,
            max_candidates=budget,
            verbose=False,
            investigation=investigation,
            probe_embedding=scan["embedding"],
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return investigation


def build_truth(
    corpus: str, probe: str, candidates: Sequence[Candidate], mirror: Optional[str]
) -> Dict[str, Any]:
    """Verdict for every discovered candidate, so any ordering can be scored."""
    with open(corpus, "r", encoding="utf-8") as handle:
        raw = {
            item.get("link"): item
            for item in json.load(handle)["search_response"]["visual_matches"]
        }
    investigation = _verify(probe, candidates, len(candidates), mirror)
    results = []
    for result in investigation.results:
        item = raw.get(result.candidate.page_url, {})
        results.append(
            {
                "page_url": result.candidate.page_url,
                "image_url": result.candidate.image_url,
                "source": result.candidate.source,
                "position": result.candidate.position,
                "is_social": result.candidate.is_social,
                "priority_score": result.candidate.priority_score,
                "status": result.status,
                "distance": None if result.distance is None else round(result.distance, 6),
                # kept to show what the ranking signals were tested against
                "thumbnail_width": item.get("thumbnail_width"),
                "thumbnail_height": item.get("thumbnail_height"),
            }
        )
    return {"corpus": corpus, "probe": probe, "mirror": mirror, "results": results}


def load_truth() -> Dict[str, Any]:
    with open(TRUTH_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------- the score


def score_ordering(
    ordered: Sequence[Candidate], truth: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """What each budget would have found, from verdicts already measured."""
    verdicts = [truth[one.page_url] for one in ordered if one.page_url in truth]
    matched = [one for one in verdicts if one["status"] == "MATCH"]
    best = min(matched, key=lambda one: one["distance"]) if matched else None

    budgets = {}
    for budget in BUDGETS:
        window = verdicts[:budget]
        inside = [one for one in window if one["status"] == "MATCH"]
        budgets[budget] = {
            "matches": len(inside),
            "best_found": bool(best is not None and best in inside),
            "best_distance": min((one["distance"] for one in inside), default=None),
            "no_face": sum(1 for one in window if one["status"] == "NO_FACE"),
            "download_failed": sum(1 for one in window if one["status"] == "DOWNLOAD_FAILED"),
        }
    return {
        "best_position": (verdicts.index(best) + 1) if best else None,
        "first_match_position": (verdicts.index(matched[0]) + 1) if matched else None,
        "total_matches": len(matched),
        "budgets": budgets,
    }


# -------------------------------------------------------- ordering A/B test


def compare_to_truth(
    investigation: face_verifier.Investigation, truth: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Every way a budgeted run disagreed with the full-corpus verdicts."""
    differences = []
    for result in investigation.results:
        url = result.candidate.page_url
        distance = None if result.distance is None else round(result.distance, 6)
        expected = truth.get(url)
        if expected is None:
            differences.append(f"{url}: not in the discovered set")
        elif result.status != expected["status"]:
            differences.append(f"{url}: status {expected['status']} -> {result.status}")
        elif distance != expected["distance"]:
            differences.append(f"{url}: distance {expected['distance']} -> {distance}")
    return differences


# ------------------------------------------------------------------- report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", help="evidence/<id>.json to replay (default: newest)")
    parser.add_argument("--probe", help="probe image (default: the matched image next to it)")
    parser.add_argument("--mirror", help="serve candidate images from this local mirror")
    parser.add_argument("--refresh", action="store_true", help="recompute the cached verdicts")
    parser.add_argument("--verify", action="store_true", help="re-run the verifier per ordering")
    args = parser.parse_args(argv)

    corpus, corpus_probe = benchmark.newest_corpus()
    evidence_path = args.evidence or corpus
    probe = args.probe or (corpus_probe if evidence_path == corpus else None)
    if probe is None:
        raise SystemExit("--evidence given without --probe")
    candidates = benchmark.load_candidates(evidence_path)

    if args.refresh or not os.path.isfile(TRUTH_PATH):
        print(f"verifying all {len(candidates)} candidates once for ground truth ...")
        payload = build_truth(evidence_path, probe, candidates, args.mirror)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(TRUTH_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    truth_payload = load_truth()
    truth = {one["page_url"]: one for one in truth_payload["results"]}
    counts: Dict[str, int] = {}
    for one in truth_payload["results"]:
        counts[one["status"]] = counts.get(one["status"], 0) + 1

    print(f"corpus     {truth_payload['corpus']} ({len(truth)} unique candidates)")
    print(f"verdicts   {counts}")
    print()

    header = "  ".join(f"top-{budget:<3}" for budget in BUDGETS)
    print(f"{'ordering':<10}  {'best@':>5}  {header}  of")
    scores: Dict[str, Any] = {}
    for name, order in ORDERINGS.items():
        score = score_ordering(order(candidates), truth)
        scores[name] = score
        cells = "  ".join(f"{score['budgets'][budget]['matches']:>6}" for budget in BUDGETS)
        print(f"{name:<10}  {str(score['best_position']):>5}  {cells}  {score['total_matches']}")
    print()
    print("best@ is the position of the closest verified match; a budget below it")
    print("would have missed it. Ranking decides order only - every verdict above")
    print("came from ArcFace cosine distance against the configured threshold.")

    differences: Dict[str, List[str]] = {}
    if args.verify:
        print()
        for name in ("legacy", "ranked"):
            ordered = ORDERINGS[name](candidates)
            for budget in BUDGETS:
                investigation = _verify(probe, ordered, budget, args.mirror)
                found = compare_to_truth(investigation, truth)
                differences[f"{name}@{budget}"] = found
                print(
                    f"{name:<8} budget {budget:>3}: {investigation.checked} investigated, "
                    f"{len(investigation.not_investigated)} not investigated, "
                    f"{'verdicts identical to the full-corpus run' if not found else found}"
                )

    out = os.path.join(RESULTS_DIR, "rank-eval.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "corpus": truth_payload["corpus"],
                "unique_candidates": len(truth),
                "verdicts": counts,
                "orderings": scores,
                "differences": differences,
            },
            handle,
            indent=2,
        )
    print(f"\nwritten    {out}")
    return 1 if any(differences.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
