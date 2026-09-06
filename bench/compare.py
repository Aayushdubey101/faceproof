"""Diff two benchmark results: correctness first, then latency.

    uv run python bench/compare.py baseline-mirror optimized-mirror

Correctness is the gate. Two runs over the same mirrored candidate set must
agree on every status, every distance, the best match and the ranking; the
timing columns only mean something once they do.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(label: str) -> Dict[str, Any]:
    path = label if label.endswith(".json") else os.path.join(RESULTS_DIR, f"{label}.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def differences(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    """Every way two runs over the same candidate set disagree."""
    found: List[str] = []
    if before["best_distance"] != after["best_distance"]:
        found.append(f"best distance {before['best_distance']} -> {after['best_distance']}")
    if before["best_page_url"] != after["best_page_url"]:
        found.append(f"best candidate {before['best_page_url']} -> {after['best_page_url']}")
    if before["statuses"] != after["statuses"]:
        found.append("status sequence changed")
    for old, new in zip(before["results"], after["results"]):
        if old["page_url"] != new["page_url"]:
            found.append(f"ranking moved: {old['page_url']} -> {new['page_url']}")
        elif old != new:
            found.append(
                f"{old['page_url']}: {old['status']}/{old['distance']} -> {new['status']}/{new['distance']}"
            )
    return found


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: compare.py <before-label> <after-label>")
    before, after = load(argv[0]), load(argv[1])
    print(f"before {before['label']:<20} {before['corpus']}")
    print(f"after  {after['label']:<20} {after['corpus']}")
    print()
    print(f"{'n':>4}  {'before':>9}  {'after':>9}  {'improvement':>11}  correctness")

    failures = 0
    for old, new in zip(before["runs"], after["runs"]):
        found = differences(old, new)
        failures += len(found)
        gain = (old["total_seconds"] - new["total_seconds"]) / old["total_seconds"] * 100
        counts = {status: new["statuses"].count(status) for status in sorted(set(new["statuses"]))}
        verdict = "identical" if not found else f"{len(found)} DIFFERENCES"
        print(
            f"{new['candidates']:>4}  {old['total_seconds']:>8.2f}s  {new['total_seconds']:>8.2f}s  "
            f"{gain:>10.1f}%  {verdict} {counts}"
        )
        for line in found:
            print(f"        ! {line}")

    print()
    print("correctness:", "no behavioural differences" if not failures else f"{failures} DIFFERENCES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
