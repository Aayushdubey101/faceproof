"""FaceProof orchestration - face scan, web discovery, blockchain anchor.

    python -m faceproof run    <image.jpg>
    python -m faceproof verify <evidence/<id>.json>

`run` walks every stage once. `verify` never repeats discovery - that is the
point: the evidence is checkable long after the search is gone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from faceproof.blockchain import registry
from faceproof.blockchain import verifier as chain_verifier
from faceproof.discovery.retrieval import upload_probe
from faceproof.discovery.reverse_search import SEARCH_ENGINE_ID, reverse_search
from faceproof.evidence import hashing, manifest
from faceproof.face import adapter
from faceproof.matching import verifier as face_verifier

EVIDENCE_DIR = "evidence"
REQUIRED_KEYS = ("evidence", "fingerprint", "anchor")

TICK = "✓"
CROSS = "✗"


def _mark(ok: bool) -> str:
    return TICK if ok else CROSS


def run(
    image_path: str,
    evidence_dir: str = EVIDENCE_DIR,
    max_candidates: int = face_verifier.MAX_CANDIDATES,
    model_name: str = adapter.DEFAULT_MODEL,
    detector_backend: str = adapter.DEFAULT_DETECTOR,
) -> Dict[str, Any]:
    """Run every stage end to end and persist an anchored evidence file."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    print(f"[1/5] Face scan             {image_path}")
    probe_scan = adapter.scan_face(image_path, model_name, detector_backend)
    probe_digest = hashing.sha256_file(image_path)
    print(f"      {TICK} Face detected       {probe_scan['faces_detected']} face(s)")
    print(
        f"      {TICK} Embedding generated {probe_scan['embedding_dimensions']}-d {model_name}"
    )

    print("[2/5] Web discovery")
    probe_url = upload_probe(image_path)
    print(f"      {TICK} Probe published     {probe_url}")
    candidates, search_response = reverse_search(probe_url)
    if not candidates:
        raise RuntimeError("reverse image search returned no candidates")
    social = sum(1 for candidate in candidates if candidate.is_social)
    print(f"      {TICK} Candidates found    {len(candidates)} ({social} on social platforms)")

    print(f"[3/5] Candidate verification (up to {max_candidates} candidates)")
    work_dir = tempfile.mkdtemp(prefix="faceproof-")
    try:
        match = face_verifier.verify_candidates(
            probe_path=image_path,
            candidates=candidates,
            work_dir=work_dir,
            model_name=model_name,
            detector_backend=detector_backend,
            max_candidates=max_candidates,
        )
        if match is None:
            raise RuntimeError("no candidate passed face verification - try another probe image")

        os.makedirs(evidence_dir, exist_ok=True)
        run_id = manifest.evidence_id(probe_digest)
        match_image_name = f"{run_id}.match.jpg"
        shutil.copyfile(match.image_path, os.path.join(evidence_dir, match_image_name))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"      {TICK} Verified match      {match.candidate.page_url}")
    print(f"      {TICK} Distance            {match.distance:.4f} < {match.threshold}")

    print("[4/5] Evidence")
    evidence = manifest.build_evidence(
        probe_digest=probe_digest,
        probe_scan=probe_scan,
        match=match,
        search_engine=SEARCH_ENGINE_ID,
        probe_image_url=probe_url,
        search_response=search_response,
        candidates_returned=len(candidates),
        candidates_checked=min(max_candidates, len(candidates)),
    )
    digest = hashing.fingerprint(evidence)
    print(f"      {TICK} Manifest built")
    print(f"      {TICK} Fingerprint         {digest}")

    print("[5/5] Blockchain")
    anchor = registry.anchor(digest)
    print(f"      {TICK} Anchored            {anchor['tx_hash']}")
    print(
        f"      {TICK} Network / block     "
        f"{anchor['network']} ({anchor['chain_id']}) / {anchor['block_number']}"
    )

    document = manifest.build_document(evidence, anchor, match_image_name, search_response)
    evidence_path = os.path.join(evidence_dir, f"{run_id}.json")
    with open(evidence_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)

    print(f"\nevidence written to         {evidence_path}")
    print(f"re-verify with              python -m faceproof verify {evidence_path}")
    return document


def _load_document(evidence_path: str) -> Dict[str, Any]:
    """Read an evidence file, rejecting anything that is not one.

    The file is untrusted input - the demo invites a reviewer to hand-edit it.
    A malformed file must abort with a clear message, and a *tampered* one must
    still reach the tamper verdict rather than blow up on a missing key.
    """
    with open(evidence_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)  # JSONDecodeError is a ValueError

    if not isinstance(document, dict):
        raise ValueError(f"{evidence_path} is not a FaceProof evidence file (expected an object)")
    missing = [key for key in REQUIRED_KEYS if key not in document]
    if missing:
        raise ValueError(
            f"{evidence_path} is not a FaceProof evidence file (missing {', '.join(missing)})"
        )
    if not isinstance(document["evidence"], dict):
        raise ValueError(f"{evidence_path} has a malformed 'evidence' block")
    return document


def verify(evidence_path: str, rpc_url: Optional[str] = None) -> bool:
    """Recompute the fingerprint locally and check it against the chain."""
    document = _load_document(evidence_path)

    evidence = document["evidence"]
    recomputed = hashing.fingerprint(evidence)
    stored = document["fingerprint"]
    local_ok = recomputed == stored
    print(f"{_mark(local_ok)} Local evidence integrity")
    print(f"    stored      {stored}")
    print(f"    recomputed  {recomputed}")

    search_ok = True
    if "search_response" in document:
        search_digest = hashing.sha256_bytes(hashing.canonical_json(document["search_response"]))
        search_ok = search_digest == evidence.get("search", {}).get("response_sha256")
        print(f"{_mark(search_ok)} Search response integrity")

    image_ok = True
    artifact = document.get("artifacts", {}).get("match_image")
    if artifact:
        image_path = os.path.join(os.path.dirname(evidence_path), artifact)
        if os.path.isfile(image_path):
            expected = evidence.get("post", {}).get("image_sha256")
            image_ok = hashing.sha256_file(image_path) == expected
            print(f"{_mark(image_ok)} Matched image integrity")

    anchor = document["anchor"]
    tx_hash = anchor.get("tx_hash") if isinstance(anchor, dict) else None
    if not tx_hash:
        raise ValueError(f"{evidence_path} has no anchor transaction hash")

    onchain = chain_verifier.verify_onchain(recomputed, tx_hash, rpc_url)
    print(f"{TICK} Blockchain anchor")
    print(f"    tx          {tx_hash}")
    print(f"    on-chain    {onchain['onchain_digest']}")
    print(f"{_mark(onchain['matches'])} Fingerprint comparison")

    passed = local_ok and search_ok and image_ok and onchain["matches"]
    print(f"\nRESULT: {'VERIFIED' if passed else 'TAMPERED'}")
    return passed


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        # the tick/cross marks are not encodable in the Windows console default codepage
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="faceproof", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the full pipeline on an image")
    run_parser.add_argument("image", help="path to the probe face image")
    run_parser.add_argument("--evidence-dir", default=EVIDENCE_DIR)
    run_parser.add_argument("--max-candidates", type=int, default=face_verifier.MAX_CANDIDATES)
    run_parser.add_argument("--model", default=adapter.DEFAULT_MODEL)
    run_parser.add_argument("--detector", default=adapter.DEFAULT_DETECTOR)

    verify_parser = subparsers.add_parser("verify", help="re-verify an evidence file")
    verify_parser.add_argument("evidence", help="path to evidence/<id>.json")

    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            run(
                image_path=args.image,
                evidence_dir=args.evidence_dir,
                max_candidates=args.max_candidates,
                model_name=args.model,
                detector_backend=args.detector,
            )
            return 0
        return 0 if verify(args.evidence) else 1
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
