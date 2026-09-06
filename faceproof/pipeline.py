"""FaceProof orchestration - face scan, web discovery, blockchain anchor.

    python -m faceproof run    <image.jpg>
    python -m faceproof verify <evidence/<id>.json>

`run` walks every stage once. `verify` never repeats discovery - that is the
point: the evidence is checkable long after the search is gone.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from faceproof.blockchain import registry
from faceproof.blockchain import verifier as chain_verifier
from faceproof.discovery import DiscoveryError, retrieval
from faceproof.discovery import candidates as candidate_ranking
from faceproof.discovery.reverse_search import SEARCH_ENGINE_ID, reverse_search
from faceproof.evidence import hashing, manifest
from faceproof.face import adapter
from faceproof.matching import verifier as face_verifier

EVIDENCE_DIR = "evidence"
REQUIRED_KEYS = ("evidence", "fingerprint", "anchor")

# the tamper demo edits this display-only field of an in-memory copy
TAMPER_FIELD = ("post", "title")
TAMPER_SUFFIX = " [edited]"

TICK = "✓"
CROSS = "✗"


def _mark(ok: bool) -> str:
    return TICK if ok else CROSS


def _kilobytes(path: str) -> int:
    return round(os.path.getsize(path) / 1024)


def run(
    image_path: str,
    evidence_dir: str = EVIDENCE_DIR,
    max_candidates: int = face_verifier.MAX_CANDIDATES,
    model_name: str = adapter.DEFAULT_MODEL,
    detector_backend: str = adapter.DEFAULT_DETECTOR,
    investigation: Optional[face_verifier.Investigation] = None,
) -> Dict[str, Any]:
    """Run every stage end to end and persist an anchored evidence file.

    Pass an `Investigation` to collect what happened to each candidate; it is
    filled in place, so it survives a run that ends with no verified match.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    print(f"[1/5] Face scan             {image_path}")
    probe_scan = adapter.scan_face(image_path, model_name, detector_backend)
    probe_digest = hashing.sha256_file(image_path)
    print(f"      {TICK} Face detected       {probe_scan['faces_detected']} face(s)")
    print(f"      {TICK} Embedding generated {probe_scan['embedding_dimensions']}-d {model_name}")

    print("[2/5] Web discovery")
    provider = retrieval.selected_provider()
    with tempfile.TemporaryDirectory(prefix="faceproof-probe-") as upload_dir:
        # only what leaves the machine is resized; the original stays the probe
        upload_path = retrieval.optimize_probe(image_path, upload_dir)
        print(
            f"      {TICK} Probe optimized     {_kilobytes(image_path)} KB -> "
            f"{_kilobytes(upload_path)} KB (max {retrieval.MAX_PROBE_PIXELS}px JPEG)"
        )
        probe_url = retrieval.upload_probe(upload_path, provider)
    print(f"      {TICK} Probe published     [{provider}] {probe_url}")
    candidates, search_response = reverse_search(probe_url)
    if not candidates:
        raise DiscoveryError(SEARCH_ENGINE_ID, "the search returned no candidates for this probe")
    # candidates arrive normalized, deduplicated and ranked by investigation
    # priority, so the budget below is spent on the most promising results
    discovered = candidate_ranking.discovered_count(candidates)
    social = sum(1 for candidate in candidates if candidate.is_social)
    print(f"      {TICK} Candidates found    {discovered} ({social} on social platforms)")
    if discovered > len(candidates):
        print(f"      {TICK} Duplicates merged   {len(candidates)} unique candidates remain")
    if investigation is not None:
        investigation.discovered = discovered
        investigation.unique = len(candidates)

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
            investigation=investigation,
            probe_embedding=probe_scan["embedding"],  # encoded once, at [1/5]
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
        candidates_returned=discovered,
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


@dataclass(frozen=True)
class Audit:
    """Everything one independent re-check of an evidence file concluded.

    A check is None when the file carries nothing to check it against - an
    older evidence file without a stored search response is not a failure.
    """

    document: Dict[str, Any]
    stored_fingerprint: str
    local_fingerprint: str
    onchain_fingerprint: str
    tx_hash: str
    search_ok: Optional[bool] = None
    image_ok: Optional[bool] = None

    @property
    def local_ok(self) -> bool:
        return self.local_fingerprint == self.stored_fingerprint

    @property
    def onchain_ok(self) -> bool:
        return self.local_fingerprint == self.onchain_fingerprint

    @property
    def passed(self) -> bool:
        checks = (self.local_ok, self.search_ok, self.image_ok, self.onchain_ok)
        return all(check is not False for check in checks)


def audit(evidence_path: str, rpc_url: Optional[str] = None) -> Audit:
    """Recompute the fingerprint locally and read the anchored one off the chain.

    The caller decides how to report it - `verify` prints, the UI renders.
    """
    document = _load_document(evidence_path)
    evidence = document["evidence"]
    recomputed = hashing.fingerprint(evidence)

    search_ok: Optional[bool] = None
    if "search_response" in document:
        search_digest = hashing.sha256_bytes(hashing.canonical_json(document["search_response"]))
        search_ok = search_digest == evidence.get("search", {}).get("response_sha256")

    image_ok: Optional[bool] = None
    artifact = document.get("artifacts", {}).get("match_image")
    if artifact:
        image_path = os.path.join(os.path.dirname(evidence_path), artifact)
        if os.path.isfile(image_path):
            expected = evidence.get("post", {}).get("image_sha256")
            image_ok = hashing.sha256_file(image_path) == expected

    anchor = document["anchor"]
    tx_hash = anchor.get("tx_hash") if isinstance(anchor, dict) else None
    if not tx_hash:
        raise ValueError(f"{evidence_path} has no anchor transaction hash")

    onchain = chain_verifier.verify_onchain(recomputed, str(tx_hash), rpc_url)
    return Audit(
        document=document,
        stored_fingerprint=str(document["fingerprint"]),
        local_fingerprint=recomputed,
        onchain_fingerprint=str(onchain["onchain_digest"]),
        tx_hash=str(tx_hash),
        search_ok=search_ok,
        image_ok=image_ok,
    )


def tamper(evidence: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Edit one display-only field of a *copy* and refingerprint it.

    Nothing is written back. The demo has to show that modification is
    detected, not damage the evidence it is demonstrating on.
    """
    section, key = TAMPER_FIELD
    modified = copy.deepcopy(evidence)
    block = modified.setdefault(section, {})
    block[key] = f"{block.get(key, '')}{TAMPER_SUFFIX}"
    return modified, hashing.fingerprint(modified)


def verify(evidence_path: str, rpc_url: Optional[str] = None) -> bool:
    """Report an `audit` on stdout and return whether it passed."""
    report = audit(evidence_path, rpc_url)

    print(f"{_mark(report.local_ok)} Local evidence integrity")
    print(f"    stored      {report.stored_fingerprint}")
    print(f"    recomputed  {report.local_fingerprint}")
    if report.search_ok is not None:
        print(f"{_mark(report.search_ok)} Search response integrity")
    if report.image_ok is not None:
        print(f"{_mark(report.image_ok)} Matched image integrity")

    print(f"{TICK} Blockchain anchor")
    print(f"    tx          {report.tx_hash}")
    print(f"    on-chain    {report.onchain_fingerprint}")
    print(f"{_mark(report.onchain_ok)} Fingerprint comparison")

    print(f"\nRESULT: {'VERIFIED' if report.passed else 'TAMPERED'}")
    return report.passed


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
