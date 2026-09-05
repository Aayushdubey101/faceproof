"""Streamlit UI for exercising the FaceProof pipeline.

A thin shell over `faceproof.pipeline` - no logic of its own, so the UI can
never disagree with the CLI. Run it with:

    uv run streamlit run app.py

ponytail: pipeline stdout is mirrored into a text block rather than rendered as
real progress widgets. Swap in st.status/st.progress if the stage output stops
being readable.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import logging
import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlsplit

import streamlit as st
from dotenv import load_dotenv

from faceproof import pipeline
from faceproof.blockchain import registry
from faceproof.discovery import DiscoveryError, retrieval
from faceproof.matching import verifier as face_verifier

EXPLORER_TX_URL = "https://sepolia.basescan.org/tx/"
PROBE_TYPES = ["jpg", "jpeg", "png", "webp"]

TAGLINE = "Discover. Verify. Fingerprint. Audit."

LOG = logging.getLogger("faceproof.app")  # developer diagnostics, not shown in the UI
CARDS_PER_ROW = 3

TRUST_MODEL = (
    "Web search discovers candidates. Face matching independently verifies the facial "
    "similarity. SHA-256 fingerprints the evidence package. Base Sepolia anchors that "
    "fingerprint so later verification can detect modification. "
    "**Blockchain verification does not prove real-world identity.**"
)

HOW_IT_WORKS = """**How FaceProof verifies candidates**

Google Lens finds visually related images. FaceProof does not trust that ranking.
Each candidate image is independently processed with ArcFace and compared against
the probe. Only candidates passing the configured threshold become evidence."""

FINGERPRINT_NOTE = (
    "FaceProof fingerprints the evidence package with SHA-256. The fingerprint is anchored "
    "on Base Sepolia so the evidence can later be independently checked for tampering."
)

SCOPE_NOTE = (
    "Face matching establishes the facial similarity result. Blockchain establishes evidence "
    "integrity and provenance. Blockchain does NOT prove that the person is who they claim to be."
)

WHAT_THIS_PROVES = """**What this proves**

- Face matching verifies that the discovered image contains a sufficiently similar face.
- SHA-256 fingerprints the evidence package.
- Base Sepolia stores the fingerprint anchor.
- Independent verification recomputes the fingerprint and compares it with the blockchain record.

Blockchain verification detects evidence modification; it does not establish real-world identity
by itself."""

# the five steps a reviewer walks in the Verify tab, in order
FLOW_STEPS = (
    ("1 · Select evidence", "the file on disk"),
    ("2 · Recompute SHA-256", "hash it again, locally"),
    ("3 · Read blockchain", "fetch the anchored digest"),
    ("4 · Compare", "local vs on-chain"),
    ("5 · Verdict", "VERIFIED or TAMPERED"),
)

# the tamper demo, as a vertical chain: what was hashed -> digest -> chain -> verdict
INTACT_CHAIN = ("ORIGINAL EVIDENCE", "SHA-256", "ON-CHAIN SHA-256")
TAMPERED_CHAIN = ("MODIFIED EVIDENCE", "NEW SHA-256", "ON-CHAIN SHA-256")

# label + badge colour per candidate outcome, keyed by the matching layer's statuses
STATUS_STYLES = {
    face_verifier.MATCH: ("✓ VERIFIED", "green"),
    face_verifier.NO_MATCH: ("✗ REJECTED", "red"),
    face_verifier.NO_FACE: ("✗ NO FACE", "red"),
    face_verifier.DOWNLOAD_FAILED: ("⚠ SKIPPED", "orange"),
}

load_dotenv()
st.set_page_config(page_title="FaceProof", page_icon="🔗", layout="wide")


class _LiveOutput(io.TextIOBase):
    """Mirror pipeline stdout into a Streamlit placeholder as it is produced."""

    def __init__(self, placeholder: Any) -> None:
        self._placeholder = placeholder
        self._chunks: List[str] = []

    def write(self, text: str) -> int:
        self._chunks.append(text)
        self._placeholder.code("".join(self._chunks), language="text")
        return len(text)


def _stream(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Tuple[Any, Optional[Exception]]:
    """Call a pipeline function with its output streamed. Returns (result, error).

    The exception itself is returned, not its text: the caller renders a
    discovery outage differently from a pipeline failure.
    """
    output = _LiveOutput(st.empty())
    try:
        with contextlib.redirect_stdout(output):
            return function(*args, **kwargs), None
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
        # full traceback to the developer console, never to the evaluator
        LOG.exception("%s failed", getattr(function, "__name__", "pipeline"))
        return None, error


def _show_error(error: Exception) -> None:
    """Render a failure for a non-developer: no traceback, no secrets."""
    if not isinstance(error, DiscoveryError):
        st.error(str(error))
        return

    st.error("Web discovery unavailable")
    lines = [
        f"- **Provider** `{error.provider}`",
        f"- **Reason** {error.reason}",
    ]
    if error.timeout:
        lines.append(f"- **Limits** {error.timeout}")
    lines.append(
        "- **Suggested** retry the run; if it repeats, set `FACE_UPLOAD_PROVIDER` to another "
        f"provider ({', '.join(retrieval.PROVIDERS)}) in `.env` and restart."
    )
    st.markdown("\n".join(lines))
    st.caption(
        "Only the external discovery step is affected. Local face verification and "
        "on-chain verification of existing evidence still work."
    )


def _read_document(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _sidebar() -> None:
    st.subheader("Environment")
    for label, name in (
        ("SerpApi key", "SERPAPI_API_KEY"),
        ("Wallet key", "BLOCKCHAIN_PRIVATE_KEY"),
    ):
        # presence only - a secret must never be rendered
        st.write(f"{'set' if os.environ.get(name) else 'missing'} - {label}")
    st.write(f"{retrieval.selected_provider()} - probe upload host")
    st.caption("Missing keys? Copy .env.example to .env and restart.")

    st.divider()
    st.subheader("Chain")
    st.write(f"{registry.NETWORK_NAME} ({registry.CHAIN_ID})")
    st.caption(os.environ.get("BLOCKCHAIN_RPC_URL") or registry.DEFAULT_RPC_URL)


def _candidate_image(result: face_verifier.CandidateResult) -> None:
    """Show what was actually checked - the downloaded bytes, else the remote URL."""
    source = result.image_bytes or result.candidate.image_url
    if source:
        st.image(source, width="stretch")


def _candidate_card(number: int, result: face_verifier.CandidateResult) -> None:
    label, color = STATUS_STYLES[result.status]
    with st.container(border=True):
        heading, badge = st.columns([2, 1], vertical_alignment="center")
        heading.markdown(f"**Candidate #{number}**")
        with badge:
            st.badge(label, color=color)

        st.caption(result.candidate.source or result.candidate.page_url)
        _candidate_image(result)

        if result.distance is None:
            # never compared - inventing a distance here would misstate the evidence
            st.markdown(f"Reason: {result.reason}")
        else:
            st.markdown(
                f"Face distance: `{result.distance:.4f}`\n\n"
                f"Threshold: `{result.threshold:.4f}`\n\n"
                f"{result.reason}"
            )
        st.link_button("Open source", result.candidate.page_url, width="stretch")


def _best_match(result: face_verifier.CandidateResult) -> None:
    st.subheader("5 · Best verified match")
    left, right = st.columns([1, 2])
    with left:
        _candidate_image(result)
    with right:
        st.badge("✓ VERIFIED MATCH", color="green")
        st.markdown(f"**Source**  \n{result.candidate.source or 'unknown'}")
        distance, threshold = st.columns(2)
        distance.metric("Distance", f"{result.distance:.4f}", border=True)
        threshold.metric("Threshold", f"{result.threshold:.4f}", border=True)
        st.caption(
            "Cosine distance below the model threshold. "
            "The distance is not an identity probability."
        )
        st.link_button("Open source", result.candidate.page_url, type="primary")


def _investigation(investigation: face_verifier.Investigation) -> None:
    """Render every candidate outcome the matching layer produced."""
    st.subheader("4 · Candidate investigation")
    st.info(HOW_IT_WORKS)

    verified = investigation.verified
    discovered, checked, matched = st.columns(3)
    discovered.metric("Candidates discovered", investigation.discovered, border=True)
    checked.metric("Candidates investigated", investigation.checked, border=True)
    matched.metric("Verified matches", len(verified), border=True)

    if verified:
        _best_match(min(verified, key=lambda result: result.distance))

    st.markdown("**Every candidate investigated**")
    for row_start in range(0, investigation.checked, CARDS_PER_ROW):
        row = investigation.results[row_start : row_start + CARDS_PER_ROW]
        for offset, (column, result) in enumerate(zip(st.columns(CARDS_PER_ROW), row)):
            with column:
                _candidate_card(row_start + offset + 1, result)


def _domain(url: str) -> str:
    return urlsplit(url).netloc or url


def _evidence_package(document: dict, evidence_id: str, heading: str = "Evidence package") -> None:
    """Show exactly what was fingerprinted - every value comes from the file."""
    evidence = document["evidence"]
    post, match = evidence.get("post", {}), evidence.get("match", {})

    st.subheader(heading)
    with st.container(border=True):
        left, right = st.columns(2)
        left.markdown(
            f"**Evidence ID**  \n`{evidence_id}`\n\n"
            f"**Source**  \n{post.get('source') or 'unknown'} - "
            f"`{_domain(str(post.get('page_url', '')))}`\n\n"
            f"**Source URL**  \n{post.get('page_url', 'unknown')}"
        )
        right.markdown(
            f"**Match status**  \n{'✓ VERIFIED MATCH' if match.get('verified') else '✗ NO MATCH'}"
            f"\n\n**Cosine distance / model threshold**  \n"
            f"`{match.get('distance', 'unknown')}` / `{match.get('threshold', 'unknown')}` "
            f"({match.get('model', 'unknown')})\n\n"
            f"**Timestamp**  \n`{evidence.get('discovered_at', 'unknown')}`"
        )
        st.markdown("**EVIDENCE FINGERPRINT** (SHA-256)")
        st.code(document["fingerprint"], language="text")
        st.caption(FINGERPRINT_NOTE)
        st.caption(SCOPE_NOTE)


def _anchor_panel(anchor: dict, fingerprint: str, heading: str = "Blockchain anchor") -> None:
    """Blockchain facts exactly as the pipeline received them from the chain."""
    st.subheader(heading)
    with st.container(border=True):
        st.badge(f"✓ ANCHORED on {anchor.get('network', 'unknown network')}", color="green")
        left, right = st.columns(2)
        left.markdown(
            f"**Network**  \n{anchor.get('network', 'unknown')}\n\n"
            f"**Chain ID**  \n`{anchor.get('chain_id', 'unknown')}`"
        )
        right.markdown(
            f"**Block number**  \n`{anchor.get('block_number', 'unknown')}`\n\n"
            f"**Gas used**  \n`{anchor.get('gas_used', 'unknown')}`"
        )
        st.markdown("**Transaction hash**")
        st.code(anchor.get("tx_hash", "none"), language="text")
        st.markdown("**Evidence fingerprint anchored**")
        st.code(fingerprint, language="text")
        if anchor.get("tx_hash") and anchor.get("chain_id") == registry.CHAIN_ID:
            st.link_button(
                "View on BaseScan", f"{EXPLORER_TX_URL}{anchor['tx_hash']}", type="primary"
            )


def _flow() -> None:
    for column, (step, detail) in zip(st.columns(len(FLOW_STEPS)), FLOW_STEPS):
        with column.container(border=True):
            st.markdown(f"**{step}**")
            st.caption(detail)


def _chain(rungs: Tuple[str, ...], ok: bool) -> None:
    """The demo sequence as a vertical chain, ending in the verdict it produced."""
    verdict = "✓ VERIFIED" if ok else "✗ TAMPERED"
    with st.container(border=True):
        st.markdown("\n\n↓\n\n".join(f"**{rung}**" for rung in (*rungs, verdict)))


def _comparison(local: str, local_label: str, onchain: str) -> None:
    st.markdown(f"**{local_label}**")
    st.code(local, language="text")
    st.markdown("**ON-CHAIN FINGERPRINT**")
    st.code(onchain, language="text")
    if local == onchain:
        st.success("✓ MATCH - VERIFIED - evidence fingerprint matches the blockchain record")
    else:
        st.error(
            "✗ MISMATCH - TAMPERED - evidence fingerprint does not match the blockchain record"
        )


def _verification(report: pipeline.Audit, tampered: bool) -> None:
    st.subheader("Verification")
    _flow()

    if tampered:
        section, key = pipeline.TAMPER_FIELD
        modified, digest = pipeline.tamper(report.document["evidence"])
        st.warning(
            f"Simulation on an in-memory copy - the evidence file on disk is untouched. "
            f"Edited field `{section}.{key}`: {modified[section][key]}"
        )
        _chain(TAMPERED_CHAIN, digest == report.onchain_fingerprint)
        st.markdown("**ORIGINAL FINGERPRINT**")
        st.code(report.local_fingerprint, language="text")
        _comparison(digest, "TAMPERED FINGERPRINT", report.onchain_fingerprint)
        st.caption("Press **Restore / verify original** to re-check the untouched file.")
        return

    _chain(INTACT_CHAIN, report.onchain_ok)
    _comparison(report.local_fingerprint, "LOCAL FINGERPRINT", report.onchain_fingerprint)
    supporting = (
        ("Search response integrity", report.search_ok),
        ("Matched image integrity", report.image_ok),
    )
    for label, ok in supporting:
        if ok is not None:  # None = this file carries nothing to check that against
            st.caption(f"{'✓' if ok else '✗'} {label}")


def _audit(path: str, tampered: bool) -> None:
    """Re-read the file and the chain, then stash the report for rendering."""
    report, error = _stream(pipeline.audit, path)
    if error:
        _show_error(error)
    st.session_state["report"] = report
    st.session_state["tampered"] = tampered


def _run_tab() -> None:
    st.subheader("1 · Input face")
    upload = st.file_uploader("Probe face image", type=PROBE_TYPES)
    max_candidates = st.slider("Candidates to verify", 5, 50, face_verifier.DEMO_CANDIDATES)
    st.caption(
        "Every candidate is downloaded and face-checked, so more candidates take longer. "
        "The first run also downloads ~130 MB of model weights."
    )

    if not st.button("Run pipeline", type="primary", disabled=upload is None):
        return

    probe_path = os.path.join(tempfile.gettempdir(), f"faceproof-probe-{upload.name}")
    with open(probe_path, "wb") as handle:
        handle.write(upload.getbuffer())

    investigation = face_verifier.Investigation()
    st.subheader("2 · Face analysis  →  3 · Web discovery")
    left, right = st.columns([1, 3])
    with left:
        st.image(upload, caption="probe")
    with right:
        document, error = _stream(
            pipeline.run, probe_path, max_candidates=max_candidates, investigation=investigation
        )

    # filled in place, so the candidate work stays visible even when the run failed
    if investigation.results:
        st.divider()
        _investigation(investigation)
        st.divider()

    if error:
        _show_error(error)
        st.caption("No evidence file was written and nothing was anchored.")
        return

    _evidence_package(
        document,
        str(document["artifacts"]["match_image"])[: -len(".match.jpg")],
        heading="6 · Evidence package",
    )
    _anchor_panel(document["anchor"], document["fingerprint"], heading="7 · Blockchain anchor")
    st.success("✓ VERIFIED - match verified, evidence fingerprinted and anchored on-chain")
    st.info(WHAT_THIS_PROVES)
    with st.expander("Raw evidence document"):
        st.json(document["evidence"])


def _verify_tab() -> None:
    paths = sorted(glob.glob(os.path.join(pipeline.EVIDENCE_DIR, "*.json")))
    if not paths:
        st.info("No evidence files yet - run the pipeline first.")
        return

    st.subheader("1 · Select evidence")
    path = st.selectbox("Evidence file", paths, index=len(paths) - 1)
    st.caption(
        "An independent audit: it reads the evidence file and the chain, not the last run. "
        "Hand-edit any value inside the file's `evidence` block, then verify again."
    )
    if st.session_state.get("audit_path") != path:
        # a report about another file says nothing about this one
        st.session_state.pop("report", None)
        st.session_state["audit_path"] = path

    document = _read_document(path)
    match_image = (document or {}).get("artifacts", {}).get("match_image")
    if match_image:
        image_path = os.path.join(os.path.dirname(path), match_image)
        if os.path.isfile(image_path):
            st.image(image_path, caption="matched image", width=220)

    check, simulate, restore = st.columns(3)
    if check.button("Verify", type="primary", width="stretch"):
        _audit(path, tampered=False)
    if simulate.button("Simulate tampering", width="stretch"):
        _audit(path, tampered=True)
    if restore.button("Restore / verify original", width="stretch"):
        _audit(path, tampered=False)

    report = st.session_state.get("report")
    if report is None:
        return

    _evidence_package(report.document, os.path.basename(path)[: -len(".json")])
    _anchor_panel(report.document["anchor"], report.stored_fingerprint)
    _verification(report, bool(st.session_state.get("tampered")))
    st.info(WHAT_THIS_PROVES)


st.title("FaceProof")
st.caption(TAGLINE)
st.markdown(TRUST_MODEL)

with st.sidebar:
    _sidebar()

run_tab, verify_tab = st.tabs(["Run", "Verify"])
with run_tab:
    _run_tab()
with verify_tab:
    _verify_tab()
