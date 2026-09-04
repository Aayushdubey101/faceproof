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
import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from faceproof import pipeline
from faceproof.blockchain import registry
from faceproof.matching import verifier as face_verifier

EXPLORER_TX_URL = "https://sepolia.basescan.org/tx/"
PROBE_TYPES = ["jpg", "jpeg", "png", "webp"]

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


def _stream(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, Optional[str]]:
    """Call a pipeline function with its output streamed. Returns (result, error)."""
    output = _LiveOutput(st.empty())
    try:
        with contextlib.redirect_stdout(output):
            return function(*args, **kwargs), None
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
        return None, str(error)


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
    st.caption("Missing keys? Copy .env.example to .env and restart.")

    st.divider()
    st.subheader("Chain")
    st.write(f"{registry.NETWORK_NAME} ({registry.CHAIN_ID})")
    st.caption(os.environ.get("BLOCKCHAIN_RPC_URL") or registry.DEFAULT_RPC_URL)


def _run_tab() -> None:
    upload = st.file_uploader("Probe face image", type=PROBE_TYPES)
    max_candidates = st.slider("Candidates to verify", 5, 50, face_verifier.MAX_CANDIDATES)
    st.caption("First run downloads ~130 MB of model weights and can take several minutes.")

    if not st.button("Run pipeline", type="primary", disabled=upload is None):
        return

    probe_path = os.path.join(tempfile.gettempdir(), f"faceproof-probe-{upload.name}")
    with open(probe_path, "wb") as handle:
        handle.write(upload.getbuffer())

    left, right = st.columns([1, 3])
    with left:
        st.image(upload, caption="probe")
    with right:
        document, error = _stream(pipeline.run, probe_path, max_candidates=max_candidates)

    if error:
        st.error(error)
        return

    anchor = document["anchor"]
    st.success(f"Anchored on {anchor['network']} in block {anchor['block_number']}")
    if anchor["chain_id"] == registry.CHAIN_ID:
        st.markdown(f"[View transaction on BaseScan]({EXPLORER_TX_URL}{anchor['tx_hash']})")
    st.json(document["evidence"])


def _verify_tab() -> None:
    paths = sorted(glob.glob(os.path.join(pipeline.EVIDENCE_DIR, "*.json")))
    if not paths:
        st.info("No evidence files yet - run the pipeline first.")
        return

    path = st.selectbox("Evidence file", paths, index=len(paths) - 1)
    st.caption("Hand-edit any value inside the file's `evidence` block, then verify again.")

    document = _read_document(path)
    match_image = (document or {}).get("artifacts", {}).get("match_image")
    if match_image:
        image_path = os.path.join(os.path.dirname(path), match_image)
        if os.path.isfile(image_path):
            st.image(image_path, caption="matched image", width=220)

    if not st.button("Verify", type="primary"):
        return

    passed, error = _stream(pipeline.verify, path)
    if error:
        st.error(error)
    elif passed:
        st.success("RESULT: VERIFIED")
    else:
        st.error("RESULT: TAMPERED")


st.title("FaceProof")
st.caption("Face scan -> real web/social post -> tamper-evident blockchain record.")

with st.sidebar:
    _sidebar()

run_tab, verify_tab = st.tabs(["Run", "Verify"])
with run_tab:
    _run_tab()
with verify_tab:
    _verify_tab()
