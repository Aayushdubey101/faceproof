"""Offline checks for the FaceProof layer. No network, no TensorFlow."""

import json
import os

import pytest
from PIL import Image

from faceproof import pipeline
from faceproof.blockchain import registry
from faceproof.blockchain import verifier as chain_verifier
from faceproof.discovery import DiscoveryError, retrieval
from faceproof.discovery import reverse_search as reverse_search_module
from faceproof.discovery.candidates import Candidate, parse_visual_matches
from faceproof.discovery.reverse_search import redact
from faceproof.evidence import hashing, manifest
from faceproof.matching import verifier as face_verifier

SAMPLE_LENS_RESPONSE = {
    "search_parameters": {"engine": "google_lens", "api_key": "secret-value"},
    "search_metadata": {"id": "abc123"},
    "visual_matches": [
        {
            "position": 1,
            "title": "Some blog post",
            "link": "https://example.com/blog/1",
            "source": "example.com",
            "thumbnail": "https://example.com/thumb1.jpg",
        },
        {
            "position": 2,
            "title": "A profile photo",
            "link": "https://www.instagram.com/p/XYZ/",
            "source": "Instagram",
            "thumbnail": "https://example.com/thumb2.jpg",
        },
        {"position": 3, "title": "no image url", "link": "https://example.com/3"},
    ],
}

SAMPLE_EVIDENCE = {
    "schema": "faceproof/v1",
    "probe": {"sha256": "a" * 64, "faces_detected": 1},
    "match": {"distance": 0.41, "threshold": 0.68, "verified": True},
}


def test_social_results_rank_first_and_incomplete_entries_drop():
    candidates = parse_visual_matches(SAMPLE_LENS_RESPONSE)

    assert len(candidates) == 2  # the entry without a thumbnail is dropped
    assert candidates[0].is_social is True
    assert candidates[0].page_url == "https://www.instagram.com/p/XYZ/"
    assert candidates[1].is_social is False


def test_redact_removes_credential_shaped_fields():
    redacted = redact(SAMPLE_LENS_RESPONSE)

    assert redacted["search_parameters"]["api_key"] == "<redacted>"
    assert "search_metadata" not in redacted
    assert SAMPLE_LENS_RESPONSE["search_parameters"]["api_key"] == "secret-value"  # not mutated


def test_fingerprint_is_key_order_independent():
    reordered = dict(reversed(list(SAMPLE_EVIDENCE.items())))

    assert hashing.fingerprint(SAMPLE_EVIDENCE) == hashing.fingerprint(reordered)
    assert len(hashing.fingerprint(SAMPLE_EVIDENCE)) == 64


def test_any_tamper_changes_the_fingerprint():
    original = hashing.fingerprint(SAMPLE_EVIDENCE)
    tampered = {**SAMPLE_EVIDENCE, "match": {**SAMPLE_EVIDENCE["match"], "distance": 0.42}}

    assert hashing.fingerprint(tampered) != original


def test_anchor_rejects_a_digest_that_is_not_32_bytes():
    try:
        registry.anchor("deadbeef")
    except ValueError as error:
        assert "32-byte" in str(error)
    else:
        raise AssertionError("short digest should have been rejected")


class _FakeEth:
    """Minimal stand-in for web3.eth - enough to build and sign one anchor tx."""

    chain_id = 31337
    gas_price = 1_000_000_000

    def __init__(self):
        from eth_account import Account

        self.account = Account()
        self.estimated = None
        self.raw_sent = None

    def get_transaction_count(self, _address):
        return 7

    def estimate_gas(self, transaction):
        self.estimated = dict(transaction)
        return 21_512

    def send_raw_transaction(self, raw):
        from hexbytes import HexBytes

        self.raw_sent = bytes(raw)
        return HexBytes(b"" * 32)

    def wait_for_transaction_receipt(self, _tx_hash, timeout=None):
        return {"status": 1, "blockNumber": 6721043, "gasUsed": 21_512}


class _FakeWeb3:
    def __init__(self):
        self.eth = _FakeEth()

    @staticmethod
    def to_hex(value):
        return "0x" + bytes(value).hex()


def test_anchor_builds_and_signs_a_self_transaction(monkeypatch):
    """Guards against web3 / eth-account API drift - the signing path is real."""
    web3 = _FakeWeb3()
    monkeypatch.setattr(registry, "connect", lambda rpc_url=None: web3)
    digest = "b" * 64

    receipt = registry.anchor(digest, private_key="0x" + "1" * 64)

    sent = web3.eth.estimated
    assert sent["from"] == sent["to"]  # self-transaction
    assert sent["value"] == 0
    assert sent["data"].hex() == digest  # the digest IS the calldata
    assert web3.eth.raw_sent  # signing produced real bytes
    assert receipt["tx_hash"] == "0x" + "11" * 32
    assert receipt["chain_id"] == 31337
    assert receipt["block_number"] == 6721043


def test_verify_rejects_a_file_that_is_not_an_evidence_document(tmp_path):
    path = tmp_path / "not-evidence.json"
    path.write_text(json.dumps({"fingerprint": "a" * 64}), encoding="utf-8")

    try:
        pipeline.verify(str(path))
    except ValueError as error:
        assert "not-evidence.json" in str(error)
        assert "evidence" in str(error) and "anchor" in str(error)
    else:
        raise AssertionError("a document missing 'evidence' and 'anchor' should be rejected")


def test_verify_reports_tampering_when_a_hash_field_is_deleted(tmp_path, monkeypatch):
    """The demo invites a reviewer to hand-edit the JSON - a deleted key must not crash."""
    (tmp_path / "run.match.jpg").write_bytes(b"pretend-jpeg")
    search_response = {"visual_matches": []}
    evidence = {
        "post": {},  # image_sha256 deleted by hand
        "search": {
            "response_sha256": hashing.sha256_bytes(hashing.canonical_json(search_response))
        },
    }
    document = {
        "evidence": evidence,
        "fingerprint": hashing.fingerprint(evidence),
        "anchor": {"tx_hash": "0x" + "1" * 64},
        "artifacts": {"match_image": "run.match.jpg"},
        "search_response": search_response,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        pipeline.chain_verifier,
        "verify_onchain",
        lambda digest, tx_hash, rpc_url=None: {"onchain_digest": digest, "matches": True},
    )

    assert pipeline.verify(str(path)) is False  # not a KeyError


def test_read_anchor_reports_a_missing_transaction_as_a_runtime_error(monkeypatch):
    from web3.exceptions import TransactionNotFound

    class _MissingTxEth:
        def get_transaction(self, _tx_hash):
            raise TransactionNotFound("not found")

    class _MissingTxWeb3:
        eth = _MissingTxEth()

    monkeypatch.setattr(chain_verifier, "connect", lambda rpc_url=None: _MissingTxWeb3())

    try:
        chain_verifier.read_anchor("0x" + "9" * 64)
    except RuntimeError as error:
        assert "cannot read anchor transaction" in str(error)
    else:
        raise AssertionError("a missing transaction should surface as RuntimeError")


def _candidate(index: int) -> Candidate:
    return Candidate(
        title=f"candidate {index}",
        page_url=f"https://example.com/{index}",
        image_url=f"https://example.com/{index}/img.jpg",
        source="example.com",
        position=index,
        is_social=False,
    )


def _fake_download(url, dest_dir, name):
    """Every candidate downloads except #4, which is unreachable."""
    if url.startswith("https://example.com/4/"):
        return None
    path = os.path.join(dest_dir, name)
    with open(path, "wb") as handle:
        handle.write(b"jpeg-bytes")
    return path


# keyed by the file name verify_candidates gives each download
FAKE_COMPARISONS = {
    "candidate_00.jpg": {"verified": True, "distance": 0.41, "threshold": 0.68},
    "candidate_01.jpg": {"verified": False, "distance": 0.89, "threshold": 0.68},
    "candidate_02.jpg": {"verified": True, "distance": 0.30, "threshold": 0.68},
    "candidate_03.jpg": ValueError("no face detected"),
}


PROBE_EMBEDDING = [0.1] * 512


def _fake_compare(probe, candidate_path, **_kwargs):
    result = FAKE_COMPARISONS[os.path.basename(candidate_path)]
    if isinstance(result, ValueError):
        raise result
    return result


def _stub_face(monkeypatch, faces_detected=1, probes=None, scans=None):
    """Stand in for the face foundation, recording what each layer was asked for."""

    def _scan(image_path, *_args, **_kwargs):
        if scans is not None:
            scans.append(image_path)
        return {
            "faces_detected": faces_detected,
            "embedding": PROBE_EMBEDDING if faces_detected == 1 else None,
            "embedding_dimensions": 512,
            "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10},
            "model": "ArcFace",
            "detector": "retinaface",
        }

    def _compare(probe, candidate_path, **kwargs):
        if probes is not None:
            probes.append(probe)
        return _fake_compare(probe, candidate_path, **kwargs)

    monkeypatch.setattr(face_verifier, "download_image", _fake_download)
    monkeypatch.setattr(face_verifier.adapter, "scan_face", _scan)
    monkeypatch.setattr(face_verifier.adapter, "compare_faces", _compare)


def test_investigation_records_every_candidate_outcome(tmp_path, monkeypatch):
    """The UI reads these statuses - a no-face or unreachable candidate must not abort."""
    _stub_face(monkeypatch)
    candidates = [_candidate(index) for index in range(5)]
    investigation = face_verifier.Investigation(discovered=len(candidates))

    match = face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=candidates,
        work_dir=str(tmp_path),
        verbose=False,
        investigation=investigation,
    )

    assert [result.status for result in investigation.results] == [
        face_verifier.MATCH,
        face_verifier.NO_MATCH,
        face_verifier.MATCH,
        face_verifier.NO_FACE,
        face_verifier.DOWNLOAD_FAILED,
    ]
    assert (investigation.discovered, investigation.checked) == (5, 5)
    assert len(investigation.verified) == 2
    assert investigation.results[-1].image_bytes is None  # nothing was downloaded to show
    assert match is not None and match.distance == 0.30  # closest verified candidate wins


def test_the_probe_is_encoded_once_and_reused_for_every_candidate(tmp_path, monkeypatch):
    """Re-encoding the probe per candidate doubled the cost of an investigation."""
    scans, probes = [], []
    _stub_face(monkeypatch, probes=probes, scans=scans)

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(4)],
        work_dir=str(tmp_path),
        verbose=False,
    )

    assert scans == ["probe.jpg"]  # encoded once, not once per candidate
    assert probes == [PROBE_EMBEDDING] * 4  # every comparison reused that vector
    assert "probe.jpg" not in probes  # never the image path, which re-runs detection


def test_a_caller_that_already_encoded_the_probe_does_not_encode_it_again(tmp_path, monkeypatch):
    """The pipeline scans the probe at [1/5]; verification must not repeat it."""
    scans, probes = [], []
    _stub_face(monkeypatch, probes=probes, scans=scans)

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(3)],
        work_dir=str(tmp_path),
        verbose=False,
        probe_embedding=PROBE_EMBEDDING,
    )

    assert scans == []
    assert probes == [PROBE_EMBEDDING] * 3


def test_a_multi_face_probe_keeps_the_exact_image_comparison(tmp_path, monkeypatch):
    """One vector cannot stand in for several probe faces - accuracy wins over speed."""
    probes = []
    _stub_face(monkeypatch, faces_detected=2, probes=probes)

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(3)],
        work_dir=str(tmp_path),
        verbose=False,
    )

    assert probes == ["probe.jpg"] * 3


def test_the_candidate_budget_is_respected(tmp_path, monkeypatch):
    """Discovery may return dozens; only the budgeted ones are investigated."""
    investigation = face_verifier.Investigation(discovered=9)
    _stub_face(monkeypatch)

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(9)],
        work_dir=str(tmp_path),
        max_candidates=3,
        verbose=False,
        investigation=investigation,
    )

    assert investigation.checked == 3  # one result per investigated candidate
    assert investigation.discovered == 9  # what discovery found is reported unchanged


def test_the_closest_verified_candidate_wins_regardless_of_order(tmp_path, monkeypatch):
    """Selection is the minimum distance, so no early stop may pre-empt it."""
    _stub_face(monkeypatch)

    match = face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(3)],
        work_dir=str(tmp_path),
        verbose=False,
    )

    # candidate 0 verifies first at 0.41, candidate 2 verifies later at 0.30
    assert match is not None and match.distance == 0.30


def test_the_probe_embedding_never_reaches_the_evidence(tmp_path, monkeypatch):
    """scan_face now returns the vector; the hashed evidence must be unchanged."""
    _stub_face(monkeypatch)
    probe_scan = face_verifier.adapter.scan_face("probe.jpg")

    match = face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(0)],
        work_dir=str(tmp_path),
        verbose=False,
    )
    evidence = manifest.build_evidence(
        probe_digest="a" * 64,
        probe_scan=probe_scan,
        match=match,
        search_engine="serpapi/google_lens",
        probe_image_url="https://files.example/probe.jpg",
        search_response={"visual_matches": []},
        candidates_returned=1,
        candidates_checked=1,
    )

    assert set(evidence["probe"]) == {
        "sha256",
        "faces_detected",
        "embedding_model",
        "detector_backend",
        "embedding_dimensions",
    }  # the vector itself is never hashed into the evidence
    assert evidence["probe"]["embedding_dimensions"] == 512


def test_verified_and_rejected_candidates_carry_their_numbers():
    verified, rejected = (
        face_verifier.CandidateResult(_candidate(0), face_verifier.MATCH, 0.41, 0.68, b"jpeg"),
        face_verifier.CandidateResult(_candidate(1), face_verifier.NO_MATCH, 0.89, 0.68, b"jpeg"),
    )

    assert verified.verified is True
    assert (verified.distance, verified.threshold) == (0.41, 0.68)
    assert rejected.verified is False
    assert rejected.reason == "Below face-match threshold"


def test_uncompared_candidates_have_no_invented_distance():
    no_face = face_verifier.CandidateResult(_candidate(0), face_verifier.NO_FACE, image_bytes=b"j")
    unavailable = face_verifier.CandidateResult(_candidate(1), face_verifier.DOWNLOAD_FAILED)

    assert (no_face.distance, no_face.threshold) == (None, None)
    assert no_face.reason == "No detectable face"
    assert (unavailable.distance, unavailable.image_bytes) == (None, None)
    assert unavailable.reason == "Candidate image unavailable"


ANCHOR = {
    "network": "Base Sepolia",
    "chain_id": 84532,
    "tx_hash": "0x" + "7" * 64,
    "block_number": 46379081,
    "gas_used": 22250,
    "from_address": "0x" + "a" * 40,
}


def _write_evidence(tmp_path):
    """A minimal but complete evidence file, anchored and internally consistent."""
    (tmp_path / "run.match.jpg").write_bytes(b"pretend-jpeg")
    search_response = {"visual_matches": []}
    evidence = {
        "schema": "faceproof/v1",
        "discovered_at": "2026-09-04T12:54:03+00:00",
        "post": {
            "page_url": "https://www.example.com/gallery/1",
            "source": "Example",
            "title": "A discovered post",
            "image_sha256": hashing.sha256_bytes(b"pretend-jpeg"),
        },
        "match": {"model": "ArcFace", "distance": 0.065, "threshold": 0.68, "verified": True},
        "search": {
            "engine": "serpapi/google_lens",
            "response_sha256": hashing.sha256_bytes(hashing.canonical_json(search_response)),
        },
    }
    document = {
        "evidence": evidence,
        "fingerprint": hashing.fingerprint(evidence),
        "anchor": ANCHOR,
        "artifacts": {"match_image": "run.match.jpg"},
        "search_response": search_response,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return path, document


def _fake_chain(monkeypatch, onchain_digest=None):
    """Stand in for Base Sepolia - the audit must never need a live RPC in tests."""
    monkeypatch.setattr(
        pipeline.chain_verifier,
        "verify_onchain",
        lambda digest, tx_hash, rpc_url=None: {
            "onchain_digest": onchain_digest or digest,
            "matches": (onchain_digest or digest) == digest,
        },
    )


def test_audit_of_an_untouched_file_is_verified(tmp_path, monkeypatch):
    path, document = _write_evidence(tmp_path)
    _fake_chain(monkeypatch)

    report = pipeline.audit(str(path))

    assert report.local_fingerprint == report.stored_fingerprint == document["fingerprint"]
    assert report.onchain_fingerprint == report.local_fingerprint
    assert (report.local_ok, report.onchain_ok) == (True, True)
    assert (report.search_ok, report.image_ok) == (True, True)
    assert report.passed is True


def test_audit_reports_a_different_onchain_fingerprint_as_tampered(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    _fake_chain(monkeypatch, onchain_digest="c" * 64)

    report = pipeline.audit(str(path))

    assert report.local_ok is True  # the file agrees with itself
    assert report.onchain_ok is False  # but not with the chain
    assert report.passed is False


def test_audit_carries_the_evidence_and_anchor_values_the_ui_renders(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    _fake_chain(monkeypatch)

    report = pipeline.audit(str(path))
    evidence, anchor = report.document["evidence"], report.document["anchor"]

    assert evidence["post"]["page_url"] == "https://www.example.com/gallery/1"
    assert (evidence["match"]["distance"], evidence["match"]["threshold"]) == (0.065, 0.68)
    assert evidence["discovered_at"] == "2026-09-04T12:54:03+00:00"
    assert (anchor["network"], anchor["chain_id"]) == (registry.NETWORK_NAME, registry.CHAIN_ID)
    assert (anchor["tx_hash"], anchor["block_number"]) == (ANCHOR["tx_hash"], 46379081)
    assert report.tx_hash == ANCHOR["tx_hash"]


def test_tamper_simulation_changes_the_fingerprint(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    _fake_chain(monkeypatch)
    report = pipeline.audit(str(path))

    modified, digest = pipeline.tamper(report.document["evidence"])

    assert modified["post"]["title"].endswith(pipeline.TAMPER_SUFFIX)
    assert digest != report.local_fingerprint
    assert digest != report.onchain_fingerprint  # the demo verdict: TAMPERED


def test_tamper_simulation_leaves_the_evidence_file_untouched(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    before = path.read_bytes()
    _fake_chain(monkeypatch)
    report = pipeline.audit(str(path))

    pipeline.tamper(report.document["evidence"])

    assert path.read_bytes() == before  # nothing was written back
    assert report.document["evidence"]["post"]["title"] == "A discovered post"  # not mutated
    assert pipeline.audit(str(path)).passed is True  # still verifies afterwards


def test_streamlit_app_module_is_importable_and_valid():
    import streamlit  # noqa: F401  - the UI dependency must be installed

    with open("app.py", encoding="utf-8") as handle:
        compile(handle.read(), "app.py", "exec")


# --- probe upload and optimization ------------------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, text="https://files.example/abc.jpg"):
        self.status_code = status_code
        self.text = text
        self.ok = 200 <= status_code < 300


class _FakePost:
    """Stands in for requests.post and records what was sent."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _probe_file(tmp_path, size=(320, 240), name="probe.jpg"):
    path = tmp_path / name
    Image.new("RGB", size, (140, 90, 60)).save(str(path), format="JPEG")
    return path


def _no_sleep(monkeypatch):
    monkeypatch.setattr(retrieval.time, "sleep", lambda _seconds: None)


def test_upload_reports_a_disabled_provider_instead_of_hanging(tmp_path, monkeypatch):
    """0x0.st answers 503 "uploads disabled" - the UI must say so, in plain words."""
    _no_sleep(monkeypatch)
    post = _FakePost(_FakeResponse(503, "uploads disabled"))
    monkeypatch.setattr(retrieval.requests, "post", post)

    with pytest.raises(DiscoveryError) as raised:
        retrieval.upload_probe(str(_probe_file(tmp_path)), "0x0")

    assert raised.value.provider == "0x0"
    assert "503" in raised.value.reason and "uploads disabled" in raised.value.reason
    assert len(post.calls) == retrieval.UPLOAD_ATTEMPTS  # bounded, not endless


def test_upload_never_contacts_a_provider_that_was_not_selected(tmp_path, monkeypatch):
    """No silent fallback chain: a dead host must not cost the demo a timeout."""
    _no_sleep(monkeypatch)
    post = _FakePost(_FakeResponse(503, "uploads disabled"))
    monkeypatch.setattr(retrieval.requests, "post", post)

    with pytest.raises(DiscoveryError):
        retrieval.upload_probe(str(_probe_file(tmp_path)), "catbox")

    endpoints = {endpoint for endpoint, _kwargs in post.calls}
    assert endpoints == {retrieval.PROVIDERS["catbox"]["endpoint"]}


def test_upload_timeout_is_reported_with_its_limits(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        retrieval.requests, "post", _FakePost(retrieval.requests.Timeout("read timed out"))
    )

    with pytest.raises(DiscoveryError) as raised:
        retrieval.upload_probe(str(_probe_file(tmp_path)), "catbox")

    assert "timed out" in raised.value.reason
    assert raised.value.timeout == retrieval.TIMEOUT_NOTE


def test_upload_returns_the_public_url_the_provider_answered(tmp_path, monkeypatch):
    post = _FakePost(_FakeResponse(200, "https://files.catbox.moe/abc.jpg\n"))
    monkeypatch.setattr(retrieval.requests, "post", post)

    url = retrieval.upload_probe(str(_probe_file(tmp_path)), "catbox")

    assert url == "https://files.catbox.moe/abc.jpg"
    endpoint, kwargs = post.calls[0]
    assert endpoint == retrieval.PROVIDERS["catbox"]["endpoint"]
    assert kwargs["timeout"] == retrieval.UPLOAD_TIMEOUT
    assert "fileToUpload" in kwargs["files"]


def test_unknown_provider_names_the_ones_that_exist(tmp_path):
    with pytest.raises(DiscoveryError) as raised:
        retrieval.upload_probe(str(_probe_file(tmp_path)), "nowhere")

    assert retrieval.PROVIDER_ENV in raised.value.reason
    assert "catbox" in raised.value.reason


def test_selected_provider_follows_the_environment(monkeypatch):
    monkeypatch.delenv(retrieval.PROVIDER_ENV, raising=False)
    assert retrieval.selected_provider() == retrieval.DEFAULT_PROVIDER

    monkeypatch.setenv(retrieval.PROVIDER_ENV, " Litterbox ")
    assert retrieval.selected_provider() == "litterbox"


def test_large_probe_is_shrunk_for_upload(tmp_path):
    """A phone-sized photo is too slow to publish; the copy must be small and upright."""
    original = tmp_path / "big.jpg"
    Image.effect_mandelbrot((4000, 3000), (-3, -2.5, 2, 2.5), 40).convert("RGB").save(
        str(original), format="JPEG", quality=95
    )
    dest = tmp_path / "out"
    dest.mkdir()

    optimized = retrieval.optimize_probe(str(original), str(dest))

    with Image.open(optimized) as image:
        assert max(image.size) == retrieval.MAX_PROBE_PIXELS
        assert image.format == "JPEG"
        assert not image.getexif()  # metadata stripped
    assert os.path.getsize(optimized) < os.path.getsize(str(original)) / 2


def test_optimizing_never_touches_the_original(tmp_path):
    """The pipeline hashes and matches against the original - it must stay byte-identical."""
    original = _probe_file(tmp_path, size=(2400, 1800), name="original.jpg")
    before = original.read_bytes()
    dest = tmp_path / "out"
    dest.mkdir()

    optimized = retrieval.optimize_probe(str(original), str(dest))

    assert original.read_bytes() == before
    assert hashing.sha256_file(str(original)) == hashing.sha256_bytes(before)
    assert os.path.dirname(optimized) == str(dest)


# --- search and error hygiene -----------------------------------------------


class _FakeGet:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def __call__(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        if self.error is not None:
            raise self.error
        return self

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_search_asks_the_engine_about_the_uploaded_probe_url(monkeypatch):
    """Discovery stays genuine: the published probe URL is what gets searched."""
    get = _FakeGet(payload=SAMPLE_LENS_RESPONSE)
    monkeypatch.setattr(reverse_search_module.requests, "get", get)

    candidates, raw = reverse_search_module.reverse_search(
        "https://files.catbox.moe/abc.jpg", api_key="secret-value"
    )

    endpoint, kwargs = get.calls[0]
    assert endpoint == reverse_search_module.SERPAPI_ENDPOINT
    assert kwargs["params"]["url"] == "https://files.catbox.moe/abc.jpg"
    assert kwargs["params"]["engine"] == "google_lens"
    assert kwargs["timeout"] == reverse_search_module.SEARCH_TIMEOUT
    assert len(candidates) == 2
    assert raw["search_parameters"]["api_key"] == "<redacted>"


def test_search_failures_never_carry_the_api_key(monkeypatch):
    """str(error) and error.request.url both embed the key - neither may reach the UI."""
    leaky = reverse_search_module.requests.HTTPError(
        "401 for url: https://serpapi.com/search?api_key=secret-value"
    )
    leaky.response = _FakeResponse(401, "")
    monkeypatch.setattr(reverse_search_module.requests, "get", _FakeGet(error=leaky))

    with pytest.raises(DiscoveryError) as raised:
        reverse_search_module.reverse_search(
            "https://files.example/abc.jpg", api_key="secret-value"
        )

    assert "secret-value" not in str(raised.value)
    assert "secret-value" not in raised.value.reason
    assert "SERPAPI_API_KEY" in raised.value.reason


def test_download_failure_does_not_end_the_investigation(tmp_path, monkeypatch):
    """One unreachable thumbnail must not cost the run its other candidates."""
    _stub_face(monkeypatch)
    investigation = face_verifier.Investigation()

    match = face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(5)],
        work_dir=str(tmp_path / "work"),
        verbose=False,
        investigation=investigation,
    )

    failed = [r for r in investigation.results if r.status == face_verifier.DOWNLOAD_FAILED]
    assert len(failed) == 1 and failed[0].candidate.page_url == "https://example.com/4"
    assert failed[0].reason == "Candidate image unavailable"
    assert failed[0].distance is None
    assert investigation.checked == 5  # every candidate still reached a conclusion
    assert match is not None and match.distance == 0.30  # the best one still wins


def test_download_returns_none_when_the_host_fails(tmp_path, monkeypatch):
    def _boom(url, **_kwargs):
        raise retrieval.requests.ConnectionError("refused")

    monkeypatch.setattr(retrieval.requests, "get", _boom)

    assert retrieval.download_image("https://example.com/x.jpg", str(tmp_path), "c.jpg") is None
