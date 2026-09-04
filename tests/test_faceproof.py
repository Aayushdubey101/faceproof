"""Offline checks for the FaceProof layer. No network, no TensorFlow."""

import json

from faceproof import pipeline
from faceproof.blockchain import registry
from faceproof.blockchain import verifier as chain_verifier
from faceproof.discovery.candidates import parse_visual_matches
from faceproof.discovery.reverse_search import redact
from faceproof.evidence import hashing

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
