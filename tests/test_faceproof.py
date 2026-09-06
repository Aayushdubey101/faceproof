"""Offline checks for the FaceProof layer. No network, no TensorFlow."""

import json
import os
import threading
import time

import pytest
from PIL import Image

import app

from faceproof import pipeline
from faceproof.blockchain import registry
from faceproof.blockchain import verifier as chain_verifier
from faceproof.discovery import DiscoveryError, retrieval
from faceproof.discovery import reverse_search as reverse_search_module
from faceproof.discovery.candidates import Candidate, discovered_count, parse_visual_matches
from faceproof.discovery.normalize import canonical_url
from faceproof.discovery.reverse_search import redact
from faceproof.evidence import hashing, manifest, provenance
from faceproof.face import adapter
from faceproof.matching import (
    MatchExplanation,
    distance_scale_percent,
    explain_candidate,
    explain_dict,
    explain_match,
    explain_result,
)
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


def _visual_match(position, link, thumbnail=None, source="example.com", **extra):
    """One entry shaped like a Google Lens result."""
    item = {"position": position, "title": f"result {position}", "link": link, "source": source}
    if thumbnail is not None:
        item["thumbnail"] = thumbnail
    item.update(extra)
    return item


def test_canonical_url_normalizes_only_what_is_safe():
    assert (
        canonical_url("HTTPS://Example.com:443/post/?b=2&a=1#anchor")
        == "https://example.com/post?a=1&b=2"
    )
    assert canonical_url("http://Example.com:80/a%2fb") == "http://example.com/a%2Fb"
    # a hashbang carries the route, so it is content and survives
    assert canonical_url("https://example.com/spa#!/photo/7") == "https://example.com/spa#!/photo/7"
    assert canonical_url("not a url at all") == "not a url at all"
    assert canonical_url("") == ""


def test_tracking_parameters_are_removed_and_content_parameters_are_not():
    tagged = "https://example.com/post?id=7&utm_source=x&fbclid=abc&igshid=zz"

    assert canonical_url(tagged) == "https://example.com/post?id=7"
    assert canonical_url("https://www.youtube.com/watch?v=abc") == (
        "https://www.youtube.com/watch?v=abc"
    )


def test_two_posts_that_differ_only_by_a_content_parameter_stay_separate():
    payload = {
        "visual_matches": [
            _visual_match(1, "https://example.com/post?id=123", "https://cdn.example.com/1.jpg"),
            _visual_match(2, "https://example.com/post?id=456", "https://cdn.example.com/2.jpg"),
        ]
    }

    assert len(parse_visual_matches(payload)) == 2  # different posts, never merged


def test_the_same_page_reached_through_a_tracking_link_is_one_candidate():
    payload = {
        "visual_matches": [
            _visual_match(1, "https://www.instagram.com/p/XYZ/", "https://cdn.example.com/1.jpg"),
            _visual_match(
                2,
                "https://www.instagram.com/p/XYZ/?utm_source=share",
                "https://cdn.example.com/2.jpg",
            ),
        ]
    }

    candidates = parse_visual_matches(payload)

    assert len(candidates) == 1
    assert candidates[0].page_url == "https://www.instagram.com/p/XYZ/"  # the original, not the key
    assert candidates[0].duplicates == ("https://www.instagram.com/p/XYZ/?utm_source=share",)
    assert discovered_count(candidates) == 2  # the search still returned two results


def test_a_duplicate_image_keeps_the_better_source_and_the_url_it_replaced():
    shared = "https://cdn.example.com/photo.jpg"
    payload = {
        "visual_matches": [
            _visual_match(1, "https://blog.example.com/post", shared, source="blog"),
            _visual_match(2, "https://www.instagram.com/p/XYZ/", shared, source="Instagram"),
        ]
    }

    candidates = parse_visual_matches(payload)

    assert len(candidates) == 1
    # the social post wins the merge despite being listed second, and the page
    # it replaced is kept rather than deleted
    assert candidates[0].page_url == "https://www.instagram.com/p/XYZ/"
    assert candidates[0].duplicates == ("https://blog.example.com/post",)


def test_ranking_is_deterministic_whatever_order_the_results_arrive_in():
    entries = [
        _visual_match(4, "https://example.com/d", "https://cdn.example.com/d.jpg"),
        _visual_match(1, "https://www.instagram.com/p/A/", "https://cdn.example.com/a.jpg"),
        _visual_match(9, "https://example.com/c", "https://cdn.example.com/c.jpg"),
        _visual_match(2, "https://www.tiktok.com/@b", "https://cdn.example.com/b.jpg"),
    ]

    ordered = [one.page_url for one in parse_visual_matches({"visual_matches": entries})]
    again = [one.page_url for one in parse_visual_matches({"visual_matches": entries})]
    shuffled = [
        one.page_url for one in parse_visual_matches({"visual_matches": list(reversed(entries))})
    ]

    assert ordered == again == shuffled  # no randomness, and no dependence on arrival order


def test_a_social_post_outranks_a_better_placed_web_page():
    payload = {
        "visual_matches": [
            _visual_match(1, "https://blog.example.com/a", "https://cdn.example.com/a.jpg"),
            _visual_match(9, "https://www.instagram.com/p/A/", "https://cdn.example.com/b.jpg"),
        ]
    }

    candidates = parse_visual_matches(payload)

    assert [one.is_social for one in candidates] == [True, False]
    assert candidates[0].priority_score > candidates[1].priority_score
    assert "social source" in candidates[0].priority_reasons


def test_a_provider_cached_thumbnail_outranks_an_origin_only_image():
    payload = {
        "visual_matches": [
            {
                "position": 1,
                "title": "origin only",
                "link": "https://a.example.com/1",
                "source": "a.example.com",
                "image": "https://origin.example.com/1.jpg",
            },
            _visual_match(2, "https://b.example.com/2", "https://cdn.example.com/2.jpg"),
        ]
    }

    candidates = parse_visual_matches(payload)

    assert candidates[0].page_url == "https://b.example.com/2"
    assert "cached thumbnail" in candidates[0].priority_reasons
    # the origin-only result is still investigated, just later - never dropped
    assert candidates[1].image_url == "https://origin.example.com/1.jpg"


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


def test_candidates_outside_the_budget_are_uninvestigated_not_rejected(tmp_path, monkeypatch):
    """A candidate the budget never reached has no verdict, and must not be given one."""
    _stub_face(monkeypatch)
    investigation = face_verifier.Investigation(discovered=4, unique=4)

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(4)],
        work_dir=str(tmp_path),
        max_candidates=2,
        verbose=False,
        investigation=investigation,
    )

    investigated = {result.candidate.page_url for result in investigation.results}
    skipped = [one.page_url for one in investigation.not_investigated]
    assert skipped == ["https://example.com/2", "https://example.com/3"]
    assert not investigated & set(skipped)  # never reported as NO_MATCH, or as anything


def _download_recording_url(url, dest_dir, name):
    """Store the URL as the file body, and fail candidate 4 the way a dead host does."""
    if _url_index(url) == 4:
        return None
    path = os.path.join(dest_dir, name)
    with open(path, "wb") as handle:
        handle.write(url.encode("utf-8"))
    return path


def _verdict_by_url(probe, candidate_path, **_kwargs):
    """The verdict follows the candidate's own image, never its slot in the run."""
    with open(candidate_path, "rb") as handle:
        index = _url_index(handle.read().decode("utf-8"))
    if index == 3:
        raise ValueError("no face detected")
    return {"verified": index % 2 == 0, "distance": index / 10, "threshold": 0.68}


def test_reordering_candidates_changes_no_verification_verdict(tmp_path, monkeypatch):
    """Ranking decides what is looked at, never what the comparison concludes."""
    _stub_face(monkeypatch)
    monkeypatch.setattr(face_verifier, "download_image", _download_recording_url)
    monkeypatch.setattr(face_verifier.adapter, "compare_faces", _verdict_by_url)
    candidates = [_candidate(index) for index in range(6)]

    def verdicts(ordered):
        investigation = face_verifier.Investigation(discovered=len(ordered))
        face_verifier.verify_candidates(
            probe_path="probe.jpg",
            candidates=ordered,
            work_dir=str(tmp_path),
            verbose=False,
            investigation=investigation,
        )
        return {
            result.candidate.page_url: (result.status, result.distance)
            for result in investigation.results
        }

    listed = verdicts(candidates)
    reranked = verdicts(list(reversed(candidates)))

    assert listed == reranked
    assert listed["https://example.com/0"] == (face_verifier.MATCH, 0.0)
    assert listed["https://example.com/1"] == (face_verifier.NO_MATCH, 0.1)
    assert listed["https://example.com/3"] == (face_verifier.NO_FACE, None)
    assert listed["https://example.com/4"] == (face_verifier.DOWNLOAD_FAILED, None)


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


def _url_index(url: str) -> int:
    """The candidate index encoded in a fake candidate URL."""
    return int(url.split("/")[-2])


def _staggered_download(delays):
    """Downloads that finish out of order: the file records which URL produced it."""

    def download(url, dest_dir, name):
        time.sleep(delays[_url_index(url)])
        path = os.path.join(dest_dir, name)
        with open(path, "wb") as handle:
            handle.write(url.encode("utf-8"))
        return path

    return download


def _compare_by_content(probe, candidate_path, **_kwargs):
    """Derive the distance from the downloaded bytes, so a swapped file shows up."""
    with open(candidate_path, "rb") as handle:
        index = _url_index(handle.read().decode("utf-8"))
    return {"verified": True, "distance": index / 10, "threshold": 0.68}


def test_out_of_order_downloads_keep_candidate_order_and_identity(tmp_path, monkeypatch):
    """Concurrent fetching must not reorder candidates or pair one with another's image."""
    _stub_face(monkeypatch)
    candidates = [_candidate(index) for index in range(6)]
    # the last candidate returns first, the first one last
    monkeypatch.setattr(
        face_verifier,
        "download_image",
        _staggered_download({index: (6 - index) * 0.02 for index in range(6)}),
    )
    monkeypatch.setattr(face_verifier.adapter, "compare_faces", _compare_by_content)
    investigation = face_verifier.Investigation(discovered=len(candidates))

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=candidates,
        work_dir=str(tmp_path),
        verbose=False,
        investigation=investigation,
    )

    results = investigation.results
    assert [result.candidate.page_url for result in results] == [c.page_url for c in candidates]
    assert [result.distance for result in results] == [index / 10 for index in range(6)]
    # the bytes recorded for a candidate came from that candidate's own image URL
    assert [result.image_bytes.decode() for result in results] == [c.image_url for c in candidates]


def test_downloads_never_exceed_the_configured_worker_count(tmp_path, monkeypatch):
    """Bounded concurrency: a large candidate budget must not open a socket per candidate."""
    _stub_face(monkeypatch)
    monkeypatch.setenv(face_verifier.DOWNLOAD_WORKERS_ENV, "2")
    lock = threading.Lock()
    live = peak = 0

    def _download(url, dest_dir, name):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        try:
            time.sleep(0.02)
            return _fake_download(url, dest_dir, name)
        finally:
            with lock:
                live -= 1

    monkeypatch.setattr(face_verifier, "download_image", _download)

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=[_candidate(index) for index in range(4)],
        work_dir=str(tmp_path),
        verbose=False,
    )

    assert peak == 2 and live == 0


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, face_verifier.DEFAULT_DOWNLOAD_WORKERS),
        ("", face_verifier.DEFAULT_DOWNLOAD_WORKERS),
        ("nonsense", face_verifier.DEFAULT_DOWNLOAD_WORKERS),
        ("1", 1),
        ("8", 8),
        ("64", face_verifier.MAX_DOWNLOAD_WORKERS),  # never unlimited threads
        ("0", 1),
        ("-3", 1),
    ],
)
def test_the_download_worker_count_stays_within_bounds(monkeypatch, configured, expected):
    monkeypatch.delenv(face_verifier.DOWNLOAD_WORKERS_ENV, raising=False)
    if configured is not None:
        monkeypatch.setenv(face_verifier.DOWNLOAD_WORKERS_ENV, configured)

    assert face_verifier.download_workers() == expected


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, face_verifier.DEFAULT_COMPARE_WORKERS),
        ("nonsense", face_verifier.DEFAULT_COMPARE_WORKERS),
        ("1", 1),
        ("4", 4),
        ("64", face_verifier.MAX_COMPARE_WORKERS),  # inference is memory-hungry, never unlimited
        ("0", 1),
    ],
)
def test_the_compare_worker_count_stays_within_bounds(monkeypatch, configured, expected):
    monkeypatch.delenv(face_verifier.COMPARE_WORKERS_ENV, raising=False)
    if configured is not None:
        monkeypatch.setenv(face_verifier.COMPARE_WORKERS_ENV, configured)

    assert face_verifier.compare_workers() == expected


def _staggered_compare(delays):
    """Comparisons that finish out of order; the verdict follows the image bytes."""

    def compare(probe, candidate_path, **_kwargs):
        with open(candidate_path, "rb") as handle:
            index = _url_index(handle.read().decode("utf-8"))
        time.sleep(delays[index])
        return {"verified": index % 2 == 0, "distance": index / 10, "threshold": 0.68}

    return compare


@pytest.mark.parametrize("workers", ["1", "2", "4"])
def test_concurrent_comparisons_keep_candidate_order_and_identity(tmp_path, monkeypatch, workers):
    """However many comparisons run at once, a verdict belongs to its own candidate."""
    _stub_face(monkeypatch)
    monkeypatch.setenv(face_verifier.COMPARE_WORKERS_ENV, workers)
    candidates = [_candidate(index) for index in range(6)]
    monkeypatch.setattr(face_verifier, "download_image", _download_recording_url)
    # the last candidate finishes comparing first, the first one last
    monkeypatch.setattr(
        face_verifier.adapter,
        "compare_faces",
        _staggered_compare({index: (6 - index) * 0.02 for index in range(6)}),
    )
    investigation = face_verifier.Investigation(discovered=len(candidates))

    match = face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=candidates,
        work_dir=str(tmp_path),
        verbose=False,
        investigation=investigation,
    )

    results = investigation.results
    # candidate 4 never downloads, so it is DOWNLOAD_FAILED and never compared
    assert [result.candidate.page_url for result in results] == [c.page_url for c in candidates]
    assert [result.distance for result in results] == [0.0, 0.1, 0.2, 0.3, None, 0.5]
    assert [result.status for result in results] == [
        face_verifier.MATCH,
        face_verifier.NO_MATCH,
        face_verifier.MATCH,
        face_verifier.NO_MATCH,
        face_verifier.DOWNLOAD_FAILED,
        face_verifier.NO_MATCH,
    ]
    assert match is not None and match.distance == 0.0  # the closest, not the first to finish


def test_every_investigated_candidate_is_compared_exactly_once(tmp_path, monkeypatch):
    """No candidate pays for RetinaFace twice inside one investigation."""
    _stub_face(monkeypatch)
    monkeypatch.setattr(face_verifier, "download_image", _download_recording_url)
    compared = []

    def _compare(probe, candidate_path, **kwargs):
        with open(candidate_path, "rb") as handle:
            compared.append(handle.read().decode("utf-8"))
        return _verdict_by_url(probe, candidate_path, **kwargs)

    monkeypatch.setattr(face_verifier.adapter, "compare_faces", _compare)
    candidates = [_candidate(index) for index in range(6)]
    investigation = face_verifier.Investigation(discovered=len(candidates))

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=candidates,
        work_dir=str(tmp_path),
        max_candidates=5,
        verbose=False,
        investigation=investigation,
    )

    # five within the budget, one of which never downloaded, and no repeats
    assert len(compared) == len(set(compared)) == 4
    assert investigation.checked == 5


def test_the_same_image_url_is_fetched_once_and_every_candidate_still_reported(
    tmp_path, monkeypatch
):
    """Two posts can carry the identical thumbnail URL - one fetch, two verdicts."""
    _stub_face(monkeypatch)
    requested = []

    def _download(url, dest_dir, name):
        requested.append(url)
        return _fake_download(url, dest_dir, name)

    monkeypatch.setattr(face_verifier, "download_image", _download)
    reposted = Candidate(
        title="a repost of candidate 0",
        page_url="https://example.com/repost",
        image_url=_candidate(0).image_url,  # a duplicate is an exact image-URL match
        source="repost.example.com",
        position=9,
        is_social=True,
    )
    candidates = [_candidate(0), _candidate(1), reposted]
    investigation = face_verifier.Investigation(discovered=len(candidates))

    face_verifier.verify_candidates(
        probe_path="probe.jpg",
        candidates=candidates,
        work_dir=str(tmp_path),
        verbose=False,
        investigation=investigation,
    )

    assert requested == [_candidate(0).image_url, _candidate(1).image_url]  # fetched once
    assert [result.candidate.page_url for result in investigation.results] == [
        "https://example.com/0",
        "https://example.com/1",
        "https://example.com/repost",
    ]
    assert [result.status for result in investigation.results] == [
        face_verifier.MATCH,
        face_verifier.NO_MATCH,
        face_verifier.MATCH,  # the repost is judged on its own, from the shared image
    ]
    assert investigation.results[2].image_bytes == investigation.results[0].image_bytes


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
        "probe": {
            "sha256": "a" * 64,
            "faces_detected": 1,
            "embedding_model": "ArcFace",
            "detector_backend": "retinaface",
            "embedding_dimensions": 512,
        },
        "post": {
            "page_url": "https://www.example.com/gallery/1",
            "source": "Example",
            "title": "A discovered post",
            "image_sha256": hashing.sha256_bytes(b"pretend-jpeg"),
        },
        "match": {
            "model": "ArcFace",
            "distance_metric": "cosine",
            "distance": 0.065,
            "threshold": 0.68,
            "verified": True,
        },
        "search": {
            "engine": "serpapi/google_lens",
            "response_sha256": hashing.sha256_bytes(hashing.canonical_json(search_response)),
            "candidates_returned": 12,
            "candidates_checked": 5,
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


def test_flask_app_serves_both_pages():
    import app as web

    web.app.config["TESTING"] = True
    with web.app.test_client() as client:
        assert client.get("/").status_code == 200
        assert client.get("/verify").status_code == 200
        assert client.get("/evidence").status_code == 200
        assert client.get("/api/run").get_json()["state"] in ("idle", "running", "done", "failed")


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


# ------------------------------------------------------- provenance timeline


def _stages(events):
    return [event.stage for event in events]


def _by_stage(events, stage):
    return next(event for event in events if event.stage == stage)


def test_timeline_covers_every_stage_in_order(tmp_path):
    _, document = _write_evidence(tmp_path)

    events = provenance.timeline(document)

    assert _stages(events) == list(provenance.STAGES)
    assert all(isinstance(event, provenance.InvestigationEvent) for event in events)


def test_a_complete_run_reports_every_stage_complete_except_the_audit(tmp_path):
    _, document = _write_evidence(tmp_path)

    events = provenance.timeline(document)

    assert [event.status for event in events[:-1]] == [provenance.COMPLETE] * 7
    # the run that wrote the evidence cannot audit itself
    assert _by_stage(events, provenance.INDEPENDENT_AUDIT).status == provenance.PENDING


def test_a_broken_document_reports_the_stage_that_broke(tmp_path):
    _, document = _write_evidence(tmp_path)
    document["evidence"]["match"]["verified"] = False  # also breaks the fingerprint
    document["evidence"]["search"]["candidates_returned"] = 0
    document["anchor"] = {}

    events = provenance.timeline(document)
    statuses = {event.stage: event.status for event in events}

    assert statuses[provenance.WEB_DISCOVERY] == provenance.FAILED
    assert statuses[provenance.FACE_VERIFICATION] == provenance.FAILED
    assert statuses[provenance.HASH_GENERATED] == provenance.FAILED
    assert statuses[provenance.BLOCKCHAIN_ANCHORED] == provenance.FAILED
    # nothing was quietly promoted to a success
    assert provenance.VERIFIED not in statuses.values()


def test_a_verified_audit_closes_the_chain(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    _fake_chain(monkeypatch)
    report = pipeline.audit(str(path))

    audit_event = _by_stage(
        provenance.timeline(report.document, report), provenance.INDEPENDENT_AUDIT
    )

    assert audit_event.status == provenance.VERIFIED
    assert audit_event.metadata["On-chain digest"] == report.onchain_fingerprint
    assert audit_event.metadata["Local fingerprint"] == report.local_fingerprint


def test_a_failed_audit_closes_the_chain_as_tampered(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    _fake_chain(monkeypatch, onchain_digest="c" * 64)
    report = pipeline.audit(str(path))

    audit_event = _by_stage(
        provenance.timeline(report.document, report), provenance.INDEPENDENT_AUDIT
    )

    assert audit_event.status == provenance.TAMPERED
    assert audit_event.metadata["On-chain digest"] == "c" * 64


def test_only_the_stage_the_document_timestamps_carries_a_time(tmp_path):
    _, document = _write_evidence(tmp_path)

    events = provenance.timeline(document)

    stamped = {event.stage: event.timestamp for event in events if event.timestamp}
    assert stamped == {provenance.EVIDENCE_CREATED: "2026-09-04T12:54:03+00:00"}


def test_summary_reads_the_document_and_invents_nothing(tmp_path):
    _, document = _write_evidence(tmp_path)

    summary = provenance.summary(document)

    assert summary["fingerprint"] == document["fingerprint"]
    assert summary["matched_url"] == "https://www.example.com/gallery/1"
    assert summary["distance"] == 0.065
    assert summary["threshold"] == 0.68
    assert summary["network"] == ANCHOR["network"]
    assert summary["candidates_discovered"] == 12
    assert summary["candidates_investigated"] == 5
    # a field the document does not carry produces no key at all, not a placeholder
    assert "local_fingerprint" not in summary
    assert "audit_result" not in summary  # no audit was passed in


def test_summary_carries_the_audit_verdict_when_one_ran(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    _fake_chain(monkeypatch)
    report = pipeline.audit(str(path))

    summary = provenance.summary(report.document, report)

    assert summary["audit_result"] == provenance.VERIFIED
    assert summary["local_fingerprint"] == summary["onchain_digest"]


# ------------------------------------------------------- integrity rendering


def _verify_html(client_payload, tmp_path, monkeypatch, evidence_dir):
    """Render the verify fragment the way /api/verify does."""
    monkeypatch.setattr(pipeline, "EVIDENCE_DIR", str(evidence_dir))
    with app.app.test_client() as client:
        response = client.post("/api/verify", json=client_payload)
    return response


def test_matching_digests_render_as_integrity_verified(tmp_path, monkeypatch):
    _write_evidence(tmp_path)
    _fake_chain(monkeypatch)

    response = _verify_html({"file": "run.json"}, tmp_path, monkeypatch, tmp_path)
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["local"] == payload["onchain"]
    assert "INTEGRITY VERIFIED" in payload["html"]
    assert "TAMPER DETECTED" not in payload["html"]
    assert ">VERIFIED<" in payload["html"]


def test_differing_digests_render_as_tamper_detected(tmp_path, monkeypatch):
    _write_evidence(tmp_path)
    _fake_chain(monkeypatch, onchain_digest="c" * 64)

    response = _verify_html({"file": "run.json"}, tmp_path, monkeypatch, tmp_path)
    payload = response.get_json()

    assert payload["ok"] is False
    assert payload["local"] != payload["onchain"]
    assert "TAMPER DETECTED" in payload["html"]
    assert "INTEGRITY VERIFIED" not in payload["html"]


def test_tamper_simulation_reports_tampered_without_touching_the_file(tmp_path, monkeypatch):
    path, document = _write_evidence(tmp_path)
    before = path.read_text(encoding="utf-8")
    _fake_chain(monkeypatch)

    response = _verify_html({"file": "run.json", "tamper": True}, tmp_path, monkeypatch, tmp_path)
    payload = response.get_json()

    assert payload["tampered"] is True and payload["ok"] is False
    assert payload["local"] != payload["original"] == document["fingerprint"]
    assert "TAMPER DETECTED" in payload["html"]
    assert "Restore Original" in payload["html"]  # the way back is offered
    assert path.read_text(encoding="utf-8") == before  # nothing on disk changed


# ------------------------------------------------------- explainable matching


def test_match_explanation_generation():
    exp = explain_candidate("MATCH", distance=0.42, threshold=0.68)

    assert exp.result == "MATCH"
    assert exp.passed is True
    assert exp.distance == 0.42
    assert exp.threshold == 0.68
    assert exp.margin == 0.26
    assert exp.margin_display == "+0.26"
    assert "0.42" in exp.decision_rule and "0.68" in exp.decision_rule
    assert exp.comparison == "distance <= threshold"
    assert any("Face detected" in r for r in exp.reasons)
    assert any("ArcFace" in r for r in exp.reasons)
    assert any("threshold" in r.lower() for r in exp.reasons)
    assert all(ok for ok, _ in exp.detailed_reasons)
    assert exp.trace[-1] == ("07", "Decision: MATCH", True)


def test_no_match_explanation_generation():
    exp = explain_candidate("NO_MATCH", distance=0.74, threshold=0.68)

    assert exp.result == "NO_MATCH"
    assert exp.passed is False
    assert exp.distance == 0.74
    assert exp.threshold == 0.68
    assert exp.margin == -0.06
    assert exp.margin_display == "-0.06"
    assert "0.74" in exp.decision_rule and "0.68" in exp.decision_rule
    assert exp.comparison == "distance > threshold"
    assert any("Face detected" in r for r in exp.reasons)
    assert any("exceeds" in r for r in exp.reasons)
    assert any(not ok for ok, _ in exp.detailed_reasons)
    assert exp.trace[-1] == ("07", "Decision: NO MATCH", False)


def test_no_face_explanation_has_no_distance_and_correct_semantics():
    exp = explain_candidate("NO_FACE")

    assert exp.result == "NO_FACE"
    assert exp.passed is False
    assert exp.distance is None
    assert exp.margin is None
    assert exp.margin_display == "—"
    assert exp.reasons == ["No usable face detected in candidate"]
    # must not claim non-identity
    for r in exp.reasons:
        assert "different person" not in r.lower()
        assert "not the same person" not in r.lower()
    # must not claim ArcFace embedding ran
    for _, desc, _ in exp.trace:
        assert "ArcFace" not in desc


def test_download_failed_explanation_has_no_distance_and_correct_semantics():
    exp = explain_candidate("DOWNLOAD_FAILED")

    assert exp.result == "DOWNLOAD_FAILED"
    assert exp.passed is False
    assert exp.distance is None
    assert exp.margin is None
    assert exp.margin_display == "—"
    assert exp.reasons == ["Candidate image could not be downloaded"]
    # must not claim non-identity
    for r in exp.reasons:
        assert "different person" not in r.lower()
        assert "not the same person" not in r.lower()
    # must not claim face detection or ArcFace ran
    for _, desc, _ in exp.trace:
        assert "Face detected" not in desc
        assert "ArcFace" not in desc


def test_positive_margin_calculation():
    exp = explain_candidate("MATCH", distance=0.40, threshold=0.68)
    assert exp.margin == 0.28
    assert exp.margin_display == "+0.28"
    assert exp.margin > 0


def test_negative_margin_calculation():
    exp = explain_candidate("NO_MATCH", distance=0.75, threshold=0.68)
    assert exp.margin == -0.07
    assert exp.margin_display == "-0.07"
    assert exp.margin < 0


def test_zero_margin_calculation():
    exp = explain_candidate("MATCH", distance=0.68, threshold=0.68)
    assert exp.margin == 0.0
    assert exp.margin_display == "+0.00"
    assert exp.passed is True


def test_authoritative_threshold_from_verification_engine_is_used():
    thr = adapter.verification_threshold("ArcFace", "cosine")
    assert thr == 0.68

    exp = explain_candidate("MATCH", distance=0.50)
    assert exp.threshold == thr
    assert exp.margin == round(0.68 - 0.50, 4)


def test_explanation_does_not_hardcode_authoritative_threshold(monkeypatch):
    monkeypatch.setattr(adapter, "verification_threshold", lambda *args, **kwargs: 0.55)
    exp = explain_candidate("MATCH", distance=0.45)
    assert exp.threshold == 0.55
    assert exp.margin == 0.10


def test_ranking_score_is_not_mixed_with_face_verification():
    candidate = Candidate(
        title="Social Post",
        page_url="https://www.instagram.com/p/test",
        image_url="https://cdn.instagram.com/test.jpg",
        source="Instagram",
        position=1,
        is_social=True,
        canonical_url="https://instagram.com/p/test",
        canonical_image_url="https://cdn.instagram.com/test.jpg",
        priority_score=4.0,
        priority_reasons=["social", "thumbnail"],
    )
    result = face_verifier.CandidateResult(
        candidate=candidate,
        status="MATCH",
        distance=0.42,
        threshold=0.68,
    )
    exp = explain_result(result)
    exp_dict = exp.as_dict()

    # Ranking score must not appear in or affect face verification distance or margin
    assert "priority_score" not in exp_dict
    assert "ranking" not in exp_dict
    assert exp.distance == 0.42
    assert exp.threshold == 0.68
    assert exp.margin == 0.26


def test_no_percentage_confidence_or_probability_generated():
    exp = explain_candidate("MATCH", distance=0.42, threshold=0.68)
    serialized = json.dumps(exp.as_dict())

    for forbidden in ["confidence", "% match", "% confidence", "probability"]:
        assert forbidden not in serialized.lower()


def test_existing_evidence_fingerprint_remains_unchanged(tmp_path):
    _, document = _write_evidence(tmp_path)
    stored_fingerprint = document["fingerprint"]
    calculated_fingerprint = hashing.fingerprint(document["evidence"])
    assert calculated_fingerprint == stored_fingerprint

    # Generating an explanation from the match dictionary does not mutate the evidence
    exp = explain_dict(document["evidence"]["match"])
    assert exp.result == "MATCH"
    assert hashing.fingerprint(document["evidence"]) == stored_fingerprint


def test_existing_candidate_statuses_remain_unchanged():
    assert face_verifier.MATCH == "MATCH"
    assert face_verifier.NO_MATCH == "NO_MATCH"
    assert face_verifier.NO_FACE == "NO_FACE"
    assert face_verifier.DOWNLOAD_FAILED == "DOWNLOAD_FAILED"


def test_existing_best_match_selection_remains_unchanged():
    cand1 = _candidate(1)
    cand2 = _candidate(2)
    cand3 = _candidate(3)
    matches = [
        face_verifier.Match(
            candidate=cand1,
            image_path="1.jpg",
            image_bytes=b"",
            distance=0.65,
            threshold=0.68,
            model="ArcFace",
            metric="cosine",
        ),
        face_verifier.Match(
            candidate=cand2,
            image_path="2.jpg",
            image_bytes=b"",
            distance=0.42,
            threshold=0.68,
            model="ArcFace",
            metric="cosine",
        ),
        face_verifier.Match(
            candidate=cand3,
            image_path="3.jpg",
            image_bytes=b"",
            distance=0.55,
            threshold=0.68,
            model="ArcFace",
            metric="cosine",
        ),
    ]
    best = min(matches, key=lambda m: m.distance)
    assert best.candidate.position == 2
    assert best.distance == 0.42


def test_explain_match_does_not_mutate_match():
    cand = _candidate(1)
    match = face_verifier.Match(
        candidate=cand,
        image_path="1.jpg",
        image_bytes=b"bytes",
        distance=0.42,
        threshold=0.68,
        model="ArcFace",
        metric="cosine",
    )
    keys_before = set(dir(match))
    exp = explain_match(match)
    keys_after = set(dir(match))

    assert keys_before == keys_after
    assert not hasattr(match, "explanation")
    assert exp.result == "MATCH"
    assert exp.distance == 0.42


def test_explain_result_does_not_mutate_candidate_result():
    cand = _candidate(1)
    result = face_verifier.CandidateResult(
        candidate=cand, status="NO_MATCH", distance=0.74, threshold=0.68
    )
    keys_before = set(dir(result))
    exp = explain_result(result)
    keys_after = set(dir(result))

    assert keys_before == keys_after
    assert not hasattr(result, "explanation")
    assert exp.result == "NO_MATCH"
    assert exp.distance == 0.74


def test_visual_distance_scale_linear_mapping():
    assert distance_scale_percent(0.0) == 0.0
    assert distance_scale_percent(0.42) == 42.0
    assert distance_scale_percent(0.68) == 68.0
    assert distance_scale_percent(1.0) == 100.0
    assert distance_scale_percent(1.25) == 100.0  # clamped
    assert distance_scale_percent(None) is None


def test_forensic_trace_stops_appropriately_on_failure():
    exp_dl = explain_candidate("DOWNLOAD_FAILED")
    assert len(exp_dl.trace) == 3
    assert exp_dl.trace[0][1] == "Candidate image retrieval attempted"
    assert exp_dl.trace[1][1] == "Candidate image could not be downloaded"
    assert not any("face detected" in step[1].lower() for step in exp_dl.trace)

    exp_nf = explain_candidate("NO_FACE")
    assert len(exp_nf.trace) == 4
    assert exp_nf.trace[1][1] == "Face detection attempted"
    assert exp_nf.trace[2][1] == "No usable face detected in candidate"
    assert not any("arcface" in step[1].lower() for step in exp_nf.trace)

    exp_match = explain_candidate("MATCH", distance=0.42, threshold=0.68)
    assert len(exp_match.trace) == 7
    assert exp_match.trace[-1][2] is True

    exp_no_match = explain_candidate("NO_MATCH", distance=0.74, threshold=0.68)
    assert len(exp_no_match.trace) == 7
    assert exp_no_match.trace[-1][2] is False


def test_blockchain_claim_is_separate_from_face_verification_claim(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)
    _fake_chain(monkeypatch)
    report = pipeline.audit(str(path))

    # Audit verifies cryptographic digest against on-chain block calldata
    assert report.passed is True
    assert report.local_fingerprint == report.onchain_fingerprint
    # It verifies evidence document immutability, not personal identity
    events = provenance.timeline(report.document, report)
    audit_event = [e for e in events if e.stage == provenance.INDEPENDENT_AUDIT][0]
    assert "matches the digest anchored on chain" in audit_event.description


def test_provenance_and_audit_behavior_remains_intact(tmp_path, monkeypatch):
    path, document = _write_evidence(tmp_path)
    _fake_chain(monkeypatch)
    report = pipeline.audit(str(path))

    summary = provenance.summary(document, report)
    assert summary["audit_result"] == provenance.VERIFIED
    assert summary["distance"] == 0.065
    assert summary["threshold"] == 0.68
    assert summary["local_fingerprint"] == report.local_fingerprint


def test_candidate_card_renders_verification_evidence():
    cand = _candidate(1)
    result = face_verifier.CandidateResult(
        candidate=cand, status="MATCH", distance=0.42, threshold=0.68
    )
    cand_json = app._candidate_json(0, result)

    with app.app.test_request_context():
        rendered = app.render_template("fragments/candidates.html", candidates=[cand_json])

    assert "Distance" in rendered
    assert "0.4200" in rendered
    assert "Threshold" in rendered
    assert "0.6800" in rendered
    assert "Margin" in rendered
    assert "+0.26" in rendered
    assert "Rank #1" in rendered


def test_candidate_card_renders_no_distance_for_no_face_or_download_failed():
    cand = _candidate(1)
    result_nf = face_verifier.CandidateResult(candidate=cand, status="NO_FACE")
    cand_nf = app._candidate_json(0, result_nf)

    with app.app.test_request_context():
        rendered_nf = app.render_template("fragments/candidates.html", candidates=[cand_nf])

    assert "Face verification unavailable" in rendered_nf
    assert "Distance" not in rendered_nf

    result_dl = face_verifier.CandidateResult(candidate=cand, status="DOWNLOAD_FAILED")
    cand_dl = app._candidate_json(1, result_dl)

    with app.app.test_request_context():
        rendered_dl = app.render_template("fragments/candidates.html", candidates=[cand_dl])

    assert "Face verification not performed" in rendered_dl
    assert "Distance" not in rendered_dl


def test_best_match_card_renders_full_explanation():
    cand = _candidate(1)
    result = face_verifier.CandidateResult(
        candidate=cand, status="MATCH", distance=0.42, threshold=0.68
    )
    cand_json = app._candidate_json(0, result)

    with app.app.test_request_context():
        rendered = app.render_template(
            "fragments/best.html", candidate=cand_json, model="ArcFace", anchor=None
        )

    assert "Verification Evidence" in rendered
    assert "Cosine distance" in rendered
    assert "0.4200" in rendered
    assert "0.6800" in rendered
    assert "+0.26" in rendered
    assert "0.42" in rendered and "0.68" in rendered  # Decision rule
    assert "Cosine Distance Scale" in rendered
    assert "Why it passed" in rendered
    assert "How the decision was made" in rendered
    assert "Candidate Priority" in rendered
    assert "Rank #1" in rendered


# -----------------------------------------------------------------------------
# TASK 7 — Cached Investigation & Deterministic Replay Tests
# -----------------------------------------------------------------------------

from faceproof.evidence.replay import (
    InvestigationNotFoundError,
    InvestigationSecurityError,
    load_replay,
    sanitize_investigation_id,
)


def test_replay_loads_existing_evidence(tmp_path):
    path, document = _write_evidence(tmp_path)
    run_id = path.stem
    replay = load_replay(run_id, evidence_dir=str(tmp_path))

    assert replay.investigation_id == run_id
    assert replay.discovered_at == "2026-09-04T12:54:03+00:00"
    assert replay.fingerprint == document["fingerprint"]
    assert replay.manifest_ok is True
    assert replay.is_match is True
    assert replay.replay_mode == "cached"


def test_replay_preserves_recorded_result(tmp_path):
    path, document = _write_evidence(tmp_path)
    replay = load_replay(path.stem, evidence_dir=str(tmp_path))

    # Match values remain identical to recorded document
    assert replay.match["distance"] == 0.065
    assert replay.match["threshold"] == 0.68
    assert replay.match["verified"] is True
    assert replay.best_candidate["distance"] == 0.065
    assert replay.best_candidate["threshold"] == 0.68
    assert replay.best_candidate["label"] == "VERIFIED"
    assert replay.best_candidate["status"] == "MATCH"


def test_replay_does_not_invoke_network_or_ml(tmp_path, monkeypatch):
    path, _ = _write_evidence(tmp_path)

    # Attach spies raising errors if invoked
    monkeypatch.setattr(
        "faceproof.discovery.reverse_search.reverse_search",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected web search call")),
    )
    monkeypatch.setattr(
        "faceproof.discovery.retrieval.download_image",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected candidate download")),
    )
    monkeypatch.setattr(
        "faceproof.face.adapter.scan_face",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected face scan")),
    )
    monkeypatch.setattr(
        "faceproof.face.adapter.compare_faces",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected face comparison")),
    )
    monkeypatch.setattr(
        "faceproof.matching.verifier.verify_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected candidate verification")),
    )
    monkeypatch.setattr(
        "faceproof.blockchain.registry.anchor",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected blockchain write")),
    )

    # Loading replay must succeed without triggering any of the above
    replay = load_replay(path.stem, evidence_dir=str(tmp_path))
    assert replay.investigation_id == path.stem
    assert replay.manifest_ok is True


def test_replay_verifies_search_response_hash(tmp_path):
    path, _ = _write_evidence(tmp_path)
    replay = load_replay(path.stem, evidence_dir=str(tmp_path))

    assert replay.search_response_ok is True
    assert replay.overall_integrity_ok is True


def test_replay_detects_tampered_search_response(tmp_path):
    path, document = _write_evidence(tmp_path)
    # Tamper with stored search_response
    document["search_response"]["tampered"] = True
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    replay = load_replay(path.stem, evidence_dir=str(tmp_path))
    assert replay.search_response_ok is False
    assert replay.overall_integrity_ok is False


def test_replay_verifies_matched_image_artifact(tmp_path):
    path, _ = _write_evidence(tmp_path)
    replay = load_replay(path.stem, evidence_dir=str(tmp_path))

    assert replay.match_image_state == "VERIFIED"
    assert replay.overall_integrity_ok is True


def test_replay_detects_missing_matched_image(tmp_path):
    path, _ = _write_evidence(tmp_path)
    # Remove the artifact image
    (tmp_path / "run.match.jpg").unlink()

    replay = load_replay(path.stem, evidence_dir=str(tmp_path))
    assert replay.match_image_state == "MISSING"


def test_replay_detects_tampered_matched_image(tmp_path):
    path, _ = _write_evidence(tmp_path)
    # Alter the image bytes
    (tmp_path / "run.match.jpg").write_bytes(b"altered-bytes")

    replay = load_replay(path.stem, evidence_dir=str(tmp_path))
    assert replay.match_image_state == "INTEGRITY_FAILED"
    assert replay.overall_integrity_ok is False


def test_replay_does_not_mutate_evidence_file(tmp_path):
    path, _ = _write_evidence(tmp_path)
    before_bytes = path.read_bytes()

    replay = load_replay(path.stem, evidence_dir=str(tmp_path))
    after_bytes = path.read_bytes()

    assert after_bytes == before_bytes
    assert replay.manifest_ok is True


def test_replay_does_not_create_new_files(tmp_path):
    path, _ = _write_evidence(tmp_path)
    before_files = set(os.listdir(tmp_path))

    _ = load_replay(path.stem, evidence_dir=str(tmp_path))
    after_files = set(os.listdir(tmp_path))

    assert after_files == before_files


def test_replay_missing_investigation_raises_not_found(tmp_path):
    import pytest

    with pytest.raises(InvestigationNotFoundError):
        load_replay("nonexistent-id", evidence_dir=str(tmp_path))


def test_replay_rejects_path_traversal():
    import pytest

    with pytest.raises(InvestigationSecurityError):
        sanitize_investigation_id("../secret")

    with pytest.raises(InvestigationSecurityError):
        sanitize_investigation_id("..\\secret")

    with pytest.raises(InvestigationSecurityError):
        sanitize_investigation_id("/absolute/path")

    with pytest.raises(InvestigationSecurityError):
        sanitize_investigation_id("id;drop table")


def test_replay_flask_routes(tmp_path, monkeypatch):
    monkeypatch.setattr("faceproof.pipeline.EVIDENCE_DIR", str(tmp_path))
    path, document = _write_evidence(tmp_path)
    client = app.app.test_client()

    # 1. UI route
    res = client.get(f"/evidence/{path.stem}/replay")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "CACHED REPLAY" in html
    assert "No new web search or face verification was performed" in html
    assert "Recorded Investigation Provenance" in html
    assert "Verification Evidence" in html
    assert "0.0650" in html
    assert "0.6800" in html

    # 2. API route
    res_api = client.get(f"/api/evidence/{path.stem}/replay")
    assert res_api.status_code == 200
    data = res_api.get_json()
    assert data["replay_mode"] == "cached"
    assert data["investigation_id"] == path.stem
    assert data["fingerprint"] == document["fingerprint"]
    assert data["manifest_ok"] is True

    # 3. 404 on missing
    res_404 = client.get("/evidence/missing-run-id/replay")
    assert res_404.status_code == 404
    assert b"Investigation Not Found" in res_404.data

    # 4. 400 on security violation
    res_sec = client.get("/evidence/..%2F..%2Fetc/replay")
    assert res_sec.status_code in (400, 404)


def test_evidence_archive_has_replay_button(tmp_path, monkeypatch):
    monkeypatch.setattr("faceproof.pipeline.EVIDENCE_DIR", str(tmp_path))
    path, _ = _write_evidence(tmp_path)
    client = app.app.test_client()

    res = client.get("/evidence")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert f"/evidence/{path.stem}/replay" in html
    assert "Replay" in html


def test_regression_replay_equality(tmp_path):
    """Ensure strict before/after equality between persisted document and replay read model."""
    path, document = _write_evidence(tmp_path)
    evidence = document["evidence"]

    replay = load_replay(path.stem, evidence_dir=str(tmp_path))

    # Core identification
    assert replay.investigation_id == path.stem
    assert replay.fingerprint == document["fingerprint"]
    assert replay.discovered_at == evidence["discovered_at"]

    # Match and metrics
    assert replay.match["distance"] == evidence["match"]["distance"]
    assert replay.match["threshold"] == evidence["match"]["threshold"]
    assert replay.match["verified"] == evidence["match"]["verified"]
    assert replay.match["model"] == evidence["match"]["model"]

    # Best candidate
    assert replay.best_candidate["distance"] == evidence["match"]["distance"]
    assert replay.best_candidate["threshold"] == evidence["match"]["threshold"]
    assert replay.best_candidate["source"] == evidence["post"]["source"]
    assert replay.best_candidate["page_url"] == evidence["post"]["page_url"]
    assert replay.best_candidate["label"] == "VERIFIED"
    assert replay.best_candidate["status"] == "MATCH"

    # Blockchain
    assert replay.anchor["tx_hash"] == document["anchor"]["tx_hash"]
    assert replay.anchor["block_number"] == document["anchor"]["block_number"]



