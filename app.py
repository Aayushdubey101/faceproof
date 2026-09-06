"""Flask UI for the FaceProof pipeline.

A thin shell over `faceproof.pipeline` - no logic of its own, so the UI can
never disagree with the CLI. Run it with:

    uv run flask --app app run --port 8000

The pipeline takes minutes, so a run happens on a background thread and the
page polls `/api/run` for stdout and per-candidate outcomes.

ponytail: one run at a time, held in process memory. This is a single-operator
demo; add a job table keyed by session id if it ever serves two people at once.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from faceproof import pipeline
from faceproof.blockchain import registry
from faceproof.discovery import DiscoveryError, retrieval
from faceproof.evidence import hashing, provenance
from faceproof.evidence.replay import (
    InvestigationNotFoundError,
    InvestigationSecurityError,
    load_replay,
)
from faceproof.face import adapter
from faceproof.matching import (
    distance_scale_percent,
    explain_candidate,
    explain_dict,
    explain_result,
)
from faceproof.matching import verifier as face_verifier

EXPLORER_TX_URL = "https://sepolia.basescan.org/tx/"
ALLOWED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
CANDIDATE_CHOICES = (5, 10, 15, 20, 25, 30, 40, 50)

# stdout lines that close a stage; timestamped as the pipeline prints them, so
# every duration the UI shows is measured rather than estimated
STAGE_MARKERS = ("Face detected", "Candidates found", "Verified match", "Fingerprint ", "Anchored ")

# label + tone per candidate outcome, keyed by the matching layer's statuses
STATUS_LABELS = {
    # red is reserved for a failed download; a rejection is a normal outcome, not an error
    face_verifier.MATCH: ("VERIFIED", "ok"),
    face_verifier.NO_MATCH: ("REJECTED", "neutral"),
    face_verifier.NO_FACE: ("NO FACE", "neutral"),
    face_verifier.DOWNLOAD_FAILED: ("SKIPPED", "bad"),
}

LOG = logging.getLogger("faceproof.app")  # developer diagnostics, never shown in the UI

load_dotenv()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


class Job(io.StringIO):
    """One pipeline run, doubling as the sink its stdout is redirected into."""

    def __init__(self, probe_path: str, max_candidates: int) -> None:
        super().__init__()
        self.probe_path = probe_path
        self.max_candidates = max_candidates
        self.investigation = face_verifier.Investigation()
        self.state = "running"
        self.document: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None
        self.started = time.perf_counter()
        self.finished: Optional[float] = None
        self.marks: Dict[str, float] = {}  # marker -> seconds into the run

    @property
    def elapsed(self) -> float:
        """Seconds this run has been going, frozen at whatever it took to end."""
        return self.finished if self.finished is not None else time.perf_counter() - self.started

    def write(self, text: str) -> int:
        """Timestamp a stage as the pipeline announces it, then store the line."""
        for marker in STAGE_MARKERS:
            if marker not in self.marks and marker in text:
                self.marks[marker] = time.perf_counter() - self.started
        return super().write(text)

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            with contextlib.redirect_stdout(self):
                self.document = pipeline.run(
                    self.probe_path,
                    max_candidates=self.max_candidates,
                    investigation=self.investigation,
                )
            self.state = "done"
        except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
            # full traceback to the developer console, never to the evaluator
            LOG.exception("pipeline run failed")
            self.error = _describe(error)
            self.state = "failed"
        finally:
            self.finished = time.perf_counter() - self.started


_JOB: Optional[Job] = None
_JOB_LOCK = threading.Lock()


@dataclass(frozen=True)
class _SimulatedAudit:
    """What an audit *would* report about the tampered copy.

    The tamper demo edits a copy in memory, so there is no real audit of it to
    read - but the two numbers that decide the verdict are real: the digest the
    edited copy hashes to, and the digest actually anchored on chain. Artifact
    checks are None because a simulated edit says nothing about the files on
    disk, which were never touched.
    """

    local_fingerprint: str
    onchain_fingerprint: str
    tx_hash: str
    search_ok: Optional[bool] = None
    image_ok: Optional[bool] = None

    @property
    def local_ok(self) -> bool:
        return False  # the copy no longer hashes to the stored fingerprint

    @property
    def onchain_ok(self) -> bool:
        return self.local_fingerprint == self.onchain_fingerprint

    @property
    def passed(self) -> bool:
        return self.onchain_ok


def _describe(error: Exception) -> Dict[str, Any]:
    """Render a failure for a non-developer: no traceback, no secrets."""
    if not isinstance(error, DiscoveryError):
        return {"title": str(error), "detail": []}
    detail = [f"Provider: {error.provider}", f"Reason: {error.reason}"]
    if error.timeout:
        detail.append(f"Limits: {error.timeout}")
    detail.append(
        f"Retry the run; if it repeats, set FACE_UPLOAD_PROVIDER to another provider "
        f"({', '.join(retrieval.PROVIDERS)}) in .env and restart."
    )
    detail.append(
        "Only web discovery is affected - local face matching and on-chain "
        "verification of existing evidence still work."
    )
    return {"title": "Web discovery unavailable", "detail": detail}


def _domain(url: Any) -> str:
    return urlsplit(str(url)).netloc or str(url)


def _evidence_files() -> List[str]:
    """Evidence file names on disk, oldest first."""
    if not os.path.isdir(pipeline.EVIDENCE_DIR):
        return []
    return sorted(name for name in os.listdir(pipeline.EVIDENCE_DIR) if name.endswith(".json"))


def _evidence_path(name: str) -> Optional[str]:
    """Resolve a client-supplied name inside the evidence directory, or None.

    The name arrives from the browser, so it is matched against the real
    listing rather than joined onto the directory.
    """
    if name not in _evidence_files():
        return None
    return os.path.join(pipeline.EVIDENCE_DIR, name)


def _evidence_index() -> List[Dict[str, Any]]:
    """Every evidence file on disk, newest first. Unreadable files are skipped."""
    items: List[Dict[str, Any]] = []
    for name in reversed(_evidence_files()):
        path = os.path.join(pipeline.EVIDENCE_DIR, name)
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            items.append({**_evidence_json(document, name[: -len(".json")]), "file": name})
        except (OSError, ValueError, KeyError, TypeError):
            LOG.warning("skipping unreadable evidence file %s", name)
    return items


def _candidate_json(index: int, result: face_verifier.CandidateResult) -> Dict[str, Any]:
    label, tone = STATUS_LABELS[result.status]
    explanation = explain_result(result)
    return {
        "number": index + 1,
        "rank": result.candidate.position,
        "is_social": result.candidate.is_social,
        "label": label,
        "tone": tone,
        "source": result.candidate.source or _domain(result.candidate.page_url),
        "page_url": result.candidate.page_url,
        # the downloaded bytes are what was actually checked; else the remote URL
        "image": f"/candidate/{index}.jpg" if result.image_bytes else result.candidate.image_url,
        "distance": result.distance,
        "threshold": result.threshold,
        "reason": result.reason,
        "status": result.status,
        "margin": explanation.margin,
        "margin_display": explanation.margin_display,
        "decision_rule": explanation.decision_rule,
        "comparison": explanation.comparison,
        "reasons": explanation.reasons,
        "detailed_reasons": explanation.detailed_reasons,
        "trace": explanation.trace,
        "distance_percent": distance_scale_percent(result.distance),
        "threshold_percent": distance_scale_percent(result.threshold) if result.threshold else 68.0,
    }


def _evidence_json(document: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Every value here comes from the evidence file - nothing is inferred."""
    evidence = document["evidence"]
    post, match = evidence.get("post", {}), evidence.get("match", {})
    artifact = document.get("artifacts", {}).get("match_image")
    return {
        "id": name,
        "source": post.get("source") or "unknown",
        "domain": _domain(post.get("page_url", "")),
        "page_url": post.get("page_url", ""),
        "verified": bool(match.get("verified")),
        "distance": match.get("distance"),
        "threshold": match.get("threshold"),
        "model": match.get("model", "unknown"),
        "discovered_at": evidence.get("discovered_at", "unknown"),
        "fingerprint": document["fingerprint"],
        "match_image": f"/evidence/{artifact}" if artifact else None,
        "title": post.get("title", ""),
        "is_social": bool(post.get("is_social")),
        # recomputed locally, so the archive can flag a modified file without a chain call
        "intact": hashing.fingerprint(evidence) == document["fingerprint"],
    }


def _anchor_json(anchor: Dict[str, Any], fingerprint: str) -> Dict[str, Any]:
    tx_hash = anchor.get("tx_hash")
    explorer = (
        f"{EXPLORER_TX_URL}{tx_hash}"
        if tx_hash and anchor.get("chain_id") == registry.CHAIN_ID
        else None
    )
    return {
        "network": anchor.get("network", "unknown"),
        "chain_id": anchor.get("chain_id", "unknown"),
        "block_number": anchor.get("block_number", "unknown"),
        "gas_used": anchor.get("gas_used", "unknown"),
        "tx_hash": tx_hash or "none",
        "fingerprint": fingerprint,
        "explorer": explorer,
    }


def _timeline(job: "Job") -> List[Dict[str, Any]]:
    """Investigation stages for the progress timeline.

    Derived from what the pipeline has already printed, so the UI shows the
    same milestones as the CLI without ever exposing raw stdout. A stage
    carries a duration only where one was actually measured - the two stages
    that share a single printed line report no time of their own.
    """
    log, investigation = job.getvalue(), job.investigation
    finished = job.state != "running"
    scan, found = job.marks.get("Face detected"), job.marks.get("Candidates found")
    verified, fingerprint = job.marks.get("Verified match"), job.marks.get("Fingerprint ")
    anchored = job.marks.get("Anchored ")

    def took(end: Optional[float], start: Optional[float]) -> Optional[float]:
        return None if end is None or start is None else round(end - start, 1)

    # the unique count is shown only when deduplication actually merged
    # something, so the line never claims work that did not happen
    discovered = f"{investigation.discovered} candidates discovered"
    if 0 < investigation.unique < investigation.discovered:
        discovered += f" ({investigation.unique} unique)"

    stages: List[Tuple[str, bool, Optional[float]]] = [
        ("Face detected", "Face detected" in log, took(scan, 0.0)),
        (f"{adapter.DEFAULT_MODEL} embedding generated", "Embedding generated" in log, None),
        ("Reverse image search completed", "Candidates found" in log, took(found, scan)),
        (discovered, investigation.discovered > 0, None),
        (
            f"{investigation.checked} candidates verified",
            finished and investigation.checked > 0,
            took(verified, found),
        ),
        ("Evidence fingerprint created", "Fingerprint " in log, took(fingerprint, verified)),
        (f"Anchored on {registry.NETWORK_NAME}", "Anchored " in log, took(anchored, fingerprint)),
    ]

    steps: List[Dict[str, Any]] = []
    running = job.state == "running"
    for label, done, seconds in stages:
        state = "done" if done else "active" if running else "pending"
        running = running and done  # only the first unfinished stage animates
        steps.append({"label": label, "state": state, "seconds": seconds if done else None})
    return steps


@app.get("/")
def index() -> str:
    return render_template(
        "run.html",
        page="run",
        default_candidates=face_verifier.DEMO_CANDIDATES,
        candidate_choices=CANDIDATE_CHOICES,
    )


@app.get("/verify")
def verify_page() -> str:
    items = _evidence_index()
    selected = request.args.get("file", "")
    if not any(item["file"] == selected for item in items):
        selected = items[0]["file"] if items else ""
    return render_template("verify.html", page="verify", items=items, selected=selected)


@app.get("/evidence")
def evidence_index() -> str:
    return render_template("evidence.html", page="evidence", items=_evidence_index())


@app.post("/api/run")
def api_run() -> Any:
    global _JOB
    upload = request.files.get("probe")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a probe image first."), 400
    if not upload.filename.lower().endswith(ALLOWED_SUFFIXES):
        return jsonify(error=f"Unsupported file type - use {', '.join(ALLOWED_SUFFIXES)}."), 400

    try:
        max_candidates = max(5, min(50, int(request.form.get("max_candidates", 10))))
    except ValueError:
        return jsonify(error="Candidate count must be a number."), 400

    with _JOB_LOCK:
        if _JOB is not None and _JOB.state == "running":
            return jsonify(error="A run is already in progress."), 409
        suffix = os.path.splitext(upload.filename)[1].lower()
        handle, probe_path = tempfile.mkstemp(prefix="faceproof-probe-", suffix=suffix)
        os.close(handle)
        upload.save(probe_path)
        _JOB = Job(probe_path, max_candidates)
        _JOB.start()
    return jsonify(ok=True)


@app.get("/api/run")
def api_run_status() -> Any:
    """Progress for the run page: counts as JSON, everything visible as Jinja.

    Rendering the cards server-side keeps one copy of the markup and lets
    autoescaping handle the candidate titles and URLs, which come off the
    open web.
    """
    job = _JOB
    if job is None:
        return jsonify(state="idle")

    candidates = [
        _candidate_json(index, result) for index, result in enumerate(job.investigation.results)
    ]
    try:
        since = max(0, int(request.args.get("since", 0)))
    except ValueError:
        since = 0

    anchor = (
        _anchor_json(job.document["anchor"], job.document["fingerprint"]) if job.document else None
    )
    payload: Dict[str, Any] = {
        "state": job.state,
        "discovered": job.investigation.discovered,
        "unique": job.investigation.unique,
        "checked": job.investigation.checked,
        "verified": sum(1 for candidate in candidates if candidate["tone"] == "ok"),
        "count": len(candidates),
        "elapsed": round(job.elapsed, 1),
        "timeline_html": render_template("fragments/timeline.html", steps=_timeline(job)),
        # only the candidates the browser has not drawn yet, so thumbnails never reload
        "candidates_html": render_template(
            "fragments/candidates.html", candidates=candidates[since:]
        ),
        "error_html": (
            render_template("fragments/error.html", error=job.error) if job.error else ""
        ),
        "result_html": "",
        "best_html": "",
    }

    verified = job.investigation.verified
    evaluated = [r for r in job.investigation.results if r.distance is not None]
    if verified:
        # a verified result always carries a distance; the fallback only satisfies the type
        best = min(verified, key=lambda result: float(result.distance or 0.0))
        payload["best_html"] = render_template(
            "fragments/best.html",
            candidate=_candidate_json(job.investigation.results.index(best), best),
            model=adapter.DEFAULT_MODEL,
            anchor=anchor,
        )
    elif evaluated and job.state in ("done", "failed"):
        best = min(evaluated, key=lambda result: float(result.distance or 999.0))
        payload["best_html"] = render_template(
            "fragments/best.html",
            candidate=_candidate_json(job.investigation.results.index(best), best),
            model=adapter.DEFAULT_MODEL,
            anchor=anchor,
        )
    if job.document:
        name = str(job.document["artifacts"]["match_image"])[: -len(".match.jpg")]
        payload["result_html"] = render_template(
            "fragments/result.html",
            evidence=_evidence_json(job.document, name),
            anchor=anchor,
            # no audit has run yet: the run that wrote this evidence cannot
            # audit itself, so the chain's last step stays open
            events=provenance.timeline(job.document),
            summary=provenance.summary(job.document),
            verify_url=url_for("verify_page", file=f"{name}.json"),
        )
    return jsonify(payload)


@app.get("/candidate/<int:index>.jpg")
def candidate_image(index: int) -> Any:
    job = _JOB
    if job is None or index >= len(job.investigation.results):
        return "", 404
    image_bytes = job.investigation.results[index].image_bytes
    if image_bytes is None:
        return "", 404
    return app.response_class(image_bytes, mimetype="image/jpeg")


@app.get("/evidence/<name>")
def evidence_artifact(name: str) -> Any:
    return send_from_directory(os.path.abspath(pipeline.EVIDENCE_DIR), name)


@app.get("/replay/<investigation_id>")
@app.get("/evidence/<investigation_id>/replay")
def replay_view(investigation_id: str) -> Any:
    """Render a completed historical investigation from persisted evidence."""
    try:
        replay = load_replay(investigation_id, evidence_dir=pipeline.EVIDENCE_DIR)
    except InvestigationSecurityError:
        return (
            render_template(
                "error.html",
                page="evidence",
                error={
                    "title": "Invalid Investigation Identifier",
                    "detail": [
                        f"The identifier '{investigation_id}' contains invalid characters or path traversal sequences.",
                        "FaceProof enforces strict filesystem isolation on evidence identifiers.",
                    ],
                },
            ),
            400,
        )
    except InvestigationNotFoundError:
        return (
            render_template(
                "error.html",
                page="evidence",
                error={
                    "title": "Investigation Not Found",
                    "detail": [
                        f"No recorded evidence exists for investigation '{investigation_id}'.",
                        "Cached replay reconstructs existing evidence and will never trigger an unintended live search.",
                        "To investigate a new face, run a fresh investigation from the home page.",
                    ],
                },
            ),
            404,
        )
    except Exception as err:
        LOG.exception("Replay failed for %s", investigation_id)
        return (
            render_template(
                "error.html",
                page="evidence",
                error={"title": "Replay Failure", "detail": [str(err)]},
            ),
            500,
        )

    anchor_json = _anchor_json(replay.anchor, replay.fingerprint)
    return render_template(
        "replay.html",
        page="evidence",
        replay=replay,
        anchor_json=anchor_json,
    )


@app.get("/api/evidence/<investigation_id>/replay")
@app.get("/api/replay/<investigation_id>")
def api_replay(investigation_id: str) -> Any:
    """JSON API endpoint returning the replay read model."""
    try:
        replay = load_replay(investigation_id, evidence_dir=pipeline.EVIDENCE_DIR)
        return jsonify(replay.as_dict())
    except InvestigationSecurityError:
        return (
            jsonify(
                error="Invalid investigation identifier",
                detail="Path traversal or invalid characters rejected.",
            ),
            400,
        )
    except InvestigationNotFoundError:
        return (
            jsonify(
                error="Investigation not found",
                detail=f"No evidence for '{investigation_id}'",
            ),
            404,
        )
    except Exception as err:
        LOG.exception("API replay failed for %s", investigation_id)
        return jsonify(error="Replay failed", detail=str(err)), 500


@app.post("/api/verify")
def api_verify() -> Any:
    payload = request.get_json(silent=True) or {}
    path = _evidence_path(str(payload.get("file", "")))
    if path is None:
        return jsonify(error="Unknown evidence file."), 404

    try:
        report = pipeline.audit(path)
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
        LOG.exception("audit failed")
        described = _describe(error)
        return jsonify(error=described["title"], detail=described["detail"]), 400

    name = os.path.basename(path)[: -len(".json")]
    result: Dict[str, Any] = {
        "evidence": _evidence_json(report.document, name),
        "anchor": _anchor_json(report.document["anchor"], report.stored_fingerprint),
        "onchain": report.onchain_fingerprint,
        "checks": [
            {"label": "Local evidence integrity", "ok": report.local_ok},
            {"label": "Search response integrity", "ok": report.search_ok},
            {"label": "Matched image integrity", "ok": report.image_ok},
        ],
        # the chain as the audit found it, so the last step reports a real verdict
        "events": provenance.timeline(report.document, report),
        "summary": provenance.summary(report.document, report),
    }

    if payload.get("tamper"):
        section, key = pipeline.TAMPER_FIELD
        modified, digest = pipeline.tamper(report.document["evidence"])
        simulated = _SimulatedAudit(digest, report.onchain_fingerprint, report.tx_hash)
        result.update(
            events=provenance.timeline(report.document, simulated),
            summary=provenance.summary(report.document, simulated),
            tampered=True,
            local=digest,
            local_label="TAMPERED FINGERPRINT",
            original=report.local_fingerprint,
            edited_field=f"{section}.{key}",
            edited_value=modified[section][key],
            ok=digest == report.onchain_fingerprint,
            checks=[],  # a simulated edit says nothing about the file's own artifacts
            restore=True,  # offer the way back; the file on disk never changed
        )
        return jsonify(_rendered(result))

    result.update(
        tampered=False,
        local=report.local_fingerprint,
        local_label="LOCAL FINGERPRINT",
        ok=report.onchain_ok,
    )
    return jsonify(_rendered(result))


def _rendered(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the server-rendered verdict card to an audit result."""
    return {**result, "html": render_template("fragments/verify_result.html", report=result)}


@app.errorhandler(413)
def _too_large(_: Exception) -> Any:
    return jsonify(error=f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."), 413


if __name__ == "__main__":
    app.run(port=8000)
