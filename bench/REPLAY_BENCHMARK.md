# FaceProof — Cached Replay Benchmark & Boundary Verification Report

Everything below is measured from the live repository and reproducible with:

```bash
uv run python bench/test_replay_perf.py
```

## 1. Executive Summary

Cached Replay reconstructs a completed FaceProof investigation from persisted evidence files (`evidence/<id>.json`). It strictly operates as a read-only projection layer over the original evidence record without executing network discovery, candidate downloads, face recognition inference, or blockchain anchoring.

Across 100 benchmark iterations over historical evidence files:
- **Average Replay Latency**: **3.13 ms** (vs. 16.45 s – 38.91 s for a fresh run)
- **Speedup**: **> 3,000x faster**
- **External Resource Spend**: **0 API credits, 0 gas, 0 HTTP requests**

---

## 2. Instrumented Boundary Monitoring

To guarantee that no hidden network or ML calls occur during replay, automated instrumented spies monitor the execution boundaries of `load_replay()`:

| Boundary Monitored | Target Function | Fresh Run | Cached Replay | Invariant Status |
|:---|:---|:---:|:---:|:---:|
| Web Discovery (SerpApi) | `reverse_search.reverse_search` | 1 | **0** | PASSED (Zero calls) |
| Candidate Retrieval | `retrieval.download_image` | 10–25 | **0** | PASSED (Zero calls) |
| Probe Face Scan & Embedding | `adapter.scan_face` | 1 | **0** | PASSED (Zero calls) |
| Face Verification (ArcFace) | `adapter.compare_faces` | 10–25 | **0** | PASSED (Zero calls) |
| Verifier Batch Pipeline | `face_verifier.verify_candidates` | 1 | **0** | PASSED (Zero calls) |
| Blockchain Registry Write | `registry.anchor` | 1 | **0** | PASSED (Zero calls) |

---

## 3. Measured Latency Breakdown

Measured on Windows 11, Intel CPU, 100 warm iterations:

- **Average Latency**: 3.13 ms
- **Minimum Latency**: 2.88 ms
- **Maximum Latency**: 4.94 ms
- **95th Percentile (p95)**: 4.06 ms

The ~3 ms execution time consists entirely of:
1. Reading the local JSON manifest (`evidence/<id>.json`)
2. In-memory SHA-256 canonical JSON computation via `hashing.fingerprint`
3. In-memory SHA-256 computation over stored search response proof
4. In-memory SHA-256 verification of local matched image artifact
5. Pure read-only reconstruction of the recorded explanation metrics (margin, decision rule, visual scale)
6. Assembling the 8-stage historical provenance timeline

---

## 4. Fresh vs. Replay Operation Matrix

| Operation | Fresh Investigation | Cached Replay | Source of Truth |
|:---|:---:|:---:|:---|
| Reverse Web Search | Yes (Google Lens / SerpApi) | **No** | Stored `search_response` proof |
| Candidate Image Download | Yes (HTTP fetch) | **No** | Stored image artifacts & URLs |
| Face Detection (RetinaFace) | Yes | **No** | Recorded probe dimensions |
| Face Verification (ArcFace) | Yes | **No** | Recorded cosine distance & threshold |
| New Evidence File Created | Yes | **No** | Existing `evidence/<id>.json` |
| New Blockchain Transaction | Yes (Base Sepolia write) | **No** | Recorded `anchor.tx_hash` |
| Independent Blockchain Audit | Yes | **Yes** | Verifiable on-demand via `/verify` |
| Local Artifact Verification | Yes | **Yes** | SHA-256 computed on demand |

---

## 5. Architectural Guarantees

1. **No Silent Live Fallback**: If an investigation ID does not exist, replay immediately returns 404 (`INVESTIGATION NOT FOUND`). It will never unexpectedly trigger a live search.
2. **Deterministic Explanation**: Decision rules, margins, and visual scale points are computed deterministically from the recorded `distance` and authoritative `threshold`. The recorded match decision is immutable.
3. **Integrity Validation**: If an evidence file or search response proof has been modified, replay flags it with `CACHED DATA INTEGRITY FAILURE`.
