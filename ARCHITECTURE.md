# FaceProof — Architecture

**Face scan → web/social discovery → candidate face verification → evidence fingerprinting → Base Sepolia anchoring → independent tamper verification**

FaceProof is an end-to-end identity-evidence pipeline. It does not treat reverse image search results as proof of identity. Web discovery produces candidates; FaceProof independently verifies candidate faces using facial embeddings before creating and anchoring evidence.

---

## 1. System Architecture

```text
                         ┌─────────────────────┐
                         │      FaceProof       │
                         │    Orchestrator      │
                         └──────────┬──────────┘
                                    │
                              Input Face
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │      Face Intelligence       │
                    │      (face_engine/)          │
                    ├──────────────────────────────┤
                    │ Detection                    │
                    │ Quality Check                │
                    │ Alignment                    │
                    │ ArcFace Embedding            │
                    └──────────────┬───────────────┘
                                   │
                             Face Embedding
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   Web Discovery     │
                         ├─────────────────────┤
                         │ Reverse Image Search│
                         │ Candidate URLs      │
                         │ Candidate Images    │
                         │ Metadata            │
                         └──────────┬──────────┘
                                    │
                               Candidates
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │   Candidate Verification     │
                    ├──────────────────────────────┤
                    │ Face Detection               │
                    │ ArcFace Embedding            │
                    │ Similarity Score             │
                    │ Threshold                    │
                    │ Ranking                      │
                    └──────────────┬───────────────┘
                                   │
                           Best Verified Match
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │  Evidence Builder   │
                         ├─────────────────────┤
                         │ URL                 │
                         │ Image               │
                         │ Text / Metadata     │
                         │ Match Score         │
                         │ Timestamp           │
                         │ SHA-256 Fingerprint │
                         └──────────┬──────────┘
                                    │
                                  Hash
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Base Sepolia     │
                         │  Evidence Registry  │
                         └──────────┬──────────┘
                                    │
                             Transaction
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Independent Verify  │
                         ├─────────────────────┤
                         │ Recompute Hash      │
                         │ Read Blockchain      │
                         │ Compare             │
                         │                     │
                         │ ✓ VERIFIED          │
                         │ ✗ TAMPERED          │
                         └─────────────────────┘
```

---

## 2. Design Principles

FaceProof follows five core principles:

1. **Discovery is not identity verification.**
   Reverse image search only produces visually related candidates.

2. **Identity is independently verified.**
   Candidate images are processed again using the same face-recognition foundation and compared against the input face.

3. **Evidence is deterministic.**
   The evidence document is canonicalized before SHA-256 hashing so the same evidence produces the same fingerprint.

4. **Blockchain stores the fingerprint, not the bulky evidence.**
   The evidence remains locally readable while its cryptographic fingerprint is anchored on Base Sepolia.

5. **Verification is independent of the original pipeline run.**
   A later verification recomputes the local fingerprint and compares it with the immutable on-chain fingerprint.

---

## 3. Repository Layer Split

The repository has two logical Python layers.

| Layer                 | Path           | Role                                                                                   |
| --------------------- | -------------- | -------------------------------------------------------------------------------------- |
| Face foundation       | `face_engine/` | Face detection, alignment, embedding and verification                                  |
| FaceProof application | `faceproof/`   | Discovery, candidate verification orchestration, evidence, hashing, blockchain and CLI |

### `face_engine/`

`face_engine/` is the internal face-recognition foundation used by FaceProof.

It provides the computer-vision functionality required by the application, including:

* face detection
* face alignment
* ArcFace embeddings
* face verification
* detector/model configuration

FaceProof accesses this layer through a small application adapter rather than spreading face-engine implementation details throughout the project.

### `faceproof/`

`faceproof/` contains the project-specific pipeline and integration logic.

All FaceProof-specific functionality belongs here.

---

## 4. End-to-End Pipeline

### Stage 1 — Face Intelligence

**Input:** probe image (`.jpg`, `.jpeg`, `.png`)

The input image is processed through `face_engine/`.

```text
Probe Image
    │
    ▼
Face Detection
    │
    ▼
Quality / Validation
    │
    ▼
Face Alignment
    │
    ▼
ArcFace Embedding
    │
    ▼
Face Embedding
```

The scan produces structured information such as:

```text
FaceScanResult
├── face_detected
├── face_count
├── facial_area
├── embedding
├── embedding_model
└── detector_backend
```

The initial pipeline requires a usable face. If no face is detected, the pipeline stops before making a web-search request.

The primary face-recognition configuration is:

```text
Embedding model: ArcFace
Detector: RetinaFace
Embedding dimensions: 512
Distance metric: cosine
```

---

### Stage 2 — Web Discovery

**Input:** probe image

Reverse image search requires an image that can be accessed by the search provider.

The discovery stage performs:

```text
Probe Image
    │
    ▼
Temporary Public Image Upload
    │
    ▼
Reverse Image Search
    │
    ▼
Candidate URLs
    │
    ├── Candidate images
    ├── Page URLs
    ├── Titles
    ├── Source metadata
    └── Search metadata
    │
    ▼
URL / Image Normalization
    │
    ▼
Deduplication
    │
    ▼
Priority Ranking
    │
    ▼
Candidate Budget
    │
    ▼
Independent Face Verification
```

Normalization, deduplication and ranking all happen **before** a single face is
compared, so the budget is spent on the most promising results rather than on
whatever the provider listed first.

> **Candidate ranking is a retrieval optimization and is not an identity
> decision. Identity verification remains based on ArcFace cosine distance and
> the configured threshold.**

#### Normalization (`faceproof/discovery/normalize.py`)

`canonical_url()` produces a comparison key: lowercased scheme and host,
default port dropped, percent-escapes uppercased, one trailing slash removed,
query parameters sorted, non-hashbang fragments dropped, and the tracking
parameters in `TRACKING_PARAMS` removed (`utm_*`, `fbclid`, `gclid`, `dclid`,
`gbraid`, `wbraid`, `msclkid`, `yclid`, `twclid`, `igshid`, `igsh`, `mc_cid`,
`mc_eid`, `_ga`, `_gl`, `ref_src`, `ref_url`).

Nothing else is removed. A parameter that can select content stays: `?id=123`
and `?id=456` are different posts, `?v=abc` is a different video, and an image
CDN's `?format=webp&name=large` is different bytes. Image URLs get exactly the
same treatment, for exactly that reason.

The candidate keeps the provider's original `page_url` and `image_url` — those
are what gets fetched and what the evidence records. The canonical form sits
beside them and is never substituted.

#### Deduplication

Two exact keys, in order: canonical page URL, then canonical image URL. No
content hashing — that would mean downloading an image to decide whether to
download it — and no perceptual hashing, which is a similarity guess and has no
place in a system whose whole claim is that identity is decided by one
measurable distance.

The survivor of a merge is the higher-priority member, so folding a
tracking-tagged copy into a plain one never costs the better source. Every URL
merged away is kept on the survivor's `duplicates`, so provenance is preserved
rather than deleted.

#### Priority ranking

`priority()` is deterministic and additive — the same response always produces
the same order, on any machine:

| Signal | Weight | Why |
|---|---|---|
| Social source (`SOCIAL_DOMAINS`) | `SOURCE_PRIORITY["social"]` = 3.0 | The evidence this project looks for; carries an author and a date a reviewer can follow |
| Provider-cached thumbnail | `CACHED_THUMBNAIL_PRIORITY` = 1.0 | An origin-only image URL is the one that answers 403 and burns budget on `DOWNLOAD_FAILED` |
| Search rank | `RANK_PRIORITY` × `max(0, 1 − (position − 1) / 25)` | The provider's own ordering, decaying to nothing by `RANK_HORIZON`, so it only ever breaks ties |

Ties break on position, then page URL, so the ordering is total. Deduplication
is the duplicate penalty: a duplicate is merged before ranking and so cannot
consume the budget twice.

The score is internal. It is never rendered as a confidence, and it never
enters the evidence document. Measurements are in `bench/CANDIDATE_RANKING.md`.

The implementation uses:

```text
SerpApi
    └── Google Lens
```

The discovery stage returns candidates that are **visually related to the probe**.

A candidate is deliberately **not considered the same person at this stage**.

This distinction is important:

```text
Reverse Search
     ↓
"Potentially related image"
     ↓
Candidate
     ↓
FaceProof verification
     ↓
"Face match / no face match"
```

The raw search response is retained as search evidence after sensitive API-key-shaped fields are removed.

---

### Stage 3 — Candidate Verification

Each discovered candidate is independently checked, up to a **candidate budget**.

Discovery routinely returns far more candidates than are worth checking, and each
candidate costs one image download plus one face encode. The budget bounds that cost:

| Setting            | Value | Where                                                 |
| ------------------ | ----- | ----------------------------------------------------- |
| `MAX_CANDIDATES`   | 25    | `matching/verifier.py` — ceiling, and the CLI default |
| `DEMO_CANDIDATES`  | 10    | `matching/verifier.py` — interactive default          |
| `--max-candidates` | any   | CLI override                                          |
| Browser UI field   | 5–50  | `app.py`                                              |

The probe embedding is **not** recomputed per candidate. It is generated once in
Stage 1 and passed into `verify_candidates(probe_embedding=...)`, so a run over N
candidates performs N candidate encodes, not N+1.

```text
Candidate
    │
    ▼
Download Candidate Image
    │
    ▼
Detect Face
    │
    ├── No face → skip candidate
    │
    ▼
Generate ArcFace Embedding
    │
    ▼
Compare With Probe Embedding
    │
    ▼
Cosine Distance
    │
    ▼
Threshold Check
    │
    ▼
Verified / Rejected
```

For each usable candidate, FaceProof records:

```text
CandidateMatch
├── candidate URL
├── image URL
├── similarity / distance
├── threshold
├── verified
└── metadata
```

Candidates are ranked by the face-comparison result.

The best verified candidate is selected based on the configured similarity/distance criteria.

#### Candidate handling

| Condition                              | Status            | Behaviour                               |
| -------------------------------------- | ----------------- | --------------------------------------- |
| Face comparison passes threshold       | `MATCH`           | Candidate becomes a verified match      |
| Face compared, distance above threshold| `NO_MATCH`        | Rejected                                |
| Candidate contains no detectable face  | `NO_FACE`         | Skipped, reason recorded                |
| Image returns 404/403, times out, or is unusable | `DOWNLOAD_FAILED` | Skipped, reason recorded      |
| Ranked below the candidate budget      | —                 | Kept in `Investigation.not_investigated`, given no status |
| No candidate passes                    | —                 | Stop; do not create blockchain evidence |

A candidate the budget never reached is **uninvestigated, not rejected**. It is
held in `Investigation.not_investigated` with no status at all, because
reporting it as `NO_MATCH` would be a claim the pipeline never tested. The UI
counts `discovered`, `unique`, `investigated` and `verified` separately for the
same reason.

Candidate images are fetched by a bounded thread pool that runs ahead of the
comparison loop, so the download of candidate *n+1* overlaps the face comparison
of candidate *n*. `ThreadPoolExecutor.map` returns results in submission order,
so candidate ordering, identity, source URL and failure status are exactly what a
sequential loop produced; a candidate whose download raises still lands as
`DOWNLOAD_FAILED` and the run continues. Two candidates sharing an identical
image URL are fetched once, within that one investigation, and still judged
separately.

Comparisons run on a second bounded pool (`FACEPROOF_COMPARE_WORKERS`, default
2, max 4), which is also `pool.map` and so also in submission order. Only the
model call happens off the main thread; recording, printing and ranking stay on
one thread, so an `Investigation` is filled in candidate order regardless of
which comparison finishes first. The measured shape of that trade — 26.8 s /
23.3 s / 22.5 s / 22.1 s at 1 / 2 / 3 / 4 workers over 25 candidates, for 1.9 /
2.5 / 2.8 / 3.0 GiB peak — is why the default is 2 rather than "as many as
there are cores": RetinaFace already uses every core for a single image, so
extra workers mostly buy memory. Verdicts are byte-identical at every setting;
see `bench/FACE_VERIFICATION_PERFORMANCE.md`.

Every outcome, including the rejections, is collected into an `Investigation`
record that is filled **in place** — so the candidate work stays visible in the UI
even when a run ends with no verified match. Nothing is silently dropped, and a
candidate that was never compared carries no distance rather than a fabricated
one.

---

## 5. Evidence Builder

Once a candidate has been independently verified, FaceProof creates a structured evidence document.

```text
Verified Candidate
       │
       ▼
Evidence Builder
       │
       ├── Probe information
       ├── Candidate URL
       ├── Candidate image information
       ├── Metadata
       ├── Match information
       ├── Search information
       └── Timestamp
       │
       ▼
Canonical Evidence
       │
       ▼
SHA-256
       │
       ▼
Evidence Fingerprint
```

The evidence contains the information required to reproduce and verify the result later.

Example:

```json
{
  "evidence": {
    "schema": "faceproof/v1",
    "discovered_at": "2026-09-04T09:44:00+00:00",

    "probe": {
      "sha256": "...",
      "faces_detected": 1,
      "embedding_model": "ArcFace",
      "detector_backend": "retinaface",
      "embedding_dimensions": 512
    },

    "post": {
      "page_url": "https://...",
      "image_url": "https://...",
      "image_sha256": "...",
      "title": "...",
      "source": "...",
      "is_social": true
    },

    "match": {
      "model": "ArcFace",
      "distance_metric": "cosine",
      "distance": 0.41,
      "threshold": 0.68,
      "verified": true
    },

    "search": {
      "engine": "serpapi/google_lens",
      "probe_image_url": "https://...",
      "response_sha256": "...",
      "candidates_returned": 42,
      "candidates_checked": 25
    }
  }
}
```

---

## 6. Evidence Fingerprinting

FaceProof uses SHA-256 to create a deterministic fingerprint of the evidence.

Before hashing, the evidence is canonicalized:

```python
json.dumps(
    evidence,
    sort_keys=True,
    separators=(",", ":")
)
```

This ensures that equivalent JSON structure produces stable bytes for hashing.

The resulting digest is:

```text
canonical evidence
       │
       ▼
UTF-8 bytes
       │
       ▼
SHA-256
       │
       ▼
32-byte digest
```

The fingerprint is stored with the evidence and anchored on-chain.

---

## 7. Search-Proof Binding

The raw reverse-search response can be large, so it is not placed directly inside the hashed evidence region.

Instead:

```text
Raw Search Response
        │
        ▼
Redacted Search Response
        │
        ├── stored locally
        │
        ▼
SHA-256
        │
        ▼
response_sha256
        │
        ▼
Included inside evidence
```

Therefore the evidence fingerprint indirectly binds the search response.

If the stored search response is modified:

```text
Modified Search Response
        │
        ▼
New SHA-256
        │
        ▼
Does not equal evidence.search.response_sha256
        │
        ▼
Verification FAIL
```

This prevents someone from modifying the stored search proof without detection.

---

## 8. Blockchain Anchoring

### Network

FaceProof uses:

```text
Network: Base Sepolia
Chain ID: 84532
```

The blockchain is used as an immutable timestamped anchor for the evidence fingerprint.

FaceProof does **not** store the entire evidence document on-chain.

Instead:

```text
Evidence
   │
   ▼
SHA-256
   │
   ▼
32-byte digest
   │
   ▼
Base Sepolia transaction
```

### Anchor transaction

The current implementation uses a zero-value self-transaction:

```python
{
    "from": me,
    "to": me,
    "value": 0,
    "data": digest_bytes
}
```

The 32-byte SHA-256 digest is stored as transaction calldata.

The resulting transaction provides:

```text
tx_hash
chain_id
block_number
gas_used
from_address
```

The transaction can be independently inspected on the Base Sepolia network.

---

## 9. Why a Transaction Instead of a Smart Contract?

A smart-contract registry was intentionally avoided for the initial implementation.

| Approach                     | Decision   | Reason                                                                                    |
| ---------------------------- | ---------- | ----------------------------------------------------------------------------------------- |
| 0-value calldata transaction | **Chosen** | Minimal infrastructure and deployment risk                                                |
| Solidity registry contract   | Deferred   | Requires contract compilation, deployment, ABI and additional transaction                 |
| IPFS / Arweave               | Deferred   | Useful for storage but does not itself provide the required blockchain fingerprint anchor |
| Event-based registry         | Deferred   | Requires a deployed contract                                                              |

The calldata design provides the core property required by the prototype:

```text
Evidence
   ↓
Hash
   ↓
Immutable blockchain transaction
   ↓
Later recomputation
   ↓
Comparison
```

### Known limitation

A calldata anchor is not naturally queryable by subject.

A verifier must already know the transaction hash.

If FaceProof later requires:

```text
"Find every blockchain record belonging to this person"
```

the upgrade path is a registry smart contract with indexed events or records.

---

## 10. Independent Verification

Verification is intentionally separated from the original pipeline.

```text
Saved Evidence
      │
      ├───────────────┐
      │               │
      ▼               ▼
Recompute Local      Read Blockchain
SHA-256              Transaction
      │               │
      └───────┬───────┘
              ▼
          Compare
              │
       ┌──────┴──────┐
       ▼             ▼
    MATCH          MISMATCH
       │             │
       ▼             ▼
  ✓ VERIFIED      ✗ TAMPERED
```

Verification checks:

1. Evidence canonicalization is reproduced.
2. SHA-256 is recomputed locally.
3. The blockchain transaction is read.
4. The on-chain digest is extracted.
5. The local digest is compared against the on-chain digest.
6. The stored search-response hash is independently checked.
7. The matched image hash is independently checked when available.

---

## 11. Tamper Detection

The primary demonstration is a controlled tamper test.

### Original

```text
Evidence
   ↓
SHA-256 = ABC123...
   ↓
Base Sepolia
   ↓
ABC123...
   ↓
✓ VERIFIED
```

### After modifying the evidence

```text
Modified Evidence
   ↓
SHA-256 = XYZ789...
   ↓
Base Sepolia
   ↓
ABC123...
   ↓
✗ TAMPERED
```

The blockchain is not being used to claim that the identity itself is "stored on-chain."

Instead, it proves that the **specific evidence fingerprint recorded at the time of verification has not changed**.

### Two ways to run the test

* **Hand-edit the file.** Change any value inside the `evidence` block of
  `evidence/<id>.json` and verify again. This is the real thing: the file on disk
  now differs from what was anchored.
* **Simulate tampering (UI).** `pipeline.tamper()` deep-copies the evidence,
  appends `" [edited]"` to `post.title` on the *copy*, and refingerprints it. The
  file on disk is never written to, so **Restore / verify original** re-verifies
  the untouched original immediately afterwards.

Both paths compare against the same on-chain digest; neither writes anything to
the chain.

---

## 12. Module Architecture

The project layer is organized around clear responsibilities.

```text
app.py                      Flask UI (thin shell over pipeline)
templates/                  base.html, run.html, verify.html, evidence.html,
                            replay.html, error.html + components/, fragments/
static/                     app.css, app.js
bench/                      measurement harnesses and their reports
faceproof/
│
├── pipeline.py
│
├── face/
│   └── adapter.py
│
├── discovery/
│   ├── reverse_search.py
│   ├── candidates.py
│   ├── normalize.py
│   └── retrieval.py
│
├── matching/
│   ├── verifier.py
│   └── explanation.py
│
├── evidence/
│   ├── manifest.py
│   ├── provenance.py
│   ├── replay.py
│   └── hashing.py
│
├── blockchain/
│   ├── registry.py
│   └── verifier.py
│
└── __main__.py
```

### Responsibilities

| Module                        | Responsibility                          |
| ----------------------------- | --------------------------------------- |
| `app.py`                      | Flask UI (run, verify and evidence pages); calls `pipeline.run` / `pipeline.audit` / `pipeline.tamper` only |
| `pipeline.py`                 | End-to-end orchestration                |
| `face/adapter.py`             | Controlled interface to `face_engine/`  |
| `discovery/reverse_search.py` | Reverse image search                    |
| `discovery/candidates.py`     | Candidate parsing, deduplication and priority ranking |
| `discovery/normalize.py`      | URL canonicalization for deduplication  |
| `discovery/retrieval.py`      | Candidate image retrieval               |
| `matching/verifier.py`        | Candidate face verification and ranking |
| `matching/explanation.py`     | Deterministic, threshold-derived explanation of each verdict |
| `evidence/manifest.py`        | Evidence document construction          |
| `evidence/provenance.py`      | Read model: investigation chain and evidence summary |
| `evidence/replay.py`          | Read-only projection of a persisted investigation, with local integrity gates |
| `evidence/hashing.py`         | Canonicalization and SHA-256            |
| `blockchain/registry.py`      | Blockchain anchoring                    |
| `blockchain/verifier.py`      | On-chain fingerprint verification       |
| `__main__.py`                 | CLI entry point                         |

---

## 13. Dependency Direction

Dependencies flow toward lower-level services.

```text
                    pipeline.py
                    /    |     \
                   /     |      \
                  ▼      ▼       ▼
              discovery matching evidence
                          │       │
                          │       ▼
                          │     hashing
                          │       │
                          │       ▼
                          │    blockchain
                          │
                          ▼
                     face/adapter
                          │
                          ▼
                     face_engine/
```

Rules:

* `pipeline.py` orchestrates but should contain minimal business logic.
* `faceproof/` may import `face_engine/`.
* `face_engine/` must not import `faceproof/`.
* Blockchain functionality stays isolated from face-recognition code.
* Discovery code does not decide whether a candidate is the same person.
* Candidate verification does not perform blockchain operations.
* Evidence construction does not perform web discovery.
* Verification can run independently from the original pipeline.
* No circular imports.

---

## 14. Data Flow

The complete data flow is:

```text
Probe Image
    │
    ▼
Face Intelligence
    │
    ├── face detection
    ├── validation
    ├── alignment
    └── ArcFace embedding
    │
    ▼
Probe Embedding
    │
    ▼
Reverse Image Search
    │
    ▼
Candidate Set
    │
    ▼
Normalization / Deduplication / Priority Ranking
    │
    ▼
Candidate Budget
    │
    ▼
Candidate Image Retrieval
    │
    ▼
Candidate Face Verification
    │
    ▼
Ranked Matches
    │
    ▼
Best Verified Match
    │
    ▼
Evidence Manifest
    │
    ▼
Canonicalization
    │
    ▼
SHA-256 Fingerprint
    │
    ▼
Base Sepolia
    │
    ▼
Transaction Hash
    │
    ▼
Saved Evidence
    │
    ▼
Independent Verification
    │
    ▼
VERIFIED / TAMPERED
```

---

## 15. Evidence Storage

Evidence is stored locally in a dedicated directory.

Example:

```text
evidence/
├── 2026-09-04T094400-abc12345.json
└── 2026-09-04T094400-abc12345.match.jpg
```

The JSON contains:

* evidence manifest
* evidence fingerprint
* blockchain transaction information
* matched-image reference
* redacted search proof

The matched image is stored separately rather than embedding binary data directly inside JSON.

---

## 16. Configuration

Configuration is provided through environment variables.

| Variable                 | Required | Purpose                                                            |
| ------------------------ | -------- | ------------------------------------------------------------------ |
| `SERPAPI_API_KEY`        | yes      | Reverse image search                                               |
| `BLOCKCHAIN_PRIVATE_KEY` | yes      | Signs blockchain anchor transaction                                |
| `BLOCKCHAIN_RPC_URL`     | no       | Base Sepolia RPC endpoint; defaults to `https://sepolia.base.org`  |
| `FACE_UPLOAD_PROVIDER`   | no       | Anonymous host that publishes the resized probe so the search provider can fetch it: `catbox` (default), `litterbox`, `0x0` |
| `FACEPROOF_DOWNLOAD_WORKERS` | no   | Candidate image downloads in flight at once; default `4`, clamped to 1-8 |
| `FACEPROOF_COMPARE_WORKERS`  | no   | Candidate face comparisons at once; default `2`, clamped to 1-4. `1` is strictly sequential; verdicts are identical either way |

Example:

```env
SERPAPI_API_KEY=your_serpapi_key

BLOCKCHAIN_RPC_URL=https://sepolia.base.org

BLOCKCHAIN_PRIVATE_KEY=your_private_key

FACE_UPLOAD_PROVIDER=catbox
```

The provider table lives in `discovery/retrieval.py`. Each entry declares the
upload endpoint and the form fields it expects, so switching hosts is a config
change, not a code change. If the selected host is rate-limited or disabled, the
run aborts with a `DiscoveryError` naming the provider and the reason, and the
suggested fix is to select another provider.

Secrets are never committed to Git.

`.env` is ignored by Git and `.env.example` contains only placeholder values.

---

## 17. Entry Points

There are two entry points and one implementation. The Flask UI in `app.py`
calls the same `pipeline.run` / `pipeline.audit` functions the CLI calls, so the
UI cannot drift from the CLI behaviour.

```bash
uv run flask --app app run --port 8000   # browser UI
python -m faceproof run probe.jpg  # CLI
```

The browser UI serves four pages — `/` (run), `/verify` (re-verify and simulate
tampering), `/evidence` (every fingerprinted case on disk) and
`/replay/<investigation_id>` (cached replay of a stored investigation, also
reachable as `/evidence/<investigation_id>/replay`) — over a small JSON surface:
`POST /api/run`, `GET /api/run`, `POST /api/verify`,
`GET /api/replay/<investigation_id>`, `GET /candidate/<index>.jpg` and
`GET /evidence/<name>`. Probe uploads are capped at 16 MB.

Verification:

```bash
python -m faceproof verify evidence/<evidence-file>.json
```

The `run` command is responsible for:

```text
scan
  ↓
discover
  ↓
verify candidates
  ↓
build evidence
  ↓
hash
  ↓
anchor
  ↓
save evidence
```

The `verify` command performs only verification and does not repeat the discovery pipeline.

---

## 18. Failure Handling

FaceProof fails safely at each stage.

| Condition                              | Behaviour                         |
| -------------------------------------- | --------------------------------- |
| Probe contains no face                 | Abort before web search           |
| Probe contains unusable face           | Abort                             |
| Temporary upload fails                 | Retry once, then abort with a provider/reason discovery error |
| Reverse search returns an error        | Abort with useful error           |
| Candidate image unavailable            | Skip candidate                    |
| Candidate image has no face            | Skip candidate                    |
| Candidate verification fails           | Continue with next candidate      |
| No verified candidate                  | Do not create blockchain evidence |
| Blockchain transaction fails           | Abort anchoring                   |
| Blockchain receipt is unsuccessful     | Do not treat evidence as anchored |
| Local evidence hash differs from chain | Report `TAMPERED`                 |
| Search-response hash differs           | Report integrity failure          |
| Matched-image hash differs             | Report integrity failure          |

Partial evidence should not be presented as successfully anchored evidence.

---

## 19. Security and Integrity Boundaries

FaceProof separates three different concepts:

### Identity similarity

```text
Probe Face
     ↕
Candidate Face
     ↓
ArcFace similarity
```

This answers:

> Does the candidate image contain a face sufficiently similar to the probe?

### Source discovery

```text
Google Lens
     ↓
Candidate URL / image
```

This answers:

> Where did reverse image search find visually related content?

### Evidence integrity

```text
Evidence
   ↓
SHA-256
   ↓
Base Sepolia
```

This answers:

> Has the recorded evidence changed since it was anchored?

These are intentionally different claims and should not be conflated.

---

## 20. Deliberate Simplifications

The prototype intentionally avoids unnecessary infrastructure.

### Calldata anchor instead of registry contract

**Current ceiling:** transaction lookup requires the transaction hash.

**Upgrade:** smart-contract registry with indexed records.

### Temporary public image hosting

**Current ceiling:** reverse search requires a publicly reachable image URL and temporary hosting can be rate-limited.

**Upgrade:** controlled object storage with signed URLs.

### Candidate thumbnails

**Current ceiling:** reverse-search providers may return lower-resolution images, which can make facial comparison noisier.

**Upgrade:** retrieve the highest-quality publicly available candidate image.

### One candidate embedded at a time

**Current ceiling:** downloads and comparisons each run on a small bounded pool
(`FACEPROOF_DOWNLOAD_WORKERS`, `FACEPROOF_COMPARE_WORKERS`), but every candidate
is still embedded as its own single-image inference call.

**Upgrade:** batch the detection and embedding pass over several candidates per
call. Measurements are in `bench/FACE_VERIFICATION_PERFORMANCE.md`.

### Web-dependent recall

**Current ceiling:** a person with little or no indexed web presence may produce no useful candidates.

**Reason:** reverse image search cannot discover content that is not indexed or accessible to the search provider.

---

## 21. Demo Sequence

The recommended demonstration follows the complete trust chain, shown below on the CLI.

### Step 1 — Run FaceProof

```bash
python -m faceproof run probe.jpg
```

Show:

```text
[1/5] Face scan
      ✓ Face detected
      ✓ ArcFace embedding generated

[2/5] Web discovery
      ✓ Google Lens search completed
      ✓ Candidates discovered

[3/5] Candidate verification
      ✓ Candidates evaluated
      ✓ Best candidate verified

[4/5] Evidence
      ✓ Evidence manifest created
      ✓ SHA-256 fingerprint generated

[5/5] Blockchain
      ✓ Base Sepolia transaction confirmed
      ✓ Evidence anchored
```

### Step 2 — Verify

```bash
python -m faceproof verify evidence/<file>.json
```

Expected:

```text
✓ Local evidence integrity
✓ Search response integrity
✓ Matched image integrity
✓ Blockchain anchor
✓ Fingerprint comparison

RESULT: VERIFIED
```

### Step 3 — Tamper

Modify one value in the evidence file.

Run verification again:

```bash
python -m faceproof verify evidence/<file>.json
```

Expected:

```text
✗ Local evidence integrity
✗ Blockchain fingerprint comparison

RESULT: TAMPERED
```

This demonstrates the central property of FaceProof: **evidence can be independently checked after the original discovery process has finished.**

---

## 22. Architecture Summary

FaceProof combines four independent capabilities:

```text
FACE INTELLIGENCE
       │
       ▼
Identify / compare faces

       +

WEB DISCOVERY
       │
       ▼
Find potentially related public content

       +

EVIDENCE INTEGRITY
       │
       ▼
Create deterministic cryptographic fingerprint

       +

BLOCKCHAIN VERIFICATION
       │
       ▼
Anchor fingerprint and detect later tampering
```

The complete trust model is:

```text
                 FaceProof
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Identity Evidence      Source Evidence
          │                     │
          └──────────┬──────────┘
                     ▼
              Evidence Manifest
                     │
                     ▼
                  SHA-256
                     │
                     ▼
              Base Sepolia
                     │
                     ▼
          Independent Verification
                     │
             ┌───────┴───────┐
             ▼               ▼
         VERIFIED         TAMPERED
```

**FaceProof does not claim that blockchain proves a person's identity.**

Instead:

> **FaceProof uses facial similarity to verify a discovered candidate and blockchain anchoring to prove the integrity of the evidence generated from that verification.**

---

### 23. Cached Investigation & Deterministic Replay Architecture

A completed FaceProof investigation is persisted as an immutable evidence record in `evidence/<investigation_id>.json`. The replay subsystem provides deterministic, zero-network, zero-ML inspection of this historical record.

### 23.1 Dual-Path Trust Architecture

```text
                    LIVE
                     │
                     ▼
              FaceProof Pipeline
                     │
                     ▼
             Evidence JSON
                     │
                     ▼
                 Archive
                     │
            ┌────────┴────────┐
            ▼                 ▼
        VERIFY             REPLAY
            │                 │
            │                 ▼
            │          Read-only projection
            │          (faceproof/evidence/replay.py)
            │                 │
            │                 ▼
            │        Provenance + Explanation
            │                 │
            └───────┬─────────┘
                    ▼
             Independent Audit
```

### 23.2 Replay Invariants

Replay is an evidence-preserving inspection system, not an expiring application cache:

```text
REPLAY ≠ RE-RUN
REPLAY ≠ FRESH SEARCH
REPLAY ≠ NEW VERIFICATION
REPLAY ≠ NEW BLOCKCHAIN ANCHOR
```

1. **No External Resource Consumption**: Replay bypasses reverse web search (SerpApi), candidate image downloads, face detection (RetinaFace), face verification (ArcFace), and blockchain transaction submissions.
2. **Read-Only Projection**: `InvestigationReplay` is a read-only projection over `evidence/<id>.json`. It never writes to disk, alters timestamps, or changes recorded decisions.
3. **Transparent UI & Attribution**: The UI prominently displays the `CACHED REPLAY` badge and the transparency notice:
   > *No new web search or face verification was performed.*
4. **Active Local Integrity Verification**: Replay validates local integrity before rendering:
   - Manifest SHA-256 is recomputed and compared against the stored fingerprint.
   - Stored search proof response SHA-256 is verified against the manifest.
   - Matched image artifact hash is verified locally (`VERIFIED`, `INTEGRITY_FAILED`, or `MISSING`).
5. **Separation of Concerns**: Replay inspects historical findings; independent audit (`/verify`) re-reads Base Sepolia on-chain calldata.


