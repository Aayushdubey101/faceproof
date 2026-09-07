# FaceProof — Latency Report

Everything below is measured, not estimated. Reproduce with:

```bash
uv run python bench/benchmark.py --label baseline
uv run python bench/benchmark.py --label optimized
uv run python bench/compare.py baseline optimized
```

**Harness.** `bench/benchmark.py` replays the SerpApi response stored inside
`evidence/2026-09-05T142220-109a1d25.json` (59 candidates), so two runs see the
same candidate corpus and the only variable is the code under test. No search
credit is spent. `--mirror bench/mirror` additionally serves candidate images
from disk, removing network variance entirely — that is the correctness control,
not a latency claim.

Machine: Windows 11, CPU inference, TensorFlow 2.21.0, Python 3.11.15.
Raw JSON per run is in `bench/results/`.

---

## 1. Baseline (before any change)

Live network, warm process, `bench/results/baseline.json`:

| n candidates | Total | Probe scan | Verification | Candidate downloads | Face comparison | Evidence |
|---:|---:|---:|---:|---:|---:|---:|
| 5  |  8.91 s | 1.34 s |  7.57 s | 1.59 s | 5.97 s | < 0.01 s |
| 10 | 16.45 s | 1.34 s | 15.12 s | 2.83 s | 12.27 s | < 0.01 s |
| 25 | 38.91 s | 1.36 s | 37.54 s | 7.04 s | 30.43 s | < 0.01 s |

Model init: **cold 15.31 s, warm 1.40 s.** The second `scan_face` in the same
process is 11x faster, which confirms the existing `cached_models` layer already
builds each model once — no second loading mechanism was added.

Stages the harness does not replay, measured once on a live run:
probe upload **4.53 s**, reverse image search **8.40 s**, probe resize **0.03 s**.
These are provider round trips, not our code.

Evidence generation is below the harness's 3-decimal rounding (< 10 ms):
canonical JSON plus one SHA-256 over a few KB.

**Bottleneck before:** face comparison, 78-81 % of verification
(30.43 s of 37.54 s at n=25, about 1.22 s per candidate). Candidate downloads
were 18-21 %, and they were *serial* — every one of them sat on the critical
path.

---

## 2. What changed

One optimization: **candidate downloads are prefetched on a bounded thread
pool** (`faceproof/matching/verifier.py`, `_prefetch`). A thumbnail is ~7 KB, so
downloading is round-trip latency, while comparison is CPU-bound — fetching
candidate *n+1* while comparing candidate *n* hides the network behind work that
had to happen anyway.

- `ThreadPoolExecutor(max_workers=download_workers())`, `pool.map` — results come
  back in submission order, so candidate ordering, identity, source URL and
  failure status are what the sequential loop produced.
- `FACEPROOF_DOWNLOAD_WORKERS`, default 4, clamped to 1-8. Never unlimited.
- Two candidates whose image URL matches **exactly** are fetched once and both
  still get their own verdict. No content or perceptual hashing — a perceptual
  hash is not an identity decision.
- A download that fails or raises still yields `DOWNLOAD_FAILED` and the run
  continues.
- Every external call keeps its existing bound: downloads `(5 s connect, 20 s
  read)`, upload `(10, 90)` with 2 attempts, search `(10, 60)`. Concurrency does
  not extend any of them; no candidate is waited on indefinitely.

**Comparison was left sequential on purpose.** Measured over 8 mirrored images at
1 / 2 / 4 comparison workers: 8.86 s / 7.77 s / 7.49 s, distances byte-identical.
A 15 % gain in exchange for several decoded images in memory and concurrent
RetinaFace/ArcFace graph use is not a trade worth its failure modes.

---

## 3. Optimized

Live network, `bench/results/optimized.json`, `FACEPROOF_DOWNLOAD_WORKERS` unset
(default 4):

| n candidates | Before | After | Improvement |
|---:|---:|---:|---:|
| 5  |  8.91 s |  7.66 s | **14.0 %** |
| 10 | 16.45 s | 12.55 s | **23.7 %** |
| 25 | 38.91 s | 28.25 s | **27.4 %** |

`(before - after) / before * 100`, straight from `bench/compare.py`.

The gain scales with candidate count because the serial download time it removes
does. Read honestly: at n=25 the download stage was 7.04 s of a 38.91 s baseline
(18 %), and mean comparison time also drifted 1.217 s to 1.071 s between the two
live runs (machine variance, not our change). The control run says the same thing
from the other side — with the network removed entirely, baseline and optimized
are the same speed at n=25 (28.32 s vs 28.16 s, 0.6 %), because there is nothing
left to overlap. And optimized-live (28.25 s) now matches optimized-mirror
(28.16 s): **the network is fully hidden.**

Per-candidate download wall time *rose* (7.04 s to 8.32 s summed at n=25) — four
transfers sharing bandwidth each take longer. Summed thread time is not latency;
the totals above are.

**Bottleneck after:** face comparison, 26.78 s of 27.03 s verification at n=25 —
**99 %**. Everything else is now noise. The remaining lever is the model, and the
model is out of scope.

---

## 4. Correctness

`bench/compare.py` diffs the two runs on best distance, best candidate, status
sequence, per-candidate ranking, status and distance:

```
   n     before      after  improvement  correctness
   5      8.91s      7.66s        14.0%  identical {'MATCH': 4, 'NO_MATCH': 1}
  10     16.45s     12.55s        23.7%  identical {'MATCH': 5, 'NO_MATCH': 5}
  25     38.91s     28.25s        27.4%  identical {'MATCH': 9, 'NO_MATCH': 16}

correctness: no behavioural differences
```

Same verdict on the mirrored corpus (`baseline-mirror` vs `optimized-mirror`),
where the bytes are identical by construction. Threshold, model, detector, metric
and ranking are untouched.

---

## 5. Tests

```
uv run pytest tests/test_faceproof.py -q
47 passed in 2.23s
```

Covering, specifically:

| Requirement | Test |
|---|---|
| Probe embedding computed once | `test_the_probe_is_encoded_once_and_reused_for_every_candidate`, `test_a_caller_that_already_encoded_the_probe_does_not_encode_it_again` |
| Candidate ordering preserved | `test_out_of_order_downloads_keep_candidate_order_and_identity` |
| Concurrency preserves identity | same test — each result's bytes came from that candidate's own URL, with downloads finishing in reverse order |
| Bounded workers | `test_downloads_never_exceed_the_configured_worker_count`, `test_the_download_worker_count_stays_within_bounds` |
| Download failure isolation | `test_download_failure_does_not_end_the_investigation` |
| Duplicate candidates | `test_the_same_image_url_is_fetched_once_and_every_candidate_still_reported` |
| Matching output unchanged | `test_the_closest_verified_candidate_wins_regardless_of_order`, `test_investigation_records_every_candidate_outcome` |
| Evidence generation | `test_evidence_document_is_stable_under_reordering`, `test_the_probe_embedding_never_reaches_the_evidence` |

---

## 6. Files changed

| File | Change |
|---|---|
| `faceproof/matching/verifier.py` | `download_workers()`, `_Fetched`, `_prefetch()`; the verification loop consumes prefetched images |
| `tests/test_faceproof.py` | 4 tests for ordering, identity, bounded concurrency and duplicates |
| `app.py` | stage timings are timestamped as the pipeline prints them; `/api/run` returns real elapsed seconds |
| `templates/`, `static/` | the timeline renders measured per-stage seconds; the hardcoded "~35 sec" estimate is gone |
| `bench/` | benchmark harness, A/B differ, this report |
| `.env.example`, `README.md`, `ARCHITECTURE.md` | `FACEPROOF_DOWNLOAD_WORKERS` |

Not touched: `face_engine/`, the pipeline stages, hashing, evidence, blockchain.

---

## 7. Architecture impact

Nothing structural. `verify_candidates` keeps its signature and its contract; the
only new concept is a private prefetch generator whose pool is created per call,
so no cache and no thread outlives an investigation and stale web evidence cannot
leak between runs. Deleting `_prefetch` and inlining `download_image` returns the
module to the sequential version.
