# FaceProof — Face Verification Performance

Download prefetching took the network off the critical path and candidate
ranking chose which candidates are worth comparing. That left face comparison as
~99 % of a run, so this is the report on taking that number apart and reducing
it.

Reproduce with:

```bash
uv run python bench/profile_verify.py --mirror bench/mirror --repeats 3
FACEPROOF_COMPARE_WORKERS=1 uv run python bench/benchmark.py --label workers-1 --mirror bench/mirror
uv run python bench/benchmark.py --label workers-2 --mirror bench/mirror
uv run python bench/compare.py workers-1 workers-2
uv run python bench/inference_experiment.py --label w1 --workers 1
uv run python bench/inference_experiment.py --compare w1-faces w2-faces w4-faces
```

Machine: Windows 11, 12 logical CPUs, CPU-only inference, TensorFlow 2.21.0,
Python 3.11.15. Corpus: `evidence/2026-09-05T142220-109a1d25.json` (59
candidates) served from `bench/mirror`, so the network is not a variable.

Nothing about the identity decision changed: RetinaFace, ArcFace, 512-d
embeddings, cosine distance, threshold 0.68.

---

## 1. Per-stage timing

`bench/profile_verify.py` wraps the engine's own seams **in-process** — no file
under `face_engine/` was edited. 25 candidates x 3 repeats = 75 comparisons:

| Stage | mean | median | p95 | total | share |
|---|---:|---:|---:|---:|---:|
| Decode (`load_image`) | 0.0031 s | 0.0016 s | 0.0061 s | 0.23 s | 0.3 % |
| **RetinaFace inference** | **0.9852 s** | **0.9883 s** | **1.2192 s** | **73.89 s** | **89.5 %** |
| Detection total (border + crop) | 0.9857 s | 0.9886 s | 1.2196 s | 73.93 s | 89.5 % |
| Alignment (`extract_face`) | 0.0004 s | 0.0003 s | 0.0009 s | 0.03 s | 0.04 % |
| ArcFace embedding | 0.1123 s | 0.1070 s | 0.1332 s | 8.42 s | 10.2 % |
| Cosine distance | 0.0001 s | 0.0001 s | 0.0001 s | 0.01 s | 0.01 % |
| **Whole comparison (wall)** | 1.1014 s | 1.0997 s | 1.3401 s | 82.60 s | 100 % |

Download is not in this table because prefetching already removed it from the
critical path; it is measured in `bench/PERFORMANCE.md` (7-8 s of thread time
across 25 candidates, fully hidden behind comparison).

```text
Primary bottleneck:    RetinaFace detection, 89.5% of a comparison
Secondary:             ArcFace embedding, 10.2%
Effectively free:      decode 0.3%, alignment 0.04%, cosine 0.01%
```

Those three "free" stages add up to **0.35 %**. There is no optimization worth
making there, whatever it looks like in the code.

Candidate images are already small: median **50,440 px** (about 275x183), min
50,225, max 50,625 — Google Lens normalizes its thumbnails. Detection is
nevertheless expensive because `face_engine` pads the image with a 50 % black
border on each side before detecting, so RetinaFace actually runs on ~550x366.
That padding lives in `face_engine/modules/detection.py` and was left alone.

---

## 2. Baseline and optimized

`bench/benchmark.py`, mirrored corpus, three interleaved repeats of each
configuration (before = `FACEPROOF_COMPARE_WORKERS=1`, after = default 2):

### Baseline

```text
 5 candidates:  7.92s, 6.60s, 9.78s    ->  median  7.92s
10 candidates: 13.58s, 12.19s, 14.67s  ->  median 13.58s
25 candidates: 32.60s, 28.98s, 28.97s  ->  median 28.98s
```

### Optimized

```text
 5 candidates:  6.30s, 6.06s, 6.62s    ->  median  6.30s
10 candidates: 11.69s, 10.73s, 12.20s  ->  median 11.69s
25 candidates: 26.50s, 24.44s, 28.18s  ->  median 26.50s
```

### Improvement

| n | Before (median) | After (median) | Improvement | Verification stage only |
|---:|---:|---:|---:|---:|
| 5 | 7.92 s | 6.30 s | **20.5 %** | 6.47 s -> 5.01 s (22.6 %) |
| 10 | 13.58 s | 11.69 s | **13.9 %** | 12.17 s -> 10.40 s (14.5 %) |
| 25 | 28.98 s | 26.50 s | **8.6 %** | 27.68 s -> 25.07 s (9.4 %) |

Medians of three runs. The individual pairs ranged from +2.7 % to
+32.3 % at the same candidate count, which is why no single pair is quoted as
the result. The tighter measurement is the isolated comparison loop, where the
only thing running is the thing being measured:

| workers | median | best | worst | vs 1 worker |
|---:|---:|---:|---:|---:|
| 1 | 28.90 s | 28.70 s | 29.45 s | — |
| 2 | 24.12 s | 24.04 s | 24.19 s | **16.5 %** |
| 4 | 22.98 s | 22.84 s | 23.26 s | 20.5 % |

Three repeats each, 25 candidates, `bench/results/inference-w{1,2,4}-faces.json`.

---

## 3. What was implemented

One change: **candidate comparisons run on a bounded pool**
(`faceproof/matching/verifier.py`, `_compared`), configured by
`FACEPROOF_COMPARE_WORKERS`, default **2**, clamped 1-4.

- Only `adapter.compare_faces` leaves the main thread. Recording, printing,
  ranking and `Match` construction stay on one thread, so an `Investigation` is
  filled in candidate order no matter which comparison finishes first.
- `pool.map` yields in submission order, so candidate identity, ordering,
  `DOWNLOAD_FAILED` and `NO_FACE` handling are what the sequential loop
  produced.
- One worker skips the pool entirely rather than paying for a thread that can
  never overlap with anything.
- Bounded, never unlimited: 4 is the ceiling and 2 is the default.

Why 2 and not 12? Because one comparison already keeps most of the machine busy:

| workers | cores busy (of 12) | median | peak memory |
|---:|---:|---:|---:|
| 1 | 7.62 | 27.80 s | 1713 MiB |
| 2 | 8.96 | 23.80 s | 2256 MiB |
| 4 | 9.54 | 22.24 s | 2784 MiB |

RetinaFace's own intra-op parallelism already occupies 7.6 cores. A second
worker recovers about 1.3 more; a fourth recovers 0.6 more and costs another
0.5 GiB. The curve is flat after 2, so the default stops there.

---

## 4. Resource usage

```text
Memory:  1713 MiB peak at 1 worker
         2256 MiB peak at 2 workers   (+543 MiB, +32%)
         2784 MiB peak at 4 workers   (+1071 MiB, +63%)
         1162 MiB of that is the two models, loaded once (process starts at 32 MiB)

CPU:     7.62 of 12 cores busy at 1 worker
         8.96 of 12 cores busy at 2 workers
         9.54 of 12 cores busy at 4 workers
```

Memory is the process working set (`GetProcessMemoryInfo`), sampled every 250 ms
during the comparison loop; CPU is `time.process_time()` over wall time. Both
are recorded in `bench/results/inference-*.json`.

The default trades **+543 MiB for -16.5 %**. Four workers would buy a further
4 % for another 528 MiB, memory the run does not need — so 4 is
available, but not the default.

---

## 5. Experiments

| Experiment | Tested | Verdict | Reason |
|---|---|---|---|
| Model caching | yes | **already correct, unchanged** | `build_model` is *called* 227 times across 75 comparisons but returns **one** RetinaFace instance and **one** ArcFace instance — verified by object identity, not call count. Cold 12-15 s, warm 1.3-1.6 s. Nothing to fix, so nothing was rewritten. |
| Probe reuse | yes | **already correct, unchanged** | The pipeline encodes the probe once in stage 1 and passes `probe_embedding=` down; `verify_candidates` never re-encodes it. Pinned by `test_the_probe_is_encoded_once_and_reused_for_every_candidate` and `test_a_caller_that_already_encoded_the_probe_does_not_encode_it_again`. |
| Preprocessing | yes | **rejected** | Decode is 0.3 % of a comparison and alignment 0.04 %. Passing decoded arrays instead of file paths would remove at most 3 ms per candidate and would change the adapter's contract for no measurable gain. |
| Image resizing | yes | **rejected** | Candidate images are already ~50,440 px (275x183), with 400 px between the smallest and the largest. There are no "unnecessarily huge" images to bound, and shrinking a 275 px thumbnail to speed up detection is exactly the correctness-for-latency trade this project refuses to make. |
| Inference concurrency | yes | **accepted at 2 workers** | 16.5 % faster, verdicts byte-identical, stable across 5 repeats. 4 workers adds 4 % for 528 MiB more; 3 sits between them. Default 2, ceiling 4. |
| Batching | yes | **rejected** | `verification.verify` handles exactly one image pair and `detection.detect_faces` one image, so batching would mean rewriting `__extract_faces_and_embeddings` and the RetinaFace client inside `face_engine/`. `face_engine/` is kept unmodified, so that is out of scope, and the ceiling is low anyway: ArcFace, the only stage with a natural batch dimension, is 10.2 % of the cost. |
| Thread tuning | yes | **rejected** | `TF_NUM_INTRAOP_THREADS=6` on 12 CPUs measured **30.66 s vs 26.82 s** — 14 % *slower*, across three consistent runs — at one worker, and 28.21 s vs 23.31 s at two. The TF default already picks the right number; forcing fewer threads starves the detector. `tf.config.threading.get_intra_op_parallelism_threads()` reports `0` (auto) either way, so the reproducible slowdown is what proves the variable took effect. |
| Per-investigation result cache | yes | **rejected as unmeasurable** | Duplicate work is already avoided twice over: candidate deduplication merges candidates whose canonical page **or** image URL matches, and `_prefetch` fetches an identical image URL once. Across all 13 stored corpora there are 0 duplicate URLs in 771 candidates, and the 59 mirrored images have 59 distinct SHA-256s. A third cache layer would have nothing to hit. Verified instead by `test_every_investigated_candidate_is_compared_exactly_once`. |
| Early stopping on a good match | not implemented | **rejected by design** | A match at 0.10 does not prove that no later candidate scores 0.05. `verify_candidates` still compares every budgeted candidate and returns the minimum distance. |

---

## 6. Correctness

```text
Status differences:      0
Distance differences:    0
Best-match differences:  0
Ranking differences:     0
```

Four independent checks.

**1. Before/after by position** — `bench/compare.py`, all three interleaved
pairs:

```
   n     before      after  improvement  correctness
   5      7.92s      6.30s        20.5%  identical {'MATCH': 4, 'NO_MATCH': 1}
  10     13.58s     11.69s        13.9%  identical {'MATCH': 5, 'NO_MATCH': 5}
  25     32.60s     26.50s        18.7%  identical {'MATCH': 9, 'NO_MATCH': 16}

correctness: no behavioural differences
```

Pairs 2 and 3 print the same verdict. Best distance `0.000000`, best source
`x.com`, at every budget, before and after.

**2. By candidate identity, including face selection** —
`bench/inference_experiment.py --compare` diffs status, distance **and the
number of candidate faces the probe was scored against**, which is the
face-selection behaviour:

```
label      workers   median     best    worst  peak MiB  cores   vs ref  correctness
w1-faces         1   28.90s   28.70s   29.45s      1614   7.62     0.0%  identical
w2-faces         2   24.12s   24.04s   24.19s      2502   8.96    16.5%  identical
w4-faces         4   22.98s   22.84s   23.26s      3082   9.54    20.5%  identical

correctness: no differences
```

**3. Determinism across repeats** — every pass of a configuration must agree
with every other pass, or the configuration is nondeterministic and cannot ship
at any speed. `results_stable: true` at 1, 2 and 4 workers, five repeats each.

**4. Ranking and the full corpus** — `bench/rank_eval.py --verify`, re-run under
the new default, reproduces the ranking report's table exactly (provider 3/4/8, legacy
4/5/9, ranked 4/5/9, best match at position 2) and reports `verdicts identical
to the full-corpus run` at every budget under both orderings.

---

## 7. Tests

```
uv run pytest tests/test_faceproof.py -q
67 passed
```

| Requirement | Test |
|---|---|
| Probe embedding reuse | `test_the_probe_is_encoded_once_and_reused_for_every_candidate`, `test_a_caller_that_already_encoded_the_probe_does_not_encode_it_again` |
| Model reuse | `test_the_probe_is_encoded_once_and_reused_for_every_candidate` at the FaceProof seam; engine-level model identity is verified by `bench/profile_verify.py` (227 calls, 1 instance each), because loading a real model needs TensorFlow and the suite is offline |
| Candidate verification consistency | `test_reordering_candidates_changes_no_verification_verdict`, `test_investigation_records_every_candidate_outcome` |
| No duplicate verification | `test_every_investigated_candidate_is_compared_exactly_once`, `test_the_same_image_url_is_fetched_once_and_every_candidate_still_reported` |
| Single-face behaviour | `test_the_probe_is_encoded_once_and_reused_for_every_candidate` |
| Multi-face behaviour | `test_a_multi_face_probe_keeps_the_exact_image_comparison` |
| No-face behaviour | `test_reordering_candidates_changes_no_verification_verdict`, `test_investigation_records_every_candidate_outcome` |
| Image preprocessing consistency | `test_out_of_order_downloads_keep_candidate_order_and_identity` — the bytes recorded for a candidate are the bytes its own URL produced |
| Concurrent candidate mapping | `test_concurrent_comparisons_keep_candidate_order_and_identity`, parametrized at 1, 2 and 4 workers with comparisons finishing in reverse order |
| Bounded workers | `test_the_compare_worker_count_stays_within_bounds` |
| Old-vs-new output equality | `bench/compare.py`, `bench/inference_experiment.py --compare`, `bench/rank_eval.py --verify` — all of section 6 |

---

## 8. Limitations

- **RetinaFace is still 89.5 % of a comparison, and it stays there.** The only
  ways past it are a different detector, a smaller input, or a batch API inside
  `face_engine/`. The first two change the identity decision; the third changes
  a file this project keeps unmodified. What was optimized is how many detections run
  at once, not how long one takes.
- **The remaining headroom is about 2.4 cores.** At 2 workers the process keeps
  8.96 of 12 cores busy; perfect scaling would be 12. Most of the gap is
  single-threaded work inside one detection, which a caller cannot parallelize
  away.
- **Memory is the price.** +543 MiB at the default. On a machine that cannot
  spare it, `FACEPROOF_COMPARE_WORKERS=1` restores the old profile exactly, at
  the old speed.
- **One machine, one corpus.** 12 logical CPUs, CPU-only inference. On a
  four-core laptop a single comparison would already saturate the machine and
  the concurrency gain would shrink; on a GPU the whole profile would look
  different.
- **End-to-end run-to-run spread is wide** — 2.7 % to 32.3 % across matched
  pairs at the same candidate count. The medians above are the honest summary;
  the isolated comparison loop, where the spread is under 1 %, is the tighter
  measurement of the same change.
