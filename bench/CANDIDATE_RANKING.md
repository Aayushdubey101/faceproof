# FaceProof — Candidate Ranking Report

Everything below is measured on the stored corpus, not estimated. Reproduce with:

```bash
uv run python bench/benchmark.py --label rank-before --mirror bench/mirror
uv run python bench/benchmark.py --label rank-after  --mirror bench/mirror
uv run python bench/compare.py rank-before rank-after
uv run python bench/rank_eval.py --mirror bench/mirror --refresh --verify
```

**Corpus.** `evidence/2026-09-05T142220-109a1d25.json` carries the full SerpApi
response it was built from: 59 candidates, replayable without spending a search
credit. `bench/mirror` serves all 59 candidate images from disk, so two runs see
identical bytes and the network cannot masquerade as a ranking effect.

**Ground truth.** `rank_eval.py --refresh` verifies **all 59** candidates once,
budget wide open, and caches the verdicts in `bench/results/groundtruth.json`.
Every quality number below is a lookup into those measured verdicts — nothing is
predicted, and no ordering is scored by a model.

Machine: Windows 11, CPU inference, TensorFlow 2.21.0, Python 3.11.15.

---

## 1. Before

Code before the ranking stage, `bench/results/rank-before.json` (mirror) and
`rank-before-live.json`:

```text
Discovered:    59
Unique:        59        (no deduplication stage existed)
Investigated:  5 / 10 / 25
Runtime:       7.12s / 12.85s / 31.35s   (mirror)
               7.57s / 14.61s / 32.47s   (live)
Best match:    distance 0.000000, x.com  (all budgets)
Verdicts:      n=5  MATCH 4, NO_MATCH 1
               n=10 MATCH 5, NO_MATCH 5
               n=25 MATCH 9, NO_MATCH 16
Preprocessing: 0.76 ms  (parse only)
```

## 2. After

`bench/results/rank-after.json`, `rank-after-b`, `rank-after-c` (mirror) and
`rank-after-live.json`:

```text
Discovered:    59
Unique:        59        (0 duplicates on this corpus - see section 3)
Investigated:  5 / 10 / 25
Runtime:       6.95s / 12.63s / 29.73s   (mirror, best of 3)
               8.10s / 14.45s / 31.90s   (mirror, worst of 3)
               7.39s / 14.07s / 33.41s   (live)
Best match:    distance 0.000000, x.com  (all budgets)
Verdicts:      n=5  MATCH 4, NO_MATCH 1
               n=10 MATCH 5, NO_MATCH 5
               n=25 MATCH 9, NO_MATCH 16
Preprocessing: 2.89 ms  (parse + normalize + deduplicate + rank)
```

### Runtime: no claim

**No speedup is claimed, because none was measured.** Three runs of the *same*
post-change code against the same single pre-change run give:

| after-run | n=5 | n=10 | n=25 |
|---|---:|---:|---:|
| `rank-after` | −13.8 % | −12.5 % | −1.8 % |
| `rank-after-b` | +2.3 % | +0.3 % | +5.2 % |
| `rank-after-c` | +2.4 % | +1.7 % | +3.8 % |

The spread between identical runs (29.73 s – 31.90 s at n=25) swallows the
before value (31.35 s) whole, so every one of those percentages is machine
noise, in both directions. That is the expected result: the budget still spends
exactly 25 face comparisons, and the ranking layer added **2.1 ms** of work to a
30-second run — about 0.007 % of it. Candidate ranking changes *which*
candidates are compared, not how fast a comparison is.

Where the time actually goes is unchanged and is documented in
`bench/PERFORMANCE.md`: face comparison is ~99 % of verification.

---

## 3. Deduplication

Measured across **every** stored corpus (13 evidence files, each
`search_response` replayed through the new `parse_visual_matches`):

```text
Raw candidates:      771
Unique candidates:   771
Duplicates removed:  0
```

Zero, on real data. Google Lens already deduplicates its own `visual_matches`:
all 59 entries of the benchmark corpus have distinct page URLs, distinct
thumbnails and distinct origin images, and so do the other twelve.

That is worth stating plainly rather than hiding: **on the corpora available,
deduplication is a no-op.** It is kept because it is exact, costs microseconds,
and the failure it prevents is real — a share link carrying `?utm_source=`
beside the plain URL would otherwise burn two of the budget's slots on one page.
Its behaviour is pinned by tests
(`test_the_same_page_reached_through_a_tracking_link_is_one_candidate`,
`test_a_duplicate_image_keeps_the_better_source_and_the_url_it_replaced`)
rather than by a number this corpus cannot produce.

Normalization is conservative by the same standard: of the 59 page URLs, 12
carry a query string, and none of those parameters are in `TRACKING_PARAMS`
(`?id=`, `?v=`, `?page=`, `?lang=`, `?locale=`, `?fl=`), so none were touched.

---

## 4. Ranking evaluation

Ground truth over all 59 candidates: **10 MATCH, 34 NO_MATCH, 15 NO_FACE, 0
DOWNLOAD_FAILED**. The closest verified match is at distance `0.000000` — the
corpus probe is that candidate's own image.

| ordering | best-match position | top-5 | top-10 | top-25 |
|---|---:|---:|---:|---:|
| provider (Google Lens order) | 2 | 3 | 4 | 8 |
| legacy (social-first, pre-ranking) | 2 | 4 | 5 | 9 |
| ranked (priority score, now) | 2 | 4 | 5 | 9 |

Matches found inside each budget, out of the 10 that exist.

```text
Top-5 success:   yes - best match at position 2 under all three orderings
Top-10 success:  yes
Top-25 success:  yes
```

The best match sits at position 2 in every ordering, so *finding the best match*
does not separate them on this corpus. What does separate them is how much of
the budget is spent usefully:

| ordering | wasted NO_FACE comparisons in top-25 | useful verdicts |
|---|---:|---:|
| provider | 6 | 19 of 25 |
| legacy | 0 | 25 of 25 |
| ranked | 0 | 25 of 25 |

Ranking keeps all 15 face-less results (stock photos, product pages, logo
thumbnails) out of the budget entirely, and buys 9 verified matches instead of 8
at the same cost.

### Honest reading

**`ranked` and `legacy` produce the identical order on this corpus, and that was
predictable.** The social bonus (3.0) dominates every other term, and within a
group the rank signal is monotone in position, so the new score reproduces
"social first, then position" exactly whenever the thumbnail term is constant —
and it is constant here, because all 59 entries carry a provider-cached
thumbnail. The measured gain of ranking is therefore **against the provider's
own order** (3→4, 4→5, 8→9 matches; 6→0 wasted comparisons), not against the
previous FaceProof release, which already applied the social heuristic.

What the ranking stage changed is that the heuristic is now an explicit, documented,
deterministic score with reasons attached, alongside normalization,
deduplication, provenance and an uninvestigated state — not that the order
moved.

### A signal that was tested and rejected

Thumbnail area looked like a plausible predictor of `NO_FACE`: a tiny crop gives
RetinaFace less to work with. Measured against the ground truth, it separates
nothing:

| status | median thumbnail area | min | max |
|---|---:|---:|---:|
| MATCH | 50 512 px² | 50 246 | 50 625 |
| NO_MATCH | 50 400 px² | 50 200 | 50 625 |
| NO_FACE | 50 325 px² | 44 400 | 50 540 |

Google Lens normalizes thumbnails to a near-constant area, so the signal carries
no information. It was not added. The raw dimensions stay in `groundtruth.json`
so the check can be repeated.

---

## 5. Correctness

Three independent checks, all on the same corpus.

**A/B by position** — `bench/compare.py`, which diffs best distance, best
candidate, the status sequence and every per-candidate ranking, status and
distance:

```
   n     before      after  improvement  correctness
   5      7.12s      6.95s         2.3%  identical {'MATCH': 4, 'NO_MATCH': 1}
  10     12.85s     12.81s         0.3%  identical {'MATCH': 5, 'NO_MATCH': 5}
  25     31.35s     29.73s         5.2%  identical {'MATCH': 9, 'NO_MATCH': 16}

correctness: no behavioural differences
```

Same verdict for `rank-before-live` vs `rank-after-live` and for both control
replicas. The percentages are noise; see section 2.

**A/B by candidate identity** — `rank_eval.py --verify` runs the real verifier
at each budget under both the legacy and the new ordering, then looks up every
investigated candidate by URL against the full-corpus verdicts:

```
legacy   budget   5: 5 investigated, 54 not investigated, verdicts identical to the full-corpus run
legacy   budget  10: 10 investigated, 49 not investigated, verdicts identical to the full-corpus run
legacy   budget  25: 25 investigated, 34 not investigated, verdicts identical to the full-corpus run
ranked   budget   5: 5 investigated, 54 not investigated, verdicts identical to the full-corpus run
ranked   budget  10: 10 investigated, 49 not investigated, verdicts identical to the full-corpus run
ranked   budget  25: 25 investigated, 34 not investigated, verdicts identical to the full-corpus run
```

**Summary:**

```text
Status differences:      0
Distance differences:    0
Best-match differences:  0   (distance 0.000000, x.com, before and after)
```

Model, detector, metric, threshold, embedding dimensionality and the
MATCH / NO_MATCH / NO_FACE / DOWNLOAD_FAILED rules are untouched. Ranking
decides what is looked at; ArcFace cosine distance against `0.68` decides what
it is.

---

## 6. Tests

```
uv run pytest tests/test_faceproof.py -q
57 passed
```

| Requirement | Test |
|---|---|
| URL normalization | `test_canonical_url_normalizes_only_what_is_safe` |
| Tracking-parameter removal | `test_tracking_parameters_are_removed_and_content_parameters_are_not` |
| Distinct content parameters stay distinct | `test_two_posts_that_differ_only_by_a_content_parameter_stay_separate` |
| Exact duplicate removal | `test_the_same_page_reached_through_a_tracking_link_is_one_candidate` |
| Duplicate provenance preserved | `test_a_duplicate_image_keeps_the_better_source_and_the_url_it_replaced` |
| Deterministic ranking | `test_ranking_is_deterministic_whatever_order_the_results_arrive_in` |
| Social/source prioritization | `test_a_social_post_outranks_a_better_placed_web_page` |
| Image availability prioritization | `test_a_provider_cached_thumbnail_outranks_an_origin_only_image` |
| Candidate budget | `test_the_candidate_budget_is_respected` |
| Uninvestigated is not rejected | `test_candidates_outside_the_budget_are_uninvestigated_not_rejected` |
| MATCH / NO_FACE / DOWNLOAD_FAILED unchanged under reordering | `test_reordering_candidates_changes_no_verification_verdict` |
| Probe embedding reuse intact | `test_the_probe_is_encoded_once_and_reused_for_every_candidate`, `test_a_caller_that_already_encoded_the_probe_does_not_encode_it_again` |

---

## 7. Limitations

- **One corpus.** Every number here comes from a single 59-candidate search for
  a single probe. The ranking weights are round numbers chosen to express a
  documented policy (social sources first), not fitted to data — but they have
  also not been validated across probes, and one corpus cannot validate them.
- **Deduplication is unmeasurable on the data available.** 0 duplicates in 771
  candidates; its correctness rests on tests, not on a measured reduction.
- **The cached-thumbnail term is untested in the field.** All 59 entries carry a
  provider thumbnail, so that weight never changed an ordering here.
- **The best match is trivially placed.** The corpus probe *is* one candidate's
  image, so the closest match is at distance 0 and near the top under every
  ordering. Top-k success does not discriminate; the NO_FACE-avoidance figure
  does, and that is what section 4 leans on.
- **Ranking is a heuristic and stays one.** A genuine match ranked below the
  budget is not investigated, and the system reports exactly that —
  uninvestigated, held in `Investigation.not_investigated` with no status, never
  `NO_MATCH`. Raising `--max-candidates` is the only thing that turns an
  uninvestigated candidate into a verdict.
