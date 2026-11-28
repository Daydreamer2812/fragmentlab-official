# Unified classical optimization — 2026-11-27

For the V14.2 curved missing-boundary update, see [14.2.0 release notes](releases/v14.2.0.md).

For the subsequent V14.1 scan improvements and paired evidence, see the
[14.1.0 release notes](releases/v14.1.0.md).

The resource caps documented here are historical for this update. See [subsequent irregular scaling](IRREGULAR_SCALE.md) for current limits.

This update keeps a single engine, default blind matching, optional explicitly
reference-assisted registration, and independent post-solve evaluation. No
OCR, trained model, text recognition, source-image matching in blind mode,
or synthesized output pixels have been introduced.

## Changes and limits

| Area | Improvement | Remaining limitation |
|---|---|---|
| Import and routing | Ignore malformed/incompatible public metadata with a warning; compare a bounded sample of actual tile dimensions; trim alpha padding before scan-strip recognition. | Incorrect masks, mixed cases, ambiguous geometry or corrupt images can still require intervention. |
| Physical paper strips | Rectify foreground shape, synchronize relative rotations using cleaned whole-strip projections, suppress column-wise background/shadow and compare local ink continuity. Balanced and High accuracy use a bounded deterministic index of observed interior half-patches; Fast omits it. Color photo strips retain color information. | Shared content and missing edge material can be ambiguous. Full placement is not verified ordering; absolute upright direction remains unknown. Maximum 128 strips. |
| Irregular / eroded pieces | Cache invariant distance transforms across spectral rotations, chunk FFT correlations (12 pairs), and verify candidate SE(2) triangles with distinct boundary arcs. | No gain in relative join accuracy on the severe DAFNE case below. Search remains expensive and partial. 128-piece deep and 256-piece physical caps are retained. |
| Reference-assisted | Balanced retries up to 32 unresolved tiny pieces against a finer index, without evicting accepted poses or relaxing the original photometric and occupied-area checks. | Requires an original image of the same scale and genuine distinct appearance. Automatic micro rescue is limited to reference images up to 8 MP. Larger native reference cases use the existing indexed path. |
| Evaluation | Wrapped DAFNE fragment ZIPs work in offline component evaluation; no matching truth IDs now raise an error rather than reporting meaningless zero scores. | Truth and genuine identity labels are needed to establish accuracy; neither enters blind inference. |

Patch recurrence uses nearest-neighbor distances over raw spatial pixel
samples inside the observed strips. It is not a fitted classifier, glyph
recognizer or language model. Only reconstructed observed raster pixels
are exported, not predictions used in seam scoring.

## Measured development-set checks

These are targeted regression/development cases, not an independent held-out
study. Solver timings vary with concurrent workload. No SOTA ranking is asserted.

### Blind physical scans (DocDataset)

Same three cases and deterministic anonymization. Truth ordering was read
after solving. An undirected join counts adjacent pieces in either order;
a directed join must have the correct left-to-right order.

| Case | Previous undirected | New undirected | Previous directed | New directed |
|---|---:|---:|---:|---:|
| real1_27 | 8/26 | 17/26 | 5/26 | 16/26 |
| real2_56 | 9/55 | 50/55 | 6/55 | 50/55 |
| real3_39 | 5/38 | 23/38 | 2/38 | 22/38 |
| **Total** | **22/119 (18.5%)** | **90/119 (75.6%)** | **13/119 (10.9%)** | **88/119 (73.9%)** |

Solver time summed over three cases: 17.57 s before, 42.68 s after.
Exact position counts after: 12/27, 0/56, 1/39; local seams are not
equivalent to complete reconstructed page order. Use Fast for cheaper
matching if the additional patch search is too slow.
[Detailed scan report](../reports/scan_strips.json).

### DAFNE with an explicitly supplied intact reference

Absolute pose criterion: center error at most 10 px and angle error at most
10 degrees; no gauge alignment. Strict pass also requires all distractors
rejected. These two development cases were selected because baseline Indexed
missed one genuine tiny fragment in each.

| Case | Previous Indexed | Balanced auto rescue | Distractors rejected | New solver time |
|---|---:|---:|---:|---:|
| Tarquinia / 2019-2-19_14.55.4 | 64/65 | 65/65 | 42/42 | 24.36 s |
| Resurrection / 2019-2-28_13.56.11 | 107/108 | 108/108 | 14/14 | 12.54 s |
| Tarquinia, portable dense-color path under browser caps | 64/65 | 65/65 | 42/42 | 23.84 s |

The portable check runs native Python with browser resource limits, not
actual browser WebAssembly. All three new runs also met the stricter
5-degree angle criterion. The historical 28 official cases remain reported
separately in [the indexed benchmark](DAFNE_INDEXED.md); they were not all
rerun with the updated default.

### DAFNE Resurrection 102 — blind, no original

87 genuine fragments and 15 distractors; the same damaged development input
as in [classical continuation results](ERODED_RECOVERY.md). A single global
rigid gauge is applied only in the offline evaluator.

| Metric | Previous | Optimized |
|---|---:|---:|
| Correct primary poses | 3/87 | 3/87 |
| Correct relative joins | 6/16 | 6/16 |
| Relative join precision | 37.5% | 37.5% |
| Non-singleton components | 14 | 14 |
| Wall time (not controlled) | 292.3 s | 283.2 s |

The closed-cycle check found no selected cycle-supported join on this input.
Its value is corroborating a plausible local seam, not inferring unobserved
structure. The difficult blind case remains **unsolved**. No addition to
the score is inferred from coverage or from alternative components.

### Grid regression

Synthetic texture, 4096 16-pixel tiles, seed 911: 4096/4096 placed,
rotation-aligned DPA = 100%, adjacency F1 = 100%, no collision pixels,
solver time 2.08 s, peak process RSS 126.75 MiB. The global quarter-turn is
a gauge ambiguity (raw DPA was zero). This does not verify 65,000-piece
real torn-image reconstruction.

A previous user-provided 432-fragment ZIP also reconstructed with automatic
18×24 grid inference, 432 placed, full coverage and zero collision pixels,
21.41 s solver time. It has 31 identical pieces and no independent truth,
so the result is correctly labeled ambiguous, not 100% accurate.

### Test status

- Native C++ path: 112 tests passed.
- Python fallback: 109 tests passed, 3 native-specific tests skipped.
- Workflow/static build and offline evaluator: 2 tests passed.
- Actual browser smoke is not executed locally (no Chromium); existing CI
  runs the real browser checks. The portable native check is not a
  substitute for the browser.

[Machine-readable measurements](../reports/optimization_round.json).

## Reproduction

```sh
python -m pip install -e '.[reference]'
python scripts/build_native.py
python -m unittest discover -s tests/algorithms
SHARDELIX_CORE=python python -m unittest discover -s tests/algorithms
python -m unittest discover -s tests/integration
python benchmarks/benchmark_scanned_strips.py DocDataset.zip --profile balanced
python benchmarks/benchmark_indexed_dafne.py 00.zip result00 --case 2019-2-19_14.55.4 --engine auto
python benchmarks/benchmark_indexed_dafne.py 24.zip result24 --case 2019-2-28_13.56.11 --engine auto
python -m shardelix reconstruct DAFNE_Resurrection_102.zip --out blind
python benchmarks/evaluate_dafne_components.py DAFNE_Resurrection_102.zip truth.zip blind
python benchmarks/benchmark_scale_v8.py --count 4096 --tile 16 --out grid4096
```

Original datasets and truth archives are not included. Blind evaluation
runs separately from reconstruction. Browser limits and incomplete-data
ambiguity remain explicit rather than weakening resource guards or claiming
unverified reconstruction accuracy.
