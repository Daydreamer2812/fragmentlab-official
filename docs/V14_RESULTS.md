# Shardelix V14 results and reproducibility

V14.0.0 packages the integrated classical reconstruction engine on main
(baseline `42cd2a75db30b219699c80388cef88518bd25d4f`) and adds a lossless
C++ chain-code indexing optimization. This is a CPU-first **hybrid**
Python / C++20 / WebAssembly framework, not a C++-only rewrite.
No OCR, ML, pretrained weights or intact source image are used in blind
reconstruction. Reference-assisted registration is optional and explicit.

## Physical document strips — V14.5

See the [current document-strip release](releases/v14.5.0.md) and
[paired real-scan and external mechanical-strip measurements](../reports/v14/scan_document_v145.json).
Historical V14.4 [release](releases/v14.4.0.md) and [report](../reports/v14/scan_document_v144.json).
Historical V14.3 [release](releases/v14.3.0.md) and [report](../reports/v14/scan_document_v143.json).
This is a separate pixel-only protocol, not an irregular-pose benchmark.

## 1. New V14 paired native benchmark

The prior exact arc index stored both forward and reverse 64-bit normalized
chain codes and performed a binary search for the reverse of every distinct
code. V14 stores a single canonical key with a two-bit direction tag:
the 62-bit normalized chain code fits because the window is at most 32.
One sort places complementary directions together for a linear scan.
Self-complementary chains are handled once per unordered pair. Rare-bucket
filtering, piece-ID direction, shard partitioning, vote thresholds and
exact raster seam verification are preserved; no candidate pruning was
added. A single-shard path also avoids redundant hash/modulo operations.
The ABI is unchanged and Python fallback remains available.

| Sequential run | Prior solver (s) | V14 solver (s) | Prior peak RSS (MiB) | V14 peak RSS (MiB) |
|--:|--:|--:|--:|--:|
| 1 | 5.482 | 3.498 | 230.3 | 212.6 |
| 2 | 5.973 | 4.309 | 229.9 | 212.7 |
| 3 | 5.585 | 3.722 | 229.8 | 212.6 |
| **Median** | **5.585** | **3.722** | **229.9** | **212.6** |

Median solver-time speedup **1.50×** (33.4% less elapsed solver time);
median peak RSS reduction **7.5%**. All six runs placed 4,096/4,096
correctly in one rigid component, with 4,095/4,095 correct accepted joins,
92,746 voted poses, 38,340 verified poses, 38,264 verified pairs,
8,064 accepted candidate seams and zero collision rejections.
Native arc-vote regression tests compare exact vote rows against the Python
reference across multiple windows, shard counts and rare-bucket caps.

Anonymous intact cell-lattice synthetic fixture: 4,096 pieces, nominal
36 px tile, seed 827, quarter-turns, fixed maximum memory estimate 512 MiB.
Each process is independent; wall time varies on a shared development runner.
Solver time excludes fixture creation, I/O, rendering and offline evaluation.
RSS includes fixture, solver and evaluation. This is not an end-to-end
browser comparison, physical scan test, or general speed guarantee.
[All raw runs and environment](../reports/v14/paired_arc.json).

### Reproduce

Use separate checkouts or worktrees of the V14 release commit and baseline
`42cd2a75db30b219699c80388cef88518bd25d4f`, with the same Python dependencies,
compiler and machine. In each checkout, run:

```sh
python -m pip install -e .
python scripts/build_native.py
PYTHONPATH=src python scripts/stress_irregular_resources.py --count 4096 --tile 36 --seed 827 --memory-mib 512 --out run.json
```

Alternate revisions, repeat at least three times, and report run-level data
and median. No ground truth or known canvas enters the matching function.

## 2. Integrated scaling evidence (historical native measurements)

[Resource scaling](RESOURCE_SCALING.md) and its [raw report](../reports/resource_scaling.json)
recorded the integrated exact-lattice solver at 8,192, 16,384, 32,768,
65,536 and 100,000 fragments with 100% exact primary pose accuracy and
all accepted joins correct. The largest run used 177.32 s solver time and
2,664.1 MiB peak RSS; a 131,072-piece resource probe abstained at preflight.
The 8,192-piece ZIP workflow measured 58.09 s, with all poses correct
and zero collision pixels. These rows predate the new V14 canonical index
and are not freshly rerun V14 results. Do not extrapolate the paired
4,096-piece speedup to these larger counts.

## 3. Blind degraded fragments (integrated development measurements)

The latest [unified missing-edge measurements](../reports/unified_missing_edge_recovery.json)
predate V14 packaging. Their baseline is `d67f830` and revised source is
`42cd2a7`. The reference image is excluded, and annotations are accessed only
after prediction.

| Experiment | Baseline / initial | Revised |
|:--|--:|--:|
| Resurrection, genuine primary poses | 3/87 | 3/87 |
| Resurrection, correct / accepted relative joins | 6/16 | 6/13 |
| Resurrection, relative join precision | 37.5% | 46.2% |
| Resurrection, joins to distractors | 1 | 0 |
| Masaccio, false relative joins | 64 | 11 |
| Masaccio, correct relative joins | 0 | 0 |
| Synthetic small erosion (11 cases), correct / accepted joins | 30/61 | 30/61 |
| Severe 137-observed synthetic case, correct / accepted joins | 0/0 | 0/0 |

Masaccio's initial result exposed a routing regression; it is not an
independently held-out case. A singleton's correct pose after a free gauge
alignment is only an anchor, not evidence of reconstruction.
The severe 137-piece synthetic case now searches beyond the previous
128-piece deep-recovery cutoff, but finds no reliable seams. These
results quantify fewer false links and bounded abstention, **not a solved
blind DAFNE challenge**. See [eroded recovery](ERODED_RECOVERY.md).

## 4. Reference-assisted DAFNE (separate historical protocol)

[Indexed DAFNE report](DAFNE_INDEXED.md) covers 28 official configurations
(25 DB1, 3 DB2): 6,027/6,029 genuine poses correct at 10 px / 10 degrees
absolute tolerance, 1,154/1,154 distractors rejected and 26/28 strict passes.
The intact artwork was explicitly provided to matching. The [catalog](DAFNE_OFFICIAL_BENCHMARK.md)
contains 1,119 configurations, not 1,119 completed tests.
Do not compare these numbers with blind reconstruction or aggregate
them with the blind benchmark.

## 5. Metrics and resource policy

- **Exact global poses:** exact rigid poses of the primary component after
  one global gauge alignment, applicable to the intact lattice benchmark.
- **Relative join precision:** correct relative joins / accepted joins,
  evaluated independently in each disconnected component. Not adjacency
  recall or F1, and undefined when no join is accepted.
- **DPA / adjacency F1:** digital-position and neighbor metrics retained in
  [V12](V12_RESULTS.md) and [V13](V13_RESULTS.md); not substitutes for physical
  pose precision or coverage.
- **Absolute pose:** reference-assisted placement in original artwork
  coordinates, with no free global alignment.

[Framework](FRAMEWORK.md) · [Resource policy](RESOURCE_SCALING.md) ·
[Browser and native limits](WEB.md) · [Datasets](DATASETS.md)

No measured SOTA ranking is claimed. Native memory estimates, ZIP and
pixel limits are guardrails, not guaranteed capacities. Actual WebAssembly
timing, real physical scan recall, arbitrary perspective and very weak
or missing edges remain limitations. Missing pixels are never synthesized.

## 6. Release verification

Local verification on Python 3.12 / native C++20:
- Native algorithm suite: 130/130 tests passed.
- Python reference suite: 124 passed, 6 native-only tests skipped.
- Static web build and evaluator: 2/2 tests passed.
- Native localhost Studio: generation, reconstruction, blind ZIP replay,
  continuous rotation with missing pieces, unsafe ZIP rejection and cancellation passed.

Actual Chromium/WebAssembly was not executed locally (Emscripten unavailable);
the existing GitHub Actions browser smoke and Pages/release workflows build it
from the tagged source. Check the workflow status for the release commit.
