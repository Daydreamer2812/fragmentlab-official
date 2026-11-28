# Scaling blind irregular/torn-fragment reconstruction

This extends the **metadata-free** irregular route, not only the known-canvas
digital generator. Use a ZIP of transparent pieces or appropriately segmented
scans, with automatic detection. No intact image, canvas dimensions, grid,
OCR, trained model, filenames encoding order, or evaluation labels are used.

## Limits and routing

| Stage | Previous | Current resource guard |
|---|---:|---:|
| Native irregular / physical import | 4,096 | Adaptive RAM/pixels; outer archive ceiling 262,144 |
| Browser irregular / physical import | 1,024 | 4,096 outer archive ceiling; 256 MiB solver budget estimate |
| Gap-aware recovery | 512 sparse | Estimated descriptor workspace and pair budget (sparse above 128) |
| Deep spectral and pictorial continuation | 128 | Estimated padded fields, descriptor and FFT workspace; sparse above 128 |
| Physical document strips | 128 | 128 |

Counts are resource guards, not certified accuracy or throughput. Browser
memory, upload and canvas budgets also apply. For up to 256 input rasters,
the previous 4 MP processing target is retained; larger sets use the adaptive
working-pixel budget. Every scan side remains capped at 1,600 pixels and
the same scaling factor is applied to all pieces. Preserving the native
lattice where feasible is especially important for intact digitally cut masks.
See [current measured resource scaling](RESOURCE_SCALING.md); the historical
benchmarks below are retained for comparison.

### Sparse assembly

1. Preprocess alpha, sidecar masks, keyed or matte scan backgrounds. Preserve
   disconnected islands within declared piece rasters.
2. Above 128 pieces use complementary integer boundary hashes at two arc
   lengths, with a compact C++ rolling chain code and memory-bounded shards. Rare shape votes nominate local quarter-turn/translation poses.
   Verify overlap and observed adjacent RGB continuity without a known canvas.
   For sufficiently large but fragmented lattice evidence, retrieve additional
   appearance-matched exact arcs. If an exact zero-overlap component contains at
   least 90% of observed pieces, skip expensive approximate search; leave the
   unanchored remainder explicit.
3. Otherwise retrieve multiscale contour/color descriptors from a sparse
   nearest-neighbor index. The index uses only observed arc/owner votes, rather
   than an N-by-N matrix. At large sizes it first queries a descriptor subset,
   then verifies full descriptor distances. Quantized rigid votes are computed
   in NumPy batches. Reject continuous matches that are ambiguous on the same
   seam, lack distinctive continuity, or do not have independent cycle support.
4. Merge by component size. A spatial hash of full raster bounding boxes
   restricts cross-component collision tests to potentially intersecting
   pieces. Integer poses use exact overlap checks; continuous poses use
   interior and full-mask sampled occupancy. No disconnected groups are
   packed into a supposed original canvas.
5. If the exact route is sparse, the bounded gap fallback searches independent
   descriptors and distributed per-piece vote budgets. Above 128 pieces,
   Balanced nominates up to six pairs per piece; High accuracy nominates up to eight; the actual degree is reduced to fit the resource budget.
   The union of incoming/outgoing nominations is retained. Cap each pair at six
   poses before geometric and photometric verification. The same deep continuation stage can process larger degraded sets with sparse nominations, subject to its field and Fourier memory estimates.
6. Render only observed pixels. Integer assembly preserves the pixel lattice
   instead of rotating the entire image again. Cut-line output is calculated
   in one pass over labels. Preview up to 256 masks; all normalized masks and
   separate components remain in exported files.

## Development benchmarks

All fixtures below are reproducible synthetic partitions of an independently
generated smooth RGB texture, not real shredded-paper scans. Filenames are
anonymous and the upload ZIP contains only piece rasters: no case metadata.
Truth is loaded separately after reconstruction. Pose scores use one rigid
global gauge and thresholds of 5 px and 3 degrees. Relative joins use 10 px
and 10 degrees within each independent group. Placement, coverage and
collision counts do not by themselves establish accuracy.

| Fixture | Observed | Correct primary poses | Collision pixels | Solver seconds |
|---|---:|---:|---:|---:|
| Jagged, quarter-turn, 512 | 512 | 512 | 0 | 12.30 |
| Wavy, quarter-turn, 15% missing of 512 | 435 | 431 | 0 | 6.68 |
| Mixed cut, quarter-turn, browser resource limits | 1,024 | 1,024 | 0 | 21.91 |
| Jagged, quarter-turn, native | 4,096 | 4,093 | 0 | 80.23 |

The 4,096-piece run left three pieces unanchored and had 4,092/4,092
correct accepted relative joins. All placed lattice poses in these fixtures
were exact after one gauge alignment. The 1,024-piece run used native Python
with browser caps, not actual WebAssembly. Timings from development runs may
vary with concurrent processes.

### Damaged continuously rotated data

Large sets of actual eroded physical tears remain a research problem.
Sparse retrieval increases the number of fragments that can be examined,
but does not infer absent material or establish global orientation. On a
synthetic severe case of 144 generated fragments (5% missing, continuous
random rotation, 2 px boundary erosion, 137 observed), the new sparse gap
search performed bounded geometric verification but did not reliably
reconstruct the original.

| Fixture | Correct / accepted relative joins (all groups) | Correct primary poses / observed | Solver seconds |
|---|---:|---:|---:|
| 144 continuously rotated, intact boundaries | 25/25 | 2/144 | 86.01 |
| 137 observed, continuous rotation + 2 px erosion | 0/8 | 1/137 | 56.04 |

The intact continuous example used 549 shortlisted gap pairs and 3,267
hypotheses; 25 verified joins survive across 20 non-singleton components.
The eroded case tried 565 pairs and 3,366 gap hypotheses but none survived
gap verification; its eight local-contact joins were false in offline
evaluation. Rendering overlap was 25 and 23 pixels, respectively. These
are partial, unverified results, not evidence of DAFNE SOTA or full recovery
of 4,096 physical tears.

The new sparse branch is deliberately conservative; false joins can corrupt
an entire component. Larger inputs with very weak evidence may produce many
independent components or a singleton rather than a cosmetically complete
wrong result. Full pose accuracy requires independent labels.

## Verification

Native C++ backend: 117 tests passed. Python fallback: 114 passed and
3 native-only tests skipped. Web artifact/build and evaluator: 2 passed.
Actual Chromium/WebAssembly smoke has not been run in the local environment.

## Reproduce

```sh
python -m pip install -e .
python scripts/build_native.py
python benchmarks/benchmark_sparse_scan.py --count 512 --tile 44 --cut jagged
python benchmarks/benchmark_sparse_scan.py --count 512 --tile 44 --cut wavy --missing .15
python benchmarks/benchmark_sparse_scan.py --count 1024 --tile 40 --cut random --browser-limits
python benchmarks/benchmark_sparse_scan.py --count 4096 --tile 36 --cut jagged
python benchmarks/benchmark_sparse_scan.py --count 144 --tile 70 --cut random --angle-mode continuous --erosion 2 --missing .05
python -m unittest discover -s tests/algorithms
SHARDELIX_CORE=python python -m unittest discover -s tests/algorithms
python -m unittest discover -s tests/integration
```

Measured evidence: [reports/irregular_scale.json](../reports/irregular_scale.json).
