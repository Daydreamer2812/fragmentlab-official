# Adaptive resource scaling of blind torn-fragment reconstruction

This update focuses on the **metadata-free irregular scan route**. It does not
replace classical free-angle/eroded-piece reconstruction with a grid or
reference image. No OCR, ML, training, known layout, piece order, source
image or evaluation poses are supplied to the solver.

## What changed

- C++20 exact chain-code index encodes normalized direction differences in
  64 bits. Complementary reverse codes route to the same hash shard. A shard
  allocation limit and vote-entry limit prevent unbounded growth. Rare-bucket
  suppression remains deliberate: highly repetitive arcs are ambiguous.
- The packed contour stream grows in one contiguous buffer (no full-array
  concatenation copy); vectorized top incident vote selection replaces the
  Python dictionary of all hypotheses. C++ exact contact verifies overlapping
  occupancy and the median/80th-percentile RGB seam distance. Both kernels
  have a Python fallback; tests compare votes across different shard counts.
- Integer quarter-turn component union uses spatial bins and exact occupied
  pixel collision checks. No fractional transforms or resampling are used
  for an exact lattice. The renderer reuses the saved canvas buffer for the
  overlay; JSON output is streamed without a duplicate complete string.
- Continuous fallback bounds descriptor workspace and prunes weak outgoing
  retrieval votes for large sets. The sparse gap stage budgets pairs according
  to available memory and retains top per-source arc votes during retrieval,
  rather than concatenating every raw nearest-neighbor return.
- Native archive bytes depend on disk headroom (one eighth of available
  space); Studio's multipart parser also reserves RAM (at most one twelfth of
  available memory). The browser's independent transport limits remain.
  Native scan diagnostics include the selected budget and process peak RSS.

## Resource policy

Native reconstruction uses 55% of estimated available memory (Linux
MemAvailable and cgroup limits where visible; Windows GlobalMemoryStatusEx;
portable fallback 512 MiB if unavailable). Set `SHARDELIX_MEMORY_MIB` to an
integer of at least 32 to request a lower budget. Working input pixels are
estimated at 18 bytes/pixel, output canvas at 24 bytes/pixel. These are
conservative **working-set estimates, not OS-enforced process limits**.
The native archive ceiling is 262,144 entries; decoded input ceiling is
330 MP. Known-canvas generation and legacy digital engines still have their
own constraints. Browser uses a conservative 256 MiB solver-memory estimate,
4,096 fragments, 32 MiB compressed upload and 8 MP canvas.

Continuous and gap matching remain conditional on descriptor and verification
cost estimates. Deep continuation now estimates padded-field and peak workspace memory, with sparse pair nomination above 128 pieces. FFT verification runs only if its estimated workspace fits. Scanned-strip mode retains its separate 128-piece limit. Remaining independent components are exported without a fake
placement. Blind upright orientation and absolute image accuracy cannot be
certified without independent evidence.

## Measurement protocol

`benchmarks/stress_irregular_resources.py` generates anonymous, quarter-turned
irregular masks on an independently generated smooth random RGB field.
Adjacent masks share irregular pixel edges. Each size uses seed 827 and
nominal tile 36 px. No layout or labels are provided to matching. After the
solver finishes, one global rigid gauge is allowed for exact primary pose
accuracy, and each accepted local join is checked against offline truth.
Every size runs in a separate Linux process. Peak RSS includes fixture,
solver and evaluation; solver time excludes input generation and evaluation.
The memory-aware in-memory benchmark excludes ZIP import and PNG export.
This is a stress benchmark of intact synthetic pixel-lattice tears,
not physical scan accuracy or a DAFNE benchmark.

### Measured solver scaling

| Input fragments | Exact primary poses | Correct accepted joins | Solver seconds | Peak RSS (MiB) |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 8,191 / 8,191 | 11.82 | 446.0 |
| 16,384 | 16,384 | 16,383 / 16,383 | 25.25 | 826.9 |
| 32,768 | 32,768 | 32,767 / 32,767 | 55.40 | 1,569.0 |
| 65,536 | 65,536 | 65,535 / 65,535 | 113.63 | 2,383.9 |
| 100,000 | 100,000 | 99,999 / 99,999 | 177.32 | 2,664.1 |

All five runs formed one component, with zero collision rejections.
The first four were measured with the intermediate two-copy packed stream.
The initial 100,000-piece trial actually hit its packed-contour memory guard;
after the one-buffer correction the same seed/count completed with three
hash shards per arc window. It received 210.69 MP of cropped raster input
against its 216.91 MP working-pixel estimate (97.1%). Its estimated RAM
budget was 3,723.6 MiB on a host with ~6,770 MiB available at startup; actual
peak RSS was 2,664.1 MiB. Matching consumed 154.88 s and component merge
22.44 s, so candidate search now dominates CPU time.

The 131,072-piece probe estimated 277.35 MP of input against 220.87 MP of
working-input budget and abstained before allocation. This is a measured
**safe resource preflight boundary**, not a proven OS memory-exhaustion
threshold. Pushing deliberately to host OOM would be unsafe. No hard
100,000-piece algorithm cap was introduced.

The complete workflow is also measured on a metadata-free 8,192-piece ZIP:
8,192 detected and correctly placed, zero collisions, 31,037,677 input bytes,
58.09 s workflow (37.59 s reconstruction including render), 468.1 MiB peak RSS.
The former 4,096-piece generated jagged fixture is kept for an apples-to-apples
comparison: 4,093 exact of 4,096, zero collisions, 80.23 → 18.06 s solver
(4.44x faster; shared development runner, not controlled hardware benchmark).

The new 4,096-piece browser archive ceiling was also tested with native C++
under browser-style budgets (NOT WebAssembly timing). A compact synthetic
fixture completed 4,096/4,096 exact with 221.1 MiB peak RSS, 22.20 s wall,
13.95 s solver/render. The older recursive jagged 4,096-piece fixture had
much larger cropped bounding boxes: its common scale fell to 0.79116 and
only 3 pieces joined. Exact original-scale evaluation is not applicable to
resampled pieces. The UI now warns explicitly that resizing may destroy
digital edge complementarity. This is why the transport ceiling is not an
accuracy guarantee. Actual browser-WASM smoke coverage exercises the new
arc path in CI, but the live browser runtime has not been run locally.

A physical-style eroded regression is included: 144 generated, 137 observed,
continuous random angles, erosion 2 px, missing 5%. The new solver still
recovers only 1 correct global pose and 0/8 correct relative joins, identical
to the prior accuracy, with 51.86 s versus 56.04 s solver time. This is a
failure case, NOT a claim that the larger exact-lattice capacity solves
damaged real-world artifacts.

## Reproduce

```sh
python scripts/build_native.py
PYTHONPATH=src python benchmarks/stress_irregular_resources.py --count 8192 --tile 36
PYTHONPATH=src python benchmarks/stress_irregular_resources.py --count 65536 --tile 36
PYTHONPATH=src python benchmarks/stress_irregular_resources.py --count 100000 --tile 36
PYTHONPATH=src python benchmarks/stress_irregular_resources.py --count 8192 --tile 36 --workflow
PYTHONPATH=src python benchmarks/stress_irregular_resources.py --count 4096 --tile 36 --workflow --browser-limits
PYTHONPATH=src python benchmarks/stress_irregular_resources.py --count 131072 --tile 36
PYTHONPATH=src python benchmarks/benchmark_sparse_scan.py --count 4096 --tile 36 --cut jagged
PYTHONPATH=src python benchmarks/benchmark_sparse_scan.py --count 144 --tile 70 --cut random --angle-mode continuous --erosion 2 --missing .05
```

For the full measured rows and resource-abstention diagnostics, see
[`reports/resource_scaling.json`](../reports/resource_scaling.json).
Exact C++ and Python tests run independently. Actual browser/WASM
runtime requires Emscripten and a browser; native tests with a browser-style
budget are not measurements of browser performance.
