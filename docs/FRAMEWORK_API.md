# Shardelix framework API

## Install and call
Python 3.11+, NumPy, SciPy, Pillow. The package is a research/framework
release candidate (alpha), not a production service. Install from the source checkout with `python -m pip install -e .`, or use `PYTHONPATH=src`.

### Native and portable wheel builds

```bash
python scripts/build_native.py  # optional local C++20 accelerator
python -m pip wheel --no-deps .
# Linux/macOS, with a C++20 compiler:
SHARDELIX_BUILD_NATIVE=1 python -m pip wheel --no-deps .
```

On Windows, set `$env:SHARDELIX_BUILD_NATIVE = "1"` in PowerShell and
use a supported MinGW C++20 compiler. Native acceleration is optional;
set `SHARDELIX_CORE=python` to force the Python fallback.
Install `.[reference]` for optional OpenCV/SIFT reference registration or
`.[pdf]` for native PDF support. Wheel CI targets Linux, macOS and Windows;
PyPI publication is not configured.

### Solver API

```python
from shardelix import Solver, ResourceLimits
solver = Solver(profile="balanced", index_mode="auto", limits=ResourceLimits(
    max_fragments=16384, max_canvas_pixels=32_000_000,
    max_input_pixels=96_000_000))
result = solver.solve("case/pieces", shape=(768, 768), cut="grid",
                      grid=(32, 32), angle_mode="continuous",
                      sampling=1, expected_count=1024)
print(result.summary())
result.save("out")
```

Input: a directory of image files or a sequence of `Fragment(id, rgba)`.
IDs must be unique opaque strings; uint8 RGBA crops, nonempty alpha, pixels
outside masks zeroed. Directory input is cropped automatically. Memory input
should already be tightly cropped/normalized outside-alpha RGB. Shape uses
(height, width), grid uses (rows, columns). A missing-aware input declares
expected full count, not the missing positions. Sampling=2..4 is valid only
when the original digital pieces were oversampled before continuous rotation.
Cut can be grid, strips, horizontal-strips, jagged, wavy, jigsaw, random, or
custom (external masks). The source and truth are intentionally not API args.

`Solver(auto_tune=True, strategy="auto", index_mode="auto", limits=...)`.
Auto uses sparse midpoint for qualifying large uniform grids and retains
legacy descriptors for long strips and small grids. A weak blind consensus
may trigger one V9-index fallback. `index_mode="legacy"` forces sparse V9
candidate retrieval, not an exact restoration of all V9 code paths.
`"midpoint"` forces the compact index; `"mixed"` is experimental. Explicit
index modes require a rectangular sparse route. `strategy="classic"` permits
up to 1,024 no-missing discrete-angle pieces. V11 vectorizes reciprocal graph voting, permits bounded verified relative-mask
growth, and tries blockwise exact edge matching for weak grids up to 1,024.
Thin-strip gap recovery is limited to widths up to 4 px, observed count up to
1,024 and missing count up to 64; uncertain gaps stay unresolved.
No ground-truth tuning inside
solver. `params` are advanced experimental internals, not stable API.

## Outputs
- `result.poses`: dict of ID → Pose(x,y,k), k clockwise quarter-turn.
- `result.rectification` and `input_affines`: continuous normalization data.
- `result.summary()`: diagnostics without rendering, accuracy_verified=False.
- `result.render()`: prediction RGBA, int32 labels, collision pixels.
- `result.save(path)`: assembly.png, assembly_with_cuts.png, cut_lines.png,
  placements.json, diagnostics.json, ambiguity.json and unanchored components where applicable.
  Ambiguity reports exact normalized duplicate classes, low-texture counts,
  relative translation slack and unanchored groups; not verified accuracy.
  Continuous outputs also contain original-input → canvas affine maps.
  Keep outputs in distinct directories for independent runs.
- `evaluate` command is a separate post-solve evaluator, never an inference input.

## V12 execution profiles and browser runtime

Solver(profile="fast" | "balanced" | "accurate") selects search effort, not
a different public solver. Default without an explicit profile preserves
legacy tune/index parameters for compatibility. Use balanced for new work.
The same Python code is bundled in the static browser engine; native-only
MILP is skipped on WebAssembly. Browser limits are documented in WEB.md.
Unknown-width digital strips support continuous angles with rectangular alpha
and common height via solve_external_document / solve-document. Missing
total width is not identifiable in this mode.

## Failure and operational guidance
Invalid counts/dimensions/images/strategy raise ValueError; I/O exceptions
propagate. A partial/ambiguous result is not an exception or measured
accuracy claim. The solver may leave fragments unplaced or output relative
groups when global location cannot be inferred. No inpainting. Exact
duplicate pieces can be impossible to identify. Use independent evidence
for DPA and adjacency F1. Local Studio runs a cancellable child process;
embed Solver in a worker process for hard runtime/RAM limits, not a web
request thread. ResourceLimits bounds file count and pixel budgets, not
exact memory. Use separate process per concurrent job; no shared mutable
global tuning state. For large grid-only evaluation, the strict complete
uniform grid cell metric avoids full pixel labels. This is not the evaluator
for missing, irregular, or continuous masks.

### Scale considerations
Upper guards: grid 262,144, strips 8,192, irregular 4,096. These are not
accuracy or latency SLAs. Canvas 110M pixels, decoded crop 330M pixels.
Full raster export may require additional ~1–2 GB for large canvases;
the large-grid stress script saves only bounded predicted previews.
Studio estimates a separate upload limit from available disk and multipart-parser RAM; file inputs use available disk. CLI/in-memory API is recommended
for large datasets. Metadata-free scan import uses an adaptive working-set budget; see [resource measurements](RESOURCE_SCALING.md). Avoid holding multiple full frames at once. No real
physical-fragment guarantee. Refer to the [current technical report](TECHNICAL_REPORT.md) for V15 evidence. Historical benchmarks are not new reruns.

## Development
Install the package with `python -m pip install -e .`, then run
`python -m unittest discover -s tests/algorithms` and
`python -m unittest discover -s tests/integration`.
The integration example is `examples/framework.py`.
Contributions should preserve blind inference and include reproducible evidence.
