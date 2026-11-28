# Unified V14 Studio — GitHub Pages and local runs

## UI and solver routing

Fragment generator → generate anonymous pieces → export fragments ZIP and independent ground truth ZIP.
Reconstruction → direct transfer or import RGBA PNGs / ZIP + public metadata → Solver(profile) → optional independent evaluator.
The source/reference image never enters blind matching. The optional **Use original image for matching** checkbox routes to [indexed reference-assisted registration](DAFNE_INDEXED.md), with the same profile control. It uses a dedicated matching image, not the evaluation-only ground truth slot. Automatic detection is the default. See [the unified framework](FRAMEWORK.md).

The benchmark switch repeats the same generated-case workflow across selected
counts, cuts, seeds and profiles. The independent evaluator reads known truth
only after solve; it is not a third solver. The browser and native Studio share
`src/shardelix/workflow.py`, `src/shardelix/uploads.py` and the reconstruction package.
All profiles use the same cut-specific algorithms and geometric validation.
Fast reduces search; a bounded rescue runs only on severe irregular under-coverage; Balanced is the default;
In reference-assisted mode Fast uses indexed registration, Balanced adds bounded tiny-fragment rescue, and High accuracy enables wider experimental micro-fragment recovery. In blind mode, High accuracy adds bounded search and verifies candidate hypotheses against
observable data. It is not a guarantee of better accuracy. Native optional MILP
is not attempted in WebAssembly; browser and native results need not be identical.

## Hosted Studio and self-hosting

**[Open the official Web Studio](https://daydreamer2812.github.io/Shardelix/).**

For a separately hosted copy (subject to the repository license):

1. Put the release source at the repository root, including .github, src, web
   and scripts. Review rights/license first.
2. Under Settings → Pages, select GitHub Actions as the build and deployment source.
3. Commit/push to main or master. Actions → Deploy Shardelix Studio to GitHub Pages
   builds the static folder and deploys it. You can dispatch the workflow manually.
4. Open the Pages URL listed by GitHub, usually
   `https://OWNER.github.io/REPO/`. Assets use relative URLs.
5. Check Actions → Browser engine smoke. It installs Chromium/Playwright on
   the runner and tests real WebAssembly, single/batch solving, blind replay,
   exports, and responsive layout. Review the retained screenshot and logs.

V14 unifies the V13 hybrid core with adaptive irregular and missing-edge recovery.
Enable Pages in repository settings; the included GitHub Actions workflow
builds and deploys the static site from main. For branch-based hosting, build
the static assets and deploy the complete generated output directory to a dedicated
hosting branch. A Pages site exposes the distributed engine ZIP as well as
the rendered UI. No account credentials or API keys are needed to run it.

## Local static preview

```bash
python scripts/build_web.py --out build/pages --wasm
python -m http.server 8767 --bind 127.0.0.1 --directory build/pages
```
Visit http://127.0.0.1:8767. Opening index.html directly via file:// is not supported.
The accelerated build requires Emscripten (em++). Omit --wasm for the Python reference build.
The build script creates the static artifact locally; generated build files are not committed.
The deployment workflow rebuilds it from source, so do not manually edit engine.zip.
The C++ worker module uses SIMD and portable single-threaded WebAssembly. The pinned Pyodide 0.27.7 runtime and its NumPy, SciPy, Pillow packages download
from cdn.jsdelivr.net. Initial loading can take time and requires internet.
There is no application/backend upload endpoint in browser mode. The input files,
working state and downloads stay in the browser worker; CDN requests load code only.

## Limits and unsupported inputs

| | Browser | Native Studio |
|---|---|---|
| Input | Raster image or zipped/image fragments | Raster, optional PDF (PyMuPDF), fragments |
| ZIP/input bytes | 32 MiB | Adaptive disk and multipart-parser RAM |
| Fragment cap | 4,096 (lower per family where applicable) | Per-family native ResourceLimits |
| Canvas | 8 million pixels | 110 MP digital; adaptive scan route |
| Decoded input | 24 million pixels | Up to 330 million pixels |
| Batch cases | 24 | 64 |
| Batch retained export | 192 MiB | Native guards |

These are protective caps, not verified WebAssembly capacity or performance
guarantees. Use modest cases first. Mobile memory may be tighter. Cancel
terminates a browser worker and the next run reloads it. The native app starts
its own cancellable subprocess. Download before starting another run.
External fragment ZIPs do not need `case.json`. Automatic setup is enabled
on each new import: upload and click Run. A shared input classifier checks
image headers, rotation-invariant cell-size multiplicities, representative
foreground evidence (alpha, solid matte/color-key borders, illumination),
and explicit same-stem mask sidecars. Valid Shardelix public geometry is
authoritative; foreign/incompatible `case.json` is ignored with a diagnostic,
not used as hidden truth.

- Equal-size or quarter-turned rectangular cells, including exact ±1-pixel
  nonuniform raster partitions (up to 1024), use the grid solver.
  For small grids (4–64 pieces, bounded canvas), up to four distinct layout
  hypotheses are compared using blind seam and coverage evidence; ambiguous
  or larger cases retain the conventional aspect-ratio prior. Original
  orientation and missing count remain mathematically ambiguous.
- Opaque variable-width strips with a common long edge are not segmented:
  color strips use RGB/chrominance seam matching, document-like strips use
  the text/ink solver. Explicit alpha rectangles with a common long axis
  can be rectified from continuous rotations; if strict rectification
  rejects them, irregular matching preserves the masks.
- Foreground-masked, keyed, or matte-background scans use document-strip
  rectification where appropriate or general contour matching. For an
  ambiguous opaque crop, auto mode retains all pixels instead of guessing a
  background. A single full opaque image is preserved as one tile.
- ZIP layouts may use `pieces/`, `fragments/`, `tiles/`, `patches/`, or
  `scans/`, inside a wrapper. Auxiliary/reference/ground-truth directories
  remain excluded. Multiple unrelated image directories remain ambiguous
  and require a primary fragment folder.
- Stale manual values never override automatic mode. Disable automatic
  setup to give a known canvas or use explicit physical scan mode if
  automatic foreground detection is ambiguous. No source image, order
  file, or filename pose labels enter inference.

The generator treats Resolution as a preferred maximum side. It can resize the working canvas to fit the requested exact count and resource limits (including sampling), then reports the actual canvas and grid. Uniform rectangular cuts may require snapping each dimension to a cell multiple. Thin cells have limited texture; infeasible counts return a descriptive error rather than silently changing the count.

DPA/F1 require a matching independent ground-truth archive. Raw DPA measures absolute poses in the original blind canvas; rotation-aligned DPA allows a single global quarter-turn of the known canvas (0/180° for nonsquare canvases). A correctly assembled image can therefore have 100% aligned DPA while being upside down. With a reference, the Studio shows raw and rotation-aligned pixel MAE on overlapping observed pixels, and can preview/export a separately labeled reference-aligned image and overlay. Reference-only alignment uses pixel agreement, not pose labels; it cannot establish DPA/F1. No reflection, local edit, or free translation is applied. The original blind PNG, cut overlay and poses are retained unchanged. If no reference is supplied, absolute upright orientation cannot generally be established.
Document strips require same-height digital rectangles, not photographs of torn
paper or arbitrary perspective. Unknown-width inference cannot know the total
missing area of strips not present in the input.

## Physical paper scans and photos (experimental)

In Reconstruction, leave **Input type → Automatic detection** for irregular
scans, one overview image, or a ZIP with images under `scans/`. Use **Physical
paper scans / photos** explicitly for several same-size opaque scan sheets,
which otherwise resemble digital tiles. Choose Digital fragments to force
the existing grid/canvas solver. No dimensions, rows/columns, or angles are
required in physical mode.

- Scan or photograph the same planar side of the paper with a uniform DPI or
  camera distance. Use a matte background that contrasts with the entire paper,
  include margins and space apart the pieces. Prefer a flatbed scanner.
- The system applies EXIF orientation, one common resolution scale, robust
  border color / optional planar illumination estimation, adaptive mask
  thresholding (including JPEG ringing suppression), morphology, connected
  components and speckle filtering. Transparent PNG alpha, high-confidence magenta chroma-key, and reliable solid backgrounds of other colors are supported (background is converted to alpha before resampling). Noisy JPEGs retain the robust threshold/illumination route.
- It matches complementary, resampled contour arcs at multiple lengths,
  estimates continuous SE(2) transforms, checks opposed tangents and seam
  color, retains alternative poses, then joins components with interior
  collision checks. Unanchored groups are not arbitrarily placed on the canvas.
- On sparse/weak exact-contact assemblies, a bounded experimental gap-aware
  stage can estimate separated rigid poses from masked inward color and gradient
  evidence. It keeps missing pixels transparent and exports independent component
  poses in `component_placements.json` (also in the result ZIP). Explicit
  piece-raster folder layouts keep islands of damaged masks together; overview
  scans still split multiple silhouettes. High accuracy enables the deeper
  no-ML continuation/FFT search; Balanced activates it automatically after
  sparse/weak matching. This may take minutes; the padded field and Fourier workspace estimates determine whether the stage runs. Above 128 pieces the pair search is sparse.
  See [eroded recovery](ERODED_RECOVERY.md).
- The preview shows detected masks; download the normalized mask ZIP, scan
  processing report, reconstruction PNG, boundary overlay and affine poses.
  Poses are local cropped working-pixel vertex coordinates → output canvas;
  crop offsets and common scale are in the report.
- Metadata-free physical/irregular mode now uses adaptive native memory
  estimates, with an outer archive ceiling of **262,144** native / **4,096**
  browser fragments (not accuracy guarantees). Browser solver memory uses a
  conservative 256 MiB estimate; actual browser/OS memory is not exposed.
  Native takes 55% of estimated available RAM, with a configurable lower
  `SHARDELIX_MEMORY_MIB` budget. The 4 MP small-set resolution target and 1,600
  px per-scan-side bound remain; larger sets use the adaptive working-pixel
  budget. Above 128 pieces, compact C++ lattice matching and integer spatial
  assembly bypass continuous search when appropriate. Sparse continuous and
  gap fallbacks estimate their descriptor and verification workspace; if the
  budget is insufficient, independent components remain explicit. Deep continuation uses the same adaptive memory policy and sparse search above 128 pieces. The mask contact sheet previews
  at most 256 pieces; all normalized rasters and component poses remain in the
  output. See [measured resource scaling](RESOURCE_SCALING.md).
  All processing is local.
- Paper touching the scan edge, overlapping/stacked pieces, near-identical
  foreground/background, strong shadows, nonplanarity, perspective and
  inconsistent DPI can cause segmentation or alignment failures. Missing
  material can cause seam translation drift. Upright orientation remains
  unverified in blind scan mode. Coverage is occupancy of an
  estimated bounding box, not original-page coverage, and DPA/F1 are not
  computed on physical scans without independently registered truth.

The reproducible simulation `PYTHONPATH=src python benchmarks/benchmark_scan.py`
includes rotated noisy JPEGs, a 2-D wavy partition and one-pixel erosion.
`reports/scan_simulation.json` reports per-piece center and rotation errors
after a common rigid gauge (the first placed piece defines the gauge).
`reports/scan_real.json` records two public hand-torn magazine scan sets from
[AutoMosaic / OpenSoft14](https://github.com/raltgz/OpenSoft14), not redistributed
here. Both were fully connected (6/6 and 5/5), with visually plausible text and
art alignment, but no independent ground-truth poses or pixel accuracy claim.
The real examples have residual visible seam gaps and small raster overlaps. In eroded or distractor-heavy artwork datasets, connected groups may remain small; the result exports all detected masks plus unanchored group membership, not fabricated full-page positions.

### Folder-wrapped scanned strip archives

ZIP files may contain a single outer collection folder (for example
`collection/0.png`, `collection/1.png`). The importer selects the primary
raster directory, not auxiliary views or ground truth. If the archive contains
multiple unrelated image directories, use `pieces/` or `scans/` to identify the input.

The physical scan route supports binary, same-size foreground masks stored
under a sibling `bg_mask_vis/`, `masks/`, `mask/`, `foreground_masks/`, or
other supported mask folder (white/255 or 1 = foreground, black/0 =
background). Names may match the image stem or use a `_mask` suffix. The mask folder is never counted as fragments.
Companion NumPy arrays and ordering/answer text files are not read by the solver.
Partial mask collections are supported; only verified same-size binary masks
are applied and remaining inputs are inferred automatically.

If the images are elongated and share a common long dimension, the engine
attempts physical document-strip rectification before general contour matching.
It rectifies the foreground silhouette per row, estimates relative 180° text
orientation from cleaned whole-strip row projections, compares shadow-normalized
ink-edge profiles, and in Balanced/High accuracy uses nearby observed interior
patches to score likely stroke continuations without recognizing characters.
V14.1 adds local shadow suppression, bounded vertical text-row registration,
shift-tolerant foreground overlap and conditionally enabled page-margin priors.
V14.3 adds a bounded explicit cut-loss hypothesis bank on high-resolution
document strips (Balanced/High accuracy), reusing observed interior pixel
patches. Reciprocal edge ranks and alternative-edge margins select a candidate
chain without labels.
V14.4 uses a bounded assignment/subtour branch-and-bound path refinement
instead of MILP. It preserves the incumbent and always returns a valid chain.
Native patch queries use at most four workers; WebAssembly stays single-threaded.
Matching-only scaling retains horizontal detail on equal-width strips and
uses isotropic anti-aliased downsampling on tall varying-width scans.
A guarded elongated-mask tolerance handles some tapered cuts.
V14.5 reuses the same observed patch predictions to score bounded global
vertical phase, six local bands, and ink-weighted residuals on tall varying-width
scans. A calibrated cut-gap winner is not overridden by coarse residuals.
A fragmented-looking local rim is admitted only with a dominant width, bounded
outward excursion and the existing interior-hole guard.
The exported strip pixels remain at rectified source resolution.
Color photographs use chromatic strip matching instead. Fast skips the patch index.
The solver optimizes a directed chain. It supports 2–128 scanned strips. For other
shapes, the irregular scan route remains available. Physical strip ordering
is experimental: aligned geometry and full placement do not prove that the
text is in correct order. The normalized strips and scan report are exportable;
the supplied ground-truth ordering is never used for inference.

No character recognition, language models, or trained models are used. Matching is based only
on image geometry and pixel continuity; consequently the method is
language-independent, but cannot resolve all ambiguous text seams or
guarantee global upright orientation.

Reproduce the real-strip benchmark using
`PYTHONPATH=src python benchmarks/benchmark_scanned_strips.py /path/to/DocDataset.zip`.
The original [DocDataset](https://people.tamu.edu/~xinli/DocReassembly/) is
not redistributed. The script anonymizes inputs before reconstruction, then
reads the answer ordering only after the pixel-only solver finishes. See
[`reports/scan_strips.json`](../reports/scan_strips.json) for measured
historical results: 90/119 correct undirected seams vs 22/119 before the optimization.
V14.1 further improves to 109/119 (91.6%) in Balanced native runs;
see the [paired scan report](../reports/v14/scan_document_v141.json).
V14.3 reaches 119/119 directed joins and 122/122 exact positions on those same
three real development pages, for both filename seeds 624 and 2026.
See the [V14.3 report](../reports/v14/scan_document_v143.json).
V14.4 reduces native processing time and adds three independent external
mechanical-strip collections. See [paired sample evidence](../reports/v14/scan_document_v144.json).
These are native pixel-only
development results, not a claim of identical browser performance or verified
accuracy on arbitrary uploads.

## Verification status

The static artifact, JS syntax, archive contents, Python tests and local
native HTTP smoke were verified during release. GitHub Actions also ran real
Chromium WebAssembly successfully after two mobile-layout fixes. The regression
covers single reconstruction, batch, ZIP metadata replay, mobile viewport and
no JS errors. The public repository runs the same browser checks in its own CI.
See the repository Actions tab for current status.

Official references:
- https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages
- https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
- https://pyodide.org/en/0.27.7/usage/index.html
