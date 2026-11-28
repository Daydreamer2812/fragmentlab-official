# Experimental gap-aware recovery for degraded fragments

## V14.2: curved missing-boundary recovery

When the weak exact-contact route cannot justify a coherent reconstruction,
Balanced/High accuracy can test morphological offsets of measured fragment
silhouettes at four scale-dependent radii. Complementary arc matching on these
temporary offset masks proposes rigid poses, which are mapped back into the
original crop coordinates and checked against actual mask collision, opposing
tangents, contiguous arc support and gap dispersion. Correlation of variations
and first differences through the missing seam establishes independent
appearance support; low-contrast or competing poses are withheld.

The search is bounded by pair-work and memory budgets. It is not a new public
solver or a dilation-based output renderer. Rounded populations do not use
this offset prior, since their lost boundaries contain little complementary
shape; a separate bridge-consistency guard rejects weak seeds while retaining
their poses as competitors. Existing stronger gap-aware selections are retained.

See [V14.2.0 release notes](releases/v14.2.0.md) and
[paired damage benchmark](../reports/v14/damaged_fragments_v142.json).

## Scope

The physical scan solver now has an adaptive reference-free pipeline for
eroded boundaries, missing material, unknown continuous rotations and
distractor fragments. It does not synthesize missing image pixels. Neither
source images nor ground-truth poses are loaded by the matching algorithm.
This is an experimental extension, not a claim of a solved DAFNE benchmark.

## Pipeline

1. Foreground normalization uses alpha, sidecar masks or reliable background
   removal. In ZIPs explicitly labeled `pieces/`, `fragments/`, `tiles/` or
   `patches/`, one raster denotes one fragment: separated islands of surviving
   material are kept in its common coordinates. Overview scans still segment
   multiple physical objects. The largest contour drives pose proposals.
2. Run the existing complementary-contour matcher first.
3. If the largest exact-contact component contains less than 60% of the input
   pieces, or a small irregular set (up to 16 pieces) has weak local contact
   evidence, attempt bounded gap-aware recovery (2–128 legacy full search;
   above 128, adaptive sparse recovery in Balanced/High accuracy).
   The workflow configuration accepts `gap_recovery: false` to disable this
   stage, or `gap_recovery: "always"` to search it regardless of the trigger.
   The latter does not guarantee selection over the exact-contact model.
4. Sample color and gradients inward from masked contours. Index multiscale
   arc descriptors and propose continuous SE(2) transformations with
   nonzero seam gaps (bounded by 36 working pixels and 35% of the typical
   fragment side). Score contiguous opposing tangent support, absence of
   overlap, extrapolated color consistency, observed interior continuation,
   gradient agreement and texture. Reject ambiguous competing poses.
5. Check local triangle-cycle agreement and refine redundant pose constraints
   with robust soft-L1 residuals. Reject a refinement if it creates collisions
   or increases residuals. Rank independent components by seam confidence,
   not only their size. The continuation stage consumes the verified gap hypotheses; only the final selected components are exported.
6. Output the primary component and transparently represent the lost material.
   No spatial relationship is assumed between disconnected components.

The optimization update also caches invariant masks and interior distance
transforms across spectral angles, chunks Fourier correlations, and checks
triangles with distinct boundary arcs before accepting borderline multi-evidence
seams. The [targeted rerun](OPTIMIZATION.md) preserved 6/16 correct relative
joins and 3/87 primary poses on the severe DAFNE case; it did not solve it.

## Output

- `assembly.png`: observed pixels of the selected connected group only.
- `component_placements.json`: every component (including singletons),
  its accepted joins and each fragment's 2×3 working crop → component
  affine matrix, crop origin, and selected model name. Intermediate competing assemblies are no longer duplicated.
  Each component uses its own independent coordinate gauge. The file
  is not a globally registered reconstruction; do not combine unrelated
  groups by their raw coordinate values.
- `scan_report.json`: exact-contact and gap-search diagnostics, detected
  counts, component sizes, search statistics and limitations.
- `placements.json`: primary component's output-canvas poses as before.

Global orientation remains ambiguous without additional information. Counts
and coverage are not accuracy. Deep continuation and sparse gap recovery use field, descriptor, and pair working-set estimates rather than a 128-piece cutoff. The physical import route uses adaptive memory with the outer archive ceilings of 262,144 native / 4,096 browser. See [scaling evidence](RESOURCE_SCALING.md).
There is no claim of 65,000-piece support for irregular eroded data.

## Evaluation

Run the reproducible synthetic benchmark (randomly rotated textured curved
pieces, 3/5/8 px erosion, missing pieces and a distractor):
```sh
PYTHONPATH=src python benchmarks/benchmark_eroded.py
```
The script scores relative pose only after assembly (10 px center tolerance,
10 degree angle tolerance), with no credit for singleton anchors. The raw
results are in [eroded_benchmark.json](../reports/eroded_benchmark.json).

[Real/development results](../reports/eroded_real_development.json) compare
the previous commit with the prototype on a DAFNE example, two damaged scans
and the MIT example, using the existing evaluation harness and same input
ZIPs. The DAFNE case was used to develop the method and is not an independent
test split. The evaluation aligns one global rigid gauge for the rendered
group; recovered pose count is not simply number of placements. We report
weak results, including unchanged accuracy on one severely damaged scan.

Neither matches on synthetic textures nor a few correct DAFNE poses justify
a broad accuracy claim. DAFNE includes 87 genuine fragments and 15 distractors
in this test, and its official challenge permits a source-image reference;
the blind setting here is different.

## Deep classical search (no machine learning)

The newer high-quality stage is implemented in `continuation.py` and
`spectral_gap.py`. Balanced automatically tries it after sparse or weak
contact recovery. The High accuracy profile forces the additional search
within the same resource budgets; Fast disables this additional stage.
Pass `deep_recovery: "always"` or `false` to override the policy in the API.

- A multi-channel **structure tensor** estimates the local direction of
  isophotes. Outside the mask, rays project back into valid interior pixels;
  unreliable rays fall back to nearest observed color. The inferred band is
  evidence only and is never rendered.
- Distinctive **line exits** (pictorial lines crossing the fragment boundary)
  supply candidate orientations and translations. Two nonparallel exits can
  determine translation by a 2x2 linear system instead of assuming contact.
- A **masked spectral search** proposes additional transformations:
  for each nominated pair, sample 24 angles and all translations on a coarse
  grid using FFT correlations of observed and extrapolated bands. A weighted
  RGB squared error, support, texture and true-mask collision are evaluated
  over the translation grid, then retained hypotheses are checked at common
  working resolution. This is a bounded proposal search, not exhaustive
  continuous global optimization.
- **Pose-mode consensus** combines evidence from point/line retrieval,
  spectral retrieval, and previous gap verification. A robust, soft curvature
  prior downweights angular pieces only when a sufficiently large majority is
  smooth and compact; no input fragments are deleted. Contextual forest growth
  rejects candidates that collide with existing members and requires multiple
  line crossings or consistent independent pose evidence. Unanchored groups
  remain available in the JSON. Primary ranking uses combined evidence with
  a weak-link penalty, not ground truth.
- Compatible pose peaks are fused around the moving-fragment centre,
  balancing proposal families instead of overcounting densely sampled Fourier
  angles. The fused pose must stay close to the strongest peak and pass
  an observed-mask collision check; confidence derives from the original mode.
- During component growth, two different anchors on distinct incoming arcs
  can resolve ambiguous individual seams. If retrieval missed a second
  candidate, the implied relative pose is independently checked directly
  against both original masks and bidirectional appearance. A narrow mid-gap
  ribbon compares contrast variation between the extrapolated fields,
  rather than merely comparing mean colour.
- Above 128 pieces the same stage uses sparse pair nomination. Sparse seeds
  require corroboration by a cycle, a distinctive verified prior seam or
  multiple line exits plus independent proposal families: fewer retrieved
  competitors do not imply higher confidence. The padded field workspace
  is estimated from available memory (96 bytes per padded pixel, plus a
  peak-field and descriptor check); reduce gap padding when it is the
  bottleneck. Fourier memory is estimated from the actual grid, unique cached
  spectra and maximum nominated neighbor batch, rather than a fixed per-piece
  multiplier. Fourier search is skipped if its estimated workspace does not fit.
  On a rounded, compact population with weak exact contact, choose the
  confidence-verified damaged components rather than a larger speculative
  contact tree. The solver can take minutes on
  challenging inputs. It uses only NumPy/SciPy and does not train or load
  a model.

Reproducible synthetic comparisons with the previous solver:
```sh
PYTHONPATH=src python benchmarks/benchmark_classical.py
```
Raw results: [classical_benchmark.json](../reports/classical_benchmark.json).
In the historical run, the selected main result was unchanged on these 11 fixtures; alternate
component predictions may have better precision at the cost of coverage.
The 8px erosion cases remain unresolved despite the larger search.

## Further research (classical only)

- Improve local structure preservation with classical exemplar/patch search
  or variational continuation, with uncertainty estimates.
- Develop multiple-hypothesis component search with exact or bounded
  combinatorial optimization, and validate longer cycles without forcing
  unrelated fragments to align.
- Introduce texture-frequency, contour curvature-scale-space and calibrated
  local likelihood evidence. Benchmark each cue by ablation.
- Evaluate on additional licensed artwork and real shredded-scan data, with
  independent test splits and separate reference-aided vs blind protocols.
  No trained networks or learned descriptors are required or planned.

### Development-set result: DAFNE Resurrection

With the same 102 input images (87 genuine, 15 distractors), comparing the
previous commit with deep classical continuation:

| Metric | Previous gap-aware | Deep classical |
|---|---:|---:|
| Correct primary poses (one global rigid gauge) | 2/87 | 3/87 |
| Correct relative joins across all independent components | 3/11 | 6/16 |
| Relative join precision | 27.3% | 37.5% |
| Non-singleton components | 7 | 14 |
| Median primary center error | 2.42 px | 1.60 px |
| Wall time, this machine | 81.5 s | 292.3 s |

The MIT regression remains 9/9 poses. The deep proposal set includes 24
sampled rotations, 898 nominated Fourier pairs and 12,572 evaluated spectral
proposals. A coherent rounded-population prior downweighted 15 pieces.
All decisions used observed rasters only; labels were read afterward by
the evaluator. Some of the secondary components are wrong, and the DAFNE
global recovery remains poor (3/87); it is not a solved challenge.

See [complete measurements](../reports/classical_real_development.json).
To reproduce the external evaluation with the separate dataset-pack truth
archive, unzip the solver's result ZIP and run:

```sh
python benchmarks/evaluate_dafne_components.py INPUT.zip TRUTH.zip RESULT_DIR
```

The evaluator accepts the DAFNE dataset-pack annotation JSON convention,
not arbitrary external truth formats. No external dataset files are bundled.

## Unified missing-edge update

The blind route now produces one final component graph, with independent
components retained; only the primary component is rendered. It no longer
duplicates the earlier competing graphs in the result JSON. The optional
reference-assisted route is unchanged.

Measured in native Balanced mode (truth read independently after inference):

| Data | Prior / initial | Revised |
|---|---:|---:|
| DAFNE Resurrection (102 observed; 87 genuine) — correct relative joins | 6/16 | 6/13 |
| Resurrection — relative precision | 37.5% | 46.2% |
| Resurrection — joins to distractors | 1 | 0 |
| Resurrection — correct primary poses | 3/87 | 3/87 |
| Masaccio (101 observed) — incorrect joins, before/after rounded-selection guard | 64 | 11 |
| Masaccio — correct relative joins | 0 | 0 |
| 11 synthetic erosion regression cases — correct relative joins | 30/61 | 30/61 |
| Severe 137-piece synthetic scan — correct relative joins | 0/0 | 0/0 |

The Masaccio case identified the bias toward larger weak contact components
and was rerun after fixing it; neither it nor the Resurrection development
case constitutes an independent held-out accuracy estimate. The 137-piece
case exercised 6.4 million padded field pixels and 66 Fourier candidate
pairs, beyond the previous continuation ceiling, but no verified seam
survived. It returned unanchored fragments instead of unconfirmed pairs.
Direct implied-seam verification was validated by a controlled regression,
not by a successful new contextual join on these two official DAFNE cases.
Thus the principal measured DAFNE gain is fewer false links, not complete
global restoration or increased genuine-pose recall.

[Raw results and limitations](../reports/unified_missing_edge_recovery.json).

To reproduce a variant from a downloaded [official DAFNE archive](https://vision.unipv.it/DAFchallenge/DAFNE_dataset/dataset_download.html), without extracting or matching the source image:

```sh
PYTHONPATH=src python benchmarks/benchmark_blind_dafne.py ARCHIVE.zip \
  --case 2019-2-20_16.30.25 --out results/blind_dafne
```

The script opens only the eroded PNGs before inference, and reads the
official annotation text afterward. No DAFNE artwork is bundled.
