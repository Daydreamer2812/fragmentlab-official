# V15.0 technical report

**Scope:** CPU-first classical computer vision, blind by default. This is
a research release, not a demonstrated state-of-the-art result. No OCR,
language model, learned descriptor or intact source image is used in blind
inference. All missing material remains unobserved. Reference-assisted
evaluation is reported separately.

## Architecture and validation

1. Segment the observed paper, rectify elongated silhouettes using masked
   linear interpolation, and separate sparse neutral document scans from
   chromatic images. No word or character decoding is performed.
2. Estimate row alignment and orientation by pixel/foreground correlation.
   Existing multiscale patch, phase, cut-gap and global path searches remain
   the primary ordering engine. Strong row consensus (at least 0.65) now
   affects the rendered pose; otherwise no vertical correction is imposed.
3. Report each selected seam's outgoing and incoming ranks and relative
   rival margin. A distinctive seam must be the unique reciprocal best
   and meet the 0.05 normalized margin. Split the proposed chain at
   unsupported seams; the user receives both the full tentative chain and
   provisional components. The threshold is an engineering diagnostic,
   not a calibrated confidence probability.
4. For eroded irregular fragments, offset contour nomination stays bounded
   by the original pair and memory limits. Refine already-verified rigid
   transformations using two rounds of translation and rotation trials
   (12 proposals per verified hypothesis). Accept only improvement of the
   observed objective, subject to collision and the physically estimated
   two-sided gap. Reject refinement that drifts from the gap residual.
5. Merge nontrivial rigid components only when two independently anchored
   verified links agree in SE(2) and all observed cross-component masks
   satisfy the collision test. Compare transforms at fragment centres,
   avoiding dependence on an arbitrary coordinate origin. Bound the
   corroboration search to 256 trials; do not force a single global layout.

Physical strip costs are dense in strip count (quadratic matrix storage);
the constrained ordering search has a bounded search budget, not a
polynomial-time global optimality guarantee. Row interpolation is linear
in output pixels. Offset contour pair nomination remains quadratic in
small damage cases but is omitted beyond the fixed pair/memory budget.
The independent-component bridge has bounded candidate trials; collision
checks are cross-component pairwise. Existing C++ sparse exact-contact
kernels and resource limits continue to serve large intact partitions.

## Paired evidence

Baseline: `ef975d64b94cb61c8e735dcf133ba587ac453bd8` (14.5.0).
The mechanical protocol uses the same anonymization seed 624, physical
matching path, Balanced profile, and global reversal metric on both
revisions. Seven pages were obtained after development and excluded from
policy selection. They are not a statistically representative sample.

| Metric | V14.5 | V15.0 |
|:--|--:|--:|
| DocDataset Real: directed joins (3 pages) | 119/119 | 119/119 |
| DocDataset Real: exact strip positions | 122/122 | 122/122 |
| Mechanical: previously observed pages (30) | 470/688 | 477/688 |
| Mechanical: new holdout pages (7) | 129/165 | 129/165 |
| Mechanical: total (37 pages) | 599/853 | 606/853 |
| Stress: undirected adjacency (8 synthetic scan cases) | 73/248 | 73/248 |
| Stress: orientation, one global gauge | 256/256 | 256/256 |
| Stress: joint directed/oriented adjacency | 61/248 | 61/248 |
| Stress: vertical offset MAE (256 pieces), orientation-gauge aligned | 2.195 px | 0.363 px |
| Continuous-rotation erosion: correct/accepted joins (4 cases) | 7/10 | 8/9 |

Distinctive mechanical seams: 120/129 correct in the 30 previously observed
pages, 55/55 in the seven new pages. Coverage is 184/853 possible joins
(21.6%); precision is 175/184 (95.1%) in this sample. These results do
not imply that a whole assembled page is correct. The hardest scan stress
cases still show poor ordering. The refined damage pose passes the
existing lost-boundary regression but one severe erosion case remains
incorrect. Development timings were affected by concurrent jobs; no
end-to-end controlled speed multiplier is inferred from them.

The anonymized DAFNE development case (107 pieces, 65 genuine) exposes
a major limitation: both tested policies yield 4/39 correct accepted
relative joins, 5 outlier joins, and only 2/65 correct primary poses under
independent component gauges. This is synthetic artwork damage, not a
physical paper benchmark. Sparse offset nomination and stricter ad hoc
component scoring were investigated but not validated, so they are not
included. The second DAFNE case and the full corpus have not completed
evaluation; no result is claimed for them.

[Per-case raw comparisons](../reports/v15/) · [Data sources](DATASETS.md)

## Acceptance checklist

| Area | Evidence / outstanding work |
|:--|:--|
| Blind default, no OCR/ML | Preserved; separate assisted mode |
| Sparse and physical document scans | Routing and pose changes tested; full-chain ordering remains imperfect |
| Damage geometry and collision | Bounded SE(2) refinement and two-anchor graph support; severe eroded/outlier cases unresolved |
| Global assembly | Conservative graph merge and strip path competition; no exact optimum guarantee |
| Confidence and abstention | Distinctive seam evidence and partial components; calibration on a larger frozen test set pending |
| Speed and memory | Masked row interpolation and lossless faster PNG encoding; original work budgets preserved; controlled end-to-end profiling pending |
| Reproducible evaluation | Frozen manifest runner, anonymized inputs, per-case pose and adjacency; full independent external suites pending |
| Framework distribution | CLI, Python and browser share the engine; portable/native wheel builds; cross-platform CI and actual browser WASM execution need CI confirmation |
| Research publication | Citation, methods, metrics, reproducible reports; no full comparative ablation or SOTA proof |
| Licensing | Maintainer decision pending; no license is assigned automatically |

To establish a defensible SOTA claim, evaluate the full mechanical corpus
and genuinely torn, physically scanned documents, including disconnected
pages, missing pieces, distractors, folds and variable illumination.
Compare against published methods with identical inputs, truth rules and
hardware/timing scope; freeze development decisions before running the
held-out set. Report coverage as well as accuracy, confidence calibration,
resource use, ablations and failure examples. See [references](REFERENCES.md).
