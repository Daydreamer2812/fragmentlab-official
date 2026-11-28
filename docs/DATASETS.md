# External document data and evaluation protocol (V5)

- LSU GVC Lab official source: https://www.ece.lsu.edu/xinli/DocReassembly/index.html
- Official download: https://www.ece.lsu.edu/xinli/DocReassembly/data/DocDataset.zip
- Research: Yongqing Liang and Xin Li, *Reassembling Shredded Document Stripes Using Word-path Metric and Greedy Composition Optimal Matching Solver*, IEEE TMM, 2019. https://ieeexplore.ieee.org/document/8844125
- User-supplied DocDataset.zip matches official bytes. SHA-256:
  d00ca3aabff4d099f716cc5f98ac9bef9e841bd2a891f8303e44cd27271ee364.
- Contents: 60 digital puzzles (20,30,40,60 strips); 3 physical scans (27,39,56);
  one mixed-orientation 36-strip puzzle. Raw benchmark is not included: use the provided ZIP.
- Blind loader accepts numeric individual PNG files and randomizes names/order.
  It never loads puzzle_img.png, order.txt or gt before solving. Public orientation
  case flag is metadata; original orientations aren't supplied. Physical scans have
  unequal strip widths and are treated as approximate order reconstruction.
- Train: 32 cases on even source document IDs. Holdout: 28 on odd source IDs.
  Group by document ID, not count, to avoid training on the same page as holdout.
  These are V5 held-out for current tuning, not entirely unseen historically.
  Frozen parameter selection is by development adjacency, tie-break order.
- Baseline is unweighted boundary L1 plus the same path machinery, with document
  margin and blank-split priors disabled; **not** a claimed full V4 comparison.
- Directed neighbor accuracy = correct ordered adjacencies / (n-1).
  Since all strips placed exactly once, equals directed adjacency F1.
  Direct order accuracy = indices correct / n; no circular shift or translation.
  Orientation case permits global reversal of the complete chain; individual
  rotation accuracy not scored (no matching orientation truth in order.txt).
- For digital cases only, exact pixel fraction and MAE use gt after prediction,
  if shape matches. Coverage of real scans is relative to inferred canvas only.
- --previews outputs assembly, cut overlay and predicted layout for each case.
  Metrics are in summary.csv/json. The archive includes summary and selected
  previews, not the original external dataset. Do not redistribute LSU raw data.


## User-supplied Cho 432
- 20 directories, each 432 individual 28x28 PNGs, with row/column GT.
- Public grid 18x24, canvas 504x672. Loader independently permutes and anonymizes IDs, quarter-turns optional.
- For the 432 benchmark, inference receives pieces and public dimensions only.
  GT is read after prediction. For 1024 recuts, original image is reconstructed
  from GT in generation ONLY, then truth is moved outside solver input.
- V6 single-consensus parameters chosen on even-numbered cases (10 images),
  frozen then tested on odd-numbered images (10). Multi-hypothesis integrated
  results are a subsequent full-suite evaluation (odd results previously observed).
- Baseline V5 measured on identical anonymous inputs and rotations. Duplicate
  metrics are diagnostic; strict ID DPA/F1 never adjusted for identical pixels.
- User-supplied archive not redistributed; all outputs reproducible locally.

## Mechanical shredding collections (V14.4)
- Paixão et al., *Self-supervised deep reconstruction of mixed
  strip-shredded text documents*, Pattern Recognition (2020):
  https://www.sciencedirect.com/science/article/pii/S0031320320303381
- Author repository and dataset download instructions:
  https://github.com/thiagopx/deeprec-pr20 (see `scripts/get_dataset.sh`).
  Use the datasets only; no training code, weights or intact documents
  are needed by Shardelix.
- The native matching-only evaluator reads each page's `strips/`, converts
  optional author-provided `masks/*.npy` to alpha, and substitutes random
  file aliases before rectification. Sequential strip identifiers are
  decoded for scoring only after inference. Whole pages, model weights
  and truth never enter matching.
- Development sample: D1/S-MARQUES D001,D021,D041,D061;
  D2/S-ISRI-OCR D001,D007,D014,D020; D3/S-CDIP D005,D015,D025,D035,
  D045,D055,D065,D075,D085,D095. Additional sampled pages not used in
  deciding the scaling rule: D1/D010, D2/D010, D3/D050.
  These 21 selected pages are not the complete published dataset.
- Primary metric: directed adjacency after at most one global reversal
  (maximum of forward and reversed-chain counts). Report direct adjacency,
  direct and reversal-aligned position accuracy separately. This does not
  verify individual strip orientation, pixel accuracy or subpixel pose.
  A rejected page contributes zero correct joins/positions, retaining
  its full denominator.
- Runtime is rectification plus matching, not conversion, rendering,
  UI routing, export or full workflow. Output canvases may exceed the
  configured application pixel budget at full source resolution.
- Reproduce locally after extracting the external archive:
  `PYTHONPATH=src python scripts/benchmark_mechanical_strips.py /path/datasets --out result.json`.
  Use `--cases D1/D001 D2/D007` for a subset. No raw external scan data is
  stored or redistributed in this repository.

### V14.5 extension
- Recheck all 21 previously evaluated pages. The earlier additional pages
  are now development data, not holdout.
- The first six new S-CDIP pages (D002,D012,D032,D052,D072,D092) were
  evaluated after choosing the initial robust-residual policy. A later
  calibrated-gap guard was introduced following a regression on the
  development D2/D001. Retest the affected D3/D002 page; report the
  first-pass result separately from the final result.
- Three further S-CDIP pages (D018,D038,D078) were evaluated after the
  final guard was fixed, with no further tuning. Keep these results
  separate from the development sample. All pages are selected samples,
  not a random representative validation of the whole collection.
- Same anonymous matching-only protocol and orientation-neutral directed
  adjacency metric as V14.4; no model or intact document needed. See
  [V14.5 release](releases/v14.5.0.md) for limitations.

## V15 frozen evaluation

Use [the example manifest](benchmark-suite.example.json) and replace the
source paths with locally acquired datasets:

```bash
python scripts/benchmark_suite.py --manifest docs/benchmark-suite.example.json --out outputs/v15-suite
python scripts/benchmark_suite.py --manifest docs/benchmark-suite.example.json --out outputs/v15-suite --resume
```

The runner provides per-stage process isolation, source/engine hashes,
version, backend, dependencies, seeds, thread policy, metrics checksums,
logs, failures and timeouts. Resume requires matching provenance and a
valid metrics checksum. A scan-stress or damage stage uses its own fixed
case seeds. For DAFNE, add a stage with kind `dafne`, the official archive
path, exact case name/substring, and seed 624. DAFNE aliases are assigned
after seeded shuffle. Evaluation uses truth after blind inference; no
external source rasters are redistributed. The seven new holdout pages
are listed in the example and reported separately from the earlier 30.
Measured results: [reports/v15](../reports/v15/) and
[technical report](TECHNICAL_REPORT.md).
