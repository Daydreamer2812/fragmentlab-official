# Historical DAFNE FFT baseline: classical matching against the original artwork

> Historical FFT baseline. For the current indexed engine, tested DB2 results, and updated limits, see [DAFNE_INDEXED.md](DAFNE_INDEXED.md).

This **opt-in, reference-assisted** experiment is isolated from blind reconstruction.
It uses no trained network, OCR, learned features, neural inpainting or pose
annotations during matching. It is not a solution to the reference-free DAFNE
problem. The previous blind development result remains 3/87 globally correct
primary poses, with 6/16 correct relative joins across components (see
[ERODED_RECOVERY.md](ERODED_RECOVERY.md)). The reference-assisted mode is a
different protocol: DAFNE's official digital anastylosis setting supplies the
unbroken artwork. Do not compare the two as equal-information experiments.

## Verified development result

Case: DAFNE_Resurrection_102, from the separated Shardelix dataset pack. The
official fresco reference is the 730 × 826 JPEG in the top level of the
Resurrection archive, **not** a ground-truth montage, available-pixels map,
generated assembly, or any image from a solutions folder.

| Metric | Measured |
|---|---:|
| Uploaded fragments | 102 |
| Genuine fragments | 87 |
| Distractors | 15 |
| Correct absolute poses (10 px and 10 degrees) | 87/87 |
| Distractors rejected | 15/15 |
| Median / maximum center error | 1.087 / 3.065 px |
| Median / maximum angular error | 0.0564 / 0.6175 degrees |
| Coverage (observed foreground pixels only) | 44.69% |
| CPU solver time (browser-limit workflow, native Python) | 45.81 s |

See [machine-readable evidence](../reports/dafne_reference_development.json),
including independent input/reference/truth SHA-256 hashes. Annotations were
loaded only after solving; absolute pose scoring does not fit a global
alignment. This is one development case, not held-out generalization or all
DAFNE/fresco scans. The elapsed time is not a measured browser WASM runtime.

## Run in Studio

1. The following steps describe the historical FFT branch; for current main use the [indexed instructions](DAFNE_INDEXED.md). Install Python dependencies, and start:
   `python -m pip install -e .`, then `python -m shardelix studio`.
2. In current Studio, enable **Use original image for matching (optional)** (Balanced runs the new indexed engine).
3. Upload the dataset's input-only `DAFNE_Resurrection_102.zip`.
4. Download the official archive from the
   [DAFNE challenge publisher](https://vision.unipv.it/DAFchallenge/ANASTYLOSIS%20DB1/PierodellaFrancesca_Resurrezione_730x826.zip).
   Extract only the original
   `PierodellaFrancesca_Resurrezione_730x826/PierodellaFrancesca_Resurrezione_730x826.JPG`
   and select it in **Matching reference image (required)**.
5. Run. No count, cut, canvas, candidate rotation or ground-truth upload is
   required. Download Results ZIP, including placements and rejection reports.

Respect the dataset publisher's research/educational terms. No dataset images
or annotations are redistributed in this branch. The reference is required at
the same pixel scale as the fragments. This first version supports at most 256
fragments, a 2-megapixel reference and the existing Studio upload limits. It
does not infer arbitrary scale, exposure changes, perspective or severe
photometric degradation. Unknown areas stay transparent. Rejected or ambiguous
fragments stay unplaced, rather than filling gaps with reference pixels.

The static web build includes the same explicit mode in its worker:
`python scripts/build_web.py --out build/pages`. Main now includes the unified workflow; use the legacy benchmark scripts to reproduce FFT numbers.

## Method

- Segment foreground using existing alpha, chroma-key and scan preprocessing.
- Coarsely sample rotations and use FFT-based masked RGB sum-of-squared
  differences to search translations on a reference pyramid. Reuse the reference
  frequency transform; score only observed fragment interiors.
- Refine candidate SE(2) poses through robust continuous least-squares and
  analytic image-gradient Jacobians. Escalate ambiguous/small fragments to a
  finer search.
- Reject high photometric error, ambiguous locations and foreground collisions.
  Render observed raster pixels only, recording candidate scores and rejections.
- No original image access in the blind route; the distinct matching-reference
  upload is required for this explicit mode. The truth slot is never substituted.

## Repeat the independent measurement

```bash
python benchmarks/benchmark_reference_dafne.py \
  /path/to/input/DAFNE_Resurrection_102.zip \
  /path/to/PierodellaFrancesca_Resurrezione_730x826.JPG \
  /path/to/ground_truth/DAFNE_Resurrection_102.zip \
  /path/to/results --browser-limits
```

The benchmark copies only the input and the original reference into a clean
solver job. Following export, it reads annotations and checks all genuine
centers/angles and distractor rejection. It exits nonzero unless all 87
genuine poses pass at the declared tolerance and no distractor is accepted.
