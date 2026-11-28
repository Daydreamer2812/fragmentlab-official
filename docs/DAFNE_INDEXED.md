# DAFNE indexed classical registration (integrated framework)

**Reference-assisted, no ML.** This mode requires the intact original artwork at the same pixel scale as the fragments. It does not solve the reference-free problem. Blind recovery is a separate path within the same framework. No annotations or solution image are supplied to the solver; the evaluator reads truth after exporting poses. Rendered pixels come only from accepted fragments.

## Algorithm

- Overlapping tiled SIFT extraction, RootSIFT normalization, classical FLANN keypoint retrieval and rigid-pose votes when OpenCV is installed.
- Reusable rotation-invariant colour/difference-of-Gaussian descriptor cKDTree over subsampled source coordinates; dense search for weakly textured inputs.
- Deterministic masked photometric scoring, nonmaximum suppression in pose space, continuous SE(2) optimization with damped robust Gauss–Newton and bilinear image derivatives.
- Candidate ambiguity and absolute residual tests; local raster occupancy prevents overlapping foreground placement without an all-pairs piece loop.
- **Indexed:** baseline validated on the 28 cases below. Balanced now retries up to 32 tiny rejected fragments (8–900 foreground pixels, original up to 8 MP) using the micro-index and the same photometric and occupancy checks. Existing accepted poses are never evicted. **Indexed-deep (experimental):** adds denser micro-fragment and border candidate lookup, plus more permissive search triggers (not a looser acceptance rule). Tested on two selected difficult DB1 cases; broader verification pending.
- A portable NumPy/SciPy backend needs no OpenCV; native mode can optionally use OpenCV. No training data, pretrained network or learned features.

These ideas are motivated by the classical DAFNE digital-anastylosis literature (SAFFO, RINCCAS and Lermé et al.), not a claimed reproduction of their published systems. See the [official DAFNE dataset](https://vision.unipv.it/DAFchallenge/DAFNE_dataset/dataset_download.html).

## Measured development results

Absolute center distance ≤10 px and absolute angle error ≤10°; no best-fit alignment.
A strict case passes only if every genuine pose is correct and every distractor rejected.
The three DB2 cases and all accepted genuine DB1 poses also satisfy the stricter 5° angle criterion.

| Official subset | Engine | Correct genuine poses | Rejected distractors | Strict cases |
|---|---|---:|---:|---:|
| Same 25 DB1 archives/cases | FFT baseline | 2873/2874 | 467/476 | 17/25 |
| Same 25 DB1 cases | Indexed | 2872/2874 | 476/476 | 23/25 |
| DB2 (all three challenge cases benchmarked here) | Indexed | 3155/3155 | 678/678 | 3/3 |
| Combined 28 cases | Indexed | 6027/6029 | 1154/1154 | 26/28 |

| DB2 | Total input | Poses | Distractors rejected | Measured solver seconds |
|---|---:|---:|---:|---:|
| Domenichino, Virgin and Unicorn | 600 | 474/474 | 126/126 | 79.71 |
| Zuccari, Cardinal Hat | 1188 | 943/943 | 245/245 | 152.46 |
| Andrea di Bonaiuto | 2045 | 1738/1738 | 307/307 | 284.96 |

All three DB2 cases were previously blocked by FFT resource guards. Largest DB2 positional error 2.857 px, largest angular error 0.2181°. On the paired DB1 cases, FFT summed solver time 3568.48 s vs indexed 387.68 s, median paired runtime ratio 8.62; runs were concurrent and not a controlled speed comparison.

Two missed tiny fragments in DB1 (Tarquinia ID 105 and Resurrection ID 121) were recovered in independent indexed-deep reruns; both selected whole cases passed (65/65 and 108/108, respectively, with 42/42 and 14/14 distractors rejected). This is targeted development evidence, not a validation of deep on all 28. On the portable NumPy/SciPy path with browser resource limits, Tarquinia scored 64/65 in default and 65/65 in deep, rejecting all 42 distractors (native Python, not actual WASM timing).

See [the optimization update](OPTIMIZATION.md) for the targeted Balanced reruns. The explicit indexed engine remains available for baseline reproduction.

See [per-case measured evidence](../reports/dafne_indexed_benchmark.json). The previous full census covers 65 archives and 1119 configurations; this is **not** a full-catalog run, and the benchmark subset is not a held-out test. No assertion of comparable published SOTA ranking is made.

## Use in native Studio

```sh
git checkout main
python -m pip install -e '.[reference]'
python -m shardelix studio
```

In Reconstruction enable **Use original image for matching (optional)**. Upload a single-case official ZIP (its `frag_eroded/` folder is recognized, not the intact artwork or annotations), or ZIP the PNGs of one DB1 case. Separately select the official original artwork as **Matching reference image**. Use Balanced first (indexed plus bounded automatic rescue); High accuracy enables broader experimental deep micro-fragment search. An archive with several DB1 configurations requires choosing a case, not mixing them. For batch testing use the script below.

Native guardrails: 8192 fragments, 64 MP reference, 330 MP decoded input. Browser: 4096 fragments, 8 MP reference, 24 MP decoded input, 32 MiB upload. Native Studio upload limit is estimated from available disk and parser RAM. These are guards, not certified throughput; RAM and real browser runtime can constrain lower. The largest verified case here is 2045 pieces. Original-scale input must be preserved. Native is recommended for DB2. The unified framework now integrates reference-assisted and blind workflows in main.

## Reproduce

Download official input and (for DB2) solution ZIPs from the publisher. Do not upload solutions to Studio.

```sh
python benchmarks/benchmark_indexed_dafne.py input.zip results --solution solution.zip --keep
python benchmarks/benchmark_indexed_dafne.py db1.zip results --all-cases --resume
python benchmarks/benchmark_indexed_dafne.py db1.zip auto --case 2019-2-19_14.55.4 --engine auto
python benchmarks/benchmark_indexed_dafne.py db1.zip deep --case 2019-2-19_14.55.4 --engine indexed-deep
```

The runner exports per-case JSON, independent pose metrics at 10° and 5°, and optionally images. Each case is checked after matching; it exits nonzero if any strict case fails. To resume use a dedicated output directory for each engine/input. The older benchmark scripts explicitly select FFT for reproducibility. No official artwork or solution files are redistributed in this repository.
