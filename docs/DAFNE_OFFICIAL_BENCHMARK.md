# Official DAFNE catalog and baseline verification

Source: [official DAFNE download page](https://vision.unipv.it/DAFchallenge/DAFNE_dataset/dataset_download.html).
Measured engine: `46e801d` (reference-assisted mode, no ML). The solver was
**not modified for this benchmark**. The original artwork is a matching input,
never the pose annotations. This experiment is not reference-free reconstruction.

## Coverage and limits

All 65 official input archives were inventoried: 37
downloaded ZIPs, 28 ZIP central directories read via
HTTP byte ranges. Across 1,119 distinct configurations there are
760,171 fragment instances. DB1 has 62 artworks × 18
variants; DB2 has 3 challenge cases. No original images, fragments, solutions or
pose annotations are redistributed in this repository.

| Status | Cases |
|---|---:|
| Count/canvas/preprocessing eligible | 170 |
| Completed and independently evaluated | 25 |
| Eligible but not run at this snapshot | 145 |
| Not run – resource-limit rejection at census | 946 |
| Actual run error, DB2 | 3 |
| Total | 1119 |

The existing workflow caps detected pieces at 256, the matching reference at
2 MP and input work pixels at 4 MP; it rejects preprocessing that changes the
fragment scale. The census found 949 unsupported cases, including the 3 actual
DB2 failures. These rows have no pose accuracy and must not be interpreted
as completed solver results. The eligible remainder was too costly to complete
in this interactive run; the runner below resumes per-case checkpoints.

## Measured reference-assisted results

| Metric | Measured cases |
|---|---:|
| Completed configurations | 25 |
| Fragment inputs | 3350 |
| Correct genuine pose | 2873/2874 (99.97%) |
| Correct distractor rejection | 467/476 (98.11%) |
| False accepted distractors | 9 |
| Strictly correct cases | 17/25 |

Absolute center error ≤10 pixels, angular error ≤10 degrees; no global alignment.
Strictly correct means every genuine pose passes and no distractor is placed.
Timing is concurrent CPU wall time, not an isolated latency benchmark.
Resurrection: 18/18 configurations, 2197/2197 genuine
poses, 231/233 distractors rejected.
The other measured frescoes expose both missed genuine pieces and false
acceptance of distractors. There is no claim that all DAFNE cases passed.

## DB2 actual errors

All 3 input archives were downloaded, their original images passed separately,
and the unmodified native workflow was invoked. Each stopped at the 256
detected-piece limit, before reconstruction.

| Case | Inputs | Result |
|---|---:|---|
| Domenichino, The Virgin and the Unicorn | 600 | Resource-limit error |
| Zuccari, Cardinal's Hat | 1188 | Resource-limit error |
| Andrea di Bonaiuto, Allegory | 2045 | Resource-limit error |

## Files and reproduction

- [Full per-case inventory and result CSV](../reports/dafne_official_cases.csv)
- [Aggregate and per-archive data](../reports/dafne_official_benchmark.json)
- [Official URLs and source dimensions](../reports/dafne_official_catalog.json)
- [Resumable benchmark runner](../benchmarks/benchmark_official_dafne.py)

```bash
python -m pip install -e .
python benchmarks/benchmark_official_dafne.py --archive-dir /path/to/archives \
    --out /path/to/benchmark_results --download --workers 4
```

The runner downloads missing eligible archives from the official publisher
for research/educational use only, enumerates their views, and solves eligible
cases one at a time or in a bounded process pool. Local archive names are the
catalog IDs (e.g. `24.zip` for Resurrection); use `--ids 24` for a single artwork.
Unsupported rows are recorded without claiming reconstruction. Case JSON
files are checkpoints. Use a new output directory if the solver changes.
For a fresh complete benchmark, do not add `--max-cases-per-archive`.

The dataset copyright restricts use to nonprofit research, education and
noncommercial purposes. Respect the publisher's terms.

## Engineering implications

Increasing the constant alone is not sufficient. The exhaustive masked FFT
rotation/translation search and all-pair collision verification are too
costly for hundreds of thousands of fragments. A scalable classical
reference-assisted solver needs feature-index candidate retrieval and local
masked photometric refinement, resolution-aware preprocessing, memory-bounded
pyramids and spatial collision indexing. Score blind reconstruction separately.
