# Benchmarks

Independent research evaluation, outside the runtime package. Run from the
repository root after installing Shardelix (`python -m pip install -e .`).
Ground truth is used only by the post-solve evaluator. Reference-assisted
experiments are separate protocols.

## Current evaluation

| Runner | Purpose |
|:--|:--|
| [benchmark_suite.py](benchmark_suite.py) | Frozen manifests, provenance and resumable independent runs |
| [benchmark_scanned_strips.py](benchmark_scanned_strips.py) | Real DocDataset physical strips |
| [benchmark_mechanical_strips.py](benchmark_mechanical_strips.py) | Mechanical document scans |
| [benchmark_scan_stress.py](benchmark_scan_stress.py) | Synthetic scan perturbations |
| [benchmark_damage_stress.py](benchmark_damage_stress.py) | Continuous rotations and eroded fragments |
| [benchmark_blind_dafne.py](benchmark_blind_dafne.py) | Blind DAFNE cases |
| [stress_irregular_resources.py](stress_irregular_resources.py) | Irregular scaling and memory guards |
| [evaluate_dafne_components.py](evaluate_dafne_components.py) | Independent component metrics |

```bash
python benchmarks/benchmark_scanned_strips.py data/external/DocDataset.zip --seed 624 --out outputs/real624.json
python benchmarks/benchmark_mechanical_strips.py data/external/deeprec/datasets --out outputs/mechanical.json
python benchmarks/benchmark_scan_stress.py data/external/DocDataset.zip --out outputs/stress.json
python benchmarks/stress_irregular_resources.py --count 4096 --tile 36 --seed 827 --memory-mib 512 --out outputs/scale4096.json
```

For multiple cases, edit [manifest.example.json](manifest.example.json) to
point at local data and run:

```bash
python benchmarks/benchmark_suite.py --manifest benchmarks/manifest.example.json --out outputs/suite
python benchmarks/benchmark_suite.py --manifest benchmarks/manifest.example.json --out outputs/suite --resume
```

Manifest inputs resolve relative to the manifest's location. The manifest
runner records the engine and input hashes, versions, backend, stage
configuration, metrics checksum, failures and timeouts.

## Reference-assisted evaluation

These are intentionally not pooled with blind scores:

- [benchmark_indexed_dafne.py](benchmark_indexed_dafne.py)
- [benchmark_official_dafne.py](benchmark_official_dafne.py)
- [benchmark_reference_dafne.py](benchmark_reference_dafne.py)

## Historical reproduction

Use the matching release or commit for exact historical reproduction;
runner filenames and experiment protocols are preserved. Runners formerly
under `scripts/` now live here; use the matching release when replaying
commands from an older checkout.

- Digital: [classical](benchmark_classical.py), [Cho](benchmark_cho.py),
  [LSU](benchmark_lsu.py), [document V12](benchmark_document12.py).
- Damage: [eroded](benchmark_eroded.py).
- Earlier scale and profiles: [V12](benchmark_v12.py),
  [profiles V12](benchmark_profiles_v12.py), [C++ V13](benchmark_cpp13.py),
  [scale V8](benchmark_scale_v8.py), [stress V9](stress_v9.py),
  [stress matrix V9](stress_matrix_v9.py).
- Earlier scans: [scan](benchmark_scan.py), [sparse scan](benchmark_sparse_scan.py).

See [datasets and protocol](../docs/DATASETS.md) and
[measured results](../reports/README.md). Keep historical reports immutable;
write new outputs to an ignored location. External inputs are not bundled.
