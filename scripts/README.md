# Development and benchmark tools

Run these scripts from the repository root after installing the package:
`python -m pip install -e .`. Keep external datasets under an ignored
local directory such as `data/external/` and generated files under
`outputs/` (create the output directory before running the examples).

## Build, QA and examples

| Tool | Purpose |
|:--|:--|
| [build_native.py](build_native.py) | Compile the optional native C++20 accelerator |
| [build_wasm.py](build_wasm.py) | Compile the WebAssembly accelerator |
| [build_web.py](build_web.py) | Build the static Web Studio |
| [smoke_studio.py](smoke_studio.py) | Local Studio smoke check |
| [qa_studio.cjs](qa_studio.cjs), [qa_dafne_reference.cjs](qa_dafne_reference.cjs) | Browser QA |
| [create_document_fixture.py](create_document_fixture.py) | Generate the deterministic document example |
| [make_photo_showcase.py](make_photo_showcase.py), [make_showcase.py](make_showcase.py) | Reproduce visual evidence |

## Recommended benchmarks

| Goal | Tool |
|:--|:--|
| Real DocDataset strips | [benchmark_scanned_strips.py](benchmark_scanned_strips.py) |
| External mechanical document scans | [benchmark_mechanical_strips.py](benchmark_mechanical_strips.py) |
| Simulated scan damage | [benchmark_scan_stress.py](benchmark_scan_stress.py) |
| Irregular scalability and memory guards | [stress_irregular_resources.py](stress_irregular_resources.py) |
| Damaged fragments | [benchmark_damage_stress.py](benchmark_damage_stress.py), [benchmark_eroded.py](benchmark_eroded.py) |
| Blind DAFNE | [benchmark_blind_dafne.py](benchmark_blind_dafne.py) |
| Explicit reference-assisted DAFNE | [benchmark_indexed_dafne.py](benchmark_indexed_dafne.py), [benchmark_official_dafne.py](benchmark_official_dafne.py), [benchmark_reference_dafne.py](benchmark_reference_dafne.py) |
| Independent component evaluation | [evaluate_dafne_components.py](evaluate_dafne_components.py) |

```bash
python scripts/benchmark_scanned_strips.py DocDataset.zip --seed 624 --out outputs/real624.json
python scripts/benchmark_mechanical_strips.py data/external/datasets --out outputs/mechanical.json
python scripts/benchmark_scan_stress.py DocDataset.zip --out outputs/stress.json
python scripts/stress_irregular_resources.py --count 4096 --tile 36 --seed 827 --memory-mib 512 --out outputs/scale4096.json
```

## Historical reproduction tools

These runners retain their original names and protocols for reproducibility.
Use the matching release or commit when reproducing old measurements.

- Digital: [benchmark_classical.py](benchmark_classical.py), [benchmark_cho.py](benchmark_cho.py), [benchmark_lsu.py](benchmark_lsu.py), [benchmark_document12.py](benchmark_document12.py)
- V12/V13: [benchmark_v12.py](benchmark_v12.py), [benchmark_profiles_v12.py](benchmark_profiles_v12.py), [benchmark_cpp13.py](benchmark_cpp13.py)
- Earlier scale and stress: [benchmark_scale_v8.py](benchmark_scale_v8.py), [stress_v9.py](stress_v9.py), [stress_matrix_v9.py](stress_matrix_v9.py)
- Earlier scan studies: [benchmark_scan.py](benchmark_scan.py), [benchmark_sparse_scan.py](benchmark_sparse_scan.py)

[Dataset protocol](../docs/DATASETS.md) · [Benchmark archive](../reports/README.md)
