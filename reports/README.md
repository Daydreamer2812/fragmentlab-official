# Benchmark archive

This directory contains measured results and evaluation diagnostics, not
runtime dependencies or copies of external datasets. Keep old results
immutable and record the exact code revision, seed, protocol, device and
timing scope when adding new measurements.

## Current evidence

| Topic | Report | Protocol / notes |
|:--|:--|:--|
| V15.0 paired evaluation | [JSON reports](v15/) | [Technical report](../docs/TECHNICAL_REPORT.md) |
| V14.5 physical-document scans | [JSON](v14/scan_document_v145.json) | [Release](../docs/releases/v14.5.0.md) |
| V14.4 mechanical scans and speed | [JSON](v14/scan_document_v144.json) | [Release](../docs/releases/v14.4.0.md) |
| V14.3 real document strips | [JSON](v14/scan_document_v143.json) | [Release](../docs/releases/v14.3.0.md) |
| V14.2 damaged fragments | [JSON](v14/damaged_fragments_v142.json) | [Release](../docs/releases/v14.2.0.md) |
| V14.1 scan development | [JSON](v14/scan_document_v141.json) | [Release](../docs/releases/v14.1.0.md) |
| V14.0.1 refactor parity | [JSON](v14/refactor_parity.json) | [Architecture](../docs/ARCHITECTURE.md) |
| V14 paired native C++ indexing | [JSON](v14/paired_arc.json) | [V14 results](../docs/V14_RESULTS.md) |
| Irregular scalability | [JSON](resource_scaling.json) | [Resource policy](../docs/RESOURCE_SCALING.md) |

## Historical and specialized experiments

| Family | Evidence |
|:--|:--|
| V13 acceleration / showcase | [Paired results](v13/paired.json), [photo showcase](v13/photo-showcase.json), [protocol](../docs/V13_RESULTS.md) |
| V12 digital cases, profiles, stress and recovery | [Case data](v12/cases/), [profiles](v12/profiles.csv), [stress](v12/stress.csv), [recovery](v12/recovery.csv), [continuous documents](v12/document_continuous.csv), [showcase](v12/showcase.json), [protocol](../docs/V12_RESULTS.md) |
| Blind damaged-fragment research | [Unified missing-edge recovery](unified_missing_edge_recovery.json), [eroded synthetic](eroded_benchmark.json), [eroded real development](eroded_real_development.json) |
| Real/simulated scans | [Real scans](scan_real.json), [simulation](scan_simulation.json), [strip scans](scan_strips.json) |
| Reference-assisted DAFNE | [Indexed results](dafne_indexed_benchmark.json), [official results](dafne_official_benchmark.json), [case CSV](dafne_official_cases.csv), [catalog](dafne_official_catalog.json), [development](dafne_reference_development.json) |
| Earlier classical and scale research | [Classical benchmark](classical_benchmark.json), [real development](classical_real_development.json), [irregular scale](irregular_scale.json), [optimization round](optimization_round.json) |

## Evaluation conventions

- Direct position accuracy, adjacency F1, directed document-strip adjacency,
  accepted-join precision and relative pose precision measure different things.
- Blind evaluation consumes truth only after reconstruction. Reference-assisted
  evaluation explicitly uses an intact source image, so it is a separate task.
- Historical results are not automatically measurements of the newest release.
  Development samples are not held-out validation.
- Record time boundaries: solver-only, matching-only, and end-to-end runtimes
  are not interchangeable. Native CPU measurements are not browser measurements.
- External images, masks, original pages, private data and generated previews
  belong outside this archive.

[Dataset protocol](../docs/DATASETS.md) · [Reproduction scripts](../scripts/README.md)
