# Shardelix documentation

[Project overview](../README.md) · [Latest release](releases/v15.0.0.md) ·
[Benchmark runners](../benchmarks/README.md) · [Raw benchmark reports](../reports/README.md)

## Start here

| Topic | Documentation |
|:--|:--|
| Install, run, integrate | [Framework API and CLI](FRAMEWORK_API.md) |
| Browser Studio and local hosting | [Web](WEB.md) |
| Public interfaces and routing | [Framework](FRAMEWORK.md) |
| Source code and module responsibilities | [Architecture](ARCHITECTURE.md) |
| Coordinate frames and metric conventions | [Coordinate conventions](COORDINATE_CONVENTIONS.md) |
| Input sizes, memory budgets and practical limits | [Resource scaling](RESOURCE_SCALING.md) |

## Development

- [Build and QA tools](../scripts/README.md)
- [Benchmark runners and frozen manifest](../benchmarks/README.md)
- [Contribution guide](../CONTRIBUTING.md)

## Algorithms and research

| Topic | Documentation |
|:--|:--|
| Latest research results and outstanding criteria | [V15 technical report](TECHNICAL_REPORT.md) · [V15 release](releases/v15.0.0.md) |
| Physical documents, historical measurements | [V14 results](V14_RESULTS.md) |
| Large intact irregular fragments | [Irregular scaling](IRREGULAR_SCALE.md) |
| Eroded boundaries and missing material | [Eroded recovery](ERODED_RECOVERY.md) |
| Optimization history | [Optimization](OPTIMIZATION.md) |
| External datasets and evaluation protocols | [Datasets](DATASETS.md) |
| Bibliography | [References](REFERENCES.md) |

## DAFNE evaluation — separate protocols

- [Indexed reference-assisted benchmark](DAFNE_INDEXED.md)
- [Official dataset catalog and historical FFT results](DAFNE_OFFICIAL_BENCHMARK.md)
- [Reference-assisted development protocol](DAFNE_REFERENCE.md)
- [Blind damaged-fragment experiments](ERODED_RECOVERY.md)

Results obtained using an intact reference image are never pooled with
blind reconstruction. External datasets are not redistributed.

## Historical results

- [Release notes, newest first](releases/README.md)
- [V14](V14_RESULTS.md) · [V13](V13_RESULTS.md) · [V12](V12_RESULTS.md)
- [Benchmark archive](../reports/README.md) · [Reproduction tools](../benchmarks/README.md)

Historical documents preserve original measurements and may describe
older behavior. For the current public interfaces prefer the latest
release and the framework API documentation.
