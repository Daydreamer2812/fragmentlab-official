# Contributing to Shardelix

Thank you for helping improve Shardelix. Keep changes deterministic, CPU-first, and honest
about measured performance.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Before opening a pull request

1. Put algorithm changes in the appropriate geometry package under
   `src/shardelix/reconstruction/`; see [architecture](docs/ARCHITECTURE.md).
   Avoid version-numbered runtime modules and parallel workflow entry points.
   Use the [script index](scripts/README.md) and [benchmark archive](reports/README.md)
   when adding an experiment; preserve historical results.
   Keep generated artifacts under `outputs/` or `data/synthetic/`.
2. Do not expose ground truth to blind inference code.
3. Add a regression test for bug fixes and seeded evaluation for solver changes.
4. Run `python -m unittest discover -s tests/v2 && python -m unittest discover -s tests/workflow`.
5. Update relevant documentation when behavior, metrics, or limitations change.

Pull requests should explain the problem, the approach, verification performed, and any known
limitations. Do not commit raw datasets, generated outputs, credentials, or private documents.
