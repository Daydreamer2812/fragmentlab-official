# V13 hybrid workbench

V13 keeps the V12 algorithm family routes and adds a portable C++20 ABI for contour tracing, batched edge descriptors, 2×2 cycle consensus and fused sparse matching scores. Native shared libraries and browser WebAssembly compile the same source. NumPy/SciPy, global assembly, pose recovery, image handling, and evaluation currently remain Python. This is a hybrid upgrade, not a full C++ rewrite.

## Workflow separation

- Generator: only generates anonymous shuffled pieces; exports public fragment ZIP and separate ground-truth ZIP.
- Reconstruction: direct handoff or upload existing fragments; optional separate reference. The solver never receives the reference directory.
- A matching pose archive (verified with SHA-256 fingerprints of the exact fragment files) enables independent DPA/F1. An image-only reference enables side-by-side viewing and raw overlapping-pixel MAE/RMSE; it cannot establish DPA/F1, and global rotation remains ambiguous.
- Optional benchmark still generates controlled cases, solves blindly, evaluates separately, and records failures.

## Regression and algorithm changes

Four C++ kernels are checked against their Python reference with randomized contour topology, reciprocal cycles, edge descriptors, and floating-point seam-score parity. A catastrophic Fast-profile irregular under-coverage guard tries one complementary verified candidate and accepts only when the blind objective improves, more pieces are placed, and collisions are zero.

The V12 256-piece synthetic jigsaw Fast failure is included in the regression suite. This is not a guarantee for all jigsaw inputs. Continuous-angle irregular fragments with missing pieces, short low-information edges at large scale, multi-gap strips, and repeated/blank document edges remain research problems. Benchmarking must keep the source and evaluation labels out of solver inputs.

## Paired native runtime (same V13 case, same algorithm)

Synthetic textured grid, seed 11226, 20-pixel cells, 90-degree rotations, one pair of runs per size on the same host. Solver time excludes input generation, file import, output rendering, and evaluation. These are native C++ results, not measured WebAssembly speedups.

| Pieces | Python reference | C++ backend | Solver speedup | Reranking speedup | DPA / F1 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 2.896 s | 1.308 s | 2.21× | 11.76× | 100% / 100% |
| 16,384 | 11.203 s | 6.540 s | 1.71× | 10.77× | 100% / 100% |

Both backends place every fragment without pixel collisions on these paired cases. Raw results: [paired.json](../reports/v13/paired.json). The 8-pixel-cell 16,384-piece stress case remains unresolved; a broader retrieval experiment was rejected because it degraded accuracy. A second residual hole search on continuous-angle jigsaw with missing pieces was also rejected because it recovered no additional pieces. Neither experimental change ships.

## Reproduction

```bash
python -m pip install -e .
python scripts/build_native.py
python -m unittest discover -s tests/algorithms
python -m unittest discover -s tests/integration
SHARDELIX_CORE=python python -m unittest discover -s tests/algorithms
python scripts/build_web.py --out build/pages --wasm
```

The last command requires Emscripten. CI builds the WASM module and asserts that the real browser uses it. The plain Python reference build works without Emscripten.
