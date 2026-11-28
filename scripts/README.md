# Development tools

Run from the repository root. Benchmark and stress runners live in
[benchmarks/](../benchmarks/README.md), not here.

| Tool | Purpose |
|:--|:--|
| [build_native.py](build_native.py) | Compile the optional C++20 accelerator |
| [build_wasm.py](build_wasm.py) | Compile the WebAssembly accelerator |
| [build_web.py](build_web.py) | Package the static Web Studio |
| [smoke_studio.py](smoke_studio.py) | Local Studio HTTP integration check |
| [qa_studio.cjs](qa_studio.cjs), [qa_dafne_reference.cjs](qa_dafne_reference.cjs) | Browser QA |
| [create_document_fixture.py](create_document_fixture.py) | Generate the deterministic document fixture |
| [make_photo_showcase.py](make_photo_showcase.py), [make_showcase.py](make_showcase.py) | Reproduce documented visual demonstrations |

```bash
python -m pip install -e .
python scripts/build_native.py
python scripts/build_web.py --out build/pages
python scripts/smoke_studio.py
```

For the WASM build, use `python scripts/build_web.py --out build/pages --wasm`
with Emscripten installed. Build output and local generated results are
ignored by Git.
