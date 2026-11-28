"""Build deterministic GitHub Pages static artifact from the shared Studio and Python package."""
from pathlib import Path
import argparse,shutil,zipfile,hashlib,tomllib
ROOT=Path(__file__).resolve().parents[1]
def build(output, wasm=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (ROOT/"pyproject.toml").open("rb") as f:
        version=tomllib.load(f)["project"]["version"]
    html=(ROOT/"src/shardelix/studio.html").read_text(encoding="utf-8")
    (output/"index.html").write_text(html.replace("__SHARDELIX_VERSION__",version),encoding="utf-8")
    shutil.copyfile(ROOT/"LICENSE",output/"LICENSE")
    shutil.copyfile(ROOT/"web/worker.js",output/"worker.js")
    shutil.copyfile(ROOT/"web/core_bridge.js",output/"core_bridge.js")
    if wasm:
        from build_wasm import build as compile_wasm
        compile_wasm(output)
    with zipfile.ZipFile(output/"engine.zip","w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        info=zipfile.ZipInfo("LICENSE",date_time=(2026,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,(ROOT/"LICENSE").read_bytes())
        for p in sorted((ROOT/"src/shardelix").rglob("*.py")):
            if "__pycache__" in p.parts:continue
            info=zipfile.ZipInfo(p.relative_to(ROOT/"src").as_posix(),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    # A single revision links the HTML, Worker, Python engine and WASM assets.
    # Cached older code cannot be mixed into a newly loaded Studio.
    h=hashlib.sha256()
    for name in ("index.html","worker.js","core_bridge.js","engine.zip","fl_core.js","fl_core.wasm"):
        asset=output/name
        if asset.exists():h.update(name.encode());h.update(asset.read_bytes())
    revision=h.hexdigest()[:20]
    html=output/"index.html"
    html.write_text(html.read_text(encoding="utf-8").replace("__BUILD_REVISION__",revision),encoding="utf-8")
    (output/".nojekyll").touch()
    print("Built",output)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default=str(ROOT/"build/pages"));p.add_argument("--wasm",action="store_true",help="Build C++ accelerator (requires Emscripten)");a=p.parse_args();build(a.out,a.wasm)
