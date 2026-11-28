"""Build the optional C++20 ctypes core for a native host or wheel."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build(output_dir: Path | None = None) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("C++20 compiler (g++ or clang++) required")
    system = platform.system()
    ext = {"Windows": ".dll", "Darwin": ".dylib"}.get(system, ".so")
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "src/shardelix"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ("_fl_core" + ext)
    linkage = ["-dynamiclib", "-fPIC"] if system == "Darwin" else ["-shared"]
    if system == "Windows":
        if "g++" in Path(compiler).name.lower():
            linkage += ["-static-libgcc", "-static-libstdc++"]
    elif system != "Darwin":
        linkage += ["-fPIC"]
    args = [compiler, "-std=c++20", "-O3", "-ffp-contract=off", *linkage,
            "-DNDEBUG", str(ROOT / "cpp/core.cpp"), "-o", str(target)]
    subprocess.run(args, check=True)
    print("Built", target)
    return target


if __name__ == "__main__":
    build()
