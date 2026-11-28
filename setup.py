"""Standard pure-Python installation; opt-in native platform wheel.

SHARDELIX_BUILD_NATIVE=1 python -m pip wheel --no-deps .
No compiler is needed for the default, portable Python wheel.
"""
from pathlib import Path
import os
import runpy

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

NATIVE = os.environ.get("SHARDELIX_BUILD_NATIVE") == "1"


class ShardelixDistribution(Distribution):
    def has_ext_modules(self):
        # ctypes does not require a CPython C extension, but native wheels
        # must never be mislabeled as architecture-independent.
        return NATIVE


class BuildPython(build_py):
    def run(self):
        super().run()
        output = Path(self.build_lib) / "shardelix"
        for suffix in (".so", ".dll", ".dylib"):
            (output / ("_fl_core" + suffix)).unlink(missing_ok=True)
        if NATIVE:
            build = runpy.run_path(str(Path(__file__).parent / "scripts/build_native.py"))["build"]
            build(output)


setup(distclass=ShardelixDistribution, cmdclass={"build_py": BuildPython})
