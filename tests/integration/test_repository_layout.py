"""Repository navigation and benchmark entry-point regression checks."""
import json
from pathlib import Path
import re
import unittest

from benchmarks.benchmark_suite import command

ROOT = Path(__file__).resolve().parents[2]


class RepositoryLayoutTests(unittest.TestCase):
    def test_local_documentation_links_resolve(self):
        files = [ROOT / name for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md")]
        for folder in ("docs", "benchmarks", "reports", "scripts"):
            files.extend((ROOT / folder).rglob("*.md"))
        files.append(ROOT / "data/README.md")
        for path in sorted(files):
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                target = target.split("#", 1)[0]
                if not target or target.startswith(("https://", "http://", "mailto:")):
                    continue
                with self.subTest(document=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue((path.parent / target).exists(), "Broken relative link")

    def test_manifest_dispatches_to_benchmark_runners(self):
        manifest = ROOT / "benchmarks/manifest.example.json"
        stages = json.loads(manifest.read_text(encoding="utf-8"))["stages"]
        self.assertTrue(stages)
        for stage in stages:
            source = stage.get("source")
            source = manifest.parent / source if source else None
            args = command(stage, source, ROOT / "outputs/layout-test")
            runner = Path(args[1])
            with self.subTest(stage=stage["name"]):
                self.assertEqual(runner.parent, ROOT / "benchmarks")
                self.assertTrue(runner.is_file())
        # DAFNE is optional, so it need not be in the sample manifest.
        args = command({"kind": "dafne", "case": "example"}, Path("archive.zip"), Path("out"))
        self.assertTrue(Path(args[1]).is_file())


if __name__ == "__main__":
    unittest.main()
