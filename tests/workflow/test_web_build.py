import unittest,tempfile,subprocess,sys,zipfile,os,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class WebBuildTests(unittest.TestCase):
    def test_static_artifact_shared_core(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            subprocess.run([sys.executable,str(ROOT/"scripts/build_web.py"),"--out",str(d)],check=True,capture_output=True)
            self.assertTrue((d/".nojekyll").exists())
            html=(d/"index.html").read_text()
            for text in ("Fragment generator","Reconstruction","Ground truth","Benchmark generated cases","Performance profile","Use original image for matching (optional)","./worker.js","__SESSION_TOKEN__"):
                self.assertIn(text,html)
            import re
            revision=re.search(r'buildRevision="([0-9a-f]{20})"',html)
            self.assertIsNotNone(revision);self.assertNotIn("__BUILD_REVISION__",html)
            self.assertIn('./worker.js?v=',html)
            worker=(d/"worker.js").read_text()
            self.assertIn('assetURL("engine.zip")',worker);self.assertIn('searchParams.set("v",revision)',worker)
            self.assertIn("browser=True",worker);self.assertIn("engine.zip",worker)
            self.assertIn("v0.27.7",worker)
            self.assertNotIn("scanOCR",html)
            self.assertNotIn("tesseract",worker.lower())
            self.assertNotIn("refineScan",worker)
            with zipfile.ZipFile(d/"engine.zip") as z:
                self.assertNotIn("shardelix/reconstruction/physical_ocr.py",z.namelist())
                for f in ("shardelix/api.py","shardelix/native_core.py","shardelix/workflow.py","shardelix/uploads.py","shardelix/reconstruction/case_io.py","shardelix/reconstruction/pose.py","shardelix/reconstruction/rectification.py","shardelix/reconstruction/irregular/frame_anchor.py","shardelix/reconstruction/strips/multi_gap.py","shardelix/reconstruction/tiles/verification.py","shardelix/reconstruction/strips/physical.py","shardelix/reconstruction/strips/document_patches.py","shardelix/reconstruction/irregular/eroded.py","shardelix/reconstruction/irregular/offset_contours.py","shardelix/reconstruction/irregular/component_graph.py","shardelix/reconstruction/irregular/continuation.py","shardelix/reconstruction/irregular/spectral_gap.py","shardelix/reconstruction/reference/fft.py","shardelix/reconstruction/reference/index.py"):
                    self.assertEqual(z.read(f),(ROOT/"src"/f).read_bytes())
            p=subprocess.run([sys.executable,"-c","import importlib,pkgutil,shardelix; import shardelix.reconstruction as r; [importlib.import_module(m.name) for m in pkgutil.walk_packages(r.__path__,r.__name__+'.')]; print(shardelix.__version__)"],cwd=d,
                         env={**os.environ,"PYTHONPATH":str(d/"engine.zip")},capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            with (ROOT/"pyproject.toml").open("rb") as project:
                version=tomllib.load(project)["project"]["version"]
            self.assertEqual(p.stdout.strip(),version)
            self.assertIn(f"v{version} · Unified",html)
            self.assertNotIn("__SHARDELIX_VERSION__",html)
            self.assertIn(f'version: "{version}"',(ROOT/"CITATION.cff").read_text())
            self.assertTrue((ROOT/"docs/releases"/f"v{version}.md").is_file())
if __name__=="__main__":unittest.main()
