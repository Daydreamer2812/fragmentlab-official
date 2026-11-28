"""Public API safety, portable resource diagnostics and command-line contract."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from shardelix import Fragment, Pose, ReconstructionResult, __version__, reconstruct_files
from shardelix.reconstruction.cli import main
from shardelix.reconstruction.output import staged_output
from shardelix.reconstruction.resource_budget import peak_rss_mib


def example_result(x=0):
    rgba = np.full((4, 6, 4), 255, dtype=np.uint8)
    return ReconstructionResult([Fragment("a", rgba)], {"a": Pose(x, 0, 0)},
                                (4, 6), {}, None, False)


class OutputTests(unittest.TestCase):
    def test_success_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result"
            target.mkdir()
            result = example_result().save(target)
            self.assertEqual(result["diagnostics"]["status"], "assembled")
            self.assertEqual(result["diagnostics"]["rendered_piece_count"], 1)
            self.assertTrue((target / "assembly.png").is_file())
            before = (target / "placements.json").read_bytes()
            with self.assertRaises(ValueError):
                example_result().save(target)
            self.assertEqual((target / "placements.json").read_bytes(), before)

    def test_failed_export_preserves_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            for existing in (False, True):
                target = Path(tmp) / "result"
                if existing:
                    target.mkdir()
                with patch("shardelix.api.write_cut_outputs", side_effect=OSError("disk error")):
                    with self.assertRaises(OSError):
                        example_result().save(target)
                self.assertEqual(target.exists(), existing)
                if existing:
                    self.assertEqual(list(target.iterdir()), [])
                self.assertEqual(list(Path(tmp).glob(".shardelix-*")), [])

    def test_destination_populated_during_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result"
            with self.assertRaises(ValueError):
                with staged_output(target) as stage:
                    (stage / "assembly.png").write_bytes(b"new")
                    target.mkdir()
                    (target / "keep.txt").write_text("keep")
            self.assertEqual((target / "keep.txt").read_text(), "keep")
            self.assertFalse((target / "assembly.png").exists())

    def test_outside_canvas_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = example_result(x=1).save(Path(tmp) / "result")
            diag = result["diagnostics"]
            self.assertEqual(diag["status"], "partial")
            self.assertEqual(diag["rendered_piece_count"], 0)
            self.assertEqual(diag["not_rendered"], ["a"])
            self.assertFalse(diag["accuracy_verified"])

    def test_reconstruct_files_copy_failure(self):
        def fake_job(root, config, notify):
            root.joinpath("result").mkdir()
            root.joinpath("result", "assembly.png").write_bytes(b"test")
            return {"state": "done"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "piece.png"
            Image.new("RGB", (8, 8)).save(image)
            target = root / "result"
            with patch("shardelix.workflow.run_job", side_effect=fake_job):
                with patch("shutil.copytree", side_effect=OSError("disk error")):
                    with self.assertRaises(OSError):
                        reconstruct_files(image, target)
                self.assertFalse(target.exists())
                state = reconstruct_files(image, target)
            self.assertEqual(state["state"], "done")
            self.assertEqual((target / "assembly.png").read_bytes(), b"test")


class ResourceTests(unittest.TestCase):
    def test_posix_units(self):
        for platform, raw in (("linux", 64 * 1024), ("darwin", 64 * 1024**2)):
            resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda x: SimpleNamespace(ru_maxrss=raw))
            with patch.object(sys, "platform", platform), patch.dict(sys.modules, resource=resource):
                self.assertEqual(peak_rss_mib(), 64.0)

    def test_windows_and_browser(self):
        with patch.object(sys, "platform", "win32"):
            with patch("shardelix.reconstruction.resource_budget._windows_peak_rss_bytes", return_value=64 * 1024**2):
                self.assertEqual(peak_rss_mib(), 64.0)
            with patch("shardelix.reconstruction.resource_budget._windows_peak_rss_bytes", return_value=None):
                self.assertIsNone(peak_rss_mib())
        with patch.object(sys, "platform", "emscripten"):
            self.assertIsNone(peak_rss_mib())

    def test_missing_resource_module(self):
        with patch.object(sys, "platform", "linux"), patch.dict(sys.modules, resource=None):
            self.assertIsNone(peak_rss_mib())


class CliTests(unittest.TestCase):
    def test_version(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as exc:
            main(["--version"])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), "shardelix " + __version__)

    def test_doctor(self):
        with patch.dict(os.environ, {"SHARDELIX_MEMORY_MIB": "128"}):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["doctor"]), 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["version"], __version__)
            self.assertTrue(report["ready"])
            self.assertIn(report["backend"], ("cpp-native", "python-reference"))
            self.assertIn("resource_budget", report)

    def test_operational_error_and_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["reconstruct", str(Path(tmp) / "missing.png"), "--out", str(Path(tmp) / "out")]
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(main(args), 2)
            self.assertIn("Provide existing fragment", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
            with self.assertRaises(ValueError):
                main(["--debug", *args])
