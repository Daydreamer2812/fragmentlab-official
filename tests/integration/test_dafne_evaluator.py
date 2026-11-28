"""Regression for the offline evaluator; truth never enters the solver."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from evaluate_dafne_components import evaluate


class TestDafneEvaluator(unittest.TestCase):
    def test_wrapped_archive_and_missing_truth_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, truth = root / "input.zip", root / "truth.zip"
            png = io.BytesIO()
            Image.new("RGBA", (10, 10), (1, 2, 3, 255)).save(png, format="PNG")
            with zipfile.ZipFile(inp, "w") as z:
                z.writestr("wrapped/fragment_0001.png", png.getvalue())
            with zipfile.ZipFile(truth, "w") as z:
                z.writestr("annotations/source_metadata.json", json.dumps({
                    "ground_truth": ["5 5 5 0"],
                    "source_id_by_piece_id": {"1": 5},
                }))
            pose = {"input_to_canvas_affine": [[1, 0, 0], [0, 1, 0]],
                    "input_to_component_affine": [[1, 0, 0], [0, 1, 0]]}
            (root / "scan_report.json").write_text(json.dumps({
                "pieces": [{"id": "a", "file": "fragment_0001.png", "crop_xy": [0, 0]}],
                "common_scale": 1,
            }))
            (root / "placements.json").write_text(json.dumps({"a": pose}))
            (root / "component_placements.json").write_text(json.dumps({
                "model": "test", "components": [{"poses": {"a": pose}, "joins": []}],
            }))
            result = evaluate(inp, truth, root)
            self.assertEqual(result["primary"]["gt_pose_count"], 1)
            self.assertEqual(result["primary"]["pose_correct"], 1)
            with zipfile.ZipFile(inp, "w") as z:
                z.writestr("wrapped/fragment_0002.png", png.getvalue())
            with self.assertRaisesRegex(ValueError, "No matching truth IDs"):
                evaluate(inp, truth, root)
