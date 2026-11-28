"""Physical scan route: segmentation, rigid geometry, independent evaluation."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from shardelix.reconstruction.irregular.scan import preprocess
from shardelix.workflow import run_job
from benchmarks.benchmark_scan import trial


class ScanTests(unittest.TestCase):
    def test_noisy_rotated_jpeg_reconstruction(self):
        row=trial(42,4,2.5)
        self.assertEqual(row["detected"],4)
        self.assertEqual(row["placed"],4)
        self.assertGreater(row["mask_iou_mean"],.98)
        self.assertEqual(row["relative_pose_correct"],4)

    def test_overview_auto_route_with_slight_lighting_gradient(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            y,x=np.indices((180,300))
            a=np.stack([20+40*x/300+10*y/180]*3,axis=2)
            a[30:110,20:100]=190;a[50:150,180:255]=220
            Image.fromarray(a.astype(np.uint8)).save(root/"upload"/"overview.png")
            s=run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True},browser=True)
            self.assertEqual(s["diagnostics"]["scan"]["detected_pieces"],2)
            self.assertEqual(s["diagnostics"]["status"],"partial")
            self.assertIsNone(s["metrics"])
            self.assertTrue({"scan_report.json","fragments.zip","assembly.png","fragments_preview.png"}.issubset(s["files"]))
            report=json.loads((root/"result"/"scan_report.json").read_text())
            self.assertFalse(report["accuracy_verified"])
            self.assertEqual([r["mask_pixels"] for r in report["pieces"]],[6400,7500])
            self.assertTrue(all(r["method"]=="border-plane" for r in report["pieces"]))
            with zipfile.ZipFile(root/"result"/"fragments.zip") as z:
                self.assertEqual(sorted(z.namelist()),["pieces/scan_001.png","pieces/scan_002.png"])

    def test_transparent_masks_and_zip_scans(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            a=np.zeros((100,150,4),np.uint8)
            a[15:85,15:55]=[100,160,200,255]
            a[20:80,85:130]=[180,70,60,255]
            a[30:35,30:35]=0
            p=root/"scan.png";Image.fromarray(a).save(p)
            with zipfile.ZipFile(root/"upload"/"input.zip","w") as z:z.write(p,"scans/scan.png")
            state=run_job(root,{"task":"blind","input_mode":"scan"},browser=True)
            self.assertEqual(state["diagnostics"]["scan"]["detected_pieces"],2)
            report=json.loads((root/"result"/"scan_report.json").read_text())
            self.assertEqual(report["pieces"][0]["mask_pixels"],70*40-25)
            self.assertEqual(report["pieces"][0]["method"],"alpha")

    def test_uniform_image_reports_no_silhouette(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/"blank.png";Image.new("RGB",(80,90),"white").save(p)
            with self.assertRaisesRegex(ValueError,"No paper silhouettes"):
                preprocess([p],root/"pieces")


if __name__=="__main__":
    unittest.main()
