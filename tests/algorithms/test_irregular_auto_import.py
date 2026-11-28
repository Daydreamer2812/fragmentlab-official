"""Automatic metadata-free color-key/irregular import and large scan budget."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from shardelix.workflow import inspect_uploads, run_job
from shardelix.reconstruction.irregular.scan import chroma_mask


def fixture_archive(path,n,size=30):
    """All opaque, equal-size RGB; the colored sentinel is the mask."""
    y,x=np.indices((size,size));mask=((x-size/2)**2+(y-size/2)**2)<(size*.34)**2
    with zipfile.ZipFile(path,"w") as z:
        for i in range(n):
            a=np.full((size,size,3),[255,0,255],np.uint8)
            a[mask]=[(40+i*73)%200+10,(20+i*37)%200+10,(90+i*13)%200+10]
            b=io.BytesIO();Image.fromarray(a).save(b,format="PNG")
            z.writestr(f"collection/fragment_{i:04d}.png",b.getvalue())
        # Metadata of arbitrary benchmarks is not a public geometry schema.
        z.writestr("collection/source_metadata.json",json.dumps({"ground_truth":"DO NOT USE", "spurious_source_ids":[1]}))
    return int(mask.sum())


class IrregularAutoImportTests(unittest.TestCase):
    def test_chroma_key_requires_rim(self):
        a=np.zeros((40,40,3),np.uint8);a[10:30,10:30]=[255,0,255]
        self.assertIsNone(chroma_mask(a))

    def test_equal_size_chroma_not_grid_even_in_digital_auto(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            path=root/"upload/keyed.zip";area=fixture_archive(path,4)
            info=inspect_uploads([path],True)
            self.assertEqual(info["observed_count"],4)
            self.assertIsNone(info["suggested_grid"])
            self.assertIsNone(info["metadata"])
            state=run_job(root,{"task":"blind","input_mode":"digital","auto_layout":True},browser=True)
            self.assertEqual(state["diagnostics"]["scan"]["detected_pieces"],4)
            self.assertIsNone(state["metrics"])
            report=json.loads((root/"result/scan_report.json").read_text())
            self.assertTrue(all(p["method"]=="magenta-chroma-key" and p["mask_pixels"]==area for p in report["pieces"]))
            self.assertEqual(report["pair_search"],"all-pairs")

    def test_large_irregular_archive_uses_index_and_retains_all_masks(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            path=root/"upload/keyed.zip";fixture_archive(path,102,size=18)
            state=run_job(root,{"task":"blind","input_mode":"digital","auto_layout":True},browser=True)
            d=state["diagnostics"];s=d["scan"]
            self.assertEqual(s["detected_pieces"],102)
            self.assertEqual(s["pair_search"],"indexed-contour-color")
            self.assertEqual(s["possible_pairs"],5151)
            self.assertLessEqual(s["searched_pairs"],1020)
            self.assertIsNone(state["metrics"])
            self.assertFalse(s["accuracy_verified"])
            with zipfile.ZipFile(root/"result/fragments.zip") as z:
                self.assertEqual(len(z.namelist()),102)
                self.assertFalse(any("metadata" in p or "truth" in p for p in z.namelist()))
            self.assertIn("unanchored_components.json",state["files"])


if __name__=="__main__":
    unittest.main()
