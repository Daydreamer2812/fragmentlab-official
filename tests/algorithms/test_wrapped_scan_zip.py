"""Folder-wrapped ZIPs and scanned document strip mask sidecars."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np
from shardelix.workflow import inspect_uploads, safe_zip, run_job


def png(a):
    buf=io.BytesIO(); Image.fromarray(a).save(buf,format="PNG"); return buf.getvalue()


def archive(path,wrong_mask=False):
    with zipfile.ZipFile(path,"w") as z:
        for i in range(4):
            h,w=120,19+i
            mask=np.zeros((h,w),np.uint8)
            for y in range(h):
                left=2+int(y//40)%2
                mask[y,left:left+14]=255
            a=np.full((h,w,3),20,np.uint8)
            a[mask>0]=240
            a[18+i*8:22+i*8,5:12]=0
            z.writestr(f"collection/{i}.png",png(a))
            z.writestr(f"collection/bg_mask_vis/{i}.png",png(mask[:-1] if wrong_mask and i==0 else mask))
            z.writestr(f"collection/bg_mask/{i}.npy",b"intentionally not parsed")
        z.writestr("collection/order.txt",b"NEVER USED FOR MATCHING")
        z.writestr("collection/evaluation/source.png",png(np.full((3,3,3),255,np.uint8)))


class WrappedScanZipTests(unittest.TestCase):
    def test_wrapped_zip_masks_and_auto_strip_route(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            path=root/"upload/collection.zip";archive(path)
            info=inspect_uploads([path],browser=True)
            self.assertEqual(info["observed_count"],4)
            self.assertEqual(info["foreground_mask_count"],4)
            self.assertIsNone(info["suggested_grid"])
            self.assertEqual(info["suggested_document"]["height"],120)
            s=run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True},browser=True)
            self.assertEqual(s["diagnostics"]["placed"],4)
            self.assertEqual(s["diagnostics"]["scan"]["kind"],"document-strips")
            self.assertEqual(s["diagnostics"]["render_collision_pixels"],0)
            self.assertIsNone(s["metrics"])
            report=json.loads((root/"result/scan_report.json").read_text())
            self.assertEqual(len(report["pieces"]),4)
            self.assertTrue(all(row["masked"] and row["rectified_wh"]==[14,120] for row in report["pieces"]))
            with Image.open(root/"result/assembly.png") as im:self.assertEqual(im.size,(56,120))
            with zipfile.ZipFile(root/"result/fragments.zip") as z:
                self.assertEqual(len(z.namelist()),4)
                self.assertFalse(any("mask_vis" in f or "order" in f for f in z.namelist()))

    def test_bad_mask_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/"bad.zip";archive(p,True)
            with self.assertRaisesRegex(ValueError,"mask dimensions"):
                safe_zip(p,root/"case",browser=True,scan=True,extract_masks=True)

    def test_plain_nested_tiles(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/"tiles.zip"
            with zipfile.ZipFile(p,"w") as z:
                for i in range(4):z.writestr(f"myfolder/tile{i}.png",png(np.full((8,8,3),255,np.uint8)))
            self.assertEqual(inspect_uploads([p],True)["observed_count"],4)
            self.assertEqual(safe_zip(p,root/"case",browser=True),(None,4))

    def test_dafne_single_case_zip_and_reference_exclusion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/"dafne.zip"
            data=png(np.full((10,10,3),255,np.uint8))
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("painting/painting.jpg",data)
                z.writestr("painting/frag_eroded/frag_eroded_0.png",data)
                z.writestr("painting/fragments.txt",b"truth is not a fragment")
            _,count,_,details=safe_zip(p,root/"case",details=True,scan=True)
            self.assertEqual(count,1)
            self.assertTrue(details["piece_rasters"])
            self.assertEqual([x.name for x in (root/"case/pieces").iterdir()],["frag_eroded_0.png"])
            with zipfile.ZipFile(p,"a") as z:
                z.writestr("painting2/frag_eroded/frag_eroded_0.png",data)
            with self.assertRaisesRegex(ValueError,"multiple cases"):
                safe_zip(p,root/"case2",scan=True)

    def test_ambiguous_dirs_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"multiple.zip"
            with zipfile.ZipFile(p,"w") as z:
                for d in ("one","two"):z.writestr(d+"/image.png",png(np.full((10,10),255,np.uint8)))
            with self.assertRaisesRegex(ValueError,"Multiple image directories"):inspect_uploads([p],True)


if __name__=="__main__":unittest.main()
