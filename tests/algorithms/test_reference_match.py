"""Reference-assisted mode must remain opt-in and separate from truth."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from shardelix.workflow import run_job


class ReferenceModeTests(unittest.TestCase):
    def test_truth_is_never_a_matching_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/"upload").mkdir();(base/"truth").mkdir()
            Image.new("RGB",(30,30),"red").save(base/"upload"/"piece.png")
            Image.new("RGB",(80,80),"red").save(base/"truth"/"source.png")
            with self.assertRaisesRegex(ValueError,"separate reference slot"):
                run_job(base,{"task":"blind","input_mode":"reference"})

    def test_absolute_registration_rotation_and_outlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/"upload").mkdir();(base/"matching_reference").mkdir()
            rng=np.random.default_rng(34)
            rgb=np.clip(gaussian_filter(rng.uniform(0,255,(124,136,3)),(.9,.9,0))*2-127,0,255).astype(np.uint8)
            Image.fromarray(rgb).save(base/"matching_reference"/"source.png")
            h,w=40,38;yy,xx=np.indices((h,w));mask=((xx-18)/18)**2+((yy-19)/19)**2<1
            a=np.dstack([rgb[42:42+h,51:51+w],mask.astype(np.uint8)*255])
            Image.fromarray(np.rot90(a,1)).save(base/"upload"/"piece.png")
            other=a.copy();other[:,:,:3]=[255,0,0]
            Image.fromarray(other).save(base/"upload"/"outlier.png")
            result=run_job(base,{"task":"blind","input_mode":"reference"},browser=True)
            self.assertEqual(result["diagnostics"]["placed"],1)
            self.assertTrue(result["diagnostics"]["matching_reference_used"])
            self.assertIsNone(result["metrics"])
            report=json.loads((base/"result"/"scan_report.json").read_text())
            poses=json.loads((base/"result"/"placements.json").read_text())
            record=next(p for p in report["pieces"] if p["file"]=="piece.png")
            pred=poses[record["id"]];A=np.array(pred["input_to_canvas_affine"])
            R=A[:,:2];pos=R@(np.array([h/2,w/2])-np.array(pred["crop_xy"]))+A[:,2]
            self.assertLess(np.linalg.norm(pos-[51+w/2,42+h/2]),2.5)
            self.assertLess(abs(np.rad2deg(np.arctan2(R[1,0],R[0,0]))-90),3)
            self.assertFalse(report["accuracy_verified"])
            with Image.open(base/"result"/"assembly.png") as img:
                self.assertEqual(img.size,(136,124))
                self.assertLess(np.count_nonzero(np.array(img)[:,:,3]),w*h)

    def test_adaptive_tiny_retry_rescues_collision_without_eviction(self):
        from unittest.mock import patch
        from shardelix.reconstruction.common import Fragment
        from shardelix.reconstruction.reference.index import IndexedMatch,reconstruct_indexed
        def match(x,y,err):
            return IndexedMatch(0.,np.array([x,y],np.float32),err,3.,"test")
        class Index:
            sift_index=None
            def __init__(self,*args,**kwargs):pass
            def match(self,p,**kwargs):
                return [match(20,20,.5 if p.id=="a" else .9 if p.id=="b" else 1.2)]
            def micro_match(self,p):
                return [match(45,45,.2)] if p.id=="b" else [match(20,20,.1)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/"reference.png";Image.new("RGB",(64,64)).save(src)
            a=np.full((8,8,4),255,np.uint8)
            ps=[Fragment(k,a.copy()) for k in ("a","b","c")]
            with patch("shardelix.reconstruction.reference.index.ReferenceIndex",Index):
                d=reconstruct_indexed(ps,src,root/"out",{"input_images":3,"common_scale":1.},adaptive_micro=True)
            self.assertEqual(d["placed"],2)
            self.assertEqual(d["micro_rescue"],{"attempted":2,"placed":1,"eligible":2})
            poses=json.loads((root/"out/placements.json").read_text())
            self.assertEqual(set(poses),{"a","b"})
            np.testing.assert_allclose(poses["a"]["input_to_canvas_affine"],[[1,0,16],[0,1,16]])
            self.assertEqual(d["render_collision_pixels"],0)


if __name__=="__main__":unittest.main()
