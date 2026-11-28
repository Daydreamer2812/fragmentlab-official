"""Gap-aware matching and preservation of missing material."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.eroded import gap_candidates, rank_components
from shardelix.reconstruction.irregular.scan import preprocess, _shape, _merge, rotation
from shardelix.workflow import run_job


from benchmarks.benchmark_eroded import eroded_fixture

class ErodedFragmentTests(unittest.TestCase):
    def test_abraded_missing_rotated_with_distractor(self):
        pieces,truth=eroded_fixture()
        shapes=[_shape(p,2.) for p in pieces]
        # No pose/label passes to the actual matcher.
        candidates,diag=gap_candidates(shapes)
        self.assertGreater(diag["evaluated"],0)
        groups,joins=_merge(shapes,candidates)
        groups=rank_components(shapes,groups,joins)
        self.assertGreaterEqual(len(groups[0]),2)
        self.assertFalse(any(pieces[-1].id in (q["a"],q["b"]) for q in joins))
        # Evaluate relative pose only after matching, and do not count an
        # arbitrary global anchor as a recovered seam.
        correct=set()
        for score,i,j,R,t,info in candidates:
            if truth[i] is None or truth[j] is None:continue
            A,a=truth[i];B,b=truth[j]
            Q=A.T@B;u=A.T@(b-a)
            center=shapes[j].points.mean(0)
            position=np.linalg.norm((R-Q)@center+t-u)
            angle=np.degrees(np.arccos(np.clip(np.trace(R.T@Q)/2,-1,1)))
            if position<10 and angle<10 and info["median_gap"]>1:
                correct.add((i,j))
        self.assertGreaterEqual(len(correct),2)
        self.assertTrue(any(len(g)==1 for g in groups))

    def test_rounded_damage_does_not_prefer_a_larger_weak_contact_tree(self):
        from unittest.mock import patch
        from shardelix.reconstruction.irregular.scan import reconstruct
        y,x=np.indices((36,36))
        mask=(x-18)**2+(y-18)**2<16**2
        rgba=np.dstack([np.repeat((70+3*x)[...,None],3,axis=-1).astype(np.uint8),
                        mask.astype(np.uint8)*255])
        pieces=[Fragment(f"p{i}",rgba.copy()) for i in range(20)]
        pose=lambda i:(np.eye(2),np.array([45.*i,0.]))
        large=[{i:pose(i) for i in range(5)}]+[{i:pose(0)} for i in range(5,20)]
        small=[{i:pose(i) for i in range(2)}]+[{i:pose(0)} for i in range(2,20)]
        edge={"a":"p0","b":"p1","score":15.,"consensus_supported":True}
        for rounded,expected in ((True,2),(False,5)):
            with tempfile.TemporaryDirectory() as tmp,                  patch("shardelix.reconstruction.irregular.scan._pair",return_value=[]),                  patch("shardelix.reconstruction.irregular.scan._merge",return_value=(large,[])),                  patch("shardelix.reconstruction.irregular.eroded.gap_candidates",return_value=([],{})),                  patch("shardelix.reconstruction.irregular.continuation.recover",return_value=(small,[edge],
                       {"attempted":True,"population_prior":rounded,"accepted_joins":1})):
                out=Path(tmp)
                diag=reconstruct(pieces,out,{"input_images":20,"warnings":[],"common_scale":1.})
                self.assertEqual(diag["placed"],expected)
                comp=json.loads((out/"component_placements.json").read_text())
                self.assertEqual(len(comp["components"][0]["members"]),expected)
                self.assertNotIn("alternative",comp)

    def test_piece_raster_preserves_disconnected_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"upload").mkdir()
            a=np.zeros((70,85,4),np.uint8)
            a[8:55,8:28]=[30,120,160,255]
            a[12:58,45:76]=[30,120,160,255]
            # Explicit pieces/ declares one fragment raster, even when the
            # surviving foreground has multiple disconnected islands.
            with zipfile.ZipFile(root/"upload"/"pieces.zip","w") as z:
                for i in range(2):
                    b=io.BytesIO();Image.fromarray(a).save(b,format="PNG")
                    z.writestr(f"pieces/p{i}.png",b.getvalue())
            state=run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True},browser=True)
            self.assertEqual(state["diagnostics"]["scan"]["detected_pieces"],2)
            report=json.loads((root/"result"/"scan_report.json").read_text())
            self.assertTrue(report["piece_rasters"])
            self.assertTrue(all(p["foreground_components"]==2 for p in report["pieces"]))
            self.assertTrue(all(p["mask_pixels"]==int((a[:,:,3]>0).sum()) for p in report["pieces"]))
            comp=json.loads((root/"result"/"component_placements.json").read_text())
            self.assertFalse(comp["accuracy_verified"])
            self.assertNotIn("alternative",comp)
            self.assertNotIn("other_model",comp)
            self.assertEqual(sum(len(g["members"]) for g in comp["components"]),2)
            self.assertIn("component_placements.json",state["files"])
            # In the rendered singleton, both disconnected islands survive.
            rendered=np.asarray(Image.open(root/"result"/"assembly.png"))
            self.assertGreaterEqual(int((rendered[:,:,3]>0).sum()),int((a[:,:,3]>0).sum())-5)
            # Overview scans remain split into separate physical silhouettes.
            p=root/"overview.png";Image.fromarray(a).save(p)
            parts,rep=preprocess([p],root/"overview")
            self.assertEqual(len(parts),2)

    def test_homogeneous_fragments_abstain(self):
        a=np.full((45,45,4),[150,120,80,255],np.uint8)
        shapes=[_shape(Fragment(str(i),a.copy()),2.) for i in range(3)]
        c,d=gap_candidates(shapes)
        self.assertEqual(c,[])
        self.assertEqual(d["reason"],"insufficient pictorial texture")


if __name__=="__main__":
    unittest.main()
