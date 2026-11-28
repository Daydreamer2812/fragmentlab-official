"""Blind-only rotations, whole-canvas evaluation, and rendered cut boundaries."""
from pathlib import Path
import tempfile,json,unittest
import numpy as np
from PIL import Image
from shardelix.reconstruction.common import Fragment,Pose,read_source,load_fragments,render
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.cli import solve_case,evaluate_case
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.overlay import cut_overlay
from shardelix.reconstruction.irregular.verified import exact_holes,verify_pair

class V4BlindTests(unittest.TestCase):
    def test_cut_overlay_is_predicted_only(self):
        a=np.full((4,6,4),255,np.uint8);labels=np.zeros((4,6),int)
        labels[:,3:]=1;labels[0,5]=-1
        p,lines=cut_overlay(a,labels,scale=3)
        self.assertEqual(p.shape,(12,18,4))
        self.assertTrue((lines[:,8,3]==255).all())
        self.assertEqual(int((lines[:,:,3]>0).sum()),12)
        self.assertTrue(np.all(p[:,8]==[255,48,35,255]))
    def test_rotated_textured_strips_without_truth(self):
        rgb=read_source(Path("data/examples/straight_texture.png"),768)
        with tempfile.TemporaryDirectory() as tmp:
            for cut in ("strips","horizontal-strips"):
                with self.subTest(cut=cut):
                    root=Path(tmp)/cut;case=root/"case";out=root/"out"
                    generate(rgb,case,64,cut,731,True)
                    away=root/"held_truth";(case/"evaluation").rename(away)
                    solve_case(case,out,tune=False)
                    self.assertTrue((out/"assembly_with_cuts.png").exists())
                    self.assertTrue((out/"cut_lines.png").exists())
                    away.rename(case/"evaluation")
                    m=evaluate_case(case,out)
                    self.assertTrue(m["pass_95"],m)
    def test_canvas_gauge_does_not_allow_translation(self):
        a=np.full((4,4,4),255,np.uint8)
        pieces=[Fragment("p",a)]
        truth={"p":Pose(0,0)}
        pred={"p":Pose(1,0)}
        m=evaluate(pieces,pred,truth,(4,8),allow_global_rotation=True,allow_translation=False)
        self.assertEqual(m["direct_position_accuracy"],0.)
        self.assertEqual(m["gauge_aligned_position_accuracy"],1.)
        self.assertFalse(m["pass_95"])
    def test_hole_fill(self):
        a=np.full((4,4,4),255,np.uint8)
        whole=np.full((12,12,4),255,np.uint8);whole[4:8,4:8]=0
        pieces=[Fragment("a",whole),Fragment("b",a)]
        p,n=exact_holes(pieces,{"a":Pose(0,0)},(12,12))
        self.assertEqual(n,1);self.assertEqual(p["b"],Pose(4,4))
        _,labels,c=render(pieces,p,(12,12));self.assertTrue((labels>=0).all());self.assertEqual(c,0)
if __name__=="__main__":unittest.main()
