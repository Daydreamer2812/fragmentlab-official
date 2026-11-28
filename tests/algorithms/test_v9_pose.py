import json,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from shardelix import Solver,Fragment,Pose
from shardelix.reconstruction.generator import generate,_split
from shardelix.reconstruction.rectification import _rectify,rotate_raster
from shardelix.reconstruction.cli import solve_case,evaluate_case
from shardelix.reconstruction.irregular.tolerant import repair_boundary_ownership
from shardelix.reconstruction.common import render,load_fragments

class Pose9Tests(unittest.TestCase):
    def test_continuous_grid_missing_blind(self):
        rng=np.random.default_rng(831)
        src=(gaussian_filter(rng.normal(size=(192,192,3)),(1.4,1.4,0))*420+127).clip(0,255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);case=root/"case";out=root/"result"
            m=generate(src,case,64,"grid",202,angle_mode="continuous",missing=.1)
            self.assertEqual(m["piece_count"],58);self.assertEqual(m["expected_count"],64)
            (case/"evaluation").rename(root/"held")
            r=solve_case(case,out)
            self.assertNotIn("poses",m);self.assertFalse(r["diagnostics"]["accuracy_verified"])
            self.assertTrue((out/"assembly_with_cuts.png").is_file())
            self.assertEqual(len(r["input_to_canvas_affine"]),r["diagnostics"]["placed"])
            (root/"held").rename(case/"evaluation")
            e=evaluate_case(case,out)
            self.assertTrue(e["pass_95"],e)
            self.assertLess(e["coverage"],.95)
            self.assertEqual(e["available_piece_recovery"],1)
    def test_irregular_mask_normalization(self):
        rng=np.random.default_rng(95);L=_split((128,128),16,rng,"random")
        for s in (1,2):
            acc=[]
            for i in range(8):
                y,x=np.nonzero(L==i);a=np.zeros((np.ptp(y)+1,np.ptp(x)+1,4),np.uint8)
                a[y-y.min(),x-x.min()]=255
                angle=float(rng.uniform(-180,180));b=rotate_raster(a,angle,s)
                yy,xx=np.nonzero(b[:,:,3]);b=b[yy.min():yy.max()+1,xx.min():xx.max()+1]
                v,r,A=_rectify(b,None,s,True)
                self.assertLess(abs((r+angle+45)%90-45),1.)
                self.assertEqual(A.shape,(2,3))
                k=int(round((r+angle)/90))%4;mask=np.rot90(v,-k)[:,:,3]
                if mask.shape==a.shape[:2]:acc.append((mask==a[:,:,3]).mean())
                else:acc.append(0.)
            self.assertGreater(np.mean(acc),.85 if s==1 else .98)
    def test_repair_reports_overlap(self):
        a=np.full((6,6,4),255,np.uint8);ps=[Fragment("a",a),Fragment("b",a.copy())]
        poses={"a":Pose(0,0),"b":Pose(5,0)}
        fixed,raw,core=repair_boundary_ownership(ps,poses,(6,11))
        self.assertEqual(raw,6);self.assertEqual(core,0)
        self.assertEqual(render(fixed,poses,(6,11))[2],0)
        self.assertTrue(np.all(ps[1].rgba==255))
    def test_validation(self):
        ps=[Fragment("a",np.full((4,4,4),255,np.uint8))]
        for kwargs in (dict(expected_count=0),dict(angle_mode="bogus"),dict(sampling=2),dict(expected_count=True)):
            with self.assertRaises(ValueError):
                Solver().solve(ps,shape=(4,4),cut="grid",grid=(1,1),**kwargs)
        with self.assertRaises(ValueError):
            Solver(strategy="classic").solve(ps,shape=(4,4),cut="grid",grid=(1,1),angle_mode="continuous")

if __name__=="__main__":unittest.main()
