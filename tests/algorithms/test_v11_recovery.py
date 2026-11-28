import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from shardelix import Fragment,Pose
from shardelix.reconstruction.tiles.sparse import candidates,_links_reference
from shardelix.reconstruction.tiles.links import reciprocal_links
from shardelix.reconstruction.tiles.consensus import edge_costs
from shardelix.reconstruction.tiles.dense_recovery import exact_mgc_candidates
from shardelix.reconstruction.irregular.anchor import anchored_complete,soft_pack
from shardelix.reconstruction.strips.single_gap import solve_thin_missing
from shardelix.reconstruction.common import render
from shardelix.reconstruction.evaluation import canvas_align
from shardelix.reconstruction.tiles.metrics import grid_objective
from shardelix.reconstruction.quality import objective

def texture(h,w,seed=51):
    rng=np.random.default_rng(seed)
    return np.clip(127+420*gaussian_filter(rng.normal(size=(h,w,3)),(1.45,1.45,0)),0,255).astype(np.uint8)
def frag(id,a):
    z=np.full((*a.shape[:2],4),255,np.uint8);z[:,:,:3]=a
    return Fragment(str(id),z)

class Recovery11Tests(unittest.TestCase):
    def test_vector_cycles_exact(self):
        a=texture(96,96);pieces=[frag(i,a[y:y+12,x:x+12]) for i,(y,x) in enumerate((y,x) for y in range(0,96,12) for x in range(0,96,12))]
        idx,cost,_=candidates(pieces,(96,96),(8,8),True,shortlist=24,index_mode="midpoint")
        old,rows,margin=_links_reference(idx,cost);new=reciprocal_links(idx,cost,rows,margin)
        self.assertEqual(old,new);self.assertTrue(any(v[4]>0 for v in new))
    def test_block_exact_mgc(self):
        a=texture(48,48);pieces=[frag(i,a[y:y+12,x:x+12]) for i,(y,x) in enumerate((y,x) for y in range(0,48,12) for x in range(0,48,12))]
        raw=edge_costs(pieces,True,metric="mgc",gradient=.65,depth=2)
        idx,cost=exact_mgc_candidates(pieces,(48,48),(4,4),True,batch=7)
        np.testing.assert_allclose(cost,np.sort(raw,axis=1)[:,:5],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(cost,np.take_along_axis(raw,idx,axis=1),rtol=1e-5,atol=1e-6)
    def test_anchor_preserves_and_verifies(self):
        a=texture(24,72);p=[frag(i,a[:,i*24:(i+1)*24]) for i in range(3)]
        anchor={"0":Pose(0,0),"2":Pose(48,0)}
        result,diag=anchored_complete(p,anchor,[{"1":Pose(0,0)}],[(10,0,1,0,24,0),(10,1,2,0,24,0)],(24,72))
        self.assertEqual(result,{"0":anchor["0"],"1":Pose(24,0),"2":anchor["2"]})
        self.assertEqual(render(p,result,(24,72))[2],0)
        q,_=anchored_complete(p,anchor,[{"1":Pose(0,0)}],[(10,0,1,0,0,0)],(24,72))
        self.assertEqual(q,anchor)
    def test_unanchored_cluster_growth(self):
        a=texture(24,72);p=[frag(i,a[:,i*24:(i+1)*24]) for i in range(3)]
        q,d=anchored_complete(p,{"1":Pose(0,0),"2":Pose(24,0)},[{"0":Pose(0,0)}],[(20,0,1,0,24,0)],(24,72))
        self.assertEqual(q,{"0":Pose(0,0),"1":Pose(24,0),"2":Pose(48,0)})
        self.assertEqual(d["anchor_translation_slack"],[0,0])
    def test_soft_packing_no_core_conflict(self):
        a=texture(24,72);p=[frag(i,a[:,i*24:(i+1)*24]) for i in range(3)]
        anchor={"0":Pose(0,0),"2":Pose(48,0)}
        q,diag=soft_pack(p,anchor,[],(24,72),rounds=2)
        self.assertEqual(q,{"0":anchor["0"],"1":Pose(24,0),"2":anchor["2"]})
        self.assertEqual(render(p,q,(24,72))[2],0)
    def test_thin_gap_unique_and_abstain(self):
        a=texture(384,64,seed=60);pieces=[];truth={}
        for i in range(32):
            if i in (6,21):continue
            z=frag(i,a[:,2*i:2*i+2]);turn=i%4
            pieces.append(Fragment(z.id,z.rotated(turn).copy()));truth[z.id]=Pose(2*i,0,(-turn)%4)
        q,d=solve_thin_missing(pieces,(384,64),(1,32),True)
        self.assertEqual(d["thin_gap_status"],"resolved-single-gap-budget")
        self.assertEqual(canvas_align(pieces,q,truth,(384,64),True),truth)
        q,d=solve_thin_missing(pieces,(384,66),(1,33),True)
        self.assertFalse(q) # unknown extra missing position must not be invented
    def test_ambiguity_export_no_accuracy_claim(self):
        import tempfile,json
        from pathlib import Path
        from shardelix import ReconstructionResult
        p=[frag(i,np.full((8,8,3),190,np.uint8)) for i in range(2)]
        r=ReconstructionResult(p,{"0":Pose(0,0)},(8,16),{"identical_piece_count":2,"_unanchored_components":[{"count":1,"poses":{"1":{"cell_x":0,"cell_y":0,"k":0}}}]},(1,2),True)
        with tempfile.TemporaryDirectory() as t:
            for dest in ("a","b"):
                root=Path(t)/dest;r.save(root)
                a=json.loads((root/"ambiguity.json").read_text())
                self.assertEqual(a["exact_equivalent_piece_count"],2);self.assertFalse(a["accuracy_verified"])
                self.assertTrue((root/"unanchored_components.json").exists())
    def test_grid_objective_streaming(self):
        a=texture(12,15000);p=[frag(i,a[:,3*i:3*i+3]) for i in range(5000)]
        poses={z.id:Pose(3*i,0) for i,z in enumerate(p) if i%11!=0}
        self.assertAlmostEqual(grid_objective(p,poses,(12,15000),(1,5000)),objective(p,poses,(12,15000)),places=7)
if __name__=="__main__":unittest.main()
