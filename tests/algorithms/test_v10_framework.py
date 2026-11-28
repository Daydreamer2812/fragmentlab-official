import unittest,tempfile,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from shardelix import Solver,Fragment,Pose,ResourceLimits
from shardelix.reconstruction.generator import generate,_split
from shardelix.reconstruction.common import load_fragments,render
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.tiles.metrics import evaluate_grid,grid_objective
from shardelix.reconstruction.quality import objective
from shardelix.reconstruction.tiles.edge_index import query_edges
from shardelix.reconstruction.overlay import cut_overlay

class Framework10Tests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(10);a=(gaussian_filter(rng.normal(size=(144,144,3)),(1.4,1.4,0))*420+127).clip(0,255).astype(np.uint8)
        return a
    def test_grid_metrics_equal_pixels(self):
        src=self.fixture()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"case";generate(src,root,64,"grid",seed=17,rotations=True)
            ps=load_fragments(root/"pieces");raw=json.loads((root/"evaluation/ground_truth.json").read_text());gt={i:Pose(**v) for i,v in raw["poses"].items()}
            poses=dict(gt);poses.pop(next(iter(poses)));a,b=list(poses)[:2];poses[a],poses[b]=poses[b],poses[a]
            for rot in (False,True):
                m=evaluate(ps,poses,gt,(144,144),rot,False);q=evaluate_grid(ps,poses,gt,(144,144),(8,8),rot)
                for key in ("direct_position_accuracy","adjacency_f1","coverage","collision_pixels","pass_95"):
                    self.assertAlmostEqual(m[key],q[key])
            self.assertAlmostEqual(objective(ps,poses,(144,144)),grid_objective(ps,poses,(144,144),(8,8)),places=7)
    def test_index_repeatability_and_types(self):
        rng=np.random.default_rng(11);a=rng.random((20,16,3),dtype=np.float32);g=rng.normal(size=a.shape).astype(np.float32)*.01
        x=query_edges(a,g,a[:,::-1],g[:,::-1],k=8,workers=1)
        y=query_edges(a,g,a[:,::-1],g[:,::-1],k=8,workers=2)
        self.assertEqual(x.dtype,np.int32);self.assertTrue(np.array_equal(x,y));self.assertTrue(np.all(x[:,0]==np.arange(len(a))))
    def test_framework_fast_index_forced(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"case";generate(self.fixture(),root,64,"grid",seed=17,rotations=True)
            (root/"evaluation").rename(Path(t)/"hidden_truth")
            s=Solver(index_mode="midpoint");r=s.solve(root/"pieces",shape=(144,144),grid=(8,8),cut="grid",rotations=True)
            self.assertEqual(r.diagnostics["index_mode"],"midpoint")
            self.assertFalse(r.summary()["accuracy_verified"])
            self.assertEqual(r.render()[0].shape,(144,144,4));r.save(Path(t)/"result")
            gt={i:Pose(**v) for i,v in json.loads((Path(t)/"hidden_truth"/"ground_truth.json").read_text())["poses"].items()}
            self.assertTrue(evaluate(r.fragments,r.poses,gt,(144,144),True)["pass_95"])
    def test_reject_mismatched_uniform_cells(self):
        p=Fragment("one",np.full((4,5,4),255,np.uint8))
        with self.assertRaises(ValueError):Solver().solve([p],shape=(5,5),grid=(1,1),cut="grid")
        with self.assertRaises(ValueError):Solver(index_mode="bad").solve([p],shape=(4,5),grid=(1,1),cut="grid")
    def test_adaptive_fallback_uses_only_observable_scores(self):
        from unittest.mock import patch
        from shardelix.reconstruction.tiles.sparse import solve_indexed_grid
        class Large(list):
            def __len__(self):return 2048
        pieces=Large()
        p={"a":Pose(0,0,0)}
        q={"a":Pose(0,0,0),"b":Pose(16,0,0)}
        def diag(mode):
            return {"index_mode":mode,"shortlist_seam_fraction":.5,
                    "reciprocal_seam_fraction":.5,"retrieval_seconds":1.0,
                    "assembly_seconds":1.0}
        with patch("shardelix.reconstruction.tiles.sparse.solve_sparse_grid",side_effect=[(p,diag("midpoint")),(q,diag("legacy"))]) as call:
            with patch("shardelix.reconstruction.tiles.metrics.grid_objective",side_effect=[.9,.4]):
                poses,d=solve_indexed_grid(pieces,(512,1024),(32,64),True)
        self.assertEqual(call.call_count,2)
        self.assertEqual(poses,q);self.assertEqual(d["index_mode"],"legacy")
        self.assertTrue(d["index_fallback_used"])
    def test_resource_limits(self):
        p=Fragment("p",np.full((4,4,4),255,np.uint8))
        with self.assertRaises(ValueError):
            Solver(limits=ResourceLimits(max_canvas_pixels=15)).solve([p],shape=(4,4),cut="grid",grid=(1,1))
        with self.assertRaises(ValueError):
            Solver(limits=ResourceLimits(max_input_pixels=15)).solve([p],shape=(4,4),cut="grid",grid=(1,1))
        with self.assertRaises(ValueError):
            Solver(limits=ResourceLimits(max_fragments=1)).solve([p],shape=(4,8),cut="grid",grid=(1,2),expected_count=2)
        with self.assertRaises(ValueError):ResourceLimits(max_fragments=999999)
    def test_overlay_pixel_equivalence(self):
        lab=np.array([[1,1,2],[1,-1,2],[3,3,2]],np.int32);a=np.zeros((3,3,4),np.uint8)
        o,c=cut_overlay(a,lab)
        expected=np.array([[False,True,False],[True,False,False],[False,True,False]])
        self.assertTrue(np.array_equal(c[:,:,3]>0,expected))
    def test_generator_exact_geometry_unequal(self):
        src=np.random.default_rng(1).integers(0,256,(29,31,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"case";meta=generate(src,root,6,"grid",seed=5,rotations=True)
            ps=load_fragments(root/"pieces");gt={i:Pose(**v) for i,v in json.loads((root/"evaluation/ground_truth.json").read_text())["poses"].items()}
            image,labels,coll=render(ps,gt,src.shape[:2]);self.assertTrue(np.array_equal(image[:,:,:3],src));self.assertEqual(coll,0)
            actual=_split(src.shape[:2],6,np.random.default_rng(5),"grid")
            # Compare partitions independent of randomized label IDs.
            self.assertEqual(len(np.unique(labels)),6)
            for i in range(6):self.assertEqual(len(np.unique(labels[actual==i])),1)
if __name__=="__main__":unittest.main()
