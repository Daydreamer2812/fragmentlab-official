import unittest,tempfile
from pathlib import Path
import numpy as np
from shardelix.reconstruction.common import load_fragments,read_source,Pose,Fragment
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.tiles.consensus import solve_consensus_grid
from shardelix.reconstruction.strips.ordering import oriented_path_milp
from shardelix.reconstruction.evaluation import evaluate
import json

class ConsensusTests(unittest.TestCase):
    def test_rotated_grid_textured_blind(self):
        src=read_source(Path("data/examples/straight_texture.png"),512)
        with tempfile.TemporaryDirectory() as t:
            c=Path(t)/"case";meta=generate(src,c,64,"grid",391,True)
            p=load_fragments(c/"pieces")
            away=Path(t)/"hidden";(c/"evaluation").rename(away)
            poses,diag=solve_consensus_grid(p,src.shape[:2],tuple(meta["grid"]),True)
            truth={i:Pose(**v) for i,v in json.loads((away/"ground_truth.json").read_text())["poses"].items()}
            m=evaluate(p,poses,truth,src.shape[:2],allow_global_rotation=True)
            self.assertTrue(m["pass_95"],m)
            self.assertEqual(m["collision_pixels"],0)
    def test_oriented_milp_chain(self):
        raw=np.full((2,2,6,6),10.)
        for q in raw:
            for m in q:np.fill_diagonal(m,np.inf)
        for i in range(5):raw[i%2,(i+1)%2,i,i+1]=.01
        l=np.full(6,20.);r=l.copy();l[0]=0;l[-1]=0
        order,turn=oriented_path_milp(raw,l,r,timeout=3)
        np.testing.assert_array_equal(order,np.arange(6))
        np.testing.assert_array_equal(turn,np.arange(6)%2)
    def test_uniform_tiles_no_false_gate(self):
        p=[Fragment(str(i),np.full((8,8,4),255,np.uint8)) for i in range(9)]
        poses,diag=solve_consensus_grid(p,(24,24),(3,3),True)
        self.assertEqual(len(poses),9);self.assertEqual(diag["identical_piece_count"],9)

if __name__=="__main__":unittest.main()
