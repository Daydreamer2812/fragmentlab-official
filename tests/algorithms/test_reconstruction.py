"""Independent stdlib tests for the v2 pipeline, with no OpenCV/pytest dependency."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

from shardelix.reconstruction.common import Pose, Fragment, render, load_fragments
from shardelix.reconstruction.generator import CUTS, generate
from shardelix.reconstruction.cli import solve_case, evaluate_case
from shardelix.reconstruction.solver import solve
from shardelix.reconstruction.irregular.lattice import contours, R, _vertex_to_pose
from shardelix.reconstruction.evaluation import evaluate


def source(h=256,w=320):
    return np.random.default_rng(271).integers(0,256,(h,w,3),dtype=np.uint8)


class ReconstructionTests(unittest.TestCase):
    def test_partition_all_families(self):
        rgb=source()
        with tempfile.TemporaryDirectory() as tmp:
            for cut in CUTS:
                with self.subTest(cut=cut):
                    case=Path(tmp)/cut
                    generate(rgb,case,37,cut,71,True)
                    pieces=load_fragments(case/"pieces")
                    truth=json.loads((case/"evaluation"/"ground_truth.json").read_text())
                    poses={i:Pose(**p) for i,p in truth["poses"].items()}
                    a,labels,c=render(pieces,poses,rgb.shape[:2])
                    self.assertEqual(len(pieces),37)
                    self.assertEqual(c,0)
                    self.assertTrue(np.all(labels>=0))
                    np.testing.assert_array_equal(a[:,:,:3],rgb)

    def test_determinism_and_metadata(self):
        rgb=source()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for key in ("a","b"):
                generate(rgb,root/key,12,"random",45,True)
            a,b=root/"a",root/"b"
            self.assertEqual((a/"case.json").read_bytes(),(b/"case.json").read_bytes())
            self.assertEqual((a/"evaluation/ground_truth.json").read_bytes(),(b/"evaluation/ground_truth.json").read_bytes())
            self.assertNotIn("seed",json.loads((a/"case.json").read_text()))
            for p in (a/"pieces").glob("*.png"):
                self.assertEqual(p.read_bytes(),(b/"pieces"/p.name).read_bytes())

    def test_custom_partition(self):
        rgb=source(24,40)
        labels=np.zeros((24,40),int);labels[:,20:]=51;labels[:12,20:]=99
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp)/"case"
            meta=generate(rgb,case,3,"random",42,True,labels)
            self.assertEqual(meta["cut"],"custom")
            pieces=load_fragments(case/"pieces")
            truth={i:Pose(**p) for i,p in json.loads((case/"evaluation/ground_truth.json").read_text())["poses"].items()}
            rendered,l,c=render(pieces,truth,rgb.shape[:2])
            np.testing.assert_array_equal(rendered[:,:,:3],rgb)
            self.assertEqual(c,0)

    def test_other_modes_rejected(self):
        a=np.zeros((3,3,4),np.uint8);a[:,:,3]=255
        for mode in ("reference","template"):
            with self.assertRaises(ValueError):
                solve([Fragment("x",a)],(3,3),mode=mode)
        with self.assertRaises(ValueError):
            solve([Fragment("x",a)],(3,3),template=np.zeros((3,3),int))

    def test_truth_is_not_needed_for_blind_solver(self):
        # Jagged shapes remain an unambiguous geometric signal even for random RGB.
        rgb=source(480,640)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/"case";out=root/"out"
            generate(rgb,case,16,"jagged",42,True)
            truth=case/"evaluation";away=root/"held_truth"
            truth.rename(away)
            solve_case(case,out,"blind",None,False)
            self.assertTrue((out/"placements.json").exists())
            away.rename(truth)
            m=evaluate_case(case,out)
            self.assertGreater(m["direct_position_accuracy"],.95)
            self.assertGreater(m["adjacency_f1"],.95)
            self.assertEqual(m["collision_pixels"],0)

    def test_reference_not_allowed_in_blind(self):
        a=np.zeros((3,3,4),np.uint8);a[:,:,3]=255
        with self.assertRaises(ValueError):
            solve([Fragment("x",a)],(3,3),mode="blind",reference=a[:,:,:3])

    def test_identical_pieces_not_falsely_claimed(self):
        rgb=np.full((40,40,3),255,np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/"case";out=root/"out"
            generate(rgb,case,16,"grid",42)
            solve_case(case,out,tune=False)
            met=evaluate_case(case,out)
            self.assertEqual(met["pixel_accuracy"],1)
            self.assertFalse(met["pass_95"])
            d=json.loads((out/"diagnostics.json").read_text())
            self.assertGreater(d["identical_piece_count"],0)

    def test_1024_piece_pixel_conservation(self):
        rgb=source(256,320)
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp)/"case"
            generate(rgb,case,1024,"jagged",28,True)
            pieces=load_fragments(case/"pieces")
            gt=json.loads((case/"evaluation/ground_truth.json").read_text())
            a,l,c=render(pieces,{i:Pose(**p) for i,p in gt["poses"].items()},rgb.shape[:2])
            self.assertEqual(len(pieces),1024)
            self.assertEqual(c,0)
            self.assertTrue((l>=0).all())
            np.testing.assert_array_equal(a[:,:,:3],rgb)

    def test_pixel_vertex_rotation(self):
        a=np.zeros((3,5,4),np.uint8);a[:,:,3]=255;p=Fragment("p",a)
        for k in range(4):
            self.assertTrue(np.array_equal(R[k]@R[-k%4],np.eye(2,dtype=int)))
            pose=_vertex_to_pose(p,k,np.array([20,20]))
            rotated=p.rotated(k)
            self.assertEqual((rotated.shape[0],rotated.shape[1]),((3,5) if k%2==0 else (5,3)))
            self.assertEqual(pose.k,k)
        loops=contours(p.mask)
        self.assertEqual(len(loops),1)
        self.assertEqual(len(loops[0][1]),16)

    def test_missing_pieces_are_penalized(self):
        a=np.full((4,4,4),255,np.uint8)
        pieces=[Fragment("a",a),Fragment("b",a)]
        truth={"a":Pose(0,0),"b":Pose(4,0)}
        metrics=evaluate(pieces,{"a":Pose(0,0)},truth,(4,8))
        self.assertEqual(metrics["direct_position_accuracy"],.5)
        self.assertEqual(metrics["adjacency_f1"],0)
        self.assertEqual(metrics["coverage"],.5)
        self.assertFalse(metrics["pass_95"])


if __name__=="__main__":
    unittest.main()
