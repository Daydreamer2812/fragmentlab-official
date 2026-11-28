"""Regression cases for count/resolution and absolute orientation in Studio."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from shardelix.reconstruction.common import Fragment, Pose, render
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.generator import fit_generation_canvas
from shardelix.reconstruction.reference.evaluation import compare_reference
from shardelix.workflow import run_job


class StudioFitOrientationTests(unittest.TestCase):
    def test_exact_counts_across_resolutions_and_cut_axes(self):
        for cut, count in (("strips", 257), ("horizontal-strips", 257), ("grid", 257),
                           ("grid", 1024)):
            for side in (128, 256, 512):
                with self.subTest(cut=cut, count=count, side=side):
                    with tempfile.TemporaryDirectory() as t:
                        root = Path(t)
                        (root / "upload").mkdir()
                        Image.new("RGB", (512, 256), (120, 30, 90)).save(root / "upload" / "test.png")
                        state = run_job(root, dict(task="generate", cut=cut, count=count,
                                                   max_side=side, angle_mode="quarter"), browser=True)
                        self.assertEqual(state["generation"]["observed"], count)
                        with zipfile.ZipFile(root / "result/fragments.zip") as z:
                            meta = json.loads(z.read("case.json"))
                            r, c = meta["grid"]
                            self.assertEqual(r*c, count)
                            self.assertLessEqual(r, meta["height"])
                            self.assertLessEqual(c, meta["width"])
                            self.assertEqual(len(z.namelist()), count+1)
                            self.assertEqual(meta["grid"], state["generation"]["grid"])
                        with Image.open(root / "result/source.png") as im:
                            self.assertEqual(im.size, (meta["width"], meta["height"]))

    def test_fit_resources_and_uniform_cells(self):
        h,w,g = fit_generation_canvas((8192,8192),4096,"grid",8_000_000,True)
        self.assertLessEqual(h*w,8_000_000)
        self.assertEqual((h%g[0],w%g[1]),(0,0))
        self.assertEqual(g[0]*g[1],4096)
        self.assertRaisesRegex(ValueError,"vertical strips",fit_generation_canvas,
                               (128,128),4096,"strips",8_000_000,True)
        for cut in ("strips","horizontal-strips","grid"):
            h,w,g=fit_generation_canvas((128,256),1025,cut,8_000_000,True)
            self.assertEqual((h%g[0],w%g[1]),(0,0))

    def test_raw_and_aligned_pose_and_image_are_distinct(self):
        # Two differently colored fragments with an upside-down blind solution.
        pieces=[Fragment("a",np.full((3,4,4),[20,80,180,255],np.uint8)),
                Fragment("b",np.full((3,4,4),[200,60,15,255],np.uint8))]
        gt={"a":Pose(0,0,0),"b":Pose(4,0,0)}
        poses={"a":Pose(4,0,2),"b":Pose(0,0,2)}
        shape=(3,8)
        metrics=evaluate(pieces,poses,gt,shape,allow_global_rotation=True)
        self.assertEqual(metrics["raw_position_accuracy"],0.)
        self.assertEqual(metrics["direct_position_accuracy"],1.)
        self.assertEqual(metrics["evaluation_rotation_cw_deg"],180)
        self.assertEqual(evaluate(pieces,poses,gt,shape)["direct_position_accuracy"],0.)
        reference,_,_=render(pieces,gt,shape)
        blind,_,_=render(pieces,poses,shape)
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);Image.fromarray(reference).save(out/"truth.png")
            Image.fromarray(blind).save(out/"assembly.png")
            Image.fromarray(blind).save(out/"assembly_with_cuts.png")
            original=(out/"assembly.png").read_bytes()
            comparison=compare_reference(out,out/"truth.png",allow_rotation=True,
                                         pose_rotation=180,pixel_cap=100)
            self.assertEqual(comparison["evaluation_rotation_cw_deg"],180)
            self.assertGreater(comparison["raw_mae"],0)
            self.assertEqual(comparison["aligned_mae"],0)
            self.assertEqual((out/"assembly.png").read_bytes(),original)
            np.testing.assert_array_equal(np.array(Image.open(out/"assembly_reference_aligned.png")),reference)
            np.testing.assert_array_equal(np.array(Image.open(out/"assembly_reference_aligned_with_cuts.png")),reference)
            raster=compare_reference(out,out/"truth.png",allow_rotation=True,pixel_cap=100)
            self.assertEqual(raster["evaluation_rotation_cw_deg"],180)

    def test_quarter_turn_and_no_mirroring(self):
        a=np.arange(4*4*4,dtype=np.uint8).reshape(4,4,4);a[:,:,3]=255
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);Image.fromarray(a).save(out/"truth.png")
            Image.fromarray(np.rot90(a,1)).save(out/"assembly.png")
            comparison=compare_reference(out,out/"truth.png",allow_rotation=True,pixel_cap=16)
            self.assertEqual(comparison["evaluation_rotation_cw_deg"],90)
            self.assertEqual(comparison["aligned_mae"],0)
            Image.fromarray(np.flip(a,axis=1)).save(out/"assembly.png")
            comparison=compare_reference(out,out/"truth.png",allow_rotation=True,pixel_cap=16)
            self.assertGreater(comparison["aligned_mae"],0)
            comparison=compare_reference(out,out/"truth.png",allow_rotation=False,pixel_cap=16)
            self.assertEqual(comparison["evaluation_rotation_cw_deg"],0)


if __name__=="__main__":
    unittest.main()
