"""Unified file API routes without leaking optional evaluation inputs."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from shardelix import reconstruct_files
from shardelix.workflow import run_job

class UnifiedFrameworkTests(unittest.TestCase):
    def test_api_automatic_grid_and_evaluation_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pieces=root/"pixels";pieces.mkdir()
            rng=np.random.default_rng(42)
            rgb=gaussian_filter(rng.uniform(0,255,(32,32,3)),(1,1,0)).astype(np.uint8)
            source=root/"source.png";Image.fromarray(rgb).save(source)
            for i in range(4):
                y,x=divmod(i,2);Image.fromarray(rgb[y*16:(y+1)*16,x*16:(x+1)*16]).save(pieces/f"{i}.png")
            result=reconstruct_files(pieces,root/"blind",ground_truth=source)
            self.assertFalse(result["diagnostics"].get("matching_reference_used",False))
            self.assertEqual(result["diagnostics"]["placed"],4)
            self.assertTrue((root/"blind"/"assembly.png").exists())
            self.assertIsNotNone(result["comparison"])
            with self.assertRaisesRegex(ValueError,"new or empty"):
                reconstruct_files(pieces,root/"blind")
    def test_api_opt_in_reference_and_shared_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pixels=root/"fragment.png";source=root/"original.png"
            rng=np.random.default_rng(34)
            rgb=np.clip(gaussian_filter(rng.uniform(0,255,(124,136,3)),(.9,.9,0))*2-127,0,255).astype(np.uint8)
            Image.fromarray(rgb).save(source)
            yy,xx=np.indices((40,38));mask=((xx-18)/18)**2+((yy-19)/19)**2<1
            Image.fromarray(np.dstack([rgb[42:82,51:89],mask.astype(np.uint8)*255])).save(pixels)
            for profile,engine in (("balanced","indexed"),("accurate","indexed-deep")):
                result=reconstruct_files(pixels,root/profile,matching_reference=source,profile=profile)
                self.assertTrue(result["diagnostics"]["matching_reference_used"])
                self.assertEqual(result["diagnostics"]["input_detection"]["engine"],engine)
                self.assertEqual(result["diagnostics"]["performance_profile"],profile)
                self.assertIsNone(result["metrics"])
                self.assertEqual(result["diagnostics"]["placed"],1)
            with self.assertRaisesRegex(ValueError,"do not mix"):
                reconstruct_files(pixels,root/"mixed",matching_reference=source,ground_truth=source)
