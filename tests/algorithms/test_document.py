"""V5 document and cut-family regression tests."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from shardelix.reconstruction.common import Fragment,render
from shardelix.reconstruction.strips.ordering import chain,solve_document_strips
from shardelix.reconstruction.generator import CUTS,generate
from shardelix.reconstruction.cli import solve_external_document

class DocumentTests(unittest.TestCase):
    def test_removed_cut_families(self):
        self.assertEqual(set(CUTS),{"grid","strips","horizontal-strips","jagged","wavy","jigsaw","random"})
        with tempfile.TemporaryDirectory() as tmp:
            for cut in ("diagonal","voronoi"):
                with self.assertRaises(ValueError):
                    generate(np.zeros((40,40,3),np.uint8),Path(tmp)/cut,4,cut)
    def test_directed_margin_anchors_chain(self):
        m=np.full((6,6),5.);np.fill_diagonal(m,np.inf)
        for i in range(5):m[i,i+1]=.1
        m[5,0]=.001
        l=np.ones(6)*20;r=l.copy();l[0]=0;r[-1]=0
        np.testing.assert_array_equal(chain(m,borders=(l,r)),np.arange(6))
    def test_arbitrary_width_external_blind_render(self):
        rand=np.random.default_rng(7)
        gray=255-((gaussian_filter(rand.random((180,120)),1)>.5)*210).astype(np.uint8)
        gray[:,0]=255;gray[:,-1]=255
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        cuts=[0,19,34,52,73,94,120]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/"pieces";folder.mkdir()
            for i in rand.permutation(len(cuts)-1):
                Image.fromarray(rgb[:,cuts[i]:cuts[i+1]]).save(folder/f"{rand.bytes(10).hex()}.png")
            result=solve_external_document(folder,root/"result")
            self.assertEqual(len(result["poses"]),6)
            self.assertEqual(result["diagnostics"]["render_collision_pixels"],0)
            self.assertEqual(result["diagnostics"]["canvas_width"],120)
            self.assertFalse(result["diagnostics"]["source_reference_used"])
            for name in ("assembly.png","assembly_with_cuts.png","cut_lines.png"):
                self.assertTrue((root/"result"/name).is_file())
    def test_uniform_document_no_crash(self):
        a=np.full((40,10,4),255,np.uint8)
        ps=[Fragment(str(i),a.copy()) for i in range(4)]
        poses,diag=solve_document_strips(ps)
        self.assertEqual(len(poses),4);self.assertEqual(diag["blank_strip_count"],4)
        _,_,collision=render(ps,poses,(40,40));self.assertEqual(collision,0)

if __name__=="__main__": unittest.main()
