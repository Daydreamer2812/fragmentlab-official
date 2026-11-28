"""V3 straight-cut tests: exact dimensions, blind isolation, and ambiguity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shardelix.reconstruction.common import read_source
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.cli import solve_case,evaluate_case
from shardelix.reconstruction.tiles.straight import _strip_assignment


class StraightTests(unittest.TestCase):
    def test_path_exact_unequal_widths(self):
        slots=np.array([2,1,1,2,1,1,2,1,1])
        perm=np.array([7,4,6,3,8,2,0,1,5])
        widths=slots[perm]
        costs=np.full((len(slots),len(slots)),10.)
        inv=np.argsort(perm)
        costs[inv[:-1],inv[1:]]=0.01
        np.fill_diagonal(costs,np.inf)
        order,_=_strip_assignment(costs,widths,slots)
        np.testing.assert_array_equal(perm[order],np.arange(len(slots)))

    def test_texture_straight_blind_without_truth(self):
        source=read_source(Path("data/examples/straight_texture.png"),768)
        with tempfile.TemporaryDirectory() as tmp:
            for cut in ("strips","horizontal-strips","grid"):
                with self.subTest(cut=cut):
                    root=Path(tmp)/cut;case=root/"case";out=root/"result"
                    meta=generate(source,case,64,cut,293)
                    if cut=="horizontal-strips":self.assertEqual(meta["grid"],[64,1])
                    away=root/"held_truth";(case/"evaluation").rename(away)
                    solve_case(case,out,"blind",None,True)
                    away.rename(case/"evaluation")
                    r=evaluate_case(case,out)
                    self.assertTrue(r["pass_95"],r)
                    self.assertEqual(r["collision_pixels"],0)
                    self.assertFalse(r["source_reference_used"])

    def test_blank_ambiguity_no_false_success(self):
        rgb=np.full((60,90,3),255,np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/"case";out=root/"result"
            generate(rgb,case,10,"horizontal-strips",47)
            solve_case(case,out,"blind",None,False)
            d=json.loads((out/"diagnostics.json").read_text())
            m=evaluate_case(case,out)
            self.assertEqual(d["identical_piece_count"],10)
            self.assertEqual(m["pixel_accuracy"],1.)
            self.assertFalse(m["pass_95"])
            self.assertEqual(m["collision_pixels"],0)

if __name__=="__main__":unittest.main()
