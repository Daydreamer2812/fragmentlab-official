"""Pixel-only strip reconstruction and export in native and browser modes."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from shardelix.workflow import run_job, inspect_uploads


class PhysicalDocumentWorkflowTests(unittest.TestCase):
    def test_pixel_only_scan_native_and_browser(self):
        poses = []
        for browser in (False, True):
            with self.subTest(browser=browser), tempfile.TemporaryDirectory() as t:
                root = Path(t)
                (root/"upload").mkdir()
                for i in range(8):
                    a = np.full((320, 48, 4), 255, np.uint8)
                    for y in range(15, 300, 23):
                        a[y:y+5, 3+i:24+i, :3] = 0
                    Image.fromarray(a).save(root/"upload"/f"p{i}.png")
                config = {"task": "blind", "input_mode": "scan"}
                # Old clients may still send this flag. It must have no effect.
                if not browser:
                    config["ocr_refine"] = True
                with patch("shutil.which", side_effect=AssertionError("External program lookup")):
                    result = run_job(root, config, browser=browser)
                d = result["diagnostics"]
                self.assertEqual(d["algorithm"], "physical-document-foreground-multiscale")
                self.assertEqual(d["scan"]["kind"], "document-strips")
                self.assertEqual(d["placed"], 8)
                matching = d["scan"]["matching"]
                self.assertTrue(matching["patch_recurrence"])
                self.assertTrue(matching["local_shadow_normalization"])
                self.assertIn("global_path", matching)
                self.assertIn("vertical_registration", matching)
                self.assertEqual(d["render_collision_pixels"], 0)
                self.assertFalse(d["accuracy_verified"])
                self.assertNotIn("ocr", d["scan"])
                self.assertIsNone(result["metrics"])
                self.assertIn("strip_components.json", result["files"])
                with zipfile.ZipFile(root/"result/result.zip") as z:
                    archived = json.loads(z.read("diagnostics.json"))
                    self.assertEqual(archived["algorithm"], d["algorithm"])
                    self.assertTrue(archived["orientation_ambiguous"])
                    self.assertNotIn("ocr", archived["scan"])
                    placements = json.loads(z.read("placements.json"))["poses"]
                    components = json.loads(z.read("strip_components.json"))
                    self.assertFalse(components["accuracy_verified"])
                    members = [member for group in components["components"] for member in group["members"]]
                    self.assertCountEqual(members, placements.keys())
                    poses.append(placements)
        self.assertEqual(poses[0], poses[1])

    def test_color_strips_with_transparent_padding_auto_route(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            y,x=np.indices((128,56))
            source=np.stack([120+80*np.sin(.13*x+.03*y),140+65*np.cos(.21*x-.06*y),
                             120+70*np.sin(.08*x-.09*y)],axis=2).astype(np.uint8)
            for i in (3,1,0,2):
                tile=np.zeros((150,100,4),np.uint8)
                tile[11:139,40:54,:3]=source[:,14*i:14*(i+1)]
                tile[11:139,40:54,3]=255
                if i%2:tile=np.rot90(tile)
                Image.fromarray(tile).save(root/"upload"/f"p{(i*7)%11}.png")
            info=inspect_uploads(sorted((root/"upload").iterdir()),True)
            self.assertEqual(info["route"],"scan-strips")
            state=run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True},browser=True)
            self.assertEqual(state["diagnostics"]["algorithm"],"physical-color-strip-chain")
            self.assertEqual(state["diagnostics"]["placed"],4)
            self.assertEqual(state["diagnostics"]["render_collision_pixels"],0)
            with Image.open(root/"result/assembly.png") as im:
                self.assertEqual(im.size,(56,128))

    def test_patch_similarity_is_bounded_and_finite(self):
        from shardelix.reconstruction.strips.physical import _patch_recurrence
        images=[]
        for k in range(4):
            a=np.zeros((92,26),np.float32)
            for y in range(9,83,13):
                a[y:y+3,3:22]=1
                a[y:y+7,7+k:9+k]=1
            images.append(a)
        c=_patch_recurrence(images,6,0)
        self.assertEqual(c.shape,(4,4))
        self.assertTrue(np.isfinite(c).all())
        self.assertGreaterEqual(float(c.min()),0)
        np.testing.assert_allclose(c,_patch_recurrence(images,6,0))
        # Dense calibrated patches compare the actual pixel centers with
        # an explicit missing-cut width, not a virtual rendered bridge.
        for gap in (2,3):
            d=_patch_recurrence(images,6,gap,inset=1)
            self.assertEqual(d.shape,(4,4))
            self.assertTrue(np.isfinite(d).all())
            self.assertGreaterEqual(float(d.min()),0)
            np.testing.assert_allclose(d,_patch_recurrence(images,6,gap,inset=1))

    def test_cut_hypothesis_uses_observed_reciprocal_evidence(self):
        from shardelix.reconstruction.strips.document_patches import (
            _chain_distinctiveness, select_cut_hypothesis,
        )
        # No filenames, ground truth or original page enters the selector.
        rng=np.random.default_rng(18)
        baseline=rng.uniform(.8,1.2,(8,8))
        strong=np.ones((8,8))
        target=np.array([3,1,6,4,0,5,7,2])
        strong[target[:-1],target[1:]]=.1
        borders=(np.zeros(8),np.zeros(8))
        with patch("shardelix.reconstruction.strips.document_patches._patch_recurrence",
                   side_effect=[strong,None]):
            c,order,diag=select_cut_hypothesis([None]*8,baseline,borders,lambda _:None)
        np.testing.assert_array_equal(order,target)
        self.assertLess(_chain_distinctiveness(c,order),0)
        self.assertEqual(diag["selected"],"gap-2-weight-1")
        self.assertEqual(len(diag["candidates"]),4)
        # Unusable patches leave the prior matcher available.
        with patch("shardelix.reconstruction.strips.document_patches._patch_recurrence",
                   return_value=None):
            c,order,diag=select_cut_hypothesis([None]*8,baseline,borders,lambda _:None)
        self.assertEqual(diag["selected"],"coarse")
        self.assertEqual(sorted(order.tolist()),list(range(8)))


    def test_local_shadow_normalization_preserves_ink(self):
        from shardelix.reconstruction.strips.physical import _local_ink
        y,x=np.indices((450,32))
        paper=.15+.07*np.sin(y/38)+.12*np.exp(-x/4)
        ink=np.zeros_like(paper,dtype=bool)
        ink[45:48,5:27]=True
        ink[117:120,3:25]=True
        ink[203:208,9:26]=True
        evidence=_local_ink(np.where(ink,.94,paper).astype(np.float32))
        self.assertGreater(float(evidence[ink].mean()),.7)
        self.assertLess(float(evidence[~ink].mean()),.03)
        self.assertTrue(np.isfinite(evidence).all())

    def test_text_row_registration_and_abstention(self):
        from shardelix.reconstruction.strips.physical import _align_text_rows, _shift_rows
        a=np.zeros((420,24),np.float32)
        for y in (23,61,96,155,187,229,286,327,378):
            a[y:y+5,3:21]=1
        expected=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5,0])
        images=[_shift_rows(a,-int(d)) for d in expected]
        offsets,d=_align_text_rows(images)
        np.testing.assert_array_equal(offsets,expected)
        self.assertTrue(d["applied"])
        offsets,d=_align_text_rows([a]*len(images))
        self.assertFalse(d["applied"])
        self.assertFalse(offsets.any())

    def test_high_resolution_matching_preserves_horizontal_detail_when_requested(self):
        from shardelix.reconstruction.strips.physical import _matching_scale
        im=np.random.default_rng(83).random((200,41),dtype=np.float32)
        self.assertIs(_matching_scale(im,200,1.0),im)
        by_height=_matching_scale(im,100,1.0)
        isotropic=_matching_scale(im,100,.5)
        self.assertEqual(by_height.shape,(100,41))
        self.assertEqual(isotropic.shape,(100,20))
        np.testing.assert_allclose(by_height[0],im[0])
        self.assertTrue(np.isfinite(isotropic).all())

    def test_rectify_long_tapered_scan_without_accepting_holes(self):
        from shardelix.reconstruction.strips.physical import _rectify, NotDocumentStrips
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"strip.png"
            a=np.full((700,55,4),255,np.uint8)
            a[:,:,3]=0
            for y in range(700):
                a[y,:int(round(28+24*y/699)),3]=255
            Image.fromarray(a).save(path)
            piece,record=_rectify(path)
            self.assertEqual(piece.rgba.shape,(700,40,4))
            self.assertTrue(record["masked"])
            a[100:600,8:18,3]=0
            Image.fromarray(a).save(path)
            with self.assertRaises(NotDocumentStrips):
                _rectify(path)

    def test_patch_phase_and_ink_weighted_residuals(self):
        from shardelix.reconstruction.strips.document_patches import (
            _patch_pair, _phase_cost, _weighted_patch_pair, _continuation_alternatives,
        )
        rng=np.random.default_rng(38)
        R=rng.random((3,72,6),dtype=np.float32)
        L=rng.random(R.shape,dtype=np.float32)
        L[1,1:]=R[0,:-1]
        PL=R.copy();PR=L.copy()
        raw=.5*(_patch_pair(PL,L)+_patch_pair(R,PR))
        result=_phase_cost(R,L,PL,PR,raw,1.)
        self.assertLess(result[0,1],.02)
        self.assertGreater(raw[0,1],result[0,1])
        scores=_continuation_alternatives(R,L,PL,PR,raw)
        self.assertEqual(set(scores),{"phase","ink-weighted","band-phase"})
        for c in scores.values():
            self.assertEqual(c.shape,(3,3))
            self.assertTrue(np.isfinite(c).all())
            self.assertGreaterEqual(float(c.min()),0)
        W=.15+.85*R[0].max(1)
        expected=(W[:,None]*(PL[0]-L[1])**2).sum()/(
            (W[:,None]*(PL[0]**2+L[1]**2)).sum()+1)
        self.assertAlmostEqual(float(_weighted_patch_pair(PL,L,R)[0,1]),float(expected),places=5)

    def test_localized_rim_requires_dominant_width_and_contiguous_mask(self):
        from shardelix.reconstruction.strips.physical import _rectify, NotDocumentStrips
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"rim.png"
            a=np.full((1000,60,4),255,np.uint8)
            a[:,:,3]=0;a[:,:40,3]=255
            a[400:550,40:60,3]=255
            Image.fromarray(a).save(path)
            piece,record=_rectify(path)
            self.assertEqual(piece.rgba.shape,(1000,40,4))
            self.assertTrue(record["localized_rim"])
            a[200:700,9:19,3]=0
            Image.fromarray(a).save(path)
            with self.assertRaises(NotDocumentStrips):_rectify(path)
            a[:,:,3]=0;a[:,:40,3]=255
            a[200:550,40:60,3]=255
            Image.fromarray(a).save(path)
            with self.assertRaises(NotDocumentStrips):_rectify(path)

    def test_alternative_hypothesis_retains_baseline_or_selects_stronger_evidence(self):
        from shardelix.reconstruction.strips.document_patches import select_cut_hypothesis
        rng=np.random.default_rng(24)
        base=rng.uniform(.8,1.2,(8,8));strong=np.ones((8,8))
        target=np.array([2,5,1,4,0,7,3,6])
        strong[target[:-1],target[1:]]=.05
        borders=(np.zeros(8),np.zeros(8))
        with patch("shardelix.reconstruction.strips.document_patches._patch_recurrence",return_value=None):
            _,order,d=select_cut_hypothesis([None]*8,base,borders,lambda _:None,
                                            alternatives={"phase":strong})
        np.testing.assert_array_equal(order,target)
        self.assertEqual(d["selected"],"phase")
        with patch("shardelix.reconstruction.strips.document_patches._patch_recurrence",return_value=None):
            _,_,d=select_cut_hypothesis([None]*8,base,borders,lambda _:None,
                                       alternatives={"identical":base.copy()})
        self.assertEqual(d["selected"],"coarse")
        # A calibrated gap winner must remain available; coarse residuals
        # evaluate a different physical gap and cannot replace it.
        with patch("shardelix.reconstruction.strips.document_patches._patch_recurrence",
                   side_effect=[strong,None]):
            _,_,d=select_cut_hypothesis([None]*8,base,borders,lambda _:None,
                                        alternatives={"phase":strong*.001})
        self.assertTrue(d["selected"].startswith("gap-2-"))
        self.assertNotIn("phase",[c["hypothesis"] for c in d["candidates"]])

    def test_global_chain_matches_small_exhaustive_oracle(self):
        import itertools
        from shardelix.reconstruction.strips.ordering import (
            _chain_energy, _priority, refine_chain_global,
        )
        costs=np.random.default_rng(7).uniform(.05,2,(6,6))
        np.fill_diagonal(costs,np.inf)
        initial=np.array([3,5,2,1,0,4])
        priority=np.zeros((7,7))
        priority[:6,:6]=_priority(costs,.5)
        def energy(order):
            return _chain_energy(priority,np.asarray(order))
        oracle=min(energy(o) for o in itertools.permutations(range(6)))
        result,diag=refine_chain_global(costs,initial,seconds=5)
        self.assertEqual(sorted(result.tolist()),list(range(6)))
        self.assertAlmostEqual(energy(result),oracle)
        self.assertTrue(diag["accepted"])
        fallback,diag=refine_chain_global(costs,initial,seconds=0)
        np.testing.assert_array_equal(fallback,initial)
        self.assertFalse(diag["accepted"])

    def test_global_refinement_with_borders_and_forbidden_arcs(self):
        import itertools
        from shardelix.reconstruction.strips.ordering import (
            _chain_energy, _priority, refine_chain_global,
        )
        for seed in (8,14,62):
            rng=np.random.default_rng(seed)
            costs=rng.uniform(.05,2,(6,6))
            np.fill_diagonal(costs,np.inf)
            costs[5,2]=costs[2,5]=np.inf
            borders=tuple(rng.uniform(0,1,6) for _ in range(2))
            initial=np.arange(6)
            priority=np.zeros((7,7))
            priority[:6,:6]=_priority(costs,.5)
            priority[6,:6],priority[:6,6]=borders
            result,diag=refine_chain_global(costs,initial,borders,top_k=6,seconds=5)
            oracle=min(_chain_energy(priority,np.asarray(o)) for o in itertools.permutations(range(6)))
            self.assertAlmostEqual(_chain_energy(priority,result),oracle)
            self.assertEqual(diag["status"],"shortlist-optimal")
            self.assertLessEqual(diag["selected_energy"],diag["initial_energy"])


if __name__ == "__main__":
    unittest.main()
