"""Regression gates for damage-aware pose, sparse scans and honest seam evidence."""
from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from shardelix.reconstruction.common import Fragment, Pose, save_rgba
from shardelix.reconstruction.irregular.component_graph import merge_supported_components
from shardelix.reconstruction.irregular.offset_contours import refine_offset
from shardelix.reconstruction.strips.physical import (
    _rectify_rows, _textlike, _ink, chain_evidence, match_rectified_strips,
)


class ReleaseQualityTests(unittest.TestCase):
    def test_sparse_neutral_document_vs_blank_and_chromatic(self):
        a=np.full((500,40,4),255,np.uint8)
        a[60:65,2:30,:3]=0  # less than 1.2% dark area
        pieces=[Fragment(str(i),a.copy()) for i in range(8)]
        self.assertTrue(_textlike(pieces,[_ink(p) for p in pieces]))
        a[:,:,:3]=255
        pieces=[Fragment(str(i),a.copy()) for i in range(8)]
        self.assertFalse(_textlike(pieces,[_ink(p) for p in pieces]))
        a[:,:,:3]=[220,20,50]
        pieces=[Fragment(str(i),a.copy()) for i in range(8)]
        self.assertFalse(_textlike(pieces,[_ink(p) for p in pieces]))

    def test_rectification_exact_for_observed_rows_and_hidden_rgb_invariance(self):
        rng=np.random.default_rng(124)
        a=rng.integers(0,256,(28,36,4),dtype=np.uint8);a[...,3]=255
        left=rng.integers(0,5,len(a)).astype(float);right=rng.integers(30,36,len(a)).astype(float)
        x=left[:,None]+(right-left)[:,None]*np.linspace(0,1,24)
        y=np.broadcast_to(np.arange(len(a))[:,None],x.shape)
        expected=np.stack([map_coordinates(a[:,:,k],(y,x),order=1,prefilter=False)
                           for k in range(4)],axis=-1)
        np.testing.assert_array_equal(_rectify_rows(a,left,right,24),expected)
        a[8:11,12:15,3]=0
        b=a.copy();b[b[...,3]==0,:3]=[255,0,255]
        np.testing.assert_array_equal(_rectify_rows(a,left,right,24),_rectify_rows(b,left,right,24))

    def test_seam_evidence_ties_and_two_piece_ambiguity(self):
        c=np.full((3,3),1.);np.fill_diagonal(c,np.inf)
        self.assertFalse(any(v['distinctive'] for v in chain_evidence(c,np.arange(3))))
        c[0,1]=.1;c[1,2]=.1
        evidence=chain_evidence(c,np.arange(3))
        self.assertTrue(all(v['distinctive'] for v in evidence))
        self.assertEqual(evidence[0]['outgoing_rank'],1)
        two=chain_evidence(np.array([[np.inf,1],[1,np.inf]]),np.arange(2))
        self.assertIsNone(two[0]['normalized_rival_margin'])
        self.assertFalse(two[0]['distinctive'])

    def test_component_graph_two_distinct_anchors_and_collision_gate(self):
        shapes=[SimpleNamespace(points=np.array([[0.,0.],[1.,0.],[1.,1.]]),
                                fragment=SimpleNamespace(id=str(i))) for i in range(4)]
        R=np.eye(2);zero=np.zeros(2)
        groups=[{0:(R,zero),1:(R,np.array([0.,10.]))},
                {2:(R,zero),3:(R,np.array([0.,10.]))}]
        seeds=[(12.,0,2,R,np.array([10.,0.]),{}),(11.,1,3,R,np.array([10.,0.]),{})]
        g,j,d=merge_supported_components(shapes,groups,[],seeds,lambda *args:0.)
        self.assertEqual(len(g),1);self.assertEqual(len(j),2)
        self.assertEqual(d['component_graph_merges'],1)
        np.testing.assert_array_equal(g[0][3][1],[10.,10.])
        self.assertEqual(len(groups),2)
        for invalid,overlap in ((seeds[:1],0.),([seeds[0],seeds[0]],0.),
            ([seeds[0],(11.,1,3,R,np.array([25.,0.]),{})],0.),(seeds,.1)):
            g,j,d=merge_supported_components(shapes,groups,[],invalid,lambda *args:overlap)
            self.assertEqual(len(g),2);self.assertFalse(j)

    def test_refine_offset_only_accepts_better_observed_objective(self):
        p=SimpleNamespace(points=np.array([[0.,0.],[0.,2.],[2.,0.]]))
        R=np.eye(2);t=np.zeros(2)
        def score(a,b,fa,fb,Q,u,limit,info):
            return (10.-np.linalg.norm(u-[2.,0.]),{})
        with patch('shardelix.reconstruction.irregular.offset_contours.verify_offset',side_effect=score), \
             patch('shardelix.reconstruction.irregular.offset_contours._overlap',return_value=0.):
            s,Q,u,_=refine_offset(p,p,None,None,R,t,12.,{},(8.,{}))
        self.assertGreater(s,8.)
        np.testing.assert_allclose(u,[2.,0.])

    def test_corrected_row_registration_only_for_strong_consensus(self):
        a=np.full((200,20,4),255,np.uint8);a[20:25,3:15,:3]=0
        pieces=[Fragment(str(i),a.copy()) for i in range(8)]
        shifts=[-4,-3,0,1,0,2,3,4]
        for consensus in (.4,.8):
            diag={'vertical_registration':{'applied':True,'row_consensus':consensus,
                    'offsets_matching_px':shifts},'matching_scale':1.0,'seam_evidence':[]}
            with patch('shardelix.reconstruction.strips.physical._textlike',return_value=True), \
                patch('shardelix.reconstruction.strips.physical._stroke_order',return_value=(np.arange(8),diag)):
                poses,matching,_=match_rectified_strips(pieces)
            self.assertEqual(matching['vertical_registration']['output_alignment_applied'],consensus>.65)
            self.assertEqual([poses[str(i)].y for i in range(8)],
                             [d-min(shifts) for d in shifts] if consensus>.65 else [0]*8)

    def test_fast_png_is_lossless(self):
        a=np.random.default_rng(77).integers(0,256,(32,64,4),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fragment.png';save_rgba(p,a,compress_level=1)
            with Image.open(p) as im:np.testing.assert_array_equal(np.asarray(im),a)


if __name__ == '__main__':
    unittest.main()
