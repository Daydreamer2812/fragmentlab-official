"""Resource-aware lattice kernel parity and integer assembly regressions."""
import unittest
from unittest.mock import patch
import numpy as np

from benchmarks.stress_irregular_resources import fixture
from shardelix.native_core import arc_votes,arc_votes_available,lattice_contact
from shardelix.reconstruction.irregular.lattice import R,propose
from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.large_scan import (
    _packed_contours,_exact_contact,lattice_candidates,merge_lattice,
    merge_spatial,lattice_box)
from shardelix.reconstruction.resource_budget import resource_budget,ResourceBudget,MIB
from shardelix.reconstruction.irregular.scan import _shape,_overlap


class ResourceTests(unittest.TestCase):
    def test_adaptive_budget(self):
        with patch("shardelix.reconstruction.resource_budget.available_memory",return_value=4*1024*MIB):
            b=resource_budget()
            self.assertEqual(b.memory_bytes,int(.55*4*1024*MIB))
            self.assertLessEqual(resource_budget(override_mib=64).memory_bytes,64*MIB)
        self.assertEqual(resource_budget(browser=True).memory_bytes,256*MIB)
        with self.assertRaises(ValueError):resource_budget(override_mib=16)

    def test_native_archive_budget(self):
        from shardelix.workflow import native_upload_limit
        from collections import namedtuple
        Usage=namedtuple("Usage","total used free")
        with patch("shardelix.workflow.shutil.disk_usage",return_value=Usage(16*1024*MIB,4*1024*MIB,12*1024*MIB)):
            with patch("shardelix.reconstruction.resource_budget.available_memory",return_value=6*1024*MIB):
                self.assertEqual(native_upload_limit(),1536*MIB)
                self.assertEqual(native_upload_limit(multipart=True),512*MIB)

    @unittest.skipUnless(arc_votes_available(),"native arc kernel unavailable")
    def test_exact_arc_shards_reference(self):
        pieces,_,_=fixture(16,36,827)
        packed=_packed_contours(pieces,resource_budget(browser=True))
        for k in (14,20,32):
            truth={tuple(q) for q in propose(pieces,k,True,max_bucket=6) if q[0]>=3}
            for shards in (1,3,7):
                result=arc_votes(packed,k,6,shards,1_000_000,100_000)
                self.assertEqual(truth,{tuple(q) for q in result})
        with self.assertRaises(MemoryError):arc_votes(packed,20,6,1,1,100_000)

    @unittest.skipUnless(arc_votes_available(),"native arc kernel unavailable")
    def test_canonical_arc_bucket_and_shard_parity(self):
        # Duplicates create populous and sometimes self-complementary
        # chain-code buckets. Preserve the reference rare-bucket rule.
        pieces,_,_=fixture(12,36,2718)
        pieces += [Fragment("duplicate-"+str(i),p.rgba.copy()) for i,p in enumerate(pieces[:3])]
        packed=_packed_contours(pieces,resource_budget(browser=True))
        for k in (6,14,24,32):
            for bucket in (2,6,16):
                truth={tuple(q) for q in propose(pieces,k,True,max_bucket=bucket) if q[0]>=3}
                for shards in (1,3):
                    result=arc_votes(packed,k,bucket,shards,1_000_000,100_000)
                    self.assertEqual(truth,{tuple(q) for q in result})

    def test_native_contact_all_rotations(self):
        rng=np.random.default_rng(827)
        a=rng.integers(40,100,(32,18,4),dtype=np.uint8);a[...,3]=255
        b=rng.integers(40,100,(32,18,4),dtype=np.uint8);b[...,3]=255
        a[7:10,15:,3]=0;b[2:5,:3,3]=0
        p=Fragment("a",a)
        for r in range(4):
            q=Fragment("b",np.ascontiguousarray(np.rot90(b,r)))
            x,y,_,_=lattice_box(q,(r,0,0));t=np.array([18-x,-y])
            if lattice_contact(p.rgba,q.rgba,r,t) is None:self.skipTest("native contact kernel unavailable")
            actual=_exact_contact(p,q,r,t,16)
            with patch("shardelix.native_core.lattice_contact",return_value=None):
                expected=_exact_contact(p,q,r,t,16)
            self.assertIsNotNone(expected)
            self.assertAlmostEqual(actual[0],expected[0],places=10)
            self.assertAlmostEqual(actual[1]["color_difference"],expected[1]["color_difference"])
            self.assertIsNone(_exact_contact(p,q,r,t-np.array([3,0]),16))

    def test_integer_component_merge_parity(self):
        pieces,_,_=fixture(25,36,827)
        candidates,_=lattice_candidates(pieces)
        shapes=[_shape(p,2.) for p in pieces]
        old,oj,_=merge_spatial(shapes,candidates,_overlap,strict=True)
        new,nj,stats=merge_lattice(pieces,candidates)
        self.assertEqual(oj,nj);self.assertEqual(len(new[0]),25)
        for g,h in zip(old,new):
            self.assertEqual(g.keys(),h.keys())
            for i in g:
                np.testing.assert_array_equal(g[i][0],h[i][0]);np.testing.assert_array_equal(g[i][1],h[i][1])
        self.assertEqual(stats["collision_rejections"],0)
        groups,joins,stats=merge_lattice(pieces,candidates,pixel_cap=60**2)
        self.assertGreater(stats["canvas_budget_rejections"],0)
        self.assertGreater(len(groups),1)
