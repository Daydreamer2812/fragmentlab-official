"""Metadata-free large torn masks, strict spatial merging, sparse gap retrieval."""
import unittest
import numpy as np

from benchmarks.benchmark_sparse_scan import trial
from benchmarks.benchmark_eroded import eroded_fixture
from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.scan import _shape,_overlap,_descriptor,_indexed_pairs,_pair,_pair_vectorized
from shardelix.reconstruction.irregular.large_scan import merge_spatial,strict_lattice_overlap,filter_continuous
from shardelix.reconstruction.irregular.eroded import gap_candidates


class SparseScanTests(unittest.TestCase):
    def test_metadata_free_over_old_piece_cap(self):
        row=trial(288,tile=44,cut="jagged",browser=True)
        self.assertEqual(row["detected"],288)
        self.assertEqual(row["placed"],288)
        self.assertEqual(row["correct_global_poses_5px_3deg"],288)
        self.assertEqual(row["collision_pixels"],0)
        self.assertEqual(row["scan"]["pair_search"],"exact-lattice")
        self.assertTrue(row["scan"]["lattice"]["selected"])

    def test_strict_merge_rejects_collision_with_other_piece(self):
        a=np.full((20,20,4),255,np.uint8)
        shapes=[_shape(Fragment(str(i),a.copy()),2.) for i in range(3)]
        info={"exact_lattice":True}
        edges=[(99,0,1,np.eye(2),np.array([20.,0]),info),
               (50,1,2,np.eye(2),np.array([-20.,0]),info)]
        groups,joins,stats=merge_spatial(shapes,edges,_overlap,strict=True)
        self.assertEqual([len(g) for g in groups],[2,1])
        self.assertEqual(len(joins),1)
        self.assertEqual(stats["collision_rejections"],1)
        self.assertTrue(strict_lattice_overlap(shapes[0],shapes[1],np.eye(2),np.array([19.,0])))
        self.assertFalse(strict_lattice_overlap(shapes[0],shapes[1],np.eye(2),np.array([20.,0])))

    def test_sparse_gap_retrieval_relative_geometry(self):
        pieces,truth=eroded_fixture()
        shapes=[_shape(p,2.) for p in pieces]
        candidates,stats=gap_candidates(shapes,sparse_degree=4)
        self.assertEqual(stats["sparse_degree"],4)
        self.assertLessEqual(stats["retrieved_pairs"],4*len(shapes))
        self.assertLessEqual(stats["evaluated"],24*len(shapes))
        self.assertTrue(all(c[1]<c[2] for c in candidates))

    def test_sparse_index_respects_shortlist_and_components(self):
        pieces,_=eroded_fixture()
        shapes=[_shape(p,2.) for p in pieces]
        scales=[5,9]
        a=[[_descriptor(s.points,k) for k in scales] for s in shapes]
        b=[[_descriptor(s.points,k,True) for k in scales] for s in shapes]
        owner=np.arange(len(shapes))//2
        pairs=_indexed_pairs(shapes,a,b,scales,keep=2,component_owner=owner)
        self.assertLessEqual(len(pairs),2*len(shapes))
        self.assertEqual(pairs,sorted(set(pairs)))
        self.assertTrue(all(i<j and owner[i]!=owner[j] for i,j in pairs))

    def test_vectorized_votes_and_continuous_confidence(self):
        pieces,truth=eroded_fixture(seed=9,erosion=2)
        shapes=[_shape(p,2.) for p in pieces]
        scales=[5,9,14,22]
        f=[[_descriptor(s.points,k) for k in scales] for s in shapes]
        r=[[_descriptor(s.points,k,True) for k in scales] for s in shapes]
        a=_pair(shapes[0],shapes[1],f[0],r[1],2.7)
        b=_pair_vectorized(shapes[0],shapes[1],f[0],r[1],2.7)
        self.assertAlmostEqual(a[0][0],b[0][0],places=5)
        candidates=[(c[0],i,j,*c[1:]) for i in range(len(shapes)) for j in range(i+1,len(shapes))
                    for c in _pair_vectorized(shapes[i],shapes[j],f[i],r[j],2.7)]
        accepted=filter_continuous(shapes,candidates)
        self.assertGreaterEqual(len(accepted),2)
        # The distractor is not supported by the observed continuity.
        self.assertTrue(all(truth[i] is not None and truth[j] is not None for _,i,j,*_ in accepted))
        correct=0
        for _,i,j,R,t,_ in accepted:
            A,u=truth[i];B,v=truth[j]
            Q=A.T@B;z=A.T@(v-u)
            error=np.linalg.norm((R-Q)@shapes[j].points.mean(0)+t-z)
            angle=abs(np.arctan2((R.T@Q)[1,0],(R.T@Q)[0,0]))
            correct+=int(error<=10 and angle<=np.deg2rad(10))
        self.assertGreaterEqual(correct,2)
