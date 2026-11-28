"""Regression tests for reference-free structure continuation and spectral SE(2)."""
import unittest
import math
import numpy as np
from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.scan import _shape,rotation
from shardelix.reconstruction.irregular.continuation import make_field,pose_difference,population_weights,confidence_forest,_closed_triangles,bridge_evidence,recover,independent_consensus
from shardelix.reconstruction.irregular.spectral_gap import fft_proposals,spectral_workspace
from benchmarks.benchmark_eroded import eroded_fixture


class ClassicalContinuationTests(unittest.TestCase):
    def test_fft_finds_rotated_damaged_pair(self):
        pieces,truth=eroded_fixture(seed=3,erosion=5)
        # Matching sees exactly two observed rasters, not their true poses.
        shapes=[_shape(p,2.) for p in pieces[:2]]
        fields=[make_field(s,26.) for s in shapes]
        hypotheses=fft_proposals(shapes,fields,[(0,1)],26.)[(0,1)]
        A,a=truth[0];B,b=truth[1]
        Q,u=A.T@B,A.T@(b-a)
        errors=[pose_difference(shapes,1,rotation(angle),t,Q,u) for angle,t,_,_ in hypotheses]
        self.assertTrue(any(d<10. and da<math.radians(10.) for d,da in errors))
        self.assertTrue(all(np.isfinite(f.rgb).all() for f in fields))

    def test_offset_recovers_eroded_seams_without_rendering_expanded_pixels(self):
        from shardelix.reconstruction.irregular.offset_contours import offset_recovery
        from shardelix.reconstruction.resource_budget import ResourceBudget
        pieces,truth=eroded_fixture(seed=3,erosion=8,keep=(0,1,2))
        original=[p.rgba.copy() for p in pieces]
        shapes=[_shape(p,2.) for p in pieces]
        fields=[make_field(s,36.) for s in shapes]
        candidates,diag=offset_recovery(shapes,fields,36.,ResourceBudget(256*1024**2,"test",None))
        self.assertTrue(diag["attempted"])
        self.assertGreaterEqual(len(candidates),2)
        for _,i,j,R,t,info in candidates:
            self.assertIsNotNone(truth[i]);self.assertIsNotNone(truth[j])
            A,a=truth[i];B,b=truth[j]
            distance,angle=pose_difference(shapes,j,R,t,A.T@B,A.T@(b-a))
            self.assertLess(distance,10.)
            self.assertLess(angle,math.radians(10.))
            self.assertGreater(info["median_gap"],2.)
        for p,raster in zip(pieces,original):
            np.testing.assert_array_equal(p.rgba,raster)

    def test_offset_search_bounds_pair_and_memory_work(self):
        from shardelix.reconstruction.irregular.offset_contours import offset_recovery
        from shardelix.reconstruction.resource_budget import ResourceBudget
        pieces,_=eroded_fixture(seed=3,erosion=5,keep=(0,1,2))
        shapes=[_shape(p,2.) for p in pieces]
        result,diag=offset_recovery(shapes*25,[],36.,ResourceBudget(256*1024**2,"test",None))
        self.assertFalse(result)
        self.assertFalse(diag["attempted"])
        self.assertEqual(diag["reason"],"offset search budget")

    def test_rounded_seed_bridge_rejects_opposite_contrast(self):
        from shardelix.reconstruction.irregular.continuation import verify_rounded_seeds
        y,x=np.indices((60,45))
        rgb=np.repeat((128+70*np.sin(y/9.))[...,None],3,axis=2).astype(np.uint8)
        alpha=np.full((60,45,1),255,np.uint8)
        shapes=[_shape(Fragment(str(i),np.concatenate([a,alpha],axis=2)),2.)
                for i,a in enumerate((rgb,rgb,255-rgb))]
        fields=[make_field(s,20.) for s in shapes]
        R=np.eye(2);t=np.array([53.,0.])
        candidates=[(10.,0,1,R,t,{"line_count":0}),(10.,0,2,R,t,{"line_count":0})]
        selected=verify_rounded_seeds(shapes,fields,candidates,20.)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0][1:3],(0,1))

    def test_fft_workspace_counts_unique_nominations(self):
        pieces,_=eroded_fixture(seed=3,erosion=3,keep=(0,1,2))
        shapes=[_shape(p,2.) for p in pieces[:3]]
        self.assertEqual(spectral_workspace(shapes,[],26.),0)
        one=spectral_workspace(shapes,[(0,1)],26.)
        self.assertGreater(one,0)
        self.assertEqual(one,spectral_workspace(shapes,[(0,1),(1,0)],26.))
        self.assertGreater(spectral_workspace(shapes,[(0,1),(0,2),(1,2)],26.),one)

    def test_population_prior_is_soft_and_permutation_equivariant(self):
        y,x=np.indices((68,68))
        circle=(x-33.)**2+(y-33.)**2<=28.**2
        triangle=(x>4)&(y>4)&(x+y<62)
        def frag(i,m):
            return Fragment(str(i), np.dstack([np.full((*m.shape,3),120,np.uint8),m.astype(np.uint8)*255]))
        shapes=[_shape(frag(i,circle),2.) for i in range(22)]+[_shape(frag(22,triangle),2.)]
        w,enabled=population_weights(shapes)
        self.assertTrue(enabled)
        self.assertGreater(w[-1],0.)
        self.assertLess(w[-1],.5)
        q,_=population_weights(shapes[::-1])
        np.testing.assert_allclose(q,w[::-1])
        self.assertTrue(np.all(w[:-1]==1.))

    def test_closed_triangle_requires_pose_consistency_and_distinct_arcs(self):
        pieces,truth=eroded_fixture(seed=3,erosion=3,keep=(0,1,2))
        shapes=[_shape(p,2.) for p in pieces[:3]]
        def mode(i,j,a,b):
            A,u=truth[i];B,v=truth[j]
            return (9.,i,j,A.T@B,A.T@(v-u),{"arc_a":a,"arc_b":b,"population_weight":1.})
        m=[mode(0,1,0,0),mode(0,2,len(shapes[0].points)//2,0),
           mode(1,2,len(shapes[1].points)//2,len(shapes[2].points)//2)]
        np.testing.assert_array_equal(_closed_triangles(shapes,m),[1,1,1])
        q=list(m[2]);q[4]=q[4]+[30,25]
        np.testing.assert_array_equal(_closed_triangles(shapes,[m[0],m[1],tuple(q)]),[0,0,0])
        q=list(m[2]);q[5]={**q[5],"arc_a":0}
        np.testing.assert_array_equal(_closed_triangles(shapes,[m[0],m[1],tuple(q)]),[0,0,0])

    def test_context_growth_requires_independent_evidence(self):
        pieces,truth=eroded_fixture(seed=3,erosion=3,keep=(0,1,2,3))
        shapes=[_shape(p,2.) for p in pieces[:4]]
        def candidate(i,j,score,lines,arcs=(0,0)):
            A,a=truth[i];B,b=truth[j]
            return (score,i,j,A.T@B,A.T@(b-a),
                    {"arc_a":arcs[0],"arc_b":arcs[1],"arc_fraction_a":.12,"arc_fraction_b":.12,
                     "families":["spectral","line-contour"],"line_count":lines,"population_weight":1.})
        seed=candidate(0,1,20.,2)
        # Correct additional piece has two crossings, a single-crossing
        # proposal cannot propagate the partial assembly on its own.
        good=candidate(1,2,15.,2,(len(shapes[1].points)//2,0))
        weak=candidate(2,3,14.,1,(len(shapes[2].points)//2,0))
        groups,joins=confidence_forest(shapes,[seed],[seed,good,weak])
        self.assertEqual(set(groups[0]),{0,1,2})
        self.assertEqual(len(joins),2)
        self.assertTrue(joins[-1]["contextual_growth"])

    def test_missing_seam_ribbon_compares_variations_not_just_mean(self):
        # Two observed rectangles from the same smooth structured texture,
        # separated by 8 unavailable columns. A distractor has reversed
        # contrast but the same mean colour.
        y,x=np.indices((60,45))
        rgb=np.repeat((128+70*np.sin(y/9.))[...,None],3,axis=2).astype(np.uint8)
        mask=np.full((60,45,1),255,np.uint8)
        a=_shape(Fragment("a",np.concatenate([rgb,mask],axis=2)),2.)
        b=_shape(Fragment("b",np.concatenate([rgb,mask],axis=2)),2.)
        c=_shape(Fragment("c",np.concatenate([255-rgb,mask],axis=2)),2.)
        fa,fb,fc=[make_field(v,20.) for v in (a,b,c)]
        R=np.eye(2);t=np.array([53.,0.])
        good=bridge_evidence(a,b,fa,fb,R,t,20.)
        bad=bridge_evidence(a,c,fa,fc,R,t,20.)
        self.assertIsNotNone(good);self.assertIsNotNone(bad)
        self.assertGreater(good[0],.75)
        self.assertLess(bad[0],-.75)
        self.assertGreater(bad[2],good[2]+20.)

    def test_two_anchors_rescue_ambiguous_frontier_without_single_seam_vote(self):
        pieces,truth=eroded_fixture(seed=3,erosion=3,keep=(0,1,4,5))
        shapes=[_shape(p,2.) for p in pieces[:4]]
        def candidate(i,j,s,arc,other,lines):
            A,a=truth[i];B,b=truth[j]
            return (s,i,j,A.T@B,A.T@(b-a),{"arc_a":arc,"arc_b":other,
                     "arc_fraction_a":.1,"arc_fraction_b":.1,"families":["spectral"],
                     "line_count":lines,"population_weight":1.})
        seed=candidate(0,1,20.,0,0,2)
        one=candidate(0,2,8.,len(shapes[0].points)//2,0,0)
        two=candidate(1,2,7.5,len(shapes[1].points)//2,len(shapes[2].points)//2,0)
        groups,_=confidence_forest(shapes,[seed],[seed,one])
        self.assertEqual(set(groups[0]),{0,1})
        groups,joins=confidence_forest(shapes,[seed],[seed,one,two])
        self.assertEqual(set(groups[0]),{0,1,2})
        self.assertGreaterEqual(joins[-1]["context_support"],1)
        # Different anchor IDs are not enough if both reuse the same arc.
        invalid=tuple(list(two[:-1])+[{**two[-1],"arc_b":0}])
        groups,_=confidence_forest(shapes,[seed],[seed,one,invalid])
        self.assertEqual(set(groups[0]),{0,1})

    def test_implied_bridge_recovers_a_missed_second_match(self):
        # A larger bottom fragment meets two top fragments across missing
        # 8 px strips. Only one bottom-fragment pose was retrieved.
        y,x=np.indices((100,108))
        rgb=np.stack([127+60*np.sin(x/7+y/100+k)+25*np.sin(x/20-y/40+k)
                      for k in range(3)],axis=2).clip(0,255).astype(np.uint8)
        shapes=[]
        for i,(xx,yy,w,h) in enumerate([(0,0,50,45),(58,0,50,45),(0,53,108,45)]):
            a=np.dstack([rgb[yy:yy+h,xx:xx+w],np.full((h,w),255,np.uint8)])
            shapes.append(_shape(Fragment(str(i),a),2.))
        fields=[make_field(s,26.) for s in shapes]
        def arc(i,xy):
            return int(np.argmin(np.linalg.norm(shapes[i].points-np.array(xy),axis=1)))
        def c(i,j,weight,t,a,b):
            return (weight,i,j,np.eye(2),np.array(t,float),{"arc_a":a,"arc_b":b,
                "arc_fraction_a":.1,"arc_fraction_b":.1,"families":["spectral"],
                "line_count":0,"population_weight":1.})
        seed=c(0,1,20.,[58,0],arc(0,[50,22]),arc(1,[0,22]))
        only=c(0,2,10.,[0,53],arc(0,[25,45]),arc(2,[25,0]))
        groups,_=confidence_forest(shapes,[seed],[seed,only])
        self.assertEqual(set(groups[0]),{0,1})
        groups,joins=confidence_forest(shapes,[seed],[seed,only],fields,26.)
        self.assertEqual(set(groups[0]),{0,1,2})
        self.assertEqual(joins[-1]["direct_bridge_support"],1)
        # Same silhouette and candidate poses, but the second anchor has
        # incompatible observed contrast. It cannot support the placement.
        corrupted=shapes[1].fragment.rgba.copy()
        corrupted[...,:3]=255-corrupted[...,:3]
        altered=list(shapes);altered[1]=_shape(Fragment("1",corrupted),2.)
        bad_fields=list(fields);bad_fields[1]=make_field(altered[1],26.)
        groups,_=confidence_forest(altered,[seed],[seed,only],bad_fields,26.)
        self.assertEqual(set(groups[0]),{0,1})

    def test_sparse_nomination_requires_more_than_two_correlated_generators(self):
        pieces,truth=eroded_fixture(seed=3,erosion=3,keep=(0,1,2))
        shapes=[_shape(p,2.) for p in pieces[:2]]
        A,a=truth[0];B,b=truth[1]
        info={"arc_a":0,"arc_b":0,"arc_fraction_a":.15,"arc_fraction_b":.15,
              "texture":30.,"line_count":1,"line_strength":.4,"population_weight":1.}
        candidates=[(12.,0,1,A.T@B,A.T@(b-a),{**info,"source":source})
                    for source in ("spectral","line-contour")]
        seeds,_=independent_consensus(shapes,candidates)
        self.assertTrue(seeds)
        seeds,_=independent_consensus(shapes,candidates,sparse=True)
        self.assertFalse(seeds)
        # A third independent crossing and sufficiently strong support
        # can make the same sparse match eligible.
        strong=[(*c[:-1],{**c[-1],"line_count":3,"line_strength":1.1}) for c in candidates]
        seeds,_=independent_consensus(shapes,strong,sparse=True)
        self.assertTrue(seeds)

    def test_continuation_uses_memory_not_fixed_piece_count(self):
        from shardelix.reconstruction.resource_budget import ResourceBudget
        # A large bland population reaches the texture check rather than
        # being rejected by the old 128 piece cutoff. No FFT is necessary.
        rgb=np.full((25,25,4),140,np.uint8);rgb[...,3]=255
        shapes=[_shape(Fragment(str(i),rgb.copy()),2.) for i in range(129)]
        ample=ResourceBudget(512*1024**2,"test",None)
        small=ResourceBudget(32*1024**2,"test",None)
        _,_,stats=recover(shapes,[],memory_budget=ample)
        self.assertEqual(stats["reason"],"insufficient pictorial texture")
        _,_,stats=recover(shapes,[],memory_budget=small)
        self.assertIn("memory budget",stats["reason"])


if __name__=="__main__":unittest.main()
