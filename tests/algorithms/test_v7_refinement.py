"""Focused deterministic invariants for V7 refinement (synthetic fixtures)."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
from shardelix.reconstruction.common import load_fragments,read_source,Pose,render
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.irregular.frontier import complete_frontier
from shardelix.reconstruction.tiles.refinement import refine_grid_lines,refine_grid_phase
from shardelix.reconstruction.tiles.consensus import edge_costs,dimension_filter,score_grid
from shardelix.reconstruction.tiles.straight import _grid_bounds
from shardelix.reconstruction.strips.refinement import refine_strip_blocks,_lcp

def fixture(src,folder,count,cut,seed=234,rot=True):
    meta=generate(src,folder,count,cut,seed,rot)
    p=load_fragments(folder/'pieces')
    t={i:Pose(**q) for i,q in json.loads((folder/'evaluation/ground_truth.json').read_text())['poses'].items()}
    return p,t,meta

class RefineV7Tests(unittest.TestCase):
    def test_lcp_exact(self):
        a=np.array([2,1,1,1,2,1,1,1])
        t=_lcp(a,a)
        self.assertEqual(t[0,4],4)
        self.assertEqual(t[0,1],0)
        self.assertEqual(t[2,6],2)

    def test_thin_strip_block_relocation(self):
        y,x=np.indices((72,160));rgb=np.stack((2*y+30*np.sin(x/7)+35,2*y+10*np.cos(x/11)+20,2*y+20*np.sin(x/9)+40),axis=-1).clip(0,255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as path:
            p,t,m=fixture(rgb,Path(path)/'case',64,'horizontal-strips')
            ys,xs=_grid_bounds(*rgb.shape[:2],*m['grid'])
            ids=sorted(t,key=lambda i:t[i].y)
            order=ids[8:16]+ids[:8]+ids[16:]
            q={id:Pose(0,int(ys[j]),t[id].k) for j,id in enumerate(order)}
            r,d=refine_strip_blocks(p,q,rgb.shape[:2],tuple(m['grid']),True,rounds=8)
            e=evaluate(p,r,t,rgb.shape[:2],False,False)
            self.assertTrue(e['pass_95'],e)
            self.assertEqual(e['collision_pixels'],0)
            self.assertLessEqual(d['strip_energy_after'],d['strip_energy_before'])

    def test_frontier_zero_collision_and_completion(self):
        src=read_source(Path('data/examples/straight_texture.png'),384)
        with tempfile.TemporaryDirectory() as path:
            p,t,m=fixture(src,Path(path)/'case',64,'jigsaw')
            # Deterministic fixture: hide several interior poses, never pass
            # the withheld placements into the completion algorithm.
            h,w=src.shape[:2]
            missing=sorted(t,key=lambda i:abs(t[i].x-w/3)+abs(t[i].y-h/3))[:6]
            q={i:v for i,v in t.items() if i not in missing}
            r,d=complete_frontier(p,q,(h,w),True)
            e=evaluate(p,r,t,(h,w),False,False)
            self.assertTrue(e['pass_95'],e)
            self.assertEqual(e['collision_pixels'],0)
            self.assertEqual(len(r),len(set(r)))

    def test_grid_line_and_dimension_constraints(self):
        src=read_source(Path('data/examples/straight_texture.png'),320)
        with tempfile.TemporaryDirectory() as path:
            p,t,m=fixture(src,Path(path)/'case',64,'grid',rot=True)
            shape=src.shape[:2];grid=tuple(m['grid']);ys,xs=_grid_bounds(*shape,*grid)
            raw=edge_costs(p,True,'mgc',.65)
            filt=dimension_filter(raw.copy(),p,np.diff(ys),np.diff(xs),True)
            self.assertTrue(np.isfinite(score_grid(p,t,shape,grid,True,filt)))
            j,k=next((j,k) for j in range(len(xs)-1) for k in range(j+1,len(xs)-1) if xs[j+1]-xs[j]==xs[k+1]-xs[k])
            q=dict(t);two=sorted((i for i,v in t.items() if v.y==ys[2] and v.x in [xs[j],xs[k]]),key=lambda i:t[i].x)
            a,b=two;q[a]=Pose(t[b].x,t[a].y,t[a].k);q[b]=Pose(t[a].x,t[b].y,t[b].k)
            r,d=refine_grid_lines(p,q,shape,grid,True,raw,rounds=2)
            self.assertLessEqual(d['grid_energy_after'],d['grid_energy_before']+1e-5)
            e=evaluate(p,r,t,shape,False,False)
            self.assertTrue(e['pass_95'],e)
            self.assertEqual(e['collision_pixels'],0)

    def test_grid_phase_roll(self):
        src=read_source(Path('data/examples/straight_texture.png'),320)[:256,:256]
        with tempfile.TemporaryDirectory() as path:
            p,t,m=fixture(src,Path(path)/'case',64,'grid',rot=True)
            shape=src.shape[:2];grid=tuple(m['grid']);ys,xs=_grid_bounds(*shape,*grid)
            raw=edge_costs(p,True,'mgc',.65)
            # Rotate an entire dimension-compatible cyclic column sequence.
            widths=np.diff(xs)
            period=next((i for i in range(1,grid[1]) if np.array_equal(np.roll(widths,i),widths)),None)
            if period is None:
                # Exact synthetic square grid for phase fixture.
                self.skipTest("Fixture has no non-trivial width period")
            q={i:Pose(int(xs[(int(np.searchsorted(xs,v.x))+period)%grid[1]]),v.y,v.k) for i,v in t.items()}
            r,d=refine_grid_phase(p,q,shape,grid,True,raw)
            e=evaluate(p,r,t,shape,True,False)
            self.assertTrue(e['pass_95'],e)
            self.assertLessEqual(d['phase_energy_after'],d['phase_energy_before'])

if __name__=='__main__':unittest.main()
