import unittest,tempfile,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from shardelix import Fragment,Pose,Solver
from shardelix.reconstruction.irregular.frame_anchor import anchor_irregular_frame
from shardelix.reconstruction.strips.multi_gap import solve_multigap_strips
from shardelix.reconstruction.evaluation import canvas_align
from shardelix.workflow import run_job,case_configs,inspect_uploads,WEB_COUNT

def image(h,w,seed=17,sigmax=2):
    a=np.random.default_rng(seed).normal(size=(h,w,3))
    return np.clip(127+550*gaussian_filter(a,(1.45,sigmax,0)),0,255).astype(np.uint8)

def frag(i,a,alpha=None):
    z=np.empty((*a.shape[:2],4),np.uint8);z[:,:,:3]=a;z[:,:,3]=255 if alpha is None else alpha
    z[z[:,:,3]==0,:3]=0
    return Fragment(str(i),z)

class V12Tests(unittest.TestCase):
    def test_multigap_held_out_and_abstain(self):
        # Development inputs were seed60; this independent deterministic case
        # tests two contiguous gaps without supplying their positions.
        a=image(384,96,seed=173,sigmax=3.5);p=[];truth={}
        for i in range(32):
            if i in (6,7,20,21):continue
            q=frag(i,a[:,3*i:3*i+3]);k=i%4
            p.append(Fragment(q.id,q.rotated(k).copy()));truth[q.id]=Pose(3*i,0,(-k)%4)
        q,d=solve_multigap_strips(p,(384,96),(1,32),True)
        # Any accepted proposal must use every piece exactly once, fit canvas,
        # and be measured by an independent evaluator.
        if q:
            self.assertEqual(canvas_align(p,q,truth,(384,96),True),truth)
        else:self.assertIn(d["multigap_status"],("ambiguous-gap-budget","low-continuity","gap-outside-correlation"))
        q,d=solve_multigap_strips(p,(384,210),(1,70),True)
        self.assertFalse(q) # non-inferable budget / wrong canvas
    def test_flat_wall_anchor_and_ambiguity(self):
        # Two distinct long straight edges on right; left boundaries jagged.
        pieces=[];poses={}
        for i in range(2):
            m=np.full((24,18),255,np.uint8);m[::2,0]=0
            z=frag(i,image(24,18,i),m);pieces.append(z);poses[z.id]=Pose(0,24*i)
        out,d=anchor_irregular_frame(pieces,poses,(48,30))
        self.assertEqual([x.x for x in out.values()],[12,12])
        self.assertEqual(d["anchor_translation_slack"],[0,0])
        p=[frag(i,image(24,18,i)) for i in range(2)]
        out,d=anchor_irregular_frame(p,poses,(48,30))
        self.assertEqual(d["anchor_translation_slack"],[0,12])
        self.assertEqual(out,poses)
    def test_unknown_width_continuous_document_geometry(self):
        from shardelix.reconstruction.rectification import rotate_raster
        from shardelix.reconstruction.strips.rectification import normalize_document_strips
        from shardelix.reconstruction.cli import solve_external_document
        widths=[7,13,20,32,41,52];angles=[-16,77,154,-112,41,9];H=240
        a=image(H,sum(widths),seed=454);pieces=[];x=0
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/"pieces";folder.mkdir()
            for i,(w,angle) in enumerate(zip(widths,angles)):
                z=rotate_raster(frag(i,a[:,x:x+w]).rgba,angle)
                pieces.append(Fragment(str(i),z));Image.fromarray(z).save(folder/f"{i}.png");x+=w
            q,est,aff,d=normalize_document_strips(pieces)
            self.assertEqual(d["estimated_strip_height"],H)
            self.assertEqual(d["estimated_strip_widths"],widths)
            self.assertLess(max(abs((est[str(i)]+a+90)%180-90) for i,a in enumerate(angles)),.1)
            result=solve_external_document(folder,root/"out",angle_mode="continuous")
            self.assertIn("input_to_canvas_affine",result)
            self.assertEqual(result["diagnostics"]["canvas_height"] if "canvas_height" in result["diagnostics"] else H,H)
            self.assertTrue(result["diagnostics"]["document_missing_unknown"])
            self.assertEqual(result["diagnostics"]["collision_pixels"],0)
    def test_profile_config(self):
        a=image(64,64);p=[frag(i,a[y:y+16,x:x+16]) for i,(y,x) in enumerate((y,x) for y in range(0,64,16) for x in range(0,64,16))]
        for profile in ("fast","balanced","accurate"):
            r=Solver(profile=profile).solve(p,shape=(64,64),cut="grid",grid=(4,4))
            self.assertEqual(r.diagnostics["performance_profile"],profile)
            self.assertEqual(r.render()[2],0)
        with self.assertRaises(ValueError):Solver(profile="unknown").solve(p,shape=(64,64),cut="grid",grid=(4,4))
    def test_portable_workflow_batch_blind_and_limits(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"demo";(root/"upload").mkdir(parents=True)
            Image.fromarray(image(96,96)).save(root/"upload/texture.png")
            conf=dict(task="demo",cut="grid",count=16,max_side=128,seed=9,angle_mode="quarter",profile="balanced")
            conf["batch"]={"counts":[16,25],"seeds":[9],"profiles":["fast","balanced"]}
            status=run_job(root,conf,browser=True)
            self.assertEqual(status["state"],"done")
            self.assertEqual(len(status["benchmark"]),4)
            self.assertTrue(all(x["status"]=="done" for x in status["benchmark"]),status)
            root2=Path(t)/"blind";(root2/"upload").mkdir(parents=True)
            import shutil
            archive=root2/"upload/pieces.zip";shutil.copyfile(root/"result/fragments.zip",archive)
            self.assertEqual(inspect_uploads([archive],True)["observed_count"],25)
            with zipfile.ZipFile(archive) as z:self.assertFalse(any("truth" in n for n in z.namelist()))
            r=run_job(root2,{"task":"blind","use_metadata":True,"profile":"balanced"},browser=True)
            self.assertIsNone(r["metrics"])
            self.assertEqual(r["diagnostics"]["piece_count"],25)
            self.assertNotIn("source.png",r["files"])
            with self.assertRaises(ValueError):case_configs({**conf,"batch":{"counts":list(range(1,26))}},True)
    def test_batch_failure_retained(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            Image.fromarray(image(32,32)).save(root/"upload/a.png")
            c=dict(task="demo",cut="grid",count=4,max_side=128,angle_mode="quarter",profile="fast",batch={"counts":[4,WEB_COUNT+1]})
            r=run_job(root,c,browser=True)
            self.assertEqual([x["status"] for x in r["benchmark"]],["done","error"])
            self.assertIn("benchmark.csv",r["files"])
if __name__=="__main__":unittest.main()
