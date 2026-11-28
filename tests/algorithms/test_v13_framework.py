import json,shutil,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from shardelix.native_core import backend,contours,loop_votes,rerank,profiles
from shardelix.reconstruction.irregular.lattice import _contours_reference
from shardelix.reconstruction.tiles.links import _loop_votes_reference
from shardelix.workflow import run_job

def texture():
    return np.clip(127+480*gaussian_filter(np.random.default_rng(2287).normal(size=(256,256,3)),(1.4,1.4,0)),0,255).astype(np.uint8)

class V13Tests(unittest.TestCase):
    def test_cpp_contours_and_cycles(self):
        if backend()=="python-reference":self.skipTest("C++ optional backend not built")
        rng=np.random.default_rng(123)
        for shape in [(1,1),(18,19),(40,80)]:
            for density in (0,.1,.5,.9,1):
                for _ in range(3):
                    mask=rng.random(shape)<density;a=contours(mask);b=_contours_reference(mask)
                    self.assertEqual(len(a),len(b))
                    for (a1,a2),(b1,b2) in zip(a,b):
                        np.testing.assert_array_equal(a1,b1);np.testing.assert_array_equal(a2,b2)
        for n in (4,16,128,4096):
            a=rng.integers(-1,n,(n,5),dtype=np.int32)
            np.testing.assert_array_equal(loop_votes(a),_loop_votes_reference(a))
    def test_cpp_edge_descriptors(self):
        if backend()=="python-reference":self.skipTest("C++ optional backend not built")
        from shardelix import Fragment
        from shardelix.reconstruction.tiles.consensus import _profiles_reference
        rng=np.random.default_rng(11)
        p=[Fragment(str(i),rng.integers(0,256,(*shape,4),np.uint8)) for i,shape in enumerate([(1,1),(1,5),(5,1),(8,13),(13,8),(8,13)])]
        a,b=profiles(p),_profiles_reference(p)
        for (ae,ag),(be,bg) in zip(a,b):
            np.testing.assert_allclose(ae,be,atol=2e-7,rtol=1e-6)
            np.testing.assert_allclose(ag,bg,atol=2e-7,rtol=1e-6)
    def test_cpp_fused_seam_scores(self):
        if backend()=="python-reference":self.skipTest("C++ optional backend not built")
        rng=np.random.default_rng(13)
        for L in (1,5,129,257):
            e,g,f,h=[rng.normal(size=(15,L,3)).astype(np.float32) for _ in range(4)]
            j=rng.integers(0,15,(15,9),dtype=np.int32);step=max(1,L//128)
            E=e[:,::step];G=g[:,::step];F=f[j,::-step];H=h[j,::-step]
            delta=E[:,None]-F;basecost=np.mean(np.abs(delta),axis=(2,3));su=np.sqrt(np.mean(G*G,axis=1)+.00012);sv=np.sqrt(np.mean(H*H,axis=2)+.00012)
            pred=(np.mean(np.abs(delta+G[:,None]),axis=2)/su[:,None]+np.mean(np.abs(-delta+H),axis=2)/sv).mean(axis=2)*.5
            scale=np.sqrt(np.maximum(.01,np.sqrt(np.mean(G*G,axis=(1,2))))[:,None]*np.maximum(.01,np.sqrt(np.mean(H*H,axis=(2,3)))))
            np.testing.assert_allclose(rerank(e,g,f,h,j,step),.45*basecost/scale+.55*pred,rtol=3e-6,atol=1e-6)
    def test_split_workflow_and_independent_reference(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);g=root/"gen";(g/"upload").mkdir(parents=True)
            Image.fromarray(texture()).save(g/"upload/source.png")
            s=run_job(g,dict(task="generate",cut="grid",count=64,angle_mode="quarter",seed=42,max_side=256),browser=True)
            self.assertEqual(s["generation"]["observed"],64);self.assertNotIn("assembly.png",s["files"])
            with zipfile.ZipFile(g/"result/fragments.zip") as z:
                self.assertTrue(all(n=="case.json" or n.startswith("pieces/") for n in z.namelist()))
            with zipfile.ZipFile(g/"result/ground_truth.zip") as z:self.assertIn("evaluation/ground_truth.json",z.namelist())
            states=[]
            for mode in ("blind","truth","raster","wrong_truth"):
                r=root/mode;(r/"upload").mkdir(parents=True);shutil.copyfile(g/"result/fragments.zip",r/"upload/fragments.zip")
                if mode!="blind":
                    (r/"truth").mkdir()
                    name="ground_truth.zip" if mode in ("truth","wrong_truth") else "source.png"
                    shutil.copyfile(g/"result"/name,r/"truth"/name)
                    if mode=="wrong_truth":
                        with zipfile.ZipFile(g/"result"/name) as src,zipfile.ZipFile(r/"truth"/name,"w") as dst:
                            for entry in src.namelist():
                                content=src.read(entry)
                                if entry=="evaluation/fragment_manifest.json":
                                    hashes=json.loads(content);hashes[next(iter(hashes))]="0"*64;content=json.dumps(hashes).encode()
                                dst.writestr(entry,content)
                state=run_job(r,{"task":"blind","use_metadata":True,"profile":"balanced"},browser=True);states.append(state)
            self.assertIsNone(states[0]["metrics"]);self.assertIsNone(states[2]["metrics"])
            self.assertTrue(states[1]["metrics"]["pass_95"],states[1])
            self.assertTrue(states[1]["comparison"]["pose_evaluated"]);self.assertIn("raw_mae",states[2]["comparison"])
            self.assertIsNone(states[3]["metrics"])
            self.assertIn("manifest",states[3]["comparison"]["warning"])
            for mode in ("truth","raster","wrong_truth"):
                a=json.loads((root/"blind/result/placements.json").read_text())["poses"];b=json.loads((root/mode/"result/placements.json").read_text())["poses"];self.assertEqual(a,b)
    def test_missing_reference_archive(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);g=root/"gen";(g/"upload").mkdir(parents=True)
            Image.fromarray(texture()).save(g/"upload/source.png")
            config=dict(task="generate",cut="grid",count=16,seed=42,angle_mode="continuous",missing=.125,sampling=1,max_side=256)
            result=run_job(g,config,browser=True)
            self.assertLess(result["generation"]["observed"],16)
            r=root/"recon";(r/"upload").mkdir(parents=True);(r/"truth").mkdir()
            shutil.copyfile(g/"result/fragments.zip",r/"upload/fragments.zip")
            shutil.copyfile(g/"result/ground_truth.zip",r/"truth/ground_truth.zip")
            state=run_job(r,dict(task="blind",profile="balanced",use_metadata=True),browser=True)
            self.assertTrue(state["comparison"]["pose_evaluated"],state["comparison"])
            self.assertIsNotNone(state["metrics"])
    def test_fast_jigsaw_rescue(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            Image.fromarray(texture()).save(root/"upload/texture.png")
            c=dict(task="demo",cut="jigsaw",count=256,seed=42,angle_mode="quarter",max_side=256,profile="fast")
            s=run_job(root,c,browser=True)
            self.assertTrue(s["metrics"]["pass_95"],s)
            self.assertEqual(s["diagnostics"]["placed"],256)
if __name__=="__main__":unittest.main()
