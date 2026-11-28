import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from shardelix import Fragment,Pose,Solver
from shardelix.reconstruction.common import load_fragments
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.cli import solve_case
from shardelix.reconstruction.irregular.lattice import _keys
from shardelix.studio import safe_zip,worker
from shardelix.reconstruction.limits import validate_constraints

class V8Tests(unittest.TestCase):
    def test_sparse_public_api_and_hidden_truth(self):
        rng=np.random.default_rng(440)
        src=(gaussian_filter(rng.normal(size=(256,256,3)),(1.2,1.2,0))*420+127).clip(0,255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as td:
            case=Path(td)/"case";out=Path(td)/"out"
            meta=generate(src,case,256,"grid",11,True)
            hidden=Path(td)/"truth";(case/"evaluation").rename(hidden)
            p=load_fragments(case/"pieces")
            result=Solver(strategy="sparse").solve(p,shape=(256,256),cut="grid",grid=tuple(meta["grid"]),rotations=True)
            result.save(out)
            gt={i:Pose(**v) for i,v in json.loads((hidden/"ground_truth.json").read_text())["poses"].items()}
            m=evaluate(p,result.poses,gt,(256,256),True)
            self.assertTrue(m["pass_95"],m)
            self.assertFalse(result.diagnostics["accuracy_verified"])
            self.assertTrue((out/"assembly_with_cuts.png").is_file())

    def test_metadata_rejected(self):
        for shape,cut,grid,rot in [((-1,64),"grid",(2,2),True),((64,64),"other",None,True),((64,64),"grid",(1,1),True),((64,64),"grid",(2,2),"false")]:
            with self.assertRaises(ValueError):
                validate_constraints(4,shape,cut,grid,rot)
        with self.assertRaises(ValueError):
            validate_constraints(2048,(1023,2048),"grid",(32,64),True)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=root/"case"
            generate(np.full((16,16,3),127,np.uint8),c,4,"grid")
            d=json.loads((c/"case.json").read_text());d["piece_count"]=5;(c/"case.json").write_text(json.dumps(d))
            with self.assertRaises(ValueError):solve_case(c,root/"result")

    def test_sparse_duplicate_warning(self):
        p=[Fragment(str(i),np.full((12,12,4),255,np.uint8)) for i in range(16)]
        with tempfile.TemporaryDirectory() as td:
            r=Solver(strategy="sparse").solve(p,shape=(48,48),cut="grid",grid=(4,4),rotations=True)
            r.save(td)
            self.assertIn(r.diagnostics["status"],{"partial","ambiguous"})
            self.assertEqual(r.diagnostics["identical_piece_count"],16)

    def test_geometry_vectorized_keys_exact(self):
        rng=np.random.default_rng(28);d=rng.integers(0,4,70,dtype=np.uint8)
        v=rng.integers(0,100,(70,2));n=len(d);k=12
        for rotations in (False,True):
            dd=np.r_[d,d[:k-1]];expected=[]
            for i in range(0,n,2):
                ds=dd[i:i+k]
                if np.count_nonzero(ds[1:]!=ds[:-1])<4:continue
                rev=(ds[::-1]+2)%4
                norm=(ds.astype(np.int16)-int(ds[0]))%4 if rotations else ds
                rnorm=(rev.astype(np.int16)-int(rev[0]))%4 if rotations else rev
                expected.append((norm.astype(np.uint8).tobytes(),rnorm.astype(np.uint8).tobytes(),tuple(v[i]),tuple(v[(i+k)%n]),int(ds[0]),int(rev[0])))
            self.assertEqual(list(_keys(v,d,k,2,rotations)),expected)

    def test_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);z=root/"x.zip"
            with zipfile.ZipFile(z,"w") as f:f.writestr("../x.png","bad")
            with self.assertRaises(ValueError):safe_zip(z,root/"case")
            self.assertFalse((root/"x.png").exists())

    def test_zip_ignores_truth(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=root/"original"
            generate(np.full((20,20,3),127,np.uint8),c,4,"grid")
            with zipfile.ZipFile(root/"x.zip","w") as z:
                for p in c.rglob("*"):
                    if p.is_file():z.write(p,p.relative_to(c))
            meta,n=safe_zip(root/"x.zip",root/"case")
            self.assertEqual(n,4)
            self.assertEqual(meta["piece_count"],4)
            self.assertFalse((root/"case/evaluation").exists())
