"""Reproducible blind benchmark: erased boundaries, missing tiles, rotations, distractor.

Run from repository root: PYTHONPATH=src python benchmarks/benchmark_eroded.py
The fixture labels are only passed to evaluation after solve.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.scan import reconstruct, rotation

def eroded_fixture(seed=9, erosion=5, keep=(0,1,2,3,4,6,7)):
    """Independent textured, curved partition, random rotation, missing part."""
    rng=np.random.default_rng(seed); y,x=np.indices((240,320))
    source=np.empty((240,320,3), np.uint8)
    for k in range(3):
        z=ndi.gaussian_filter(rng.normal(size=x.shape), 9)
        z=z/(z.std()+1.e-6)*42+128+35*np.sin(x/(17+4*k)+y/(26-2*k)+k)
        source[:,:,k]=np.clip(z,0,255)
    xx=x+9*np.sin(y/21)+5*np.sin(y/9)
    yy=y+7*np.sin(x/22)
    labels=np.clip((xx//80).astype(int),0,3)+4*np.clip((yy//120).astype(int),0,1)
    pieces=[]; truth=[]
    for i in keep:
        mask=ndi.binary_erosion(labels==i, iterations=erosion)
        ys,xs=np.nonzero(mask)
        x0,x1=xs.min(),xs.max()+1; y0,y1=ys.min(),ys.max()+1
        a=np.dstack([source,mask.astype(np.uint8)*255])[y0:y1,x0:x1].copy()
        a[~mask[y0:y1,x0:x1]]=0
        h,w=a.shape[:2];k=int(rng.integers(4))
        pieces.append(Fragment(f"part_{i}",np.rot90(a,k).copy()))
        truth.append((rotation(k*np.pi/2),np.array([x0,y0])+[(0,0),(w,0),(w,h),(0,h)][k]))
    a=pieces[2].rgba.copy()
    a[:,:,:3]=np.where(a[:,:,3,None]>0,[220,40,12],0)
    pieces.append(Fragment("unrelated",a));truth.append(None)
    return pieces,truth

def score(pieces,truth,components):
    by_id={p.id:i for i,p in enumerate(pieces)}
    labels={p.id:v for p,v in zip(pieces,truth)}
    correct=total=outlier=0
    for group in components:
        mat={k:np.asarray(v["input_to_component_affine"]) for k,v in group["poses"].items()}
        for e in group["joins"]:
            i,j=e["a"],e["b"];total+=1
            if labels[i] is None or labels[j] is None:
                outlier+=1;continue
            A,a=labels[i];B,b=labels[j];I,J=mat[i],mat[j]
            Q=A.T@B;u=A.T@(b-a)
            Rp=I[:,:2].T@J[:,:2];tp=I[:,:2].T@(J[:,2]-I[:,2])
            h,w=pieces[by_id[j]].mask.shape;center=np.array([w/2,h/2])
            err=np.linalg.norm((Rp-Q)@center+tp-u)
            angle=abs(np.arctan2((Rp.T@Q)[1,0],(Rp.T@Q)[0,0]))*180/np.pi
            if err<=10 and angle<=10:correct+=1
    return {"correct_relative_joins":correct,"accepted_joins":total,
            "outlier_joins":outlier,"precision":correct/total if total else None}

def trial(seed,erosion,keep):
    pieces,truth=eroded_fixture(seed,erosion,keep)
    result=[]
    for enabled in (False,True,"always"):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);report={"input_images":len(pieces),"warnings":[],"common_scale":1.0}
            t=time.perf_counter()
            d=reconstruct(pieces,out,report,gap_recovery=enabled)
            elapsed=time.perf_counter()-t
            comp=json.loads((out/"component_placements.json").read_text())
            result.append({"seed":seed,"erosion_px":erosion,"genuine_pieces":len(keep),
                           "missing_pieces":8-len(keep),"distractors":1,"gap_enabled":enabled,
                           "mode":comp["model"],"wall_seconds":elapsed,"placed":d["placed"],
                           "gap_recovery":d["scan"]["gap_recovery"],"alternative":(None if "alternative" not in comp else {"model":comp["alternative"]["model"],**score(pieces,truth,comp["alternative"]["components"])}),**score(pieces,truth,comp["components"])})
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("outputs/benchmarks/eroded_benchmark.json"))
    args=parser.parse_args()
    rows=[]
    for seed in (3,9,17):
        for erosion in (3,5,8):
            rows.extend(trial(seed,erosion,(0,1,2,3,4,6,7)))
    for seed in (3,17):
        rows.extend(trial(seed,5,(0,1,4,7)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2))
    for enabled in (False,True,"always"):
        q=[r for r in rows if r["gap_enabled"]==enabled]
        print("gap",enabled,"correct joins",sum(r["correct_relative_joins"] for r in q),
              "accepted",sum(r["accepted_joins"] for r in q),"outlier joins",sum(r["outlier_joins"] for r in q))

if __name__=="__main__":main()
