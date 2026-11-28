"""V12 controlled blind gap and frame experiments; evaluate only after solve.

Use --baseline-src /path/to/V11/src for same-input comparison. Seeds 173, 181,
191, 199 are independent of development seed60. Cases are synthetic covariance
textures, not a guarantee for physical/text or arbitrary natural images.
"""
import os,sys,subprocess,shutil,math,time,json,csv
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.common import json_write

def fixture(case,seed,width,missing,angle,axis):
    n=32;h=384;w=n*width
    src=np.clip(127+550*gaussian_filter(np.random.default_rng(seed).normal(size=(h,w,3)),(1.45,3.5,0)),0,255).astype(np.uint8)
    if axis=="horizontal-strips":src=src.transpose(1,0,2)
    meta=generate(src,case,n,axis,seed,angle_mode=angle)
    p=case/"evaluation/ground_truth.json";truth=json.loads(p.read_text());old=truth["poses"]
    labels=np.full(src.shape[:2],-1,np.int32);ids=[]
    for ident,v in old.items():
        idx=v["x"]//width if axis=="strips" else v["y"]//width
        if idx in missing:
            (case/"pieces"/(ident+".png")).unlink();continue
        j=len(ids);ids.append(ident)
        if axis=="strips":labels[:,idx*width:(idx+1)*width]=j
        else:labels[idx*width:(idx+1)*width,:]=j
    meta.update(piece_count=len(ids),expected_count=n,angle_mode=angle,sampling=1)
    truth.update(poses={i:old[i] for i in ids},label_ids=ids,case=meta)
    json_write(case/"case.json",meta);json_write(p,truth);np.savez_compressed(case/"evaluation/truth_labels.npz",labels=labels)
def run(case,out,src):
    env={**os.environ,"PYTHONPATH":str(src),"OPENBLAS_NUM_THREADS":"1"}
    code="""from pathlib import Path
import sys
from shardelix.reconstruction.cli import solve_case,evaluate_case
c,o=map(Path,sys.argv[1:]);t=c/'evaluation';h=c.parent/'held';t.rename(h)
try:solve_case(c,o)
finally:h.rename(t)
evaluate_case(c,o)
"""
    p=subprocess.run([sys.executable,"-c",code,str(case),str(out)],env=env,capture_output=True,text=True,timeout=90)
    if p.returncode:return {"error":p.stderr[-600:]}
    m=json.loads((out/"metrics.json").read_text());d=json.loads((out/"diagnostics.json").read_text())
    return {**{k:m[k] for k in ("direct_position_accuracy","adjacency_f1","available_piece_recovery","coverage","collision_pixels","pass_95","runtime_seconds")},**{k:d.get(k) for k in ("algorithm","thin_gap_status","multigap_status","multigap_accepted","frame_anchor_status")}}
def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline-src",type=Path);p.add_argument("--out",type=Path,default=Path("outputs/benchmarks/v12/recovery.csv"));a=p.parse_args()
    root=Path(__file__).resolve().parents[1];variants=[("V12",root/"src")]
    if a.baseline_src:variants.insert(0,("V11",a.baseline_src.resolve()))
    rows=[]
    with TemporaryDirectory() as t:
        t=Path(t)
        for idx,(seed,width,miss,angle,axis) in enumerate([
            (173,3,(6,7,20,21),"quarter","strips"),
            (181,3,(6,7,21),"quarter","horizontal-strips"),
            (191,4,(6,7,20,21),"quarter","strips"),
            (199,5,(6,21),"quarter","strips"),
            (173,3,(6,7,20,21),"continuous","strips"),
            (181,3,(6,7,21),"continuous","horizontal-strips"),
            (191,4,(0,6,7),"quarter","strips"),
            (199,3,(0,31,21),"continuous","strips")]):
            c=t/str(idx)/"case";fixture(c,seed,width,miss,angle,axis)
            for ver,src in variants:
                try:r=run(c,t/str(idx)/ver,src)
                except subprocess.TimeoutExpired:r={"error":"timeout"}
                row=dict(case=idx+1,seed=seed,width=width,removed=" ".join(map(str,miss)),angle=angle,cut=axis,version=ver,**r);rows.append(row)
                print(json.dumps(row),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
