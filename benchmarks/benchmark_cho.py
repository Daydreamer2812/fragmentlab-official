"""User-supplied Cho 432 benchmark. Randomized IDs; truth used only after solving.
Optional 1024 recut reconstructs original source in separate generation stage."""
from __future__ import annotations
import argparse,hashlib,json,time,csv
from pathlib import Path
import numpy as np
from PIL import Image
from shardelix.reconstruction.common import Fragment,Pose,load_fragments,render,save_rgba,json_write
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.overlay import write_cut_outputs
from shardelix.reconstruction.solver import solve
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.cli import solve_case,evaluate_case
from shardelix.reconstruction.tiles.straight import solve_straight
from shardelix.reconstruction.rotation import solve_rotated_tiles

def features(pieces):
    grads=[];white=[];std=[];keys=[]
    for p in pieces:
        im=p.rgba[:,:,:3].astype(np.float32)/255
        directions=[float(np.abs(np.diff(im,axis=k)).mean()) for k in (0,1) if im.shape[k]>1]
        gr=float(np.mean(directions)) if directions else 0.
        grads.append(float(gr));std.append(float(im.std(axis=(0,1)).mean()))
        white.append(float((im.min(axis=2)>.95).mean()))
        keys.append(hashlib.sha256(p.rgba.tobytes()).hexdigest())
    return dict(mean_gradient=float(np.mean(grads)),median_gradient=float(np.median(grads)),
                low_gradient_fraction=float(np.mean(np.array(grads)<.0075)),white_pixel_fraction=float(np.mean(white)),
                exact_duplicate_fraction=1-len(set(keys))/len(keys),mean_color_std=float(np.mean(std)))

def load_anon(case,rotate,seed=874):
    src=load_fragments(case/"pieces");rng=np.random.default_rng(seed);pieces=[];mapping={}
    for i in rng.permutation(len(src)):
        p=src[i];k=int(rng.integers(4)) if rotate else 0
        ident="p_"+rng.bytes(12).hex();pieces.append(Fragment(ident,p.rotated(k).copy()));mapping[ident]=(p.id,k)
    return pieces,mapping

def run(case,out,method,rotate,previews=True,params=None):
    pieces,mapping=load_anon(case,rotate)
    shape=(504,672);grid=(18,24)
    start=time.perf_counter()
    if method=="v5":
        # Exact V5 defaults via unchanged legacy path; rotated V5 same single strategy.
        if rotate:poses,diag=solve_rotated_tiles(pieces,shape,grid,samples=24,gradient_weight=.25,top_k=3)
        else:
            from shardelix.reconstruction.solver import _solve_v5
            poses,diag=_solve_v5(pieces,shape,"grid",grid,False,auto_tune=True)
    elif method=="candidate":
        from shardelix.reconstruction.tiles.consensus import solve_consensus_grid
        poses,diag=solve_consensus_grid(pieces,shape,grid,rotate,**(params or {}))
    else:poses,diag=solve(pieces,shape,"grid",grid,rotate,auto_tune=True)
    runtime=time.perf_counter()-start
    if previews:
        rgba,labels,_=render(pieces,poses,shape);save_rgba(out/"assembly.png",rgba);write_cut_outputs(out,rgba,labels)
        json_write(out/"placements.json",{k:p.as_dict() for k,p in poses.items()})
    # Ground truth is loaded only now.
    truth0=json.loads((case/"ground_truth.json").read_text())["pieces"]
    original={r["piece_id"]:(r["row"],r["col"]) for r in truth0}
    truth={k:Pose(original[v][1]*28,original[v][0]*28,-r%4) for k,(v,r) in mapping.items()}
    m=evaluate(pieces,poses,truth,shape,allow_global_rotation=rotate,allow_translation=False)
    m.update(features(pieces),case=case.name,method=method,rotations=rotate,runtime_seconds=runtime,diagnostics=diag,
             split="development" if int(case.name.split("_")[1])%2==0 else "holdout")
    json_write(out/"summary.json",m)
    print(case.name,method,"rot" if rotate else "up",round(m["direct_position_accuracy"]*100,2),round(m["adjacency_f1"]*100,2),m["pass_95"],round(runtime,2),flush=True)
    return m

def source_from_case(case):
    # GENERATION ONLY. Not passed to solve. Legal for new synthetic cut.
    truth0=json.loads((case/"ground_truth.json").read_text())["pieces"]
    src=np.zeros((504,672,3),np.uint8)
    for r in truth0:
        a=np.array(Image.open(case/"pieces"/(r["piece_id"]+".png")).convert("RGB"))
        y,x=r["row"]*28,r["col"]*28;src[y:y+28,x:x+28]=a
    return src

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--case");p.add_argument("--phase",choices=["all","development","holdout"],default="all")
    p.add_argument("--method",choices=["v5","candidate","v6"],default="v6")
    p.add_argument("--rotations",action="store_true");p.add_argument("--params",type=json.loads)
    p.add_argument("--recut",type=int);p.add_argument("--resize-width",type=int);p.add_argument("--cut",default="grid");p.add_argument("--features-only",action="store_true")
    a=p.parse_args();cases=sorted(a.root.glob("cho_*_432"))
    if a.case:cases=[c for c in cases if c.name==a.case]
    if a.phase!="all":cases=[c for c in cases if (int(c.name.split("_")[1])%2==0)==(a.phase=="development")]
    if not cases:raise ValueError("No cases")
    for c in cases:
        folder=a.out/c.name
        if a.features_only:print(c.name,json.dumps(features(load_fragments(c/"pieces"))));continue
        if a.recut:
            source=source_from_case(c);case=folder/"case";res=folder/"result"
            if a.resize_width:
                h,w=source.shape[:2];source=np.array(Image.fromarray(source).resize((a.resize_width,round(h*a.resize_width/w)),Image.Resampling.BICUBIC))
            generate(source,case,a.recut,a.cut,911,a.rotations)
            hide=folder/"held_truth";(case/"evaluation").rename(hide)
            try:solve_case(case,res,tune=True,params=a.params)
            finally:hide.rename(case/"evaluation")
            m=evaluate_case(case,res);m.update(case=c.name,rotations=a.rotations,cut=a.cut,pieces=a.recut,**features(load_fragments(case/"pieces")))
            json_write(folder/"summary.json",m)
            print(c.name,a.recut,a.cut,a.rotations,round(m["direct_position_accuracy"]*100,2),round(m["adjacency_f1"]*100,2),m["pass_95"],flush=True)
        else:run(c,folder,a.method,a.rotations,params=a.params)
if __name__=="__main__":main()
