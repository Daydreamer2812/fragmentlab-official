"""Scaling protocol. Independent source generation, blind solve, then evaluation.

Run from repository root: PYTHONPATH=src python benchmarks/benchmark_scale_v8.py ...
Elapsed solver includes the blind route; I/O and evaluation measured separately.
"""
import argparse,csv,json,os,platform,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from shardelix.reconstruction.common import load_fragments,read_source,render,save_rgba,Pose,json_write
from shardelix.reconstruction.generator import generate,dimensions
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.solver import solve
try:
    from shardelix.reconstruction.solver import _solve_v7
except ImportError:
    _solve_v7=solve  # original V7 checkout for apples-to-apples comparison
from shardelix.reconstruction.overlay import write_cut_outputs
def textured(h,w,seed):
    """Distinct, spatially correlated synthetic texture; not a real-photo claim."""
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(h,w,3)).astype(np.float32)
    gaussian_filter(a,(1.45,1.45,0),output=a)
    a*=420
    a+=127
    return a.clip(0,255).astype(np.uint8)

def run(args):
    n=args.count;cut=args.cut;r,c=dimensions(n,1,1)
    if args.shape: h,w=map(int,args.shape.split("x"))
    elif cut=="strips":h,w=384,n*2
    elif cut=="horizontal-strips":h,w=n*2,384
    else:h,w=r*args.tile,c*args.tile
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    case=root/"case";res=root/"result"
    if args.source.startswith("cho:"):
        from benchmark_cho import source_from_case
        source=source_from_case(Path(args.cho_root)/f"cho_{args.source.split(':')[1]}_432")
        source=np.asarray(Image.fromarray(source).resize((w,h),Image.Resampling.BICUBIC))
    elif args.source=="texture":source=textured(h,w,args.seed+1000)
    else:source=np.asarray(Image.fromarray(read_source(Path(args.source))).resize((w,h),Image.Resampling.LANCZOS))
    start=time.perf_counter()
    meta=generate(source,case,n,cut,args.seed,True)
    del source
    ps=load_fragments(case/"pieces");shape=(h,w);grid=tuple(meta["grid"]) if meta["grid"] else None
    generate_load=time.perf_counter()-start
    hidden=root/"held_truth";(case/"evaluation").rename(hidden)
    try:
        start=time.perf_counter()
        fn=_solve_v7 if args.method=="v7" else solve
        poses,diag=fn(ps,shape,cut,grid,True,auto_tune=True)
        elapsed=time.perf_counter()-start
        print(json.dumps({"phase":"solved","n":n,"cut":cut,"elapsed":elapsed,"placed":len(poses),"algorithm":diag["algorithm"]}),flush=True)
        start=time.perf_counter()
        rgba,labels,_=render(ps,poses,shape);save_rgba(res/"assembly.png",rgba);write_cut_outputs(res,rgba,labels)
        if max(h,w)>1600:
            with Image.open(res/"assembly_with_cuts.png") as im:
                im.thumbnail((1400,1400));im.convert("RGB").save(root/"preview.jpg",quality=88)
        else:Image.open(res/"assembly_with_cuts.png").convert("RGB").save(root/"preview.jpg",quality=88)
        del rgba,labels
        json_write(res/"placements.json",{"mode":"blind","poses":{i:p.as_dict() for i,p in poses.items()},"diagnostics":diag})
    finally:hidden.rename(case/"evaluation")
    gt={i:Pose(**p) for i,p in json.loads((case/"evaluation/ground_truth.json").read_text())["poses"].items()}
    metrics=evaluate(ps,poses,gt,shape,True)
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except ImportError:rss=None
    metrics.update(source=args.source,cut=cut,seed=args.seed,method=args.method,shape=list(shape),
                   solver_seconds=elapsed,generate_load_seconds=generate_load,
                   render_evaluate_seconds=time.perf_counter()-start,peak_process_rss_mib=rss,
                   algorithm=diag["algorithm"],diagnostics=diag,python=platform.python_version())
    json_write(root/"summary.json",metrics)
    print(json.dumps({k:metrics[k] for k in ["source","piece_count","cut","method","direct_position_accuracy","adjacency_f1","coverage","pass_95","solver_seconds","peak_process_rss_mib"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--count",type=int,required=True);p.add_argument("--source",default="texture")
    p.add_argument("--cho-root",default="../../grid_work/cho/cho_432");p.add_argument("--seed",type=int,default=911)
    p.add_argument("--tile",type=int,default=24);p.add_argument("--shape");p.add_argument("--cut",default="grid")
    p.add_argument("--method",choices=["v7","v8"],default="v8");p.add_argument("--out",required=True);run(p.parse_args())
