"""Reproducible, isolated stress runner. Truth is moved out before blind solve."""
from __future__ import annotations
import argparse,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark_scale_v8 import textured
from shardelix.reconstruction.generator import generate,dimensions
from shardelix.reconstruction.cli import solve_case,evaluate_case
from shardelix.reconstruction.common import json_write,read_source
def run(a):
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    n=a.count;r,c=dimensions(n,1,1)
    if a.cut=="strips":h,w=(160 if a.angle!="quarter" else 384),n*a.tile
    elif a.cut=="horizontal-strips":h,w=n*a.tile,(160 if a.angle!="quarter" else 384)
    else:h,w=r*a.tile,c*a.tile
    case=root/"case";out=root/"result"
    t=time.perf_counter()
    if a.reuse_input:
        m=json.loads((case/"case.json").read_text())
        if (m["height"],m["width"])!=(h,w) or m.get("expected_count",m["piece_count"])!=n or m["cut"]!=a.cut:
            raise ValueError("Existing generated input differs from requested public parameters")
        if out.exists() and any(out.iterdir()):raise FileExistsError("Reuse INPUT only, not an existing result")
    else:
        if a.source=="texture":src=textured(h,w,a.seed+1000)
        else:
            if a.source.startswith("cho:"):
                from benchmark_cho import source_from_case
                src=source_from_case(Path(a.cho_root)/f"cho_{a.source.split(':')[1]}_432")
            else:src=read_source(Path(a.source))
            src=np.array(Image.fromarray(src).resize((w,h),Image.Resampling.LANCZOS))
        generate(src,case,n,a.cut,a.seed,angle_mode=a.angle,missing=a.missing,sampling=a.sampling)
        del src
    gen=time.perf_counter()-t;held=root/"truth-held-out";(case/"evaluation").rename(held)
    try:
        t=time.perf_counter();result=solve_case(case,out);elapsed=time.perf_counter()-t
    finally:held.rename(case/"evaluation")
    t=time.perf_counter();m=evaluate_case(case,out);evaluation=time.perf_counter()-t
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1048576 if sys.platform=="darwin" else 1024)
    except ImportError:rss=None
    m.update(source=a.source,count=n,cut=a.cut,angle_mode=a.angle,sampling=a.sampling,missing_rate=a.missing,
             generation_seconds=gen,solve_export_seconds=elapsed,evaluation_seconds=evaluation,
             peak_process_rss_mib=rss,shape=[h,w],tile=a.tile,
             algorithm=result["diagnostics"]["algorithm"],normalization_seconds=result["diagnostics"].get("normalization_seconds",0))
    json_write(root/"summary.json",m)
    with Image.open(out/"assembly_with_cuts.png") as im:
        im.thumbnail((1200,1200));im.convert("RGB").save(root/"preview.jpg",quality=90)
    print(json.dumps(m),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--count",type=int,required=True);p.add_argument("--cut",required=True)
    p.add_argument("--angle",default="quarter",choices=["none","quarter","continuous"]);p.add_argument("--sampling",type=int,default=1)
    p.add_argument("--missing",type=float,default=0);p.add_argument("--tile",type=int,default=24)
    p.add_argument("--seed",type=int,default=911);p.add_argument("--source",default="texture");p.add_argument("--cho-root",default="../../grid_work/cho/cho_432")
    p.add_argument("--reuse-input",action="store_true",help="Reuse existing generated input, always solve fresh");p.add_argument("--out",required=True);run(p.parse_args())
