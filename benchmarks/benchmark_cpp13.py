"""Paired C++ vs reference benchmark; identical generated inputs, isolated subprocesses."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
from benchmark_scale_v8 import textured
from shardelix.reconstruction.generator import generate,dimensions

def measure(case,output,profile):
    from shardelix.reconstruction.cli import solve_case,evaluate_case
    from shardelix.native_core import backend
    truth=case.parent/"held";(case/"evaluation").rename(truth)
    try:
        t=time.perf_counter();result=solve_case(case,output,profile=profile);elapsed=time.perf_counter()-t
    finally:truth.rename(case/"evaluation")
    m=evaluate_case(case,output)
    d=result["diagnostics"]
    print(json.dumps(dict(backend=backend(),solver_seconds=d["runtime_seconds"],solve_export_seconds=elapsed,placed=m["placed"],dpa=m["direct_position_accuracy"],f1=m["adjacency_f1"],collision_pixels=m["collision_pixels"],pass_95=m["pass_95"],index_query_seconds=d.get("index_query_seconds"),rerank_seconds=d.get("rerank_seconds"),assembly_seconds=d.get("assembly_seconds"),fast_rescue=d.get("fast_rescue_v13"))))
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path("outputs/benchmarks/v13/paired.json"));p.add_argument("--counts",default="4096,16384")
    p.add_argument("--measure",nargs=3);a=p.parse_args()
    if a.measure:measure(Path(a.measure[0]),Path(a.measure[1]),a.measure[2]);return
    rows=[]
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        for n in map(int,a.counts.split(",")):
            rr,cc=dimensions(n,1,1);case=root/f"grid-{n}";generate(textured(rr*20,cc*20,12226),case,n,"grid",11226,True)
            for mode in ("python","auto"):
                r=subprocess.run([sys.executable,__file__,"--measure",str(case),str(root/f"out-{n}-{mode}"),"balanced"],env={**os.environ,"SHARDELIX_CORE":mode},capture_output=True,text=True,check=True)
                row=json.loads(r.stdout.strip().splitlines()[-1]);row.update(count=n,cut="grid",tile=20,seed=11226);rows.append(row);print(json.dumps(row),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+"\n")
if __name__=="__main__":main()
