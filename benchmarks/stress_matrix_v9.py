"""Bounded per-case stress suite, with wall-clock timeout and no forced pass.

Run from repo root: PYTHONPATH=src python benchmarks/stress_matrix_v9.py
--preset scale --out outputs/stress_v9 --jobs 1 --timeout 240

Each worker owns a new directory. All partial failures/timeouts remain in the
CSV. Use one worker for comparable latency; parallel workers measure contention.
"""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from shardelix.reconstruction.generator import CUTS

def matrix(preset):
    for cut in CUTS:
        if preset=="quick":
            yield cut,64,"quarter",1,0,24 if cut not in {"strips","horizontal-strips"} else 4
            yield cut,64,"continuous",1,.1,32 if cut not in {"strips","horizontal-strips"} else 12
            if cut not in {"grid","strips","horizontal-strips"}:yield cut,64,"continuous",2,.1,32
        else:
            if cut=="grid":
                for n in (1024,4096,16384,32768,65536):yield cut,n,"quarter",1,0,20 if n==65536 else 16
                for n in (1024,4096,8192,16384):yield cut,n,"continuous",1,.1,24
                for miss in (.2,.3):yield cut,4096,"continuous",1,miss,24
            elif cut in {"strips","horizontal-strips"}:
                for n in (1024,4096,8192):yield cut,n,"quarter",1,0,2
                for n in (256,1024,4096):yield cut,n,"continuous",1,0,8 if n>=1024 else 12
                yield cut,256,"continuous",1,.1,12
            else:
                for n in (1024,2048,4096):yield cut,n,"quarter",1,0,24
                for s in (1,2):
                    for n in (256,1024,2048):yield cut,n,"continuous",s,.1,32

def main():
    p=argparse.ArgumentParser();p.add_argument("--preset",choices=("quick","scale"),default="quick")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,default=911)
    p.add_argument("--source",default="texture");p.add_argument("--cuts",nargs="+",choices=CUTS,default=list(CUTS))
    p.add_argument("--jobs",type=int,default=1);p.add_argument("--timeout",type=int,default=240)
    a=p.parse_args()
    if not 1<=a.jobs<=8 or a.timeout<1:raise ValueError("Bad jobs/timeout")
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError("Use a new output directory")
    a.out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[1]
    specs=[v for v in matrix(a.preset) if v[0] in a.cuts]
    def worker(v):
        cut,n,angle,sampling,missing,tile=v;name=f"{cut}_{n}_{angle}_s{sampling}_m{int(missing*100)}_t{tile}"
        out=a.out/name;log=a.out/(name+".log")
        cmd=[sys.executable,str(root/"benchmarks/stress_v9.py"),"--out",str(out),"--cut",cut,"--count",str(n),"--angle",angle,"--sampling",str(sampling),"--missing",str(missing),"--tile",str(tile),"--seed",str(a.seed),"--source",a.source]
        env={**os.environ,"PYTHONPATH":str(root/"src"),"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
        t=time.monotonic();status="ok"
        with log.open("wb") as f:
            try:
                proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=a.timeout)
                if proc.returncode:status=f"error-{proc.returncode}"
            except subprocess.TimeoutExpired:status="timeout"
        row=dict(case=name,count=n,cut=cut,angle_mode=angle,sampling=sampling,missing_rate=missing,tile=tile,
                 status=status,wall_seconds=time.monotonic()-t,jobs=a.jobs,source=a.source,seed=a.seed)
        if (out/"summary.json").is_file():row.update(json.loads((out/"summary.json").read_text()))
        print(name,status,row.get("direct_position_accuracy"),row.get("runtime_seconds"),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows=list(pool.map(worker,specs))
    columns=["case","cut","count","source","angle_mode","sampling","missing_rate","tile","seed","status","pass_95","direct_position_accuracy","adjacency_f1","coverage","available_piece_recovery","runtime_seconds","solve_export_seconds","wall_seconds","peak_process_rss_mib","raw_boundary_conflicts","core_conflicts","jobs"]
    with (a.out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore");w.writeheader();w.writerows(rows)
    print("Passed",sum(r.get("pass_95",False) for r in rows),"/",len(rows),"see",a.out/"summary.csv")
if __name__=="__main__":main()
