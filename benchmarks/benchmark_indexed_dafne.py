"""Run official DAFNE DB1/DB2 using explicit reference-assisted registration.

The original image and fragments are copied to an isolated solver job. The
solution ZIP (DB2) or annotations (DB1) is opened only after reconstruction.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, re, shutil, sys, tempfile, time, zipfile
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image
from shardelix.workflow import run_job

def evaluate(input_archive,solution_archive,prefix,shape_map,out):
    if solution_archive:
        with zipfile.ZipFile(solution_archive) as z:
            gt=z.read(next(n for n in z.namelist() if n.endswith("/fragments.txt"))).decode()
    else:
        with zipfile.ZipFile(input_archive) as z:gt=z.read(prefix+"/fragments.txt").decode()
    truth={}
    for line in gt.strip().splitlines():
        k,x,y,a=line.split()[:4];truth[int(k)]=(float(x),float(y),float(a))
    scan=json.loads((out/"scan_report.json").read_text())
    poses=json.loads((out/"placements.json").read_text())
    names={q["id"]:Path(q["file"]).stem for q in scan["pieces"]}
    err=[];angles=[];correct=0;correct5=0;wrong=[];distractors=[];seen=set()
    for key,p in poses.items():
        name=names[key];i=int(name.rsplit("_",1)[1]);seen.add(i)
        if i not in truth:distractors.append(i);continue
        h,w=shape_map[name]
        A=np.asarray(p["input_to_canvas_affine"]);R=A[:,:2]
        c=R@(np.array([w/2,h/2])-np.array(p["crop_xy"]))+A[:,2]
        x,y,a=truth[i];e=float(np.linalg.norm(c-[x,y]))
        angle=float(abs((np.degrees(np.arctan2(R[1,0],R[0,0]))+a+180)%360-180))
        err.append(e);angles.append(angle)
        if e<=10 and angle<=5:correct5+=1
        if e<=10 and angle<=10:correct+=1
        else:wrong.append({"id":i,"position_error_px":e,"angle_error_deg":angle})
    n=len(shape_map);d=n-len(truth)
    return {"input_count":n,"genuine_count":len(truth),"distractor_count":d,
            "correct_pose_count":correct,"correct_pose_count_10px_5deg":correct5,"pose_accuracy":correct/len(truth),
            "distractors_rejected":d-len(distractors),"distractors_placed":len(distractors),
            "distractor_rejection":(d-len(distractors))/d if d else None,
            "median_position_error_px":float(np.median(err)) if err else None,
            "median_angle_error_deg":float(np.median(angles)) if angles else None,
            "max_position_error_px":max(err,default=None),"max_angle_error_deg":max(angles,default=None),
            "missing_genuine":sorted(set(truth)-seen),"incorrect_genuine":wrong,"false_accepted":distractors,
            "strict_pass":correct==len(truth) and not distractors,"position_tolerance_px":10,"angle_tolerance_deg":10}

def run(input_archive,solution_archive,out,*,case=None,engine="indexed",browser=False,keep=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with zipfile.ZipFile(input_archive) as z:
        groups=collections.defaultdict(list)
        for n in z.namelist():
            if "/frag_eroded/" in n and n.lower().endswith(".png"):
                groups[n.split("/frag_eroded/")[0]].append(n)
        if case is None:prefix,ns=min(groups.items(),key=lambda q:len(q[1]))
        else:prefix,ns=next((p,n)for p,n in groups.items() if p.endswith(case))
        ref=next(n for n in z.namelist() if n.lower().endswith((".jpg",".jpeg")) and len(n.strip("/").split("/"))<=2)
        with tempfile.TemporaryDirectory(prefix="dafne_indexed_",dir=out) as tmp:
            root=Path(tmp);(root/"upload").mkdir();(root/"matching_reference").mkdir()
            reference=root/"matching_reference"/Path(ref).name
            reference.write_bytes(z.read(ref));shapes={}
            for n in ns:
                p=root/"upload"/Path(n).name;p.write_bytes(z.read(n))
                with Image.open(p) as im:shapes[p.stem]=(im.height,im.width)
            def update(s):
                stage=s.get("stage","")
                m=re.search(r"(\d+)/(\d+)",stage)
                if not m or int(m.group(1))%100==0:print(stage,flush=True)
            state=run_job(root,{"task":"blind","input_mode":"reference","reference_engine":None if engine=="auto" else engine},update,browser=browser)
            assert state["diagnostics"]["matching_reference_used"] is True
            generated=root/"result"
            # Copy the solver report before reading any independent annotations.
            for n in ("placements.json","scan_report.json","diagnostics.json"):
                shutil.copy2(generated/n,out/n)
            if keep:
                for n in ("assembly.png","assembly_with_cuts.png","cut_lines.png"):
                    shutil.copy2(generated/n,out/n)
            report=evaluate(input_archive,solution_archive,prefix,shapes,generated)
            diag=state["diagnostics"]
            report.update(case=prefix,protocol="reference-assisted; absolute pose; no alignment; no ML",
                          engine=diag["algorithm"],matching_backend=diag.get("matching_backend"),
                          runtime_seconds=diag["runtime_seconds"],coverage=diag["coverage"],
                          elapsed_seconds=time.perf_counter()-started,browser_limits=browser,
                          input_sha256=hashlib.sha256(Path(input_archive).read_bytes()).hexdigest(),
                          reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
            (out/"benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report,indent=2),flush=True)
            return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_archive",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--solution",type=Path);p.add_argument("--case");p.add_argument("--engine",choices=["auto","indexed","indexed-deep","fft"],default="indexed")
    p.add_argument("--browser-limits",action="store_true");p.add_argument("--keep",action="store_true")
    p.add_argument("--all-cases",action="store_true",help="Run every case in this archive with per-case checkpoints")
    p.add_argument("--resume",action="store_true",help="Reuse completed benchmark.json checkpoints")
    a=p.parse_args()
    if a.all_cases:
        if a.case:p.error("--all-cases cannot be combined with --case")
        with zipfile.ZipFile(a.input_archive) as z:
            cases=sorted({n.split("/frag_eroded/")[0] for n in z.namelist() if "/frag_eroded/" in n and n.lower().endswith(".png")})
        results=[]
        for case in cases:
            dest=a.out/Path(case).name
            try:
                checkpoint=dest/"benchmark.json"
                if a.resume and checkpoint.exists():
                    r=json.loads(checkpoint.read_text())
                else:
                    r=run(a.input_archive,a.solution,dest,case=case,engine=a.engine,browser=a.browser_limits,keep=a.keep)
                results.append(r)
            except Exception as exc:
                results.append({"case":case,"strict_pass":False,"error":str(exc)})
                print(f"{case}: {exc}",file=sys.stderr,flush=True)
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/"catalog_results.json").write_text(json.dumps(results,indent=2)+"\n")
        return 0 if results and all(r["strict_pass"] for r in results) else 1
    r=run(a.input_archive,a.solution,a.out,case=a.case,engine=a.engine,browser=a.browser_limits,keep=a.keep)
    return 0 if r["strict_pass"] else 1
if __name__=="__main__":raise SystemExit(main())
