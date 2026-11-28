"""Evaluate DAFNE reference mode with independent truth opened only after solve.

Usage: python benchmarks/benchmark_reference_dafne.py INPUT.zip SOURCE.jpg TRUTH.zip OUTPUT
Input/annotation archives follow the separated Shardelix research dataset format.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from shardelix.workflow import run_job


def evaluate(input_zip, truth_zip, result):
    """Absolute reference-coordinate scoring: no alignment or oracle rescue."""
    with zipfile.ZipFile(input_zip) as z:
        shapes={}
        for name in z.namelist():
            if name.startswith("pieces/") and name.lower().endswith((".png",".jpg",".jpeg")):
                with z.open(name) as f, Image.open(f) as im:
                    shapes[Path(name).stem]=im.size
    with zipfile.ZipFile(truth_zip) as z:
        raw=json.loads(z.read("annotations/source_metadata.json"))
    truth={int(x[0]):list(map(float,x[1:])) for line in raw["ground_truth"] for x in [line.split()]}
    ids={f"fragment_{int(k):04d}":truth[v] for k,v in raw["source_id_by_piece_id"].items() if v in truth}
    placements=json.loads((result/"placements.json").read_text())
    scan=json.loads((result/"scan_report.json").read_text())
    names={q["id"]:Path(q["file"]).stem for q in scan["pieces"]}
    seen=set();errors=[];angles=[];correct=0;outliers=[];bad=[]
    for key,q in placements.items():
        name=names[key]
        if name in seen:raise ValueError("Duplicate source fragment placement")
        seen.add(name)
        if name not in ids:outliers.append(name);continue
        mat=np.asarray(q["input_to_canvas_affine"])
        w,h=shapes[name]
        center=np.array([(w-1)/2,(h-1)/2])
        R=mat[:,:2];t=mat[:,2]-R@np.asarray(q["crop_xy"])
        x,y,a=ids[name];theta=np.deg2rad(-a)
        G=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        gt_center=np.array([x,y])+G@(center-np.array([w/2,h/2]))
        err=float(np.linalg.norm(R@center+t-gt_center))
        delta=float(abs(np.rad2deg(np.angle(complex(R[0,0],R[1,0])/np.exp(1j*theta)))))
        errors.append(err);angles.append(delta)
        if err<=10 and delta<=10:correct+=1
        else:bad.append(name)
    n_outliers=len(shapes)-len(ids)
    d=json.loads((result/"diagnostics.json").read_text())
    return dict(protocol="reference-assisted; absolute pose; no alignment; no ML",
        case="DAFNE_Resurrection_102",input_count=len(shapes),genuine_count=len(ids),
        distractor_count=n_outliers,correct_pose_count=correct,
        pose_accuracy=correct/len(ids),placed=len(placements),
        distractors_placed=len(outliers),distractors_rejected=n_outliers-len(outliers),
        median_position_error_px=float(np.median(errors)) if errors else None,
        max_position_error_px=max(errors,default=None),
        median_angle_error_deg=float(np.median(angles)) if angles else None,
        max_angle_error_deg=max(angles,default=None),position_tolerance_px=10,angle_tolerance_deg=10,
        missing_genuine=sorted(set(ids)-seen),incorrect_pose=bad,outliers_placed=outliers,
        runtime_seconds=d["runtime_seconds"],coverage=d["coverage"])


def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("input_zip",type=Path);arg.add_argument("reference",type=Path)
    arg.add_argument("truth_zip",type=Path);arg.add_argument("output",type=Path)
    arg.add_argument("--browser-limits",action="store_true")
    a=arg.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dafne-match-") as tmp:
        root=Path(tmp)
        (root/"upload").mkdir();(root/"matching_reference").mkdir()
        shutil.copy2(a.input_zip,root/"upload"/a.input_zip.name)
        shutil.copy2(a.reference,root/"matching_reference"/a.reference.name)
        state=run_job(root,{"task":"blind","input_mode":"reference","reference_engine":"fft","profile":"balanced"},
                      lambda s:print(s["stage"],flush=True),browser=a.browser_limits)
        assert state["state"]=="done" and state["diagnostics"]["matching_reference_used"]
        shutil.copytree(root/"result",a.output,dirs_exist_ok=True)
    # Only now is the independent annotation ZIP read.
    report=evaluate(a.input_zip,a.truth_zip,a.output)
    report.update(browser_limits=a.browser_limits,elapsed_seconds=time.perf_counter()-started,
                  input_sha256=hashlib.sha256(a.input_zip.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
                  truth_sha256=hashlib.sha256(a.truth_zip.read_bytes()).hexdigest())
    (a.output/"benchmark_reference.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report["pose_accuracy"]==1. and not report["distractors_placed"] else 1

if __name__=="__main__":raise SystemExit(main())
