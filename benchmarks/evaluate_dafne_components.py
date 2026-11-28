"""Evaluate a DAFNE archive after blind reconstruction.

Usage:
  python benchmarks/evaluate_dafne_components.py INPUT.zip TRUTH.zip RESULT_DIR
TRUTH.zip must contain annotations/source_metadata.json in the separate
dataset-pack format. It is NEVER read by the reconstruction engine.
RESULT_DIR contains component_placements.json, placements.json, scan_report.json.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def affine(data):
    a=np.asarray(data,float)
    return complex((a[0,0]+a[1,1])/2,(a[1,0]-a[0,1])/2),complex(a[0,2],a[1,2])


def evaluate(input_zip,truth_zip,result):
    dims={}
    with zipfile.ZipFile(input_zip) as z:
        for path in z.namelist():
            if Path(path).stem.startswith("fragment_") and path.lower().endswith((".png",".jpg",".jpeg")):
                name=Path(path).stem
                if name in dims:raise ValueError(f"Duplicate fragment ID in input ZIP: {name}")
                with Image.open(io.BytesIO(z.read(path))) as im:dims[name]=im.size
    with zipfile.ZipFile(truth_zip) as z:
        annotation=json.loads(z.read("annotations/source_metadata.json"))
    locations={int(p[0]):list(map(float,p[1:]))for line in annotation["ground_truth"]for p in [line.split()]}
    gt={}
    for k,v in annotation["source_id_by_piece_id"].items():
        if v not in locations:continue
        name=f"fragment_{int(k):04d}"
        if name not in dims:continue
        x,y,a=locations[v];w,h=dims[name];r=np.exp(-1j*np.deg2rad(a))
        gt[name]=(r,complex(x,y)-r*complex(w/2,h/2))
    if not gt:raise ValueError("No matching truth IDs found in input archive; check dataset annotation and fragment filenames")
    report=json.loads((result/"scan_report.json").read_text())
    comp=json.loads((result/"component_placements.json").read_text())
    primary=json.loads((result/"placements.json").read_text())
    piece_map={p["id"]:p for p in report["pieces"]};scale=report["common_scale"]
    def poses(raw,key):
        output={}
        for id,p in raw.items():
            name=Path(piece_map[id]["file"]).stem
            r,t=affine(p[key]);x,y=p.get("crop_xy",piece_map[id]["crop_xy"])
            output[id]=(name,r*scale,t-r*complex(x,y))
        return output
    def align(pred):
        names=[(name,r,t)for name,r,t in pred.values()if name in gt]
        if not names:return {"pose_correct":0,"gt_pose_count":len(gt),"outliers_placed":len(pred)}
        center=np.array([complex((dims[n][0]-1)/2,(dims[n][1]-1)/2)for n,_,_ in names])
        r=np.array([r for _,r,_ in names]);t=np.array([t for _,_,t in names])
        G=np.array([gt[n][0]for n,_,_ in names]);u=np.array([gt[n][1]for n,_,_ in names])
        prediction=r*center+t;target=G*center+u
        best=None
        for i in range(len(names)):
            q=G[i]/r[i];shift=target[i]-q*prediction[i]
            d=abs(q*prediction+shift-target);a=abs(np.angle(q*r/G,deg=True))
            count=int(np.count_nonzero((d<=10)&(a<=10)))
            score=(count,int(np.count_nonzero((d<=1)&(a<=2))),-float(np.median(d)))
            if best is None or score>best[0]:best=(score,d,a)
        score,d,a=best
        return {"pose_correct":score[0],"gt_pose_count":len(gt),"genuine_placed":len(names),
                "outliers_placed":len(pred)-len(names),"median_position_error_px":float(np.median(d)),
                "median_angle_error_deg":float(np.median(a))}
    def components(model):
        joins=correct=outliers=nontrivial=0;groups=[]
        for group in model["components"]:
            pred=poses(group["poses"],"input_to_component_affine")
            if len(pred)>1:
                nontrivial+=len(pred);groups.append({"size":len(pred),"correct_global_gauge_poses":align(pred)["pose_correct"]})
            for e in group["joins"]:
                joins+=1
                a,A,u=pred[e["a"]];b,B,v=pred[e["b"]]
                if a not in gt or b not in gt:outliers+=1;continue
                GA,gu=gt[a];GB,gv=gt[b]
                r,t=B/A,(v-u)/A;G,g=GB/GA,(gv-gu)/GA
                w,h=dims[b];center=complex((w-1)/2,(h-1)/2)
                correct+=int(abs((r-G)*center+t-g)<=10 and abs(np.angle(r/G,deg=True))<=10)
        return {"model":model["model"],"accepted_relative_joins":joins,"correct_relative_joins":correct,
                "outlier_joins":outliers,"relative_join_precision":correct/joins if joins else None,
                "nontrivial_pieces":nontrivial,"nontrivial_groups":len(groups),"groups":groups}
    result={"primary":align(poses(primary,"input_to_canvas_affine")),"components":components(comp)}
    for k in ("alternative","other_model"):
        if k in comp:result[k]=components(comp[k])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_zip",type=Path);p.add_argument("truth_zip",type=Path);p.add_argument("result",type=Path)
    a=p.parse_args();print(json.dumps(evaluate(a.input_zip,a.truth_zip,a.result),indent=2))
