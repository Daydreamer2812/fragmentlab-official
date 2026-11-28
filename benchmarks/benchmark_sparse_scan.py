"""Stress metadata-free torn-piece reconstruction; truth is offline only.

Synthetic irregular partitions, anonymous filenames, independent quarter-turns,
optional missing pieces. This is not a benchmark of damaged physical scans.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.irregular.lattice import R
from shardelix.reconstruction.irregular.scan import rotation
from PIL import Image
from shardelix.workflow import run_job


def trial(count, tile=44, cut="jagged", seed=827, missing=0., browser=False, profile="balanced",angle_mode="quarter",erosion=0):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/"generated";upload=root/"upload";upload.mkdir()
        rng=np.random.default_rng(seed+1)
        side=math.ceil(math.sqrt(count))*tile
        # Distinct textured neighborhoods, not markers or a known grid.
        rgb=np.stack([gaussian_filter(rng.normal(size=(side,side)),3.) for _ in range(3)],2)
        rgb=rgb/(rgb.std(axis=(0,1))+.00001)*43+128
        rgb=np.clip(rgb+rng.normal(0,1.5,rgb.shape),0,255).astype(np.uint8)
        generate(rgb,source,count,cut,seed,rotations=True,missing=missing,angle_mode=angle_mode)
        truth_path=source/"evaluation"/"ground_truth.json"
        with zipfile.ZipFile(upload/"fragments.zip","w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted((source/"pieces").glob("*.png")):
                if erosion:
                    a=np.array(Image.open(p).convert("RGBA"))
                    mask=binary_erosion(a[...,3]>=128,iterations=erosion)
                    a[~mask]=0;a[...,3]=mask.astype(np.uint8)*255
                    Image.fromarray(a).save(p)
                z.write(p,"pieces/"+p.name)
        start=time.perf_counter()
        state=run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True,"profile":profile},
                      lambda s:print(s,flush=True),browser=browser)
        wall=time.perf_counter()-start
        diag=state["diagnostics"];out=root/"result"
        truth=json.loads(truth_path.read_text())["poses"]
        report=json.loads((out/"scan_report.json").read_text())
        poses=json.loads((out/"placements.json").read_text())
        entries={p["id"]:p for p in report["pieces"]}
        if report["common_scale"]!=1.:
            # Ground truth is in original pixels. Do not label resized
            # geometry as an exact pose failure/success using that gauge.
            return {"generated":count,"observed":len(truth),"detected":diag["piece_count"],"placed":diag["placed"],
                    "status":"resampled-input; exact original-pixel pose evaluation unavailable",
                    "common_scale":report["common_scale"],"exact_global_poses":None,
                    "collision_pixels":diag["render_collision_pixels"],"seconds":round(wall,3),
                    "solver_seconds":diag["runtime_seconds"],"scan":diag["scan"],"angle_mode":angle_mode,
                    "erosion_px":erosion,"seed":seed,"tile":tile,"cut":cut,"missing":missing,
                    "browser_limits":browser,"profile":profile}
        def truth_affine(fid):
            row=entries[fid];name=Path(row["file"]).stem
            t=truth[name];x,y=row["crop_xy"]
            with Image.open(source/"pieces"/row["file"]) as im:w,h=im.size
            if angle_mode=="continuous":
                A=rotation(np.deg2rad(t["input_ccw_deg"]))
                offset=np.array([t["x"]+t["width"]/2,t["y"]+t["height"]/2])-A@np.array([w/2,h/2])
            else:
                A=R[t["k"]];corners=np.array([[0,0],[w,0],[0,h],[w,h]])
                offset=np.array([t["x"],t["y"]])-np.min(corners@A.T,axis=0)
            return np.column_stack([A,offset+A@np.array([x,y])])
        gt_affines={fid:truth_affine(fid) for fid in entries}
        predicted=[];true=[];centers=[]
        for fid,pose in poses.items():
            G=gt_affines[fid];P=np.array(pose["input_to_canvas_affine"])
            predicted.append(P);true.append(G);centers.append(np.asarray(entries[fid]["crop_wh"])/2)
        pred=np.asarray(predicted);gt=np.asarray(true)
        # One rigid gauge for the primary component, choose most-supported
        # reference anchor after inference; never align disconnected groups.
        centers=np.asarray(centers)
        pred_center=np.einsum("nij,nj->ni",pred[:,:,:2],centers)+pred[:,:,2]
        gt_center=np.einsum("nij,nj->ni",gt[:,:,:2],centers)+gt[:,:,2]
        best=(0,0)
        for i in range(len(pred)):
            Q=gt[i,:,:2]@pred[i,:,:2].T
            t=gt_center[i]-Q@pred_center[i]
            err=np.linalg.norm(pred_center@Q.T+t-gt_center,axis=1)
            da=np.abs(np.arctan2((Q@pred[:,:,:2])[:,1,0],(Q@pred[:,:,:2])[:,0,0])-np.arctan2(gt[:,1,0],gt[:,0,0]))
            da=np.abs((da+np.pi)%(2*np.pi)-np.pi)
            score=(int(np.sum((err<=5)&(da<=np.deg2rad(3)))),int(np.sum((err<=1.e-5)&(da<=1.e-5))))
            best=max(best,score)
        comp=json.loads((out/"component_placements.json").read_text())
        def relative(groups):
            right=total=0
            for g in groups:
                pred={k:np.array(v["input_to_component_affine"]) for k,v in g["poses"].items()}
                for e in g["joins"]:
                    i,j=e["a"],e["b"];P,Q=pred[i],pred[j]
                    A,B=gt_affines[i],gt_affines[j]
                    rel=P[:,:2].T@Q[:,:2];rr=A[:,:2].T@B[:,:2]
                    u=P[:,:2].T@(Q[:,2]-P[:,2]);v=A[:,:2].T@(B[:,2]-A[:,2])
                    center=np.asarray(entries[j]["crop_wh"])/2
                    da=abs(np.arctan2((rel.T@rr)[1,0],(rel.T@rr)[0,0]))
                    right+=int(np.linalg.norm((rel-rr)@center+u-v)<=10 and da<=np.deg2rad(10))
                    total+=1
            return {"correct":right,"accepted":total}
        joins=relative(comp["components"])
        alternatives={k:{"model":comp[k]["model"],**relative(comp[k]["components"])} for k in ("alternative","other_model") if k in comp}
        return {"generated":count,"observed":len(truth),"detected":diag["piece_count"],"placed":diag["placed"],
                "correct_global_poses_5px_3deg":best[0],"exact_global_poses":best[1],
                "collision_pixels":diag["render_collision_pixels"],"seconds":round(wall,3),
                "solver_seconds":diag["runtime_seconds"],"scan":diag["scan"],
                "relative_joins_10px_10deg":joins,"alternatives":alternatives,"angle_mode":angle_mode,"erosion_px":erosion,
                "seed":seed,"tile":tile,"cut":cut,"missing":missing,"browser_limits":browser,"profile":profile}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count",type=int,default=512);p.add_argument("--tile",type=int,default=44)
    p.add_argument("--cut",choices=("jagged","wavy","jigsaw","random"),default="jagged")
    p.add_argument("--angle-mode",choices=("quarter","continuous"),default="quarter");p.add_argument("--erosion",type=int,default=0)
    p.add_argument("--missing",type=float,default=0.);p.add_argument("--seed",type=int,default=827)
    p.add_argument("--browser-limits",action="store_true");p.add_argument("--profile",default="balanced",choices=("fast","balanced","accurate"))
    p.add_argument("--out",type=Path);a=p.parse_args()
    r=trial(a.count,a.tile,a.cut,a.seed,a.missing,a.browser_limits,a.profile,a.angle_mode,a.erosion)
    text=json.dumps(r,indent=2);print(text)
    if a.out:a.out.write_text(text+"\n")
