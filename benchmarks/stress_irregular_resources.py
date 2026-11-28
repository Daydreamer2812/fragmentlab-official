"""Blind in-memory resource stress; independent synthetic torn-edge fixture.

Run one size per process for isolated peak RSS. Ground truth is used only
after inference to count exact poses under a single rigid global gauge.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.lattice import R
from shardelix.reconstruction.irregular.large_scan import lattice_candidates,merge_lattice,lattice_box
from shardelix.reconstruction.resource_budget import resource_budget,MIB


def fixture(count,tile,seed):
    rng=np.random.default_rng(seed)
    cols=math.ceil(math.sqrt(count));rows=math.ceil(count/cols)
    H,W=rows*tile,cols*tile;amp=max(2,tile//7)
    v=rng.integers(-amp,amp+1,(cols+1,H),dtype=np.int32)
    h=rng.integers(-amp,amp+1,(rows+1,W),dtype=np.int32)
    v[[0,-1]]=0;h[[0,-1]]=0
    texture=rng.integers(45,211,(H//6+2,W//6+2,3),dtype=np.uint8)
    pieces=[];truth={}
    for k in range(count):
        i,j=divmod(k,cols)
        x0=max(0,j*tile-amp);x1=min(W,(j+1)*tile+amp)
        y0=max(0,i*tile-amp);y1=min(H,(i+1)*tile+amp)
        X,Y=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1))
        mask=(X>=j*tile+v[j,Y])&(X<(j+1)*tile+v[j+1,Y])&(Y>=i*tile+h[i,X])&(Y<(i+1)*tile+h[i+1,X])
        a=X//6;b=Y//6;fx=(X%6)[...,None]/6;fy=(Y%6)[...,None]/6
        rgb=(1-fy)*((1-fx)*texture[b,a]+fx*texture[b,a+1])+fy*((1-fx)*texture[b+1,a]+fx*texture[b+1,a+1])
        rgba=np.zeros((*mask.shape,4),np.uint8);rgba[mask,:3]=rgb[mask];rgba[...,3]=mask*255
        yy,xx=np.nonzero(mask);dx,dy=int(xx.min()),int(yy.min())
        rgba=rgba[dy:yy.max()+1,dx:xx.max()+1];x0+=dx;y0+=dy
        rot=int(rng.integers(4));a=np.ascontiguousarray(np.rot90(rgba,-rot%4))
        fid=f"fragment-{int(rng.integers(2**62)):016x}";p=Fragment(fid,a)
        rr=(-rot)%4;lx,ly,_,_=lattice_box(p,(rr,0,0))
        truth[fid]=(rr,x0-lx,y0-ly)
        pieces.append(p)
    return [pieces[i] for i in rng.permutation(len(pieces))],truth,[W,H]


def run(count,tile,seed,memory_mib=None):
    budget=resource_budget(override_mib=memory_mib)
    # Conservative estimated raster area, not an arbitrary count ceiling.
    estimated=count*(tile+2*(tile//7))**2
    if estimated>budget.working_pixels:
        return {"count":count,"tile":tile,"status":"preflight-memory-budget","estimated_input_pixels":estimated,"budget":budget.describe()}
    t=time.perf_counter();pieces,truth,wh=fixture(count,tile,seed)
    prepared=time.perf_counter();print(f"Fixture {count}: {prepared-t:.2f}s, RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MiB",flush=True)
    try:
        c,stats=lattice_candidates(pieces,lambda s:print(s,flush=True),degree=16,budget=budget)
    except MemoryError as exc:
        return {"count":count,"tile":tile,"status":"resource-budget-exhausted","phase":"exact contour index",
                "reason":str(exc),"fixture_seconds":round(prepared-t,3),"elapsed_seconds":round(time.perf_counter()-t,3),
                "peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),"budget":budget.describe()}
    matched=time.perf_counter();print(f"Verified {len(c)} seams in {matched-prepared:.2f}s",flush=True)
    groups,joins,spatial=merge_lattice(pieces,c,pixel_cap=budget.canvas_pixels)
    done=time.perf_counter();print(f"Assembly {len(groups[0])}/{count} in {done-matched:.2f}s",flush=True)
    gauges={};best=0
    for gi,g in enumerate(groups):
        votes=Counter()
        for k,(P,p) in g.items():
            fid=pieces[k].id;r,x,y=truth[fid]
            Q=R[r]@P.T;u=np.array([x,y])-Q@p
            key=tuple(np.rint(np.r_[Q.ravel(),u]).astype(int))
            gauges[fid]=key;votes[key]+=1
        if gi==0:best=votes.most_common(1)[0][1]
    correct=sum(gauges[j["a"]]==gauges[j["b"]] for j in joins)
    return {"status":"completed","scope":"blind exact lattice solver, in-memory (not ZIP/PNG or physical scan)","count":count,"tile":tile,"seed":seed,"input_pixels":sum(p.rgba.shape[0]*p.rgba.shape[1] for p in pieces),"source_wh":wh,"placed":len(groups[0]),"exact_global_poses":best,"correct_relative_joins":correct,"accepted_joins":len(joins),"components":len(groups),"fixture_seconds":round(prepared-t,3),"match_seconds":round(matched-prepared,3),"merge_seconds":round(done-matched,3),"solver_seconds":round(done-prepared,3),"peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),"budget":budget.describe(),"index":stats,"merge":spatial}



def workflow(count,tile,seed,browser=False):
    """End-to-end anonymous archive import, preprocessing, PNG and ZIP export."""
    import io
    import tempfile
    import zipfile
    from PIL import Image
    from shardelix.workflow import run_job
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);upload=root/"upload";upload.mkdir()
        pieces,truth,wh=fixture(count,tile,seed)
        archive=upload/"fragments.zip"
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_STORED) as z:
            for p in pieces:
                buf=io.BytesIO();Image.fromarray(p.rgba).save(buf,format="PNG")
                z.writestr("pieces/"+p.id+".png",buf.getvalue())
        del pieces
        start=time.perf_counter()
        state=run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True,"profile":"balanced"},lambda s:print(s,flush=True),browser=browser)
        wall=time.perf_counter()-start
        d=state["diagnostics"];out=root/"result"
        report=json.loads((out/"scan_report.json").read_text())
        if report["common_scale"]!=1.:raise ValueError("Expected unchanged pixel scale for this fixture")
        poses=json.loads((out/"placements.json").read_text());gauges=Counter()
        for p in report["pieces"]:
            if p["id"] not in poses:continue
            name=Path(p["file"]).stem;r,x,y=truth[name]
            A=R[r];t=np.array([x,y])+A@np.array(p["crop_xy"])
            P=np.asarray(poses[p["id"]]["input_to_canvas_affine"])
            Q=A@P[:,:2].T;u=t-Q@P[:,2]
            gauges[tuple(np.rint(np.r_[Q.ravel(),u]).astype(int))]+=1
        return {"status":"completed","scope":"end-to-end blind ZIP import, scan normalization, exact solver and PNG/ZIP export",
                "count":count,"tile":tile,"seed":seed,"browser_limits":browser,"upload_bytes":archive.stat().st_size,
                "detected":d["piece_count"],"placed":d["placed"],"exact_global_poses":gauges.most_common(1)[0][1],
                "collision_pixels":d["render_collision_pixels"],"solver_seconds":d["runtime_seconds"],
                "workflow_seconds":round(wall,3),"peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
                "scan":d["scan"]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--count",type=int,required=True);p.add_argument("--tile",type=int,default=36);p.add_argument("--seed",type=int,default=827);p.add_argument("--memory-mib",type=int);p.add_argument("--out",type=Path);p.add_argument("--workflow",action="store_true");p.add_argument("--browser-limits",action="store_true");a=p.parse_args()
    result=workflow(a.count,a.tile,a.seed,a.browser_limits) if a.workflow else run(a.count,a.tile,a.seed,a.memory_mib);s=json.dumps(result,indent=2);print(s,flush=True)
    if a.out:a.out.write_text(s+"\n")
