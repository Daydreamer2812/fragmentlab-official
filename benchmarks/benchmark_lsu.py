"""Real LSU DocDataset. Numeric pieces only; ignore puzzle_img.png and all GT
until after prediction. Fixed dev/holdout by source document, not by strip count.
"""
from __future__ import annotations
import argparse,csv,json,re,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from shardelix.reconstruction.common import Fragment,render,save_rgba,json_write
from shardelix.reconstruction.strips.ordering import solve_document_strips,DEFAULT
from shardelix.reconstruction.overlay import write_cut_outputs

def split(name):
    if name.startswith("real"):return "physical"
    if name=="doc3_36":return "orientation"
    return "development" if int(re.match(r"doc(\d+)_",name)[1])%2==0 else "holdout"

def load_case(folder):
    files=sorted(f for f in folder.glob("*.png") if f.stem.isdigit())
    rng=np.random.default_rng(20261126)
    pieces=[];mapping={}
    for i in rng.permutation(len(files)):
        f=files[i];ident="p_"+rng.bytes(12).hex()
        pieces.append(Fragment(ident,np.array(Image.open(f).convert("RGBA"))))
        mapping[ident]=f.stem
    return pieces,mapping

def metrics(pred,truth,rotations=False):
    def score(order):
        n=len(truth);actual=set(zip(truth[:-1],truth[1:]));edges=set(zip(order[:-1],order[1:]))
        return sum(a==b for a,b in zip(order,truth))/n,len(actual&edges)/max(len(actual),1)
    direct,adj=score(pred);rev,radj=score(pred[::-1])
    if rotations and (radj,rev)>(adj,direct):return rev,radj,True
    return direct,adj,False

def run(folder,params=None,output=None,baseline=False):
    pieces,mapping=load_case(folder);part=split(folder.name)
    rotate=part=="orientation"
    start=time.perf_counter()
    # Both methods receive only anonymous pieces. No proof of pixel truth for physical/orientation.
    from shardelix.reconstruction.strips.ordering import _solve_document_v5
    solver=_solve_document_v5 if baseline else solve_document_strips
    poses,diag=solver(pieces,rotations=rotate,**(params or {}))
    elapsed=time.perf_counter()-start
    shape=(max(p.rotated(poses[p.id].k).shape[0] for p in pieces),
           sum(p.rotated(poses[p.id].k).shape[1] for p in pieces))
    rgba,labels,collision=render(pieces,poses,shape)
    if output:
        save_rgba(output/"assembly.png",rgba);write_cut_outputs(output,rgba,labels)
        json_write(output/"placements.json",{"poses":{k:v.as_dict() for k,v in poses.items()},"diagnostics":diag})
    # Evaluation starts only after poses/rendering are fixed.
    truth=(folder/"order.txt").read_text().split()
    pred=[mapping[k] for k in sorted(poses,key=lambda k:poses[k].x)]
    if set(pred)!=set(truth):raise ValueError(f"Truth ID mismatch {folder.name}")
    direct,adj,reverse=metrics(pred,truth,rotate)
    import hashlib
    from collections import Counter
    classes={}
    for p in pieces:
        classes[mapping[p.id]]=min(hashlib.sha256(np.ascontiguousarray(p.rotated(k)).tobytes()).hexdigest() for k in (range(4) if rotate else (0,)))
    tc=[classes[k] for k in truth];pc=[classes[k] for k in pred]
    def eq_score(order):
        a=Counter(zip(order[:-1],order[1:]));b=Counter(zip(tc[:-1],tc[1:]))
        return sum(x==y for x,y in zip(order,tc))/len(tc),sum((a&b).values())/max(1,len(tc)-1)
    co,ca=eq_score(pc)
    if rotate:
        ro,ra=eq_score(pc[::-1])
        if (ra,ro)>(ca,co):co,ca=ro,ra
    result=dict(case=folder.name,split=part,pieces=len(pieces),direct_order_accuracy=direct,
                equivalence_order_accuracy=co,equivalence_neighbor_accuracy=ca,exact_duplicate_fraction=1-len(set(tc))/len(tc),
                directed_neighbor_accuracy=adj,adjacency_f1=adj,reversed_global=reverse,
                runtime_seconds=elapsed,method="document" if not baseline else "boundary-l1",
                pass_95_order_and_adjacency=bool(direct>.95 and adj>.95))
    result.update(coverage=float((labels>=0).mean()),collision_pixels=collision,
                  blank_strips=diag["blank_strip_count"])
    if part in {"development","holdout"}:
        source=folder.parent.parent/"gt"/(folder.name.split("_")[0]+".png")
        if source.is_file():
            gt=np.array(Image.open(source).convert("RGB"))
            if gt.shape[:2]==shape:
                result.update(exact_pixel_fraction=float(np.all(rgba[:,:,:3]==gt,axis=2).mean()),
                              pixel_mae=float(np.abs(rgba[:,:,:3].astype(np.int16)-gt.astype(np.int16)).mean()/255))
    alternatives=[]
    for alt in diag.get("equally_scored_alternatives",[]):
        ap=[mapping[k] for k in sorted(alt["poses"],key=lambda k:alt["poses"][k]["x"])]
        ao,aa,ar=metrics(ap,truth,rotate)
        ec,ea=eq_score([classes[k] for k in ap])
        if rotate:
            rc,ra=eq_score([classes[k] for k in ap[::-1]])
            if (ra,rc)>(ea,ec):ec,ea=rc,ra
        alternatives.append({"hypothesis":alt["hypothesis"],"direct_order_accuracy":ao,"directed_neighbor_accuracy":aa,"equivalence_order_accuracy":ec,"equivalence_neighbor_accuracy":ea,"blind_energy":alt["blind_energy"]})
        if output:
            from shardelix.reconstruction.common import Pose
            pp={k:Pose(**v) for k,v in alt["poses"].items()}
            im,lb,_=render(pieces,pp,shape)
            altout=output/("alternative_"+alt["hypothesis"])
            save_rgba(altout/"assembly.png",im);write_cut_outputs(altout,im,lb)
    if alternatives:result["equally_scored_alternatives"]=alternatives
    if output:json_write(output/"metrics.json",result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--stripes-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--phase",choices=["train","evaluate"],default="evaluate")
    p.add_argument("--params",type=json.loads);p.add_argument("--previews",action="store_true")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    folders=sorted(x.parent for x in a.stripes_root.glob("*/order.txt"))
    if a.phase=="train":
        folders=[f for f in folders if split(f.name)=="development"]
        configs=[dict(metric="l1",normalization=.5,margins=0,blank_split=False),dict(metric="gradient",depth=1,dark=10,boundary=0,normalization=.5,margins=0)]
        configs += [dict(metric="gradient",depth=1,dark=w,boundary=b,normalization=q,margins=m)
                    for w,b,q,m in [(10,0,.5,4),(10,0,.5,8),(10,0,.5,16),(10,0,.8,8),(10,0,.2,8),(3,0,.5,8),(20,0,.5,8),(10,.15,.5,8)]]
        results=[]
        for config in configs:
            rows=[run(f,config) for f in folders]
            row=dict(params=config,cases=len(rows),mean_order=float(np.mean([r["direct_order_accuracy"] for r in rows])),
                     mean_adj=float(np.mean([r["directed_neighbor_accuracy"] for r in rows])))
            results.append(row);print(row,flush=True)
        results.sort(key=lambda r:(r["mean_adj"],r["mean_order"]),reverse=True)
        json_write(a.out/"training.json",results)
        return
    params=a.params or DEFAULT
    rows=[]
    for f in folders:
        for meth,cfg in [("boundary-l1",dict(metric="l1",normalization=.5,margins=0,blank_split=False)),("document",params)]:
            if split(f.name)=="physical" and meth=="document":cfg={**cfg,"shift":4}
            out=a.out/"cases"/f.name if a.previews and meth=="document" else None
            r=run(f,cfg,out,meth=="boundary-l1");rows.append(r)
            print(r["case"],meth,round(r["direct_order_accuracy"],3),round(r["directed_neighbor_accuracy"],3),flush=True)
    json_write(a.out/"summary.json",rows)
    with (a.out/"summary.csv").open("w") as fp:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
