"""Reproducible scan simulation. Truth is kept outside the reconstruction inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from shardelix.reconstruction.irregular.scan import preprocess, reconstruct, rotation


def fixture(folder, seed, count=4, noise=1.4, jpeg=True, layout="strips", loss=0):
    rng=np.random.default_rng(seed)
    h,w=240,320
    y,x=np.indices((h,w))
    rgb=np.stack([130+48*np.sin(x*.05+y*.035+k)+33*np.cos(y*.071-x*.032+k) for k in range(3)],2).clip(25,230).astype(np.uint8)
    labels=np.zeros((h,w),int)
    for i in range(1,count):
        edge=w*i/count+13*np.sin(y[:,0]*.056+i)+8*np.sin(y[:,0]*.12+i*2)+4*np.sin(y[:,0]*.32+i)
        labels+=x>edge[:,None]
    if layout=="grid":
        if count!=4:raise ValueError("Grid fixture requires four pieces")
        labels=(x>(w/2+15*np.sin(y*.065)+9*np.sin(y*.14)+4*np.cos(y*.31))).astype(int)
        labels+=2*(y>(h/2+15*np.sin(x*.047)+8*np.sin(x*.13)+4*np.cos(x*.28)))
    truth={}
    for i in range(count):
        theta=float(rng.uniform(-np.pi,np.pi));R=rotation(theta)
        yy,xx=np.nonzero(labels==i)
        pts=np.column_stack([xx+.5,yy+.5])@R.T
        t=16-pts.min(axis=0)
        ww,hh=np.ceil(pts.max(axis=0)+t+16).astype(int)
        swap=np.array([[0,1],[1,0]])
        M=swap@R.T@swap; offset=swap@(R.T@(np.array([.5,.5])-t)-.5)
        mask=ndi.affine_transform((labels==i).astype(np.uint8),M,offset,output_shape=(hh,ww),order=0)>0
        if loss:mask=ndi.binary_erosion(mask,iterations=loss)
        content=np.stack([ndi.affine_transform(rgb[:,:,k],M,offset,output_shape=(hh,ww),order=1) for k in range(3)],axis=2)
        scan=np.full((hh,ww,3),24.,float)
        scan[mask]=content[mask]
        scan+=rng.normal(0,noise,scan.shape)
        name=f"sheet_{i:02d}."+("jpg" if jpeg else "png")
        Image.fromarray(np.uint8(scan.clip(0,255))).save(folder/name,quality=92)
        truth[name]=(R,t,mask,np.array([xx.mean()+.5,yy.mean()+.5]))
    return truth


def trial(seed,count=4,noise=1.4,layout="strips",loss=0):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);inputs=root/"scans";inputs.mkdir();out=root/"out";out.mkdir()
        truth=fixture(inputs,seed,count,noise,layout=layout,loss=loss)
        start=time.perf_counter()
        pieces,report=preprocess(sorted(inputs.iterdir()),root/"pieces")
        d=reconstruct(pieces,out,report)
        poses=json.loads((out/"placements.json").read_text())
        estimates=[];ious=[]
        for row in report["pieces"]:
            piece=next(p for p in pieces if p.id==row["id"])
            R,t,mask,center=truth[row["file"]]
            x,y=row["crop_xy"];h,w=piece.mask.shape
            pm=np.zeros_like(mask);pm[y:y+h,x:x+w]=piece.mask
            ious.append(float((pm&mask).sum()/(pm|mask).sum()))
            if piece.id in poses:
                T=np.array(poses[piece.id]["input_to_canvas_affine"])
                Q=T[:,:2]@R;u=T[:,:2]@(t-[x,y])+T[:,2]
                estimates.append((Q,u,center))
        # Choose gauge by anchor; evaluate all placed relative poses against truth.
        Q0,u0,_=estimates[0]
        errors=[];angles=[]
        for Q,u,center in estimates:
            errors.append(float(np.linalg.norm(Q0.T@(Q@center+u-u0)-center)))
            diff=Q0.T@Q
            angles.append(abs(float(np.degrees(np.arctan2(diff[1,0],diff[0,0])))))
        correct=sum(e<=5 and a<=3 for e,a in zip(errors,angles))
        return {"seed":seed,"layout":layout,"edge_erosion_px":loss,"noise_sigma":noise,"expected":count,"detected":len(pieces),"placed":d["placed"],
                "mask_iou_mean":round(float(np.mean(ious)),4),"relative_pose_correct":correct,
                "center_errors_px":[round(e,2) for e in errors],"angle_errors_deg":[round(a,2) for a in angles],
                "seconds":round(time.perf_counter()-start,3)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path);a=ap.parse_args()
    rows=[trial(seed,n,noise) for seed,n,noise in [(11,4,1.4),(42,4,2.5),(71,4,4),(123,6,2.5),(2026,6,4)]]
    rows += [trial(806,4,2.5,"grid"),trial(991,4,4,"grid",1),trial(144,6,4,"strips",1)]
    report={"definition":"Known synthetic truth; center error <=5 px and relative rotation <=3 degrees, after one global rigid gauge.","cases":rows}
    print(json.dumps(report,indent=2))
    if a.output:a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
