"""Paired synthetic physical strip stress test on composed DocDataset crops.

Run: PYTHONPATH=src python scripts/benchmark_scan_stress.py DocDataset.zip --out result.json
The eight deterministic development cases combine paper shadows, eroded edges,
relative translations, unknown quarter turns and uneven strip widths. Only
anonymized scanned pixels enter the solver; truth is used for evaluation only.
This is not an independent held-out dataset. Original images are not redistributed.
"""
import argparse,io,json,zipfile,tempfile,time
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter,shift
from shardelix.reconstruction.strips.physical import reconstruct_strips

def create_scans(src,n,seed,severity,folder):
    rng=np.random.default_rng(seed)
    im=src.convert('L');im.thumbnail((1000,1400),Image.Resampling.LANCZOS)
    a=np.asarray(im,np.float32)/255
    h,w=a.shape
    # Anonymize independent of sorted order; only evaluator retains the ID list.
    cuts=np.round(np.linspace(0,w,n+1)).astype(int)
    for i in range(1,n):cuts[i]+=int(rng.integers(-3,4))
    ids=[];geometry={}
    for i,(x0,x1) in enumerate(zip(cuts[:-1],cuts[1:])):
        alias=rng.bytes(8).hex();ids.append(alias)
        tile=a[:,x0:x1].copy();width=tile.shape[1]
        dy=0
        if severity:
            dy=int(rng.integers(-3,4)) if severity==1 else int(rng.integers(-5,6))
            tile=shift(tile,(dy,0),
                       order=1,mode='constant',cval=1.,prefilter=False)
        y=np.arange(h)[:,None];x=np.arange(width)[None,:]
        wave=(.025 if severity==1 else .07)*np.sin(y/(70+rng.uniform(0,50))+rng.uniform(0,6))
        base=rng.uniform(.88,.99) if severity==1 else rng.uniform(.72,.98)
        side=(.06 if severity==1 else .15)*(np.exp(-x/2)+np.exp(-(width-1-x)/2))
        img=np.clip(tile*(base+wave-side)+rng.normal(0,.008 if severity==1 else .015,tile.shape),0,1)
        if severity==2:img=gaussian_filter(img,.35)
        pad=6;rgba=np.zeros((h,width+2*pad,4),np.uint8)
        for row in range(h):
            off=pad+int(round((.5 if severity==1 else 1.5)*np.sin(row/85+rng.uniform(0,.02))))
            left=int(severity==2 and (row//37+i)%7==0);right=int(severity==2 and (row//51+i)%8==0)
            k=(img[row,left:width-right]*255).astype(np.uint8)
            rgba[row,off+left:off+width-right,:3]=k[:,None]
            rgba[row,off+left:off+width-right,3]=255
        turn=int(rng.integers(0,4));rgba=np.rot90(rgba,turn)
        geometry[alias]={"rectified_half_turn":turn//2,"content_shift_y":dy}
        Image.fromarray(rgba).save(folder/(alias+'.png'))
    return ids,geometry

def evaluate(order,truth):
    edges={frozenset((a,b)) for a,b in zip(truth,truth[1:])}
    return {'correct':sum(frozenset((a,b)) in edges for a,b in zip(order,order[1:])),
            'total':len(truth)-1,'positions_aligned':max(sum(a==b for a,b in zip(order,truth)),sum(a==b for a,b in zip(order,truth[::-1])))}

def geometry_metrics(poses, order, truth, geometry):
    """One shared 180-degree gauge for order, orientation and row offsets.

    Measure known synthetic content translations, not an asserted real-scan
    physical DPI/pose accuracy. A globally inverted page is equivalent.
    """
    candidates=[]
    for gauge in (0,1):
        expected=truth if gauge==0 else truth[::-1]
        edges=set(zip(expected,expected[1:]))
        valid={key for key,p in poses.items() if p["k"]%4==2*(geometry[key]["rectified_half_turn"]^gauge)}
        joined=sum((a,b) in edges and a in valid and b in valid for a,b in zip(order,order[1:]))
        ys=np.array([poses[k]["y"]-(-1 if gauge==0 else 1)*geometry[k]["content_shift_y"] for k in sorted(valid)])
        err=ys-np.median(ys) if len(ys) else ys
        r={"orientation_correct":len(valid),"orientation_total":len(truth),"joint_directed_oriented_joins":joined,
           "joint_total_joins":len(truth)-1,"row_alignment_scored":len(ys),
           "row_alignment_mae_px":float(np.mean(abs(err))) if len(ys) else None,
           "row_alignment_rmse_px":float(np.sqrt(np.mean(err**2))) if len(ys) else None,"gauge_turns":gauge*2}
        candidates.append(((joined,len(valid),sum(a==b for a,b in zip(order,expected))),r))
    return max(candidates,key=lambda q:q[0])[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[]
    with zipfile.ZipFile(a.archive) as z,tempfile.TemporaryDirectory() as d:
        for doc,n,seed,severity in [('pageA',20,617,1),('pageA',32,619,2),('pageB',24,721,1),('pageB',36,725,2),('pageC',24,813,1),('pageC',40,819,2),('pageD',32,917,1),('pageD',48,919,2)]:
            case=Path(d)/f'{doc}-{n}';case.mkdir();scans=case/'scans';scans.mkdir()
            pages={'pageA':(13,17,21,23),'pageB':(15,7,23,13),'pageC':(21,15,17,7),'pageD':(23,17,13,21)}
            ims=[]
            for num in pages[doc]:
                with Image.open(io.BytesIO(z.read(f'DocDataset/gt/doc{num}.png'))) as source:
                    ims.append(source.convert('L'))
            page=Image.new('L',(max(im.width for im in ims),sum(im.height+12 for im in ims)),255);y=0
            for im in ims:page.paste(im,(0,y));y+=im.height+12
            truth,geometry=create_scans(page,n,seed,severity,scans)
            start=time.monotonic()
            try:
                diag=reconstruct_strips(sorted(scans.glob('*.png')),case/'pieces',case/'result',profile='balanced')
                poses=json.loads((case/'result/placements.json').read_text())['poses'];order=sorted(poses,key=lambda i:poses[i]['x'])
                result={'case':f'{doc}-{n}','seed':seed,'severity':severity,**evaluate(order,truth),**geometry_metrics(poses,order,truth,geometry),'seconds':diag['runtime_seconds'],'algorithm':diag['algorithm']}
            except Exception as exc:result={'case':f'{doc}-{n}','seed':seed,'severity':severity,'error':str(exc),'seconds':time.monotonic()-start}
            rows.append(result);print(json.dumps(result),flush=True)
    a.out.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
