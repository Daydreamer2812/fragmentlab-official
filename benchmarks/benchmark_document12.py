"""Unknown-width continuous digital strips: shape/pose and strict order test."""
import numpy as np,tempfile,json,csv
from PIL import Image
from pathlib import Path
from shardelix.reconstruction.rectification import rotate_raster
from shardelix.reconstruction.cli import solve_external_document
from shardelix.reconstruction.common import Pose,load_fragments
from shardelix.reconstruction.strips.rectification import normalize_document_strips
from shardelix.reconstruction.evaluation import canvas_align
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for source in ('document','texture'):
    for seed in (501,502):
        r=np.random.default_rng(seed);widths=r.integers(9,27,size=12);H=320;W=int(widths.sum())
        if source=='document':a=np.array(Image.open(ROOT/'data/examples/text_document.png').convert('RGB').resize((W,H)))
        else:a=np.clip(127+550*gaussian_filter(r.normal(size=(H,W,3)),(1.45,1.45,0)),0,255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'pieces';f.mkdir();truth={};x=0
            for j,w in enumerate(widths):
                t=float(r.uniform(-180,180));ident=r.bytes(9).hex();z=np.full((H,w,4),255,np.uint8);z[:,:,:3]=a[:,x:x+w]
                Image.fromarray(rotate_raster(z,t)).save(f/(ident+'.png'));truth[ident]=(x,t,int(w));x+=int(w)
            result=solve_external_document(f,root/'out',angle_mode='continuous')
            pieces=load_fragments(f);pp,angles,_,d=normalize_document_strips(pieces)
            gt={i:Pose(x,0,int(round((t+angles[i])/90))%4) for i,(x,t,w) in truth.items()}
            ps={i:Pose(**q) for i,q in result['poses'].items()};aligned=canvas_align(pp,ps,gt,(H,W),True)
            score=sum(aligned.get(i)==p for i,p in gt.items())/len(gt)
            errors=[abs(((t+angles[i]+90)%180)-90) for i,(x,t,w) in truth.items()]
            row=dict(source=source,seed=seed,pieces=12,height=H,width=W,estimated_height=d['estimated_strip_height'],
                     estimated_width=sum(d['estimated_strip_widths']),width_accuracy=sum(p.rgba.shape[1]==truth[p.id][2] for p in pp)/len(pp),
                     angle_mae_deg=float(np.mean(errors)),direct_position_accuracy=score,
                     collision_pixels=result['diagnostics']['collision_pixels'],runtime_seconds=result['diagnostics']['runtime_seconds'])
            rows.append(row);print(json.dumps(row),flush=True)
out=ROOT/'outputs/benchmarks/v12/document_continuous.csv';out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
