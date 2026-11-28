"""Read only original and fragment PNGs for reconstruction, then independently score.
Uses the unmodified Shardelix native reference-assisted workflow.
All downloadable archives and variants are enumerated; unsupported and pending
cases are reported separately. No DAFNE artwork is redistributed.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,collections,json,math,sys,time,zipfile,tempfile,shutil,re,urllib.request
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shardelix.workflow import run_job
def cases(z):
    groups=collections.defaultdict(list)
    for n in z.namelist():
        if '/frag_eroded/' in n and n.lower().endswith('.png'):
            groups[n.split('/frag_eroded/')[0]].append(n)
    return dict(groups)
def original(z):
    refs=[n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png')) and len(n.strip('/').split('/'))<=2 and '/frag_eroded/' not in n]
    if len(refs)!=1:raise ValueError(f'expected one top-level original, candidates {refs[:8]}')
    return refs[0]
def verify(z,pre,im_shapes,placements,scan):
    g={}
    for row in z.read(pre+'/fragments.txt').decode().splitlines():
        if row.strip():
            id,x,y,a=row.split()[:4];g[int(id)]=(float(x),float(y),float(a))
    names={q['id']:Path(q['file']).stem for q in scan['pieces']}
    good=bad=spurious=0;es=[];angles=[]
    for id,q in placements.items():
        name=names[id];i=int(name.rsplit('_',1)[1])
        if i not in g:spurious+=1;continue
        x,y,a=g[i];w,h=im_shapes[name]
        mat=np.asarray(q['input_to_canvas_affine']);R=mat[:,:2]
        p=R@(np.array([w/2,h/2])-np.array(q['crop_xy']))+mat[:,2]
        e=float(np.linalg.norm(p-[x,y]));d=float(abs(np.degrees(np.angle(complex(R[0,0],R[1,0])/np.exp(-1j*np.radians(a))))))
        es.append(e);angles.append(d)
        if e<=10 and d<=10:good+=1
        else:bad+=1
    n=len(im_shapes);outliers=n-len(g)
    return {'genuine':len(g),'distractors':outliers,'correct':good,'incorrect_genuine':bad,'false_accepted':spurious,'distractors_rejected':outliers-spurious,'pose_accuracy':good/len(g),'median_position_error_px':float(np.median(es)) if es else None,'median_angle_error_deg':float(np.median(angles)) if angles else None}
def run(item):
    zp,pre,root=item
    started=time.perf_counter()
    out=Path(root)/('case_'+re.sub(r'[^\w.-]','_',pre.rsplit('/',1)[-1])+'.json')
    if out.exists():return json.loads(out.read_text())
    row={'archive':Path(zp).name,'case':pre.rsplit('/',1)[-1],'protocol':'reference-assisted, no ML','engine':'reference-assisted-masked-se2'}
    try:
        with zipfile.ZipFile(zp) as z:
            ns=cases(z)[pre];row['input_count']=len(ns)
            ref=original(z)
            with z.open(ref) as f,Image.open(f) as im:row['source_wh']=list(im.size)
            if len(ns)>256 or math.prod(row['source_wh'])>2_000_000:
                raise ValueError('Current reference solver limit: 256 pieces / 2 megapixel source')
            with tempfile.TemporaryDirectory(prefix='dafne_case_',dir=root) as t:
                d=Path(t);(d/'upload').mkdir();(d/'matching_reference').mkdir()
                (d/'matching_reference'/Path(ref).name).write_bytes(z.read(ref))
                im_shapes={}
                for n in ns:
                    p=d/'upload'/Path(n).name;p.write_bytes(z.read(n))
                    with Image.open(p) as im:im_shapes[p.stem]=im.size
                state=run_job(d,{'task':'blind','input_mode':'reference','reference_engine':'fft','profile':'balanced'})
                # No ground-truth text has been read before solve finished.
                diag=state['diagnostics'];r=d/'result'
                assert diag.get('matching_reference_used') is True
                row.update(status='completed',placed=diag['placed'],runtime_seconds=diag['runtime_seconds'],coverage=diag['coverage'])
                row.update(verify(z,pre,im_shapes,json.loads((r/'placements.json').read_text()),json.loads((r/'scan_report.json').read_text())))
    except Exception as e:row.update(status='error',error=str(e))
    row['wall_seconds']=time.perf_counter()-started
    out.write_text(json.dumps(row,indent=2)+'\n')
    return row
def census(z):
    """Inspect only image dimensions and archive membership (not annotations)."""
    ref=original(z)
    with z.open(ref) as f, Image.open(f) as im:wh=im.size
    rows=[]
    for pre,ns in cases(z).items():
        reasons=[];total=0;max_side=0
        if len(ns)>256:reasons.append("fragment_count>256")
        if math.prod(wh)>2_000_000:reasons.append("reference_pixels>2000000")
        if not reasons:
            for n in ns:
                with z.open(n) as f, Image.open(f) as im:
                    w,h=im.size;total+=w*h;max_side=max(max_side,w,h)
            if total>4_000_000 or max_side>1600:
                reasons.append("preprocessing_rescales_fragments")
        rows.append(dict(prefix=pre,case=pre.rsplit("/",1)[-1],input_count=len(ns),
                         source_wh=list(wh),raw_pixels=total if not reasons or total else None,
                         preflight="unsupported" if reasons else "eligible",reasons=reasons))
    return rows

def fetch(url,path):
    tmp=path.with_suffix(".part")
    try:
        with urllib.request.urlopen(url,timeout=180) as r,tmp.open("wb") as f:shutil.copyfileobj(r,f,1024*1024)
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-dir",type=Path,required=True,help="Official input ZIPs, named by catalog ID (e.g. 24.zip)")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--download",action="store_true",help="Download missing eligible DB1 archives; research/educational use only")
    p.add_argument("--ids",type=int,nargs="+",help="Restrict to selected official catalog IDs")
    p.add_argument("--max-cases-per-archive",type=int,default=None,help="Optional smoke-test limit; omit for a complete run")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);a.archive_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(__file__).resolve().parents[1]/"reports/dafne_official_catalog.json").read_text())
    summary=[]
    for archive in manifest["archives"]:
        i=archive["id"]
        if a.ids is not None and i not in a.ids:continue
        record={"id":i,"name":archive["name"],"db":archive["db"],"url":archive["url"]}
        path=a.archive_dir/f"{i:02d}.zip"
        if not path.exists() and archive["catalog_limit"]:
            record.update(status="not_run_catalog_limit",reason="min count >256 or reference >2 MP")
        else:
            try:
                if not path.exists() and a.download:fetch(archive["url"],path)
                if not path.exists():
                    record.update(status="not_run_archive_missing")
                else:
                    dest=a.out/f"{i:02d}";dest.mkdir(exist_ok=True)
                    with zipfile.ZipFile(path) as z:rows=census(z)
                    (dest/"census.json").write_text(json.dumps(rows,indent=2))
                    todo=sorted([r for r in rows if r["preflight"]=="eligible"],key=lambda r:r["input_count"])
                    if a.max_cases_per_archive is not None:todo=todo[:a.max_cases_per_archive]
                    with ProcessPoolExecutor(max_workers=a.workers) as pool:
                        for f in as_completed([pool.submit(run,(str(path),r["prefix"],str(dest))) for r in todo]):
                            r=f.result();print(i,r["case"],r["status"],r.get("correct"),r.get("genuine"),r.get("false_accepted"),flush=True)
                    record.update(status="inventoried",total_cases=len(rows),eligible_cases=sum(r["preflight"]=="eligible" for r in rows),
                                  solved_cases=sum(json.loads(q.read_text())["status"]=="completed" for q in dest.glob("case_*.json")),
                                  failures=sum(json.loads(q.read_text())["status"]=="error" for q in dest.glob("case_*.json")))
            except Exception as e:record.update(status="archive_error",error=str(e))
        summary.append(record)
        (a.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
        print("ARCHIVE",record,flush=True)
if __name__=="__main__":main()
