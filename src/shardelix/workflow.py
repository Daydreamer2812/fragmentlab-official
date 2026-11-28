"""Shared native/WebAssembly workflow: generate, dispatch, evaluate and export."""
from __future__ import annotations
import csv,itertools,json,shutil,time,zipfile,hashlib,math
from pathlib import Path
from shardelix.reconstruction.limits import GRID_MAX,MAX_PIXELS,RECTANGULAR,max_count,validate_constraints

from shardelix.uploads import (IMAGE_EXT, MAX_UPLOAD, WEB_UPLOAD, WEB_COUNT, WEB_PIXELS, WEB_INPUT_PIXELS,
                      native_upload_limit, safe_zip, public_metadata, inspect_uploads,
                      _check_preview, _pixel_budget, _strip_hint, _suggest_grid)

WEB_BATCH=24
WEB_BATCH_OUTPUT_BYTES=192*1024**2
NATIVE_BATCH=64
DOWNLOADS={"assembly.png","assembly_with_cuts.png","assembly_reference_aligned.png","assembly_reference_aligned_with_cuts.png","cut_lines.png","placements.json","unanchored_components.json","component_placements.json","strip_components.json","ambiguity.json","diagnostics.json","metrics.json","source.png","result.zip","fragments.zip","benchmark.csv","benchmark.json","fragments_preview.png","ground_truth.zip","comparison.json","scan_report.json"}

def profiles(conf):
    profile=conf.get("profile")
    if profile not in (None,"fast","balanced","accurate","custom"):
        raise ValueError("Unknown performance profile")
    tune=bool(conf.get("tune",True)) if profile in (None,"custom") else profile!="fast"
    idx=conf.get("index_mode","auto") if profile in (None,"custom") else "auto"
    strategy=conf.get("strategy","auto") if profile in (None,"custom") else "auto"
    if idx not in {"auto","legacy","midpoint","mixed"} or strategy not in {"auto","sparse","classic"}:
        raise ValueError("Invalid custom solver parameters")
    params={}
    if strategy=="classic":raise ValueError("Classic strategy is available through the Python API, not this workflow")
    if strategy=="sparse" or idx!="auto":params["strategy"]="sparse"
    if idx!="auto":params["index_mode"]=idx
    return profile,tune,params

def _generate_case(root,files,conf,update,browser=False,source_preview=None):
    from shardelix.reconstruction.common import read_source,save_rgba
    from shardelix.reconstruction.generator import generate,fit_generation_canvas
    from PIL import Image
    import numpy as np
    case=root/"case";out=root/"result";out.mkdir(parents=True,exist_ok=True)
    angle=conf.get("angle_mode","quarter" if conf.get("rotations",True) else "none")
    rotations=angle!="none";sampling=int(conf.get("sampling",1));missing=float(conf.get("missing",0))
    if angle not in ("none","quarter","continuous"):raise ValueError("Invalid rotation mode")
    if sampling not in (1,2,3,4) or (sampling!=1 and angle!="continuous"):raise ValueError("Invalid sampling")
    count_cap=WEB_COUNT if browser else GRID_MAX
    pixel_cap=WEB_PIXELS if browser else MAX_PIXELS
    if len(files)!=1:raise ValueError("Select one source image or PDF")
    cut=conf["cut"];n=int(conf["count"])
    if not 1<=n<=min(max_count(cut),count_cap):raise ValueError("Piece count exceeds backend limit")
    page=int(conf.get("page",1));max_side=int(conf.get("max_side",2048))
    if not (max_side==0 or 128<=max_side<=8192):raise ValueError("Resolution must be original or 128–8192")
    if browser and files[0].suffix.lower()==".pdf":raise ValueError("Use an image or convert PDF page to PNG for browser mode")
    if files[0].suffix.lower()!=".pdf":_pixel_budget(files,WEB_INPUT_PIXELS if browser else MAX_PIXELS)
    src=read_source(files[0],max_side,page)
    initial=src.shape[:2]
    if browser and angle=="continuous":
        pixel_cap=min(pixel_cap,WEB_INPUT_PIXELS//(2*sampling*sampling))
    h,w,grid=fit_generation_canvas(initial,n,cut,pixel_cap,n>1024 or angle=="continuous" or bool(missing))
    resized=(h,w)!=initial
    if resized:
        update(f"Auto-fitting canvas: {initial[1]} × {initial[0]} → {w} × {h}")
        src=np.asarray(Image.fromarray(src).resize((w,h),Image.Resampling.LANCZOS))
    if grid and min(h//grid[0],w//grid[1])<2:
        update("Generating thin cells (limited texture evidence)")
    validate_constraints(n,(h,w),cut,grid,rotations)
    save_rgba(source_preview if source_preview is not None else out/"source.png",src)
    update("Generating shuffled fragments")
    generate(src,case,n,cut,int(conf.get("seed",42)),rotations,angle_mode=angle,missing=missing,sampling=sampling,grid=grid)
    del src
    meta=json.loads((case/"case.json").read_text())
    return meta,case,out,{"preferred_size":[initial[1],initial[0]],"auto_fitted":resized,"grid":list(grid) if grid else None,"thin_cells":bool(grid and min(h//grid[0],w//grid[1])<2)}

def _public_archive(case,out):
    with zipfile.ZipFile(out/"fragments.zip","w",zipfile.ZIP_STORED) as z:
        if (case/"case.json").exists():z.write(case/"case.json","case.json")
        for p in sorted((case/"pieces").iterdir()):z.write(p,"pieces/"+p.name)

def _generate_only(root,files,conf,notify,browser=False):
    from PIL import Image,ImageDraw
    from shardelix.reconstruction.common import json_write
    started=time.perf_counter()
    update=lambda stage:notify({"state":"running","stage":stage,"case":1,"total_cases":1,"elapsed_seconds":time.perf_counter()-started})
    meta,case,out,fit_info=_generate_case(root,files,conf,update,browser)
    update("Packaging fragments and separate ground truth")
    _public_archive(case,out)
    with zipfile.ZipFile(out/"ground_truth.zip","w",zipfile.ZIP_DEFLATED) as z:
        z.write(out/"source.png","source.png")
        z.writestr("evaluation/fragment_manifest.json",json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((case/"pieces").iterdir())},separators=(",",":")))
        for f in sorted((case/"evaluation").iterdir()):
            if f.name in ("ground_truth.json","truth_labels.npz"):z.write(f,"evaluation/"+f.name)
    # Contact sheet uses generated shuffled fragments only, bounded preview size.
    paths=sorted((case/"pieces").iterdir());n=len(paths)
    show=min(96,n);cols=min(8,show);rows=(show+cols-1)//cols
    tile=128;sheet=Image.new("RGB",(cols*tile,rows*tile),(240,242,239));draw=ImageDraw.Draw(sheet)
    for j,f in enumerate(paths[:show]):
        with Image.open(f) as im:
            im=im.convert("RGBA");im.thumbnail((tile-14,tile-30))
            x=(j%cols)*tile+(tile-im.width)//2;y=(j//cols)*tile+(tile-25-im.height)//2
            sheet.paste(im,(x,y),im);draw.text(((j%cols)*tile+8,(j//cols)*tile+tile-21),f"{j+1:03d}",fill=(75,85,80))
    sheet.save(out/"fragments_preview.png",optimize=True)
    shutil.rmtree(case)
    return {"state":"done","stage":"Fragments generated","generation":{"observed":n,"expected":meta.get("expected_count",n),"width":meta["width"],"height":meta["height"],"cut":meta["cut"],"angle_mode":meta.get("angle_mode","quarter" if meta["rotations"] else "none"),"preview_count":show,"seconds":time.perf_counter()-started,**fit_info},"files":sorted(p.name for p in out.iterdir() if p.name in DOWNLOADS)}

def _evaluate_external(case,out,meta,truth_dir,browser,allow_rotation):
    """Called only AFTER solve and render; truth never enters solver inputs."""
    from PIL import Image
    import numpy as np
    from shardelix.reconstruction.common import json_write
    from shardelix.reconstruction.case_io import evaluate_case
    files=list(truth_dir.iterdir()) if truth_dir.exists() else []
    if not files:return
    if len(files)!=1:raise ValueError("Select one reference image or ground-truth ZIP")
    f=files[0];reference=None;warning=None;pose_evaluated=False;pose_rotation=None
    if f.suffix.lower()==".zip":
        allowed={"source.png","evaluation/ground_truth.json","evaluation/truth_labels.npz","evaluation/fragment_manifest.json"}
        limit=WEB_PIXELS if browser else MAX_PIXELS
        with zipfile.ZipFile(f) as z:
            seen=set()
            for e in z.infolist():
                if e.is_dir():continue
                if e.filename not in allowed or e.filename in seen:raise ValueError("Ground-truth ZIP must contain only the generated evaluation files")
                seen.add(e.filename)
                if e.file_size>limit*8+1024*1024:raise ValueError("Ground-truth entry too large")
                p=(out/"source.png") if e.filename=="source.png" else case/e.filename
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(e) as src,p.open("wb") as dst:shutil.copyfileobj(src,dst,1024*1024)
        reference=out/"source.png" if (out/"source.png").exists() else None
        # Verify the pose archive belongs to these fragment IDs and canvas; no silent partial scoring.
        if meta is not None and (case/"evaluation/ground_truth.json").exists():
            t=json.loads((case/"evaluation/ground_truth.json").read_text())
            poses=t.get("poses",{})
            ids={p.stem for p in (case/"pieces").iterdir()}
            manifest_path=case/"evaluation/fragment_manifest.json"
            manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((case/"pieces").iterdir())}
            if (not isinstance(poses,dict) or set(poses)!=ids or t.get("case",{}).get("height")!=meta["height"]
                    or t.get("case",{}).get("width")!=meta["width"] or manifest!=actual):
                warning="Ground-truth manifest does not match the imported fragment bytes and dimensions."
            else:
                try:
                    if "expected_count" in meta:
                        labels=case/"evaluation/truth_labels.npz"
                        with zipfile.ZipFile(labels) as z:
                            if len(z.infolist())!=1 or z.infolist()[0].file_size>limit*8+65536:raise ValueError("Invalid label archive")
                        with np.load(labels,allow_pickle=False) as z:
                            if z["labels"].shape!=(meta["height"],meta["width"]):raise ValueError("Truth canvas mismatch")
                    metrics=evaluate_case(case,out);pose_evaluated=True;pose_rotation=metrics.get("evaluation_rotation_cw_deg")
                except (ValueError,KeyError,TypeError,OSError) as exc:warning="Pose evaluation unavailable: "+str(exc)[:180]
        else:warning="No compatible per-piece ground truth; image comparison only."
    elif f.suffix.lower() in IMAGE_EXT:
        reference=f
    else:raise ValueError("Ground truth must be an image or generated truth ZIP")
    comparison={"pose_evaluated":pose_evaluated,"warning":warning}
    if reference:
        from shardelix.reconstruction.reference.evaluation import compare_reference
        measured=compare_reference(out,reference,allow_rotation=allow_rotation,
                                   pose_rotation=pose_rotation,pixel_cap=WEB_INPUT_PIXELS if browser else MAX_PIXELS)
        comparison.update(measured)
        if warning:comparison["warning"]=" ".join(filter(None,[warning,measured.get("warning")]))
    json_write(out/"comparison.json",comparison)

def _scan_case(root,files,conf,update,browser=False,truth_dir=None):
    from shardelix.reconstruction.irregular.scan import preprocess,reconstruct,MAX_SCAN_PIECES,WEB_SCAN_PIECES
    from shardelix.reconstruction.common import json_write
    from shardelix.native_core import backend
    from shardelix.reconstruction.resource_budget import resource_budget
    memory_budget=resource_budget(browser=browser)
    started=time.perf_counter()
    case=root/"case";out=root/"result";out.mkdir(parents=True,exist_ok=True)
    piece_rasters=False
    if len(files)==1 and files[0].suffix.lower()==".zip":
        _,_,_,layout=safe_zip(files[0],case,browser=browser,scan=True,extract_masks=True,details=True)
        piece_rasters=layout["piece_rasters"]
        (case/"pieces").rename(case/"scans")
        paths=sorted((case/"scans").iterdir())
    else:
        if any(p.suffix.lower() not in IMAGE_EXT for p in files):raise ValueError("Scan mode accepts images or one ZIP")
        paths=files
    hint=_strip_hint([lambda p=p:p.open("rb") for p in paths],len(paths),browser)
    if hint:
        from shardelix.reconstruction.strips.physical import reconstruct_strips,NotDocumentStrips
        update("Rectifying scanned paper strips")
        try:
            d=reconstruct_strips(paths,case/"pieces",out,update,pixel_cap=WEB_PIXELS if browser else MAX_PIXELS,profile=conf.get("profile") or "balanced")
        except NotDocumentStrips:
            update("Strip silhouette inconsistent; using irregular scan processing")
        else:
            _public_archive(case,out)
            d.update(compute_backend=backend(),workflow_elapsed_seconds=time.perf_counter()-started,performance_profile="scan-document-auto",
                     input_detection={"route":"scan-strips","evidence":conf.get("input_evidence",{}),"warning":conf.get("metadata_warning")})
            json_write(out/"diagnostics.json",d)
            if truth_dir is not None and truth_dir.exists() and any(truth_dir.iterdir()):
                json_write(out/"comparison.json",{"pose_evaluated":False,"warning":"Scan strip poses are relative to rectified rasters, not the original digital ground truth."})
            return d,None,out
    update("Segmenting paper from scan backgrounds")
    pieces,report=preprocess(paths,case/"pieces",pixel_limit=WEB_INPUT_PIXELS if browser else MAX_PIXELS*3,
                             preserve_opaque=conf.get("auto_preserve_opaque",False),piece_rasters=piece_rasters,
                             max_pieces=WEB_SCAN_PIECES if browser else MAX_SCAN_PIECES,
                             work_pixels=min(memory_budget.working_pixels,4_000_000 if len(paths)<=256 else WEB_INPUT_PIXELS if browser else MAX_PIXELS*3))
    d=reconstruct(pieces,out,report,update,pixel_cap=WEB_PIXELS if browser else min(MAX_PIXELS*3,memory_budget.canvas_pixels),
                  gap_recovery=conf.get("gap_recovery",True),deep_recovery=conf.get("deep_recovery","always" if conf.get("profile")=="accurate" else conf.get("profile")!="fast"),
                  profile=conf.get("profile") or "balanced",memory_budget=memory_budget)
    _public_archive(case,out)
    from shardelix.reconstruction.resource_budget import peak_rss_mib
    peak_rss=peak_rss_mib()
    d.update(compute_backend=backend(),workflow_elapsed_seconds=time.perf_counter()-started,performance_profile="scan-auto",
             process_peak_rss_mib=peak_rss,
             input_detection={"route":"irregular","evidence":conf.get("input_evidence",{}),"warning":conf.get("metadata_warning")})
    json_write(out/"diagnostics.json",d)
    if truth_dir is not None and truth_dir.exists() and any(truth_dir.iterdir()):
        json_write(out/"comparison.json",{"pose_evaluated":False,"warning":"Scan mode does not evaluate digital ground truth: arbitrary rotation, scale and torn-edge loss require a separate physical benchmark."})
    return d,None,out

def _reference_case(root,files,conf,update,browser=False,matching_reference_dir=None):
    """Opt-in matching reference, isolated from the evaluation-only truth slot."""
    from shardelix.reconstruction.irregular.scan import preprocess
    from shardelix.reconstruction.reference.fft import reconstruct_reference
    from shardelix.reconstruction.reference.index import reconstruct_indexed, MAX_REFERENCE_PIECES, MAX_REFERENCE_PIXELS, MAX_REFERENCE_INPUT_PIXELS
    from shardelix.reconstruction.common import json_write
    from shardelix.native_core import backend
    started=time.perf_counter()
    refs=sorted(matching_reference_dir.iterdir()) if matching_reference_dir and matching_reference_dir.is_dir() else []
    if len(refs)!=1 or refs[0].suffix.lower() not in IMAGE_EXT:
        raise ValueError("Upload one original matching reference image in the separate reference slot")
    case=root/"case";out=root/"result";out.mkdir(parents=True,exist_ok=True)
    if len(files)==1 and files[0].suffix.lower()==".zip":
        _,_,_,layout=safe_zip(files[0],case,browser=browser,scan=True,extract_masks=True,details=True)
        (case/"pieces").rename(case/"scans")
        paths=sorted((case/"scans").iterdir());piece_rasters=layout["piece_rasters"]
    else:
        if any(p.suffix.lower() not in IMAGE_EXT for p in files):
            raise ValueError("Reference-assisted mode accepts fragment images or one ZIP")
        paths=files;piece_rasters=True
    update("Segmenting reference-assisted fragments")
    # One public profile controls the internal recovery strategy. Preserve
    # explicit engine overrides for reproducible historical benchmarks only.
    profile=conf.get("profile") or "balanced"
    engine=conf.get("reference_engine") or ("indexed-deep" if profile=="accurate" else "indexed")
    if engine not in {"fft","indexed","indexed-deep"}:raise ValueError("Unknown reference recovery engine")
    legacy=engine=="fft"
    cap=256 if legacy else (4096 if browser else MAX_REFERENCE_PIECES)
    pieces,report=preprocess(paths,case/"pieces",pixel_limit=WEB_INPUT_PIXELS if browser else MAX_REFERENCE_INPUT_PIXELS,
                             preserve_opaque=True,piece_rasters=piece_rasters,max_pieces=cap,preserve_scale=not legacy)
    if report["common_scale"]!=1.:
        raise ValueError("Reference-assisted registration needs unchanged fragment scale; reduce input size and reference by the same factor")
    if legacy:
        d=reconstruct_reference(pieces,refs[0],out,report,update,pixel_cap=2_000_000)
    else:
        d=reconstruct_indexed(pieces,refs[0],out,report,update,pixel_cap=WEB_PIXELS if browser else MAX_REFERENCE_PIXELS,
                              piece_cap=cap,use_sift=not browser,deep=engine=="indexed-deep",
                              adaptive_micro=conf.get("reference_engine") is None and profile=="balanced")
    _public_archive(case,out)
    d.update(compute_backend=backend(),workflow_elapsed_seconds=time.perf_counter()-started,
             performance_profile=profile,input_detection={"route":"reference-assisted","engine":engine})
    json_write(out/"diagnostics.json",d)
    return d,None,out


def _dispatch_case(root,files,conf,update,browser=False,truth_dir=None,matching_reference_dir=None):
    """Route observed input, keeping reference-assisted matching explicit."""
    detected=None
    if conf.get("task")=="blind":
        profiles(conf)
        mode=conf.get("input_mode","digital")
        if mode not in ("digital","auto","scan","reference"):raise ValueError("Unknown reconstruction input mode")
        if mode=="reference":
            return _reference_case(root,files,conf,update,browser,matching_reference_dir)
        if mode=="scan":
            return _scan_case(root,files,conf,update,browser,truth_dir)
        if conf.get("auto_layout",False):
            detected=inspect_uploads(files,browser)
            if detected["route"] in ("irregular","scan-strips"):
                update("Detected foreground fragments — selecting geometry-based reconstruction")
                return _scan_case(root,files,{**conf,"detected_route":detected["route"],"auto_preserve_opaque":True,
                                                              "input_evidence":detected.get("input_evidence",{}),"metadata_warning":detected.get("metadata_warning")},update,browser,truth_dir)
    return _digital_case(root,files,conf,update,browser,truth_dir,detected)

def _digital_case(root,files,conf,update,browser=False,truth_dir=None,detected=None):
    """Geometry-known digital cases and inferred grid/strip inputs."""
    from shardelix.reconstruction.common import json_write
    from shardelix.reconstruction.generator import generate,dimensions
    from shardelix.reconstruction.case_io import solve_case,evaluate_case,solve_external_document
    from PIL import Image
    import numpy as np
    case=root/"case";out=root/"result";out.mkdir(parents=True,exist_ok=True)
    task=conf.get("task","demo");rotations=bool(conf.get("rotations",True))
    angle=conf.get("angle_mode","quarter" if rotations else "none")
    sampling=int(conf.get("sampling",1));missing=float(conf.get("missing",0))
    if angle not in ("none","quarter","continuous"):raise ValueError("Invalid rotation mode")
    if sampling not in (1,2,3,4) or (sampling!=1 and angle!="continuous"):raise ValueError("Invalid sampling")
    rotations=angle!="none"
    count_cap=WEB_COUNT if browser else GRID_MAX
    pixel_cap=WEB_PIXELS if browser else MAX_PIXELS
    profile,tune,params=profiles(conf)
    started=time.perf_counter();import_geometry=None;layout_trials=[]
    if task=="demo":
        meta,case,out,fit_info=_generate_case(root,files,conf,update,browser,source_preview=root/"source-preview.png")
    elif task=="blind":
        meta=None
        if len(files)==1 and files[0].suffix.lower()==".zip":
            meta,n=safe_zip(files[0],case,browser=browser,
                            extract_masks=bool(detected and detected.get("metadata") and detected["metadata"].get("cut") not in RECTANGULAR))
        else:
            (case/"pieces").mkdir(parents=True)
            for f in files:
                if f.suffix.lower() not in IMAGE_EXT:raise ValueError("Blind mode accepts images or a ZIP")
                shutil.copy2(f,case/"pieces"/f.name)
            n=len(files)
        if conf.get("auto_layout",False):
            # Resolve again from the actual extracted fragments, not the UI's
            # cached inspection or old manual fields. Never inspect ground truth.
            if meta is not None:
                import_geometry="archive-metadata"
            elif detected and detected["route"] in ("opaque-strips","continuous-strips"):
                import_geometry="inferred-strips"
                angle="continuous" if detected["route"]=="continuous-strips" else "quarter"
                rotations=True;sampling=1;params={}
            else:
                update("Auto-detecting fragment geometry")
                suggestion=detected["suggested_grid"] if detected else _suggest_grid((lambda p=p:p.open("rb") for p in sorted((case/"pieces").iterdir())),n,browser)
                if not suggestion:raise ValueError("Fragment geometry changed between inspection and extraction; reselect the archive.")
                layout=suggestion["layouts"][0]
                if len(suggestion["layouts"])>1 and n<=64 and profile!="fast" and not params:
                    from shardelix.reconstruction.common import load_fragments
                    from shardelix.reconstruction.input_detection import select_visual_grid
                    update("Checking plausible grid layouts using blind seams")
                    layout,layout_trials=select_visual_grid(suggestion["layouts"],load_fragments(case/"pieces"),profile)
                # Unknown square-tile orientation is resolved by quarter-turn
                # matching; there is no need to specify a rotation mode.
                meta=dict(height=layout["height"],width=layout["width"],piece_count=n,
                          cut="grid",rotations=True,grid=[layout["rows"],layout["cols"]])
                import_geometry="inferred-grid"
        elif meta is None or not conf.get("use_metadata",True):
            cut=conf.get("cut","grid")
            if cut=="document":meta=None
            else:
                h,w=int(conf["height"]),int(conf["width"])
                expected=int(conf.get("expected_count",0)) or n
                grid=[1,expected] if cut=="strips" else [expected,1] if cut=="horizontal-strips" else [int(conf["rows"]),int(conf["cols"])] if cut=="grid" else None
                if cut=="grid":expected=grid[0]*grid[1]
                meta=dict(height=h,width=w,piece_count=n,cut=cut,rotations=rotations,grid=grid)
                if expected!=n or angle=="continuous":meta.update(expected_count=expected,angle_mode=angle,sampling=sampling)
        if meta is not None:
            meta=public_metadata(meta)
            # The metadata is authoritative when the import toggle is enabled.
            angle=meta.get("angle_mode","quarter" if meta["rotations"] else "none")
            rotations=angle!="none"
            json_write(case/"case.json",meta)

    else:raise ValueError("Invalid workflow")
    if meta is not None:
        if meta["height"]*meta["width"]>pixel_cap:raise ValueError("Canvas exceeds backend pixel budget")
        if meta.get("expected_count",meta["piece_count"])>count_cap:raise ValueError("Count exceeds backend limit")
        if params.get("strategy")=="sparse" and (not meta.get("grid") or meta["height"]%meta["grid"][0] or meta["width"]%meta["grid"][1]):
            raise ValueError("Explicit sparse/index settings require a uniform rectangular grid")
    elif params:
        raise ValueError("Custom grid settings do not apply to document strips")
    if browser:_pixel_budget(list((case/"pieces").iterdir()),WEB_INPUT_PIXELS if meta is not None else WEB_PIXELS)
    update("Matching and assembling — CPU")
    held=None
    # Enforce independent evaluation by moving truth OUTSIDE the case before solve.
    if task=="demo":
        held=root/"truth-held-out";(case/"evaluation").rename(held)
    try:
        if meta is None:
            try:
                result=solve_external_document(case/"pieces",out,rotations,profile=profile,angle_mode=angle,sampling=sampling,max_canvas_pixels=pixel_cap,color_aware=import_geometry=="inferred-strips")
            except ValueError:
                if not detected or detected["route"]!="continuous-strips":raise
                update("Strict strip rectification failed — preserving masks and trying irregular matching")
                shutil.rmtree(case,ignore_errors=True);shutil.rmtree(out,ignore_errors=True)
                return _scan_case(root,files,{**conf,"detected_route":"irregular","auto_preserve_opaque":True,
                                                "input_evidence":detected.get("input_evidence",{}),
                                                "metadata_warning":"Continuous strip estimate rejected; used irregular matching."},update,browser,truth_dir)
        else:
            result=solve_case(case,out,tune=tune,params=params or None,profile=profile)
    finally:
        if held is not None:held.rename(case/"evaluation")
    update("Rendering and evaluating")
    if task=="demo":
        (root/"source-preview.png").rename(out/"source.png")
        from shardelix.reconstruction.reference.evaluation import compare_reference
        metrics=evaluate_case(case,out)
        comparison=compare_reference(out,out/"source.png",allow_rotation=rotations,
                                     pose_rotation=metrics.get("evaluation_rotation_cw_deg"),pixel_cap=pixel_cap)
        json_write(out/"comparison.json",{**comparison,"pose_evaluated":True})
    elif truth_dir is not None:
        try:_evaluate_external(case,out,meta,truth_dir,browser,rotations)
        except Exception as exc:
            json_write(out/"comparison.json",{"warning":"Reference evaluation failed: "+str(exc)[:300],"pose_evaluated":False})
    d=json.loads((out/"diagnostics.json").read_text())
    m=json.loads((out/"metrics.json").read_text()) if (out/"metrics.json").exists() else None
    _public_archive(case,out)
    d["workflow_elapsed_seconds"]=time.perf_counter()-started
    d["performance_profile"]=profile or "legacy"
    if import_geometry:
        d["import_geometry_mode"]=import_geometry
        if meta:
            d["import_geometry"]={k:meta[k] for k in ("width","height","grid","cut","piece_count")}
        else:
            with Image.open(out/"assembly.png") as assembled:
                w,h=assembled.size
            d["import_geometry"]={"width":w,"height":h,"grid":None,"cut":"document","piece_count":n}
    if detected:
        d["input_detection"]={"route":detected["route"],"evidence":detected.get("input_evidence",{}),"warning":detected.get("metadata_warning")}
        if layout_trials:d["input_detection"]["layout_trials"]=layout_trials
    from shardelix.native_core import backend
    d["compute_backend"]=backend()
    # Peak RSS is the process high-water mark (not an incremental per-case value).
    from shardelix.reconstruction.resource_budget import peak_rss_mib
    d["process_peak_rss_mib"]=peak_rss_mib()
    json_write(out/"diagnostics.json",d)
    return d,m,out

def case_configs(conf,browser=False):
    if not conf.get("batch"):return [dict(conf)]
    if conf.get("task")!="demo":raise ValueError("Batch benchmark requires a source image")
    b=conf["batch"]
    if not isinstance(b,dict):raise ValueError("Invalid batch settings")
    choices=[b.get("counts",[conf["count"]]),b.get("seeds",[conf.get("seed",42)]),b.get("cuts",[conf["cut"]]),b.get("angles",[conf.get("angle_mode","quarter")]),b.get("profiles",[conf.get("profile","balanced")]),b.get("missing_rates",[conf.get("missing",0)])]
    cap=WEB_BATCH if browser else NATIVE_BATCH
    total=1
    for c in choices:
        if not isinstance(c,list) or not c:raise ValueError("Every benchmark axis must be nonempty")
        total*=len(c)
        if total>cap:raise ValueError(f"Batch limit is {cap} cases")
    out=[]
    for n,seed,cut,angle,profile,missing in itertools.product(*choices):
        out.append({**conf,"batch":None,"count":int(n),"seed":int(seed),"cut":cut,"angle_mode":angle,"sampling":conf.get("sampling",1) if angle=="continuous" else 1,"profile":profile,"missing":float(missing)})
    return out

def run_job(root:Path,conf,notify=lambda s:None,*,browser=False):
    """Single source or batch; all failures retained in the benchmark report."""
    from shardelix.reconstruction.common import json_write
    files=sorted((root/"upload").iterdir());_check_preview(files,browser)
    if conf.get("task")=="generate":return _generate_only(root,files,conf,notify,browser)
    result=root/"result";result.mkdir(exist_ok=True)
    configs=case_configs(conf,browser);batch=bool(conf.get("batch"))
    rows=[];last=None;started=time.perf_counter();retained=0
    for i,c in enumerate(configs):
        dest=root/"runs"/f"case-{i+1:03d}";dest.mkdir(parents=True)
        def update(stage):
            notify({"state":"running","stage":stage,"case":i+1,"total_cases":len(configs),"elapsed_seconds":time.perf_counter()-started})
        update("Validating inputs")
        row={k:c.get(k) for k in ("count","seed","cut","angle_mode","profile","missing","sampling")}
        row["case"]=i+1
        try:
            if browser and batch and retained>=WEB_BATCH_OUTPUT_BYTES:raise ValueError("Browser benchmark retained-output budget exhausted; use native Studio")
            d,m,o=_dispatch_case(dest,files,c,update,browser,root/"truth",root/"matching_reference")
            row.update(status="done",algorithm=d.get("algorithm"),runtime_seconds=d.get("runtime_seconds"),
                elapsed_seconds=d.get("workflow_elapsed_seconds"),peak_process_rss_mib=d.get("process_peak_rss_mib"),placed=d.get("placed"),coverage=d.get("coverage"),**{k:m.get(k) for k in ("direct_position_accuracy","raw_position_accuracy","evaluation_rotation_cw_deg","adjacency_f1","pass_95","collision_pixels")} if m else {})
            if last is not None and batch:
                for name in ("fragments.zip","source.png"):
                    (last[2]/name).unlink(missing_ok=True)
            last=(d,m,o)
            retained=sum(p.stat().st_size for p in (root/"runs").rglob("*") if p.is_file()) if browser and batch else 0
        except Exception as exc:
            row.update(status="error",error=str(exc)[:500])
            if not batch:raise
        rows.append(row)
        # Keep only benchmark outputs, not all generated input/GT on disk.
        shutil.rmtree(dest/"case",ignore_errors=True)
    if last:
        d,m,o=last
        shutil.copytree(o,result,dirs_exist_ok=True)
    else:d,m={},None
    if batch:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (result/"benchmark.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        json_write(result/"benchmark.json",{"cases":rows,"total":len(rows),"completed":sum(r["status"]=="done" for r in rows),"passed":sum(r.get("pass_95") is True for r in rows),"elapsed_seconds":time.perf_counter()-started,"native_peak_rss_note":"Process high-water mark, cumulative across cases; unavailable in WASM."})
    if not batch and last is None:raise ValueError("No completed result")
    with zipfile.ZipFile(result/"result.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in result.rglob("*"):
            if p.is_file() and p.name not in ("result.zip","fragments.zip","source.png"):z.write(p,p.relative_to(result))
        if batch:
            for directory in (root/"runs").iterdir():
                for p in (directory/"result").rglob("*"):
                    if p.is_file() and p.name not in ("fragments.zip","source.png"):
                        z.write(p,str(Path("cases")/directory.name/p.relative_to(directory/"result")))
    shutil.rmtree(root/"runs",ignore_errors=True)
    return {"state":"done","stage":"Benchmark complete" if batch else "Reconstruction exported",
            "diagnostics":{k:v for k,v in d.items() if k in ("algorithm","runtime_seconds","status","placed","piece_count","rendered_piece_count","not_rendered","coverage","collision_pixels","render_collision_pixels","identical_piece_count","sparse_high_consensus","reciprocal_seam_fraction","expected_count","missing_count","angle_mode","sampling","available_piece_recovery","normalization_seconds","index_mode","index_fallback_used","index_query_seconds","retrieval_seconds","assembly_seconds","performance_profile","process_peak_rss_mib","workflow_elapsed_seconds","frame_anchor_status","multigap_status","extra_verification","compute_backend","fast_rescue_v13","import_geometry_mode","import_geometry","input_detection","scan","matching_reference_used","accuracy_verified","matching_backend","micro_rescue","orientation_ambiguous")},
            "metrics":m,"comparison":json.loads((result/"comparison.json").read_text()) if (result/"comparison.json").exists() else None,"benchmark":rows if batch else None,"files":sorted(p.name for p in result.iterdir() if p.name in DOWNLOADS)}
