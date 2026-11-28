"""File-case operations shared across Python, CLI and Studio."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from shardelix.reconstruction.common import Pose,load_fragments,save_rgba,render,json_write
from shardelix.reconstruction.evaluation import evaluate
from shardelix.reconstruction.overlay import write_cut_outputs

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def solve_case(folder:Path,output:Path,mode="blind",reference=None,tune=True,params=None,template=None,profile=None):
    """Solve using only case.json (public constraints) and pieces/.

    The old Python signature rejects any non-blind input. CLI exposes no modes.
    """
    if mode!="blind" or reference is not None or template is not None:
        raise ValueError("Blind only: source/reference/template are prohibited")
    from shardelix.api import Solver
    meta=read_json(folder/"case.json")
    if not isinstance(meta,dict):raise ValueError("case.json must contain an object")
    for k in ("height","width","piece_count","cut","rotations"):
        if k not in meta:raise ValueError(f"Missing public metadata: {k}")
    from shardelix.reconstruction.limits import validate_constraints,max_count
    shape=(meta["height"],meta["width"])
    grid=tuple(meta["grid"]) if meta.get("grid") is not None else None
    observed=meta["piece_count"];expected=meta.get("expected_count",observed)
    if type(observed) is not int or type(expected) is not int or not 1<=observed<=expected:
        raise ValueError("Invalid public piece counts")
    validate_constraints(expected,shape,meta["cut"],grid,meta["rotations"])
    pieces=load_fragments(folder/"pieces",max_fragments=min(observed,max_count(meta["cut"])))
    if observed!=len(pieces):
        raise ValueError("Declared piece_count differs from input files")
    solver=Solver(auto_tune=tune,profile=profile)
    result=solver.solve(pieces,shape=shape,cut=meta["cut"],grid=grid,rotations=meta["rotations"],params=params,angle_mode=meta.get("angle_mode"),sampling=meta.get("sampling",1),expected_count=meta.get("expected_count"))
    return result.save(output)

def evaluate_case(folder,output):
    meta=read_json(folder/"case.json")
    result=read_json(output/"placements.json")
    # Only evaluator can open truth, AFTER solver and cut overlay completed.
    truth=read_json(folder/"evaluation"/"ground_truth.json")
    pieces=load_fragments(folder/"pieces")
    poses={i:Pose(**p) for i,p in result["poses"].items()}
    if "expected_count" in meta:
        from shardelix.reconstruction.pose_evaluation import evaluate_pose
        metrics=evaluate_pose(pieces,poses,truth,meta,result,np.load(folder/"evaluation"/"truth_labels.npz")["labels"])
    else:
        gt={i:Pose(**p) for i,p in truth["poses"].items()}
        metrics=evaluate(pieces,poses,gt,(meta["height"],meta["width"]),
                         allow_global_rotation=meta["rotations"],allow_translation=False)
    metrics.update(mode="blind",cut=meta["cut"],seed=truth["seed"],rotations=meta["rotations"],
                   runtime_seconds=result["diagnostics"]["runtime_seconds"],
                   source_reference_used=False,cut_template_used=False)
    json_write(output/"metrics.json",metrics)
    return metrics

def solve_external_document(folder:Path,output:Path,rotations=False,params=None,profile=None,angle_mode=None,sampling=1,max_canvas_pixels=None,color_aware=False):
    """Blind external document strips, unequal widths permitted. Input must contain only pieces."""
    from shardelix.reconstruction.output import staged_output

    with staged_output(output) as stage:
        return _solve_external_document(folder, stage, rotations, params, profile,
                                        angle_mode, sampling, max_canvas_pixels, color_aware)


def _solve_external_document(folder, output, rotations, params, profile,
                             angle_mode, sampling, max_canvas_pixels, color_aware):
    from shardelix.reconstruction.strips.ordering import solve_document_strips
    import time
    pieces=load_fragments(folder,max_fragments=1024)
    if len(pieces)>1024:raise ValueError("External document solver supports up to 1024 strips")
    start=time.perf_counter()
    angles=affines=None;extra={}
    if angle_mode=="continuous":
        from shardelix.reconstruction.strips.rectification import normalize_document_strips
        pieces,angles,affines,extra=normalize_document_strips(pieces,sampling)
        rotations=True
    elif angle_mode not in (None,"none","quarter"):
        raise ValueError("Invalid external document rotation mode")
    elif angle_mode is not None:rotations=angle_mode=="quarter"
    if angle_mode!="continuous" and sampling!=1:raise ValueError("Sampling only applies to continuous mode")
    from shardelix.reconstruction.strips.ordering import is_documentlike
    if color_aware and not is_documentlike(pieces):
        from shardelix.reconstruction.rotation import solve_unknown_color_strips
        poses,diag=solve_unknown_color_strips(pieces)
    elif profile=="fast":
        from shardelix.reconstruction.strips.ordering import _solve_document_v5
        poses,diag=_solve_document_v5(pieces,rotations=rotations,**(params or {}))
    else:
        poses,diag=solve_document_strips(pieces,rotations=rotations,**(params or {}))
    diag["performance_profile"]=profile or "legacy"
    shape=(max(p.rotated(poses[p.id].k).shape[0] for p in pieces),
           sum(p.rotated(poses[p.id].k).shape[1] for p in pieces))
    from shardelix.reconstruction.limits import MAX_PIXELS
    if shape[0]*shape[1]>min(MAX_PIXELS,max_canvas_pixels or MAX_PIXELS):
        raise ValueError("Estimated document canvas exceeds pixel budget")
    if angles is not None:
        from shardelix.api import ReconstructionResult
        diag.update(extra,angle_mode="continuous",sampling=sampling,expected_count=len(pieces),missing_count=0,
                    algorithm="estimated-document-pose-v12-"+diag.get("algorithm","document"),
                    mode="blind",runtime_seconds=time.perf_counter()-start,orientation_ambiguous=True,document_missing_unknown=True)
        return ReconstructionResult(pieces,poses,shape,diag,None,True,angles,affines).save(output)
    rgba,labels,collision=render(pieces,poses,shape)
    save_rgba(output/"assembly.png",rgba)
    write_cut_outputs(output,rgba,labels)
    for alt in diag.get("equally_scored_alternatives",[]):
        pp={k:Pose(**v) for k,v in alt["poses"].items()}
        aa,ll,_=render(pieces,pp,shape)
        dest=output/("alternative_"+alt["hypothesis"])
        save_rgba(dest/"assembly.png",aa);write_cut_outputs(dest,aa,ll)
    diag.update(mode="blind",runtime_seconds=time.perf_counter()-start,
                render_collision_pixels=collision,coverage=float((labels>=0).mean()),
                canvas_height=shape[0],canvas_width=shape[1],placed=len(poses),piece_count=len(pieces),
                status="assembled" if len(poses)==len(pieces) and collision==0 else "partial",accuracy_verified=False)
    result={"mode":"blind","poses":{k:v.as_dict() for k,v in poses.items()},"diagnostics":diag}
    json_write(output/"placements.json",result);json_write(output/"diagnostics.json",diag)
    return result
