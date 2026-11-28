"""Public blind API. Results explicitly distinguish completeness from accuracy."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from collections import Counter
import hashlib
import numpy as np
from shardelix.reconstruction.common import Fragment,Pose,render,save_rgba,json_write,load_fragments
from shardelix.reconstruction.overlay import write_cut_outputs
from shardelix.reconstruction.output import require_empty_output, staged_output
from shardelix.reconstruction.solver import solve
from shardelix.reconstruction.limits import validate_constraints,MAX_PIXELS,ResourceLimits,max_count

def result_status(pieces,poses,shape,diagnostics,labels,collision,rotations):
    # Canonical rotational copies are indistinguishable from observable data.
    identical=diagnostics.get("identical_piece_count")
    if identical is None:
        keys=Counter(min(hashlib.sha256(str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).digest()
                     for a in [p.rotated(k) for k in range(4 if rotations else 1)]) for p in pieces)
        identical=sum(v for v in keys.values() if v>1)
    coverage=float((labels>=0).mean())
    # Match render()'s bounds test without a second scan or sorting the canvas.
    rendered_ids = set()
    for piece in pieces:
        pose = poses.get(piece.id)
        if pose is None:
            continue
        h, w = piece.rgba.shape[:2]
        if pose.k % 2:
            h, w = w, h
        if 0 <= pose.x <= shape[1] - w and 0 <= pose.y <= shape[0] - h:
            rendered_ids.add(piece.id)
    not_rendered = sorted(set(poses) - rendered_ids)
    ambiguous=identical>0 or diagnostics.get("sparse_high_consensus") is False or any(diagnostics.get("anchor_translation_slack",[0,0]))
    status="partial" if len(poses)<len(pieces) or not_rendered or coverage<.9999 or collision else "ambiguous" if ambiguous else "assembled"
    return {"status":status,"placed":len(poses),"piece_count":len(pieces),"coverage":coverage,
            "collision_pixels":int(collision),"identical_piece_count":identical,
            "accuracy_verified":False,
            "rendered_piece_count":len(rendered_ids), "not_rendered":not_rendered,
            "unplaced":[p.id for p in pieces if p.id not in poses],
            "note":"Completeness and seam consensus are not ground-truth accuracy."}

@dataclass(slots=True)
class ReconstructionResult:
    fragments: Sequence[Fragment]
    poses: dict[str,Pose]
    shape: tuple[int,int]
    diagnostics: dict
    grid: tuple[int,int]|None
    rotations: bool
    rectification: dict[str,float]|None=None
    input_affines: dict|None=None

    def summary(self) -> dict:
        """Lightweight, JSON-safe summary. No ground-truth confidence claim."""
        return {k:v for k,v in {**self.diagnostics,"placed":len(self.poses),
            "piece_count":len(self.fragments),"accuracy_verified":False,
            "shape":list(self.shape)}.items() if not k.startswith("_")}

    def render(self):
        """Predicted image, label map, collision count; no file I/O."""
        return render(self.fragments,self.poses,self.shape)

    def save(self,output: Path|str):
        """Export a complete result to a new or empty directory.

        A failed export leaves no misleading partial result. Existing results
        are never overwritten.
        """
        with staged_output(output) as stage:
            result = self._save(stage)
        return result

    def _save(self, output: Path):
        """Write artifacts within an isolated staging directory."""
        rgba,labels,coll=render(self.fragments,self.poses,self.shape)
        diag={**self.diagnostics,**result_status(self.fragments,self.poses,self.shape,self.diagnostics,labels,coll,self.rotations)}
        scale=3 if self.grid and min(self.grid)==1 and self.shape[0]*self.shape[1]*9<=MAX_PIXELS and (self.shape[1]//self.grid[1] if self.grid[0]==1 else self.shape[0]//self.grid[0])<4 else 1
        groups=diag.pop("_unanchored_components",None)
        if groups is not None:
            json_write(output/"unanchored_components.json",{"note":"Relative grid-cell positions only; absolute location and orientation unknown", "components":groups})
        if diag.get("identical_piece_count",0) or groups is not None or diag.get("sparse_high_consensus") is False or any(diag.get("anchor_translation_slack",[0,0])):
            from shardelix.reconstruction.ambiguity import describe_ambiguity
            json_write(output/"ambiguity.json",describe_ambiguity(self.fragments,diag,self.rotations))
        diag.update(render_collision_pixels=coll,cut_overlay_scale=scale)
        result={"mode":"blind","poses":{i:p.as_dict() for i,p in self.poses.items()},"diagnostics":diag}
        if self.rectification is not None:
            result["rectify_ccw_deg"]=self.rectification
            result["pose_convention"]="rectify_ccw_deg then k clockwise quarter-turns; x/y normalized raster top-left"
            diag["expected_full_count"]=diag.get("expected_count",len(self.fragments))
            diag["available_piece_recovery"]=len(self.poses)/len(self.fragments)
            # A known missing input is not a solver error; do not assert accuracy.
            if diag.get("missing_count",0) and len(self.poses)==len(self.fragments) and not coll and not diag["not_rendered"] and not diag["identical_piece_count"] and diag.get("sparse_high_consensus") is not False and not any(diag.get("anchor_translation_slack",[0,0])):
                diag["status"]="assembled-with-holes"
            if self.input_affines is not None:
                lookup={p.id:p for p in self.fragments}
                matrices={}
                for i,p in self.poses.items():
                    h,w=lookup[i].rgba.shape[:2];k=p.k%4
                    R=np.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]]][k])
                    t=np.array([(0,0),(h-1,0),(w-1,h-1),(0,w-1)][k])+[p.x,p.y]
                    A=np.asarray(self.input_affines[i],float);B=R@A;B[:,2]+=t
                    matrices[i]=B.tolist()
                result["input_to_canvas_affine"]=matrices
                result["affine_convention"]="2x3 matrix maps original input file pixel centers (x,y,1) to canvas centers; no hidden truth."
        save_rgba(output/"assembly.png",rgba);write_cut_outputs(output,rgba,labels,scale=scale)
        for alt in diag.get("equally_scored_alternatives",[]):
            pp={k:Pose(**v) for k,v in alt["poses"].items()}
            aa,ll,_=render(self.fragments,pp,self.shape);dest=output/("alternative_"+alt["hypothesis"])
            save_rgba(dest/"assembly.png",aa);write_cut_outputs(dest,aa,ll,scale=scale)
        json_write(output/"placements.json",result);json_write(output/"diagnostics.json",diag)
        self.diagnostics={**diag,**({"_unanchored_components":groups} if groups is not None else {})}
        return result

@dataclass(frozen=True,slots=True)
class Solver:
    """CPU classical solver. Provide public canvas/cut metadata, never source."""
    auto_tune: bool=True
    strategy: str="auto"  # auto, classic, sparse
    index_mode: str="auto"  # auto, legacy, midpoint, mixed
    limits: ResourceLimits = ResourceLimits()
    profile: str | None = None  # fast, balanced, accurate, custom

    def solve(self,fragments: Sequence[Fragment]|Path|str,*,shape: tuple[int,int],cut: str="random",
              grid: tuple[int,int]|None=None,rotations: bool=False,params: dict|None=None,
              angle_mode: str|None=None,sampling: int=1,expected_count: int|None=None):
        import time
        started=time.perf_counter()
        if self.profile not in (None,"fast","balanced","accurate","custom"):raise ValueError("Unknown performance profile")
        auto_tune=self.auto_tune if self.profile in (None,"custom") else self.profile!="fast"
        if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
            raise ValueError("Invalid expected_count")
        self.limits.check(expected_count, shape)
        if isinstance(fragments,(str,Path)):
            fragments=load_fragments(Path(fragments), max_fragments=min(self.limits.max_fragments,max_count(cut)), max_input_pixels=self.limits.max_input_pixels)
        fragments=list(fragments)
        self.limits.check(len(fragments), shape)
        if not fragments or any(not isinstance(p,Fragment) or not isinstance(p.id,str) or not p.id or not isinstance(p.rgba,np.ndarray) for p in fragments):
            raise ValueError("Expected nonempty Fragment objects with string IDs and ndarray pixels")
        if len({p.id for p in fragments})!=len(fragments):raise ValueError("Duplicate fragment IDs")
        if any(p.rgba.ndim!=3 or p.rgba.shape[2]!=4 or p.rgba.dtype!=np.uint8 or not p.mask.any() for p in fragments):
            raise ValueError("Invalid RGBA fragment")
        if sum(p.rgba.shape[0]*p.rgba.shape[1] for p in fragments)>self.limits.max_input_pixels:
            raise ValueError("Combined fragment crop pixels exceed resource budget")
        fragments=sorted(fragments,key=lambda p:p.id)
        if self.index_mode not in {"auto","legacy","midpoint","mixed"}:raise ValueError("Invalid edge index mode")
        if self.strategy not in {"auto","classic","sparse"}:raise ValueError("Unknown strategy")
        if params is not None and not isinstance(params,dict):raise ValueError("params must be a dictionary")
        if self.index_mode!="auto":
            if not grid or self.strategy=="classic":raise ValueError("Explicit edge index requires rectangular sparse strategy")
            params={**(params or {}),"strategy":"sparse","index_mode":self.index_mode}
        if type(rotations) is not bool:raise ValueError("rotations must be a boolean")
        angle_mode=angle_mode or ("quarter" if rotations else "none")
        if angle_mode not in {"none","quarter","continuous"}:raise ValueError("Invalid angle mode")
        rotations=angle_mode!="none"
        if type(sampling) is not int or sampling not in (1,2,3,4) or sampling!=1 and angle_mode!="continuous":raise ValueError("Invalid sampling")
        if expected_count is None:expected_count=len(fragments)
        if type(expected_count) is not int or expected_count<len(fragments):raise ValueError("Invalid expected_count")
        if angle_mode=="continuous" or expected_count!=len(fragments):
            if self.strategy=="classic":raise ValueError("Classic route does not support continuous angles or missing input")
            if self.strategy not in {"auto","sparse"}:raise ValueError("Unknown strategy")
            if self.strategy=="sparse" and not grid:raise ValueError("Sparse strategy requires grid")
            from shardelix.reconstruction.pose import solve_pose
            pp,poses,diag,angles,affines=solve_pose(fragments,shape,cut,grid,angle_mode,sampling,expected_count,rotations,params,auto_tune)
            if self.profile=="accurate" and grid and not params:
                from shardelix.reconstruction.tiles.verification import verify_extra_grid
                poses,diag=verify_extra_grid(pp,poses,diag,shape,grid,rotations,expected_count)
            diag.update(performance_profile=self.profile or "legacy",runtime_seconds=time.perf_counter()-started)
            return ReconstructionResult(pp,poses,shape,diag,grid,rotations,angles,affines)
        validate_constraints(len(fragments),shape,cut,grid,rotations)
        for p in fragments:
            if grid and not p.mask.all():raise ValueError("Rectangular cuts require fully opaque rectangular fragments")
            if grid and shape[0]%grid[0]==0 and shape[1]%grid[1]==0:
                cell=(shape[0]//grid[0],shape[1]//grid[1]);dims=p.rgba.shape[:2]
                if dims!=cell and (not rotations or dims!=cell[::-1]):raise ValueError("Piece dimensions do not match public uniform cells")
        if self.strategy=="sparse":
            if not grid:raise ValueError("Sparse strategy requires a rectangular grid")
            params={"strategy":"sparse",**(params or {})}
        if self.strategy=="classic":
            from shardelix.reconstruction.solver import solve_classic
            from shardelix.reconstruction.limits import LEGACY_MAX
            if len(fragments)>LEGACY_MAX:raise ValueError("Classic strategy is limited to 1024 pieces")
            poses,diag=solve_classic(fragments,shape,cut,grid,rotations,auto_tune=auto_tune,params=params)
        else:
            poses,diag=solve(fragments,shape,cut,grid,rotations,auto_tune=auto_tune,params=params)
        # Fast normally evaluates one hypothesis. On catastrophic irregular
        # under-coverage, try one complementary verified candidate instead of
        # returning an almost empty placement. Acceptance is strictly blind.
        if (self.profile=="fast" and not grid and cut in {"jigsaw","wavy","random","jagged"}
                and not params and self.strategy!="classic" and 8<=len(fragments)<=1024
                and len(poses)<.60*len(fragments)):
            from shardelix.reconstruction.irregular.verified import solve_verified
            from shardelix.reconstruction.quality import objective,orient_to_canvas
            q,d=solve_verified(fragments,shape,rotations,k=16,max_bucket=18,pack=False)
            if rotations:q=orient_to_canvas(fragments,q,shape)
            a,b=objective(fragments,poses,shape),objective(fragments,q,shape)
            accepted=(len(q)>len(poses) and b>a+1e-8 and render(fragments,q,shape)[2]==0)
            trial={"accepted":bool(accepted),"initial_placed":len(poses),"alternative_placed":len(q),"initial_objective":a,"alternative_objective":b}
            if accepted:poses,diag=q,d
            diag["fast_rescue_v13"]=trial
        if self.profile=="accurate" and not params and self.strategy!="classic":
            if grid:
                from shardelix.reconstruction.tiles.verification import verify_extra_grid
                poses,diag=verify_extra_grid(fragments,poses,diag,shape,grid,rotations,expected_count)
            elif 8<=len(fragments)<=1024 and len(poses)<.98*len(fragments):
                from shardelix.reconstruction.irregular.verified import solve_verified
                from shardelix.reconstruction.quality import objective,orient_to_canvas
                q,d=solve_verified(fragments,shape,rotations,k=24,max_bucket=24,pack=False)
                if rotations:q=orient_to_canvas(fragments,q,shape)
                a,b=objective(fragments,poses,shape),objective(fragments,q,shape)
                accept=b>a and len(q)>=len(poses) and render(fragments,q,shape)[2]==0
                if accept:poses,diag=q,d
                diag["extra_verification"]={"accepted":bool(accept),"primary_objective":a,"alternative_objective":b}
        diag.update(performance_profile=self.profile or "legacy",runtime_seconds=time.perf_counter()-started)
        return ReconstructionResult(fragments,poses,shape,diag,grid,rotations)


def reconstruct_files(fragments, output, *, matching_reference=None, ground_truth=None,
                      profile="balanced", input_mode="auto", notify=lambda status: None):
    """Reconstruct file inputs with the same auto-routing as Studio.

    Accept one ZIP, one image, a directory of fragment images, or a sequence
    of image paths. Add a matching_reference explicitly to enable assisted
    registration. ground_truth is evaluation-only, and is never a substitute
    for matching_reference. Existing Solver.solve remains the geometry API.
    """
    import shutil
    import tempfile
    from shardelix.workflow import run_job
    from shardelix.uploads import IMAGE_EXT, native_upload_limit

    if profile not in {"fast", "balanced", "accurate"}:
        raise ValueError("Choose fast, balanced or accurate")
    if input_mode not in {"auto", "digital", "scan"}:
        raise ValueError("Choose auto, digital or scan; supply matching_reference explicitly for assisted mode")
    if matching_reference is not None and ground_truth is not None:
        raise ValueError("Use independent benchmark evaluation for reference-assisted results; do not mix matching and evaluation inputs")
    if isinstance(fragments, (str, Path)):
        path = Path(fragments)
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT) if path.is_dir() else [path]
    else:
        files = [Path(p) for p in fragments]
    files = [Path(p) for p in files]
    if not files or any(not p.is_file() for p in files):
        raise ValueError("Provide existing fragment image files or one ZIP")
    if len({p.name for p in files}) != len(files):
        raise ValueError("Duplicate fragment filenames")
    if not (len(files) == 1 and files[0].suffix.lower() == ".zip") and any(p.suffix.lower() not in IMAGE_EXT for p in files):
        raise ValueError("Provide fragment images or one ZIP")
    refs = [(slot, Path(p)) for slot, p in
            (("matching_reference", matching_reference), ("truth", ground_truth)) if p is not None]
    if any(not p.is_file() for _, p in refs):
        raise ValueError("Reference/evaluation file does not exist")
    if sum(p.stat().st_size for p in files + [p for _, p in refs]) > native_upload_limit():
        raise ValueError("Combined input exceeds native upload budget")
    output = require_empty_output(output)
    with tempfile.TemporaryDirectory(prefix="shardelix-") as tmp:
        root = Path(tmp)
        (root / "upload").mkdir()
        for p in files:
            shutil.copy2(p, root / "upload" / p.name)
        for slot, p in refs:
            (root / slot).mkdir()
            shutil.copy2(p, root / slot / p.name)
        state = run_job(root, {"task":"blind",
                              "input_mode":"reference" if matching_reference is not None else input_mode,
                              "auto_layout":True, "use_metadata":True, "profile":profile}, notify)
        with staged_output(output) as stage:
            shutil.copytree(root / "result", stage, dirs_exist_ok=True)
    return state
