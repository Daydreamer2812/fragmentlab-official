"""Command-line adapter for the shared Shardelix framework."""
from __future__ import annotations
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from shardelix.reconstruction.common import read_source
from shardelix.reconstruction.generator import CUTS,generate
from shardelix.reconstruction.case_io import read_json,solve_case,evaluate_case,solve_external_document

def _source_opts(p):
    p.add_argument("source",type=Path)
    p.add_argument("--max-side",type=int,default=2048,help="0 preserves resolution")
    p.add_argument("--page",type=int,default=1,help="PDF page (1-based)")
def _gen_opts(p):
    _source_opts(p)
    p.add_argument("--pieces",type=int,default=64)
    p.add_argument("--cut",choices=CUTS,default="random")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--rotations",action="store_true",help="Independent random quarter-turns (0/90/180/270 degrees)")
    p.add_argument("--angle-mode",choices=["none","quarter","continuous"],help="Continuous angles in [-180,180)");p.add_argument("--missing",type=float,default=0.,help="Random missing input fraction [0,1)");p.add_argument("--sampling",type=int,choices=[1,2,3,4],default=1,help="Sample factor before raster rotation (1=native; >1=oversampled)")
    p.add_argument("--fit-grid",action="store_true",help="Resize source by a few pixels to uniform rectangular cells; generation only")
    p.add_argument("--label-map",type=Path,help="Custom unlabeled raster partition (.npy), generator only")
def _solve_opts(p):
    p.add_argument("--no-tune",action="store_true",help="Single blind hypothesis")
    p.add_argument("--profile",choices=["fast","balanced","accurate","custom"],help="Performance policy")
    p.add_argument("--params",type=json.loads,help="Explicit classical solver parameter JSON")

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        return _main(args)
    except (ValueError, OSError, MemoryError, RuntimeError) as exc:
        if "--debug" in args:
            raise
        print(f"shardelix: {exc}", file=sys.stderr)
        return 2


def _doctor(require_native=False):
    import importlib.util
    import platform
    import scipy
    import PIL
    from shardelix import __version__
    from shardelix.native_core import backend
    from shardelix.reconstruction.resource_budget import peak_rss_mib, resource_budget

    report = {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "backend": backend(),
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__, "Pillow": PIL.__version__},
        "optional_modules": {name: importlib.util.find_spec(name) is not None
                             for name in ("cv2", "pymupdf", "fitz")},
        "peak_process_rss_mib": peak_rss_mib(),
    }
    try:
        report["resource_budget"] = resource_budget().describe()
    except (ValueError, MemoryError) as exc:
        report["resource_error"] = str(exc)
    report["ready"] = "resource_error" not in report and (
        not require_native or report["backend"] == "cpp-native")
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


def _main(argv):
    from shardelix import __version__
    parser=argparse.ArgumentParser(prog="shardelix",description="Classical CPU fragment reconstruction (blind by default)")
    parser.add_argument("--version", action="version", version=f"shardelix {__version__}")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for operational errors")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("doctor", help="Check runtime, backend and resource budget")
    p.add_argument("--require-native", action="store_true", help="Fail unless the C++ backend is active")
    p=sub.add_parser("studio",help="Launch local browser UI");p.add_argument("--port",type=int,default=8765);p.add_argument("--no-browser",action="store_true")
    p=sub.add_parser("generate");_gen_opts(p);p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("reconstruct",help="Automatically reconstruct files or a ZIP; original image is optional")
    p.add_argument("fragments",type=Path,nargs="+");p.add_argument("--out",type=Path,required=True)
    p.add_argument("--reference",type=Path,help="Explicit original image for matching, not blind reconstruction")
    p.add_argument("--ground-truth",type=Path,help="Independent evaluation only, not matching")
    p.add_argument("--profile",choices=["fast","balanced","accurate"],default="balanced")
    p.add_argument("--input-mode",choices=["auto","digital","scan"],default="auto")
    p=sub.add_parser("solve-document");p.add_argument("--pieces",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--rotations",action="store_true");p.add_argument("--params",type=json.loads);p.add_argument("--angle-mode",choices=["none","quarter","continuous"]);p.add_argument("--sampling",type=int,default=1);p.add_argument("--profile",choices=["fast","balanced","accurate","custom"])
    p=sub.add_parser("solve");_solve_opts(p);p.add_argument("--case",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("evaluate");p.add_argument("--case",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--require-95",action="store_true")
    p=sub.add_parser("run");_gen_opts(p);_solve_opts(p);p.add_argument("--out",type=Path,required=True);p.add_argument("--require-95",action="store_true")
    p=sub.add_parser("benchmark");_source_opts(p);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--counts",type=int,nargs="+",default=[64,256,1024]);p.add_argument("--cuts",choices=CUTS,nargs="+",default=list(CUTS))
    p.add_argument("--seeds",type=int,nargs="+",default=[42,142]);p.add_argument("--rotations",action="store_true")
    p.add_argument("--no-tune",action="store_true");p.add_argument("--keep-cases",action="store_true");p.add_argument("--require-95",action="store_true")
    p=sub.add_parser("tune");_source_opts(p);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cut",choices=CUTS,default="random");p.add_argument("--pieces",type=int,default=64)
    p.add_argument("--train-seeds",type=int,nargs="+",default=[10,20]);p.add_argument("--holdout-seeds",type=int,nargs="+",default=[30,40])
    p.add_argument("--holdout-source",type=Path);p.add_argument("--rotations",action="store_true")
    p.add_argument("--max-trials",type=int,default=12);p.add_argument("--require-95",action="store_true")
    a=parser.parse_args(argv)
    if a.command=="doctor":
        return _doctor(a.require_native)
    if a.command=="studio":
        from shardelix.studio import serve
        serve(a.port,not a.no_browser);return 0
    if a.command=="reconstruct":
        from shardelix import reconstruct_files
        inputs=a.fragments[0] if len(a.fragments)==1 else a.fragments
        state=reconstruct_files(inputs,a.out,matching_reference=a.reference,ground_truth=a.ground_truth,
                                profile=a.profile,input_mode=a.input_mode)
        print(json.dumps(state,indent=2));return 0
    if a.command in {"generate","run"}:
        src=read_source(a.source,a.max_side,a.page)
        fitted_grid=None
        if a.fit_grid:
            from shardelix.reconstruction.limits import RECTANGULAR,MAX_PIXELS
            from shardelix.reconstruction.generator import fit_generation_canvas
            from PIL import Image
            if a.cut in RECTANGULAR and not a.label_map:
                hh,ww,fitted_grid=fit_generation_canvas(src.shape[:2],a.pieces,a.cut,MAX_PIXELS,True)
                src=np.asarray(Image.fromarray(src).resize((ww,hh),Image.Resampling.LANCZOS))
        case=a.out if a.command=="generate" else a.out/"case"
        m=generate(src,case,a.pieces,a.cut,a.seed,a.rotations,np.load(a.label_map,allow_pickle=False) if a.label_map else None,angle_mode=a.angle_mode,missing=a.missing,sampling=a.sampling,grid=fitted_grid)
        if a.command=="generate":print(json.dumps(m,indent=2));return 0
        solve_case(case,a.out/"result",tune=not a.no_tune,params=a.params,profile=a.profile)
        r=evaluate_case(case,a.out/"result");print(json.dumps(r,indent=2));return int(a.require_95 and not r["pass_95"])
    if a.command=="solve-document":
        r=solve_external_document(a.pieces,a.out,a.rotations,a.params,profile=a.profile,angle_mode=a.angle_mode,sampling=a.sampling);print(json.dumps(r["diagnostics"],indent=2));return 0
    if a.command=="solve":
        r=solve_case(a.case,a.out,tune=not a.no_tune,params=a.params,profile=a.profile);print(json.dumps(r["diagnostics"],indent=2));return 0
    if a.command=="evaluate":
        r=evaluate_case(a.case,a.out);print(json.dumps(r,indent=2));return int(a.require_95 and not r["pass_95"])
    if a.command=="benchmark":
        from shardelix.reconstruction.benchmark import benchmark
        rows=benchmark(read_source(a.source,a.max_side,a.page),a.out,a.counts,a.cuts,a.seeds,a.rotations,not a.no_tune,a.keep_cases)
    elif a.command=="tune":
        from shardelix.reconstruction.benchmark import tune
        rows=tune(read_source(a.source,a.max_side,a.page),a.out,a.pieces,a.cut,a.train_seeds,a.holdout_seeds,a.rotations,
                  read_source(a.holdout_source,a.max_side,a.page) if a.holdout_source else None,a.max_trials)
    else:return 2
    return int(a.require_95 and not all(r.get("pass_95",False) for r in rows))

if __name__=="__main__":raise SystemExit(main())
