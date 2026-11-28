"""Evaluate mechanically shredded pages from Paixão et al. (PR 2020).

Dataset: https://github.com/thiagopx/deeprec-pr20 (scripts/get_dataset.sh)
Run PYTHONPATH=src python scripts/benchmark_mechanical_strips.py /path/datasets --out result.json
Only the individual strip pixels and optional supplied foreground masks are
read by reconstruction. Sequential IDs are decoded for scoring after solving.
No intact pages or model weights are used. Timings measure rectification and
matching, not file conversion, rendering or export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from shardelix.reconstruction.strips import physical as p


def evaluate(folder, seed=624, profile="balanced"):
    files=sorted(f for f in (folder/"strips").iterdir()
                 if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp"))
    rng=np.random.default_rng(seed)
    aliases={};pieces=[];rectify=[]
    # Alias mapping is based on IDs but randomized independently of order.
    # Sort the completely anonymous inputs before reading pixels.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for f in files:
            alias=rng.bytes(8).hex()
            aliases[alias]=f.stem
            with Image.open(f) as src:im=src.convert("RGBA")
            mask=folder/"masks"/(f.stem+".npy")
            if mask.is_file():
                m=np.load(mask,allow_pickle=False)
                if m.shape!=(im.height,im.width):raise ValueError("Mask dimensions differ")
                im.putalpha(Image.fromarray((m>0).astype(np.uint8)*255))
            im.save(root/(alias+".png"),compress_level=1)
        started=time.perf_counter()
        try:
            for f in sorted(root.glob("*.png")):
                piece,record=p._rectify(f);pieces.append(piece);rectify.append(record)
            heights=[len(piece.rgba) for piece in pieces]
            if max(heights)-min(heights)>max(3,.035*max(heights)):
                raise p.NotDocumentStrips("Rectified strips lack comparable long dimensions")
            images=[p._ink(piece) for piece in pieces]
            if p._textlike(pieces,images):
                same,flipped=p._projection_cost([p._clean_ink(im) for im in images])
                signs=p._signed_tree(same,flipped,neighbors=12)
                ims=[np.rot90(im,2) if signs[i] else im for i,im in enumerate(images)]
                order,matching=p._stroke_order(ims,profile,lambda _:None)
                predicted=[pieces[i].id for i in order]
                turns=[int(signs[i]) for i in order]
                route="physical-document"
            else:
                poses,matching=p.solve_unknown_color_strips(pieces)
                predicted=sorted(poses,key=lambda a:poses[a].x)
                turns=[poses[a].k//2 for a in predicted]
                route="physical-color"
        except (ValueError,RuntimeError) as exc:
            return {"case":f"{folder.parent.parent.name}/{folder.name}",
                    "pieces":len(files),"joins":len(files)-1,"error":str(exc),
                    "matching_seconds":time.perf_counter()-started}
        elapsed=time.perf_counter()-started
        # The published dataset convention gives the true sequence by
        # filename; decode those identifiers only after prediction.
        truth=sorted(aliases.values())
        pred=[aliases[a] for a in predicted]
        edges=set(zip(truth,truth[1:]))
        directed=sum((a,b) in edges for a,b in zip(pred,pred[1:]))
        reverse=sum((b,a) in edges for a,b in zip(pred,pred[1:]))
        # Global 180-degree orientation is not identifiable by local seams.
        # Keep directed score and orientation-neutral score separate.
        direct_pos=sum(a==b for a,b in zip(pred,truth))
        rev_pos=sum(a==b for a,b in zip(pred[::-1],truth))
        return {"case":f"{folder.parent.parent.name}/{folder.name}","pieces":len(files),
                "joins":len(files)-1,"directed_correct":directed,
                "gauge_aligned_correct":max(directed,reverse),
                "positions_correct":direct_pos,"gauge_aligned_positions":max(direct_pos,rev_pos),
                "relative_orientation_flips":sum(turns),"route":route,
                "matching_seconds":elapsed,"matching":matching}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--cases",nargs="*",help="Examples: D1/D001 D2/D007")
    parser.add_argument("--seed",type=int,default=624)
    parser.add_argument("--profile",choices=("fast","balanced","accurate"),default="balanced")
    args=parser.parse_args()
    folders=sorted(p for p in args.root.glob("D*/mechanical/D*") if (p/"strips").is_dir())
    if args.cases:folders=[p for p in folders if f"{p.parent.parent.name}/{p.name}" in args.cases]
    if not folders:raise ValueError("No mechanical document folders found")
    rows=[]
    for folder in folders:
        row=evaluate(folder,args.seed,args.profile);rows.append(row)
        print(json.dumps(row),flush=True)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps({"profile":args.profile,"seed":args.seed,
            "source":"https://github.com/thiagopx/deeprec-pr20","results":rows},indent=2)+"\n")
if __name__=="__main__":
    main()
