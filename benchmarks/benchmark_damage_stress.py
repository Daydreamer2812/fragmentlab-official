"""Additional damaged-fragment checks with continuous rotation.

Run from repository root: PYTHONPATH=src python benchmarks/benchmark_damage_stress.py --out stress.json
The deterministic generator uses the existing synthetic curved partition,
then applies continuous rotations. The evaluator receives truth only after
reconstruct() finishes. Each independent component has its own SE(2) gauge.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from benchmark_eroded import eroded_fixture, score
from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.scan import reconstruct, rotation


def continuous_fixture(seed, erosion):
    pieces, truth = eroded_fixture(seed, erosion)
    rng = np.random.default_rng(seed+1000)
    out, poses = [], []
    for i, (p, pose) in enumerate(zip(pieces, truth)):
        angle = float(rng.uniform(-180, 180))
        q = rotation(np.deg2rad(angle))
        src = Image.fromarray(p.rgba)
        raw = np.asarray(src.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)).copy()
        # Preserve only reliable foreground and crop the transparent pad.
        mask = raw[...,3] >= 192
        y,x = np.nonzero(mask)
        lo = np.array([x.min(),y.min()])
        hi = np.array([x.max()+1,y.max()+1])
        raw[...,3] = mask.astype(np.uint8)*255
        raw[~mask] = 0
        h,w = raw.shape[:2]
        shift = np.array(src.size)/2-q@np.array([w,h])/2+q@lo
        crop = raw[lo[1]:hi[1],lo[0]:hi[0]].copy()
        out.append(Fragment(f"piece_{rng.bytes(5).hex()}",crop))
        if pose is None: poses.append(None)
        else:
            A,a = pose
            poses.append((A@q,A@shift+a))
    order = rng.permutation(len(out))
    return [out[i] for i in order], [poses[i] for i in order]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for seed,erosion in ((31,5),(31,8),(41,5),(41,8)):
        pieces,truth=continuous_fixture(seed,erosion)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            start=time.monotonic()
            result=reconstruct(pieces,out,{"input_images":len(pieces),"warnings":[],"common_scale":1.},
                               deep_recovery="always")
            components=json.loads((out/"component_placements.json").read_text())
            item={"seed":seed,"erosion_px":erosion,"genuine_pieces":len(pieces)-1,
                  "model":components["model"],"wall_seconds":time.monotonic()-start,
                  "selected":score(pieces,truth,components["components"]),
                  "offset":result["scan"]["deep_recovery"].get("offset_recovery",{})}
            rows.append(item)
            print(json.dumps(item),flush=True)
    args.out.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
