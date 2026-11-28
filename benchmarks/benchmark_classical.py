"""Compare the previous gap solver with the high-quality classical stage.

Ground truth is used ONLY by the evaluator after reconstruct() returns.
Run: PYTHONPATH=src python benchmarks/benchmark_classical.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import time

from benchmark_eroded import eroded_fixture, score
from shardelix.reconstruction.irregular.scan import reconstruct


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("outputs/benchmarks/classical_benchmark.json"))
    args=parser.parse_args()
    rows=[]
    cases=[(seed,erosion,(0,1,2,3,4,6,7)) for seed in (3,9,17) for erosion in (3,5,8)]
    cases += [(seed,5,(0,1,4,7)) for seed in (3,17)]
    for seed,erosion,keep in cases:
        pieces,truth=eroded_fixture(seed,erosion,keep)
        for mode in (False,"always"):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                t=time.perf_counter()
                result=reconstruct(pieces,out,{"input_images":len(pieces),"warnings":[],"common_scale":1.},
                                   gap_recovery=True,deep_recovery=mode)
                components=json.loads((out/"component_placements.json").read_text())
                item={"seed":seed,"erosion_px":erosion,"genuine_pieces":len(keep),"method":"deep" if mode else "previous",
                      "wall_seconds":time.perf_counter()-t,"mode":components["model"],
                      "primary_size":result["placed"],"selected":score(pieces,truth,components["components"]),
                      "deep_recovery":result["scan"]["deep_recovery"]}
                for key in ("alternative","other_model"):
                    if key in components:
                        model=components[key]
                        item[key]={"model":model["model"],**score(pieces,truth,model["components"])}
                rows.append(item)
                print(seed,erosion,item["method"],item["mode"],item["selected"],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":main()
