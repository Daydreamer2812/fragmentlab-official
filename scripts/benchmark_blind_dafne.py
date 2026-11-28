"""Run one official DAFNE variant in blind mode, without source artwork.

Usage: PYTHONPATH=src python scripts/benchmark_blind_dafne.py ARCHIVE.zip \
       --case CASE_SUFFIX --out OUTPUT_DIRECTORY

Only fragment PNGs enter the solver. Annotation text is accessed AFTER the
workflow returns, and is used solely by the existing relative-pose evaluator.
A missing/outlier piece is not counted as a successful placement. The source
archive is not redistributed; obtain it from the official dataset website.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile
import random

from benchmark_official_dafne import cases
from evaluate_dafne_components import evaluate
from shardelix.workflow import run_job


def run(archive, suffix, out, seed=624):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        found = [(case, entries) for case, entries in cases(z).items()
                 if case.rsplit("/", 1)[-1].endswith(suffix)]
        if len(found) != 1:
            raise ValueError(f"Expected one case matching {suffix!r}, found {len(found)}")
        case, entries = found[0]
        with tempfile.TemporaryDirectory(prefix="blind-dafne-") as tmp:
            root = Path(tmp)
            (root / "upload").mkdir()
            inp = root / "upload" / "fragments.zip"
            mapping = {}
            with zipfile.ZipFile(inp, "w", zipfile.ZIP_DEFLATED) as fragments:
                order=sorted(entries)
                random.Random(seed).shuffle(order)
                for member in order:
                    # Assign independent opaque aliases; archive order and
                    # source sequence IDs cannot affect the matching order.
                    idx = len(mapping) + 1
                    mapping[str(idx)] = int(Path(member).stem.rsplit("_", 1)[-1])
                    fragments.writestr(f"pieces/fragment_{idx:04d}.png", z.read(member))
            start = time.perf_counter()
            state = run_job(root, {"task": "blind", "input_mode": "auto",
                                   "auto_layout": True, "profile": "balanced"},
                            lambda event: print(event.get("stage", ""), flush=True),
                            browser=False)
            elapsed = time.perf_counter() - start
            # Ground truth is not accessible to run_job: separate archive,
            # read the official poses only after inference completed.
            truth = root / "truth.zip"
            with zipfile.ZipFile(truth, "w") as tz:
                tz.writestr("annotations/source_metadata.json", json.dumps({
                    "source_id_by_piece_id": mapping,
                    "ground_truth": [v for v in z.read(case + "/fragments.txt").decode().splitlines()
                                     if v.strip()],
                }))
            result = root / "result"
            metrics = evaluate(inp, truth, result)
            diag = state["diagnostics"]
            if diag.get("matching_reference_used"):
                raise RuntimeError("Unexpected reference-assisted result")
            record = {"protocol": "blind auto/Balanced; independent SE(2) gauge per component; 10 px/10 degree tolerance",
                      "archive": archive.name, "case": case, "input_count": len(entries), "seed":seed,
                      "wall_seconds": elapsed, "reference_used": False, **metrics,
                      "deep_recovery": diag.get("scan", {}).get("deep_recovery", {}),
                      "common_scale": diag.get("scan", {}).get("common_scale")}
            (out / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record, indent=2), flush=True)
            return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed",type=int,default=624)
    args = parser.parse_args()
    run(args.archive, args.case, args.out, args.seed)
