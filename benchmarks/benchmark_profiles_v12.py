"""Same-input Fast/Balanced/High accuracy comparison via V12 batch workflow."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse,shutil
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from shardelix.workflow import run_job
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path("outputs/benchmarks/v12/profiles.csv"));a=p.parse_args()
    with TemporaryDirectory() as t:
        root=Path(t);(root/"upload").mkdir()
        src=np.clip(127+480*gaussian_filter(np.random.default_rng(2287).normal(size=(256,256,3)),(1.4,1.4,0)),0,255).astype(np.uint8)
        Image.fromarray(src).save(root/"upload/texture.png")
        conf={"task":"demo","cut":"grid","count":64,"seed":42,"max_side":256,"angle_mode":"quarter","profile":"balanced",
              "batch":{"counts":[64,256],"seeds":[42],"cuts":["grid","jigsaw"],"profiles":["fast","balanced","accurate"]}}
        result=run_job(root,conf,lambda s:print(s["case"],s["stage"],flush=True))
        a.out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/"result/benchmark.csv",a.out)
        print("completed",sum(r["status"]=="done" for r in result["benchmark"]),"passed",sum(r.get("pass_95") is True for r in result["benchmark"]))
if __name__=="__main__":main()
