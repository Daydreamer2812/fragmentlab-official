"""Run frozen external and synthetic benchmark manifests with provenance.

Example: python benchmarks/benchmark_suite.py --manifest benchmarks.json --out results
Each named stage runs in a separate process; errors and timeouts are recorded.
No ground-truth or original matching image is passed to blind reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_digest(path):
    if path.is_file():return digest(path)
    # External mechanical datasets are directories. Hash only input rasters
    # and masks; the selected manifest cases define the evaluated subset.
    h=hashlib.sha256()
    for p in sorted(path.glob('D*/mechanical/D*/*/*')):
        if not p.is_file() or p.parent.name not in ('strips','masks'):continue
        h.update(p.relative_to(path).as_posix().encode());h.update(digest(p).encode())
    return h.hexdigest()


def engine_digest():
    h=hashlib.sha256()
    paths=[ROOT/'pyproject.toml',ROOT/'setup.py',*ROOT.glob('cpp/*.cpp'),
           *ROOT.glob('src/shardelix/**/*.py'),*ROOT.glob('benchmarks/*.py')]
    for p in sorted(paths):
        if not p.is_file():continue
        h.update(p.relative_to(ROOT).as_posix().encode());h.update(digest(p).encode())
    return h.hexdigest()


def command(stage, source, out):
    kind=stage['kind'];profile=stage.get('profile','balanced');seed=int(stage.get('seed',624))
    scripts={'mechanical':'benchmark_mechanical_strips.py','docdataset':'benchmark_scanned_strips.py',
             'scan-stress':'benchmark_scan_stress.py','damage':'benchmark_damage_stress.py',
             'dafne':'benchmark_blind_dafne.py'}
    if kind not in scripts:raise ValueError(f'Unknown benchmark kind: {kind}')
    args=[sys.executable,str(ROOT/'benchmarks'/scripts[kind])]
    if kind!='damage':
        if source is None:raise ValueError(f'{kind} requires source')
        args.append(str(source))
    if kind in ('mechanical','docdataset'):
        args+=['--seed',str(seed),'--profile',profile]
    if kind=='mechanical' and stage.get('cases'):args+=['--cases',*stage['cases']]
    if kind=='dafne':args+=['--case',stage['case'],'--seed',str(seed)]
    return args+['--out',str(out)]


def run(manifest_path, output, resume=False):
    import numpy, scipy, PIL
    from shardelix import __version__
    from shardelix.native_core import backend
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    names=[s['name'] for s in manifest['stages']]
    if not names or len(names)!=len(set(names)) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+',n) for n in names):
        raise ValueError('Stage names must be unique alphanumeric identifiers')
    info={'version':__version__,'engine_sha256':engine_digest(),'backend':backend(),
          'python':platform.python_version(),'platform':platform.platform(),
          'cpu_count':os.cpu_count(),'numpy':numpy.__version__,'scipy':scipy.__version__,'Pillow':PIL.__version__,
          'thread_policy':{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
          'protocol':'Frozen manifest; dataset-specific truth is used by evaluators after blind inference.'}
    # Capture the actual optional accelerator bytes, not just its source.
    native=sorted((ROOT/'src/shardelix').glob('_fl_core.*'))
    info['native_sha256']={p.name:digest(p) for p in native if p.is_file()}
    summary={**info,'stages':[]}
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),**info['thread_policy']}
    for stage in manifest['stages']:
        name=stage['name'];kind=stage['kind'];folder=output/name;folder.mkdir(exist_ok=True)
        destination=folder/('dafne' if kind=='dafne' else 'metrics.json')
        metrics=destination/'metrics.json' if kind=='dafne' else destination
        meta=folder/'run.json';source=stage.get('source')
        source=(manifest_path.parent/Path(source)).resolve() if source else None
        record={'name':name,'kind':kind,'configuration':stage,'status':'pending',
                'metrics':str(metrics.relative_to(output))}
        start=time.perf_counter()
        try:
            if source is not None and not source.exists():raise FileNotFoundError(str(source))
            signature={**info,'stage':stage,'source_sha256':source_digest(source) if source else None}
            if resume and meta.exists() and metrics.exists():
                old=json.loads(meta.read_text(encoding='utf-8'))
                if old.get('signature')==signature and old.get('status')=='completed' and old.get('metrics_sha256')==digest(metrics):
                    summary['stages'].append(old);continue
            record['signature']=signature
            metrics.unlink(missing_ok=True)
            args=command(stage,source,destination)
            with (folder/'runner.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=float(stage.get('timeout_seconds',1800)))
            if result.returncode or not metrics.exists():raise RuntimeError(f'benchmark exit {result.returncode}; see runner.log')
            json.loads(metrics.read_text(encoding='utf-8'))
            record.update(status='completed',metrics_sha256=digest(metrics))
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.TimeoutExpired) as exc:
            record.update(status='failed',error=str(exc))
        record['wall_seconds']=time.perf_counter()-start
        meta.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        summary['stages'].append(record)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        print(name,record['status'],flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return 0 if all(s['status']=='completed' for s in summary['stages']) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(ROOT/'src'))
    raise SystemExit(run(a.manifest.resolve(),a.out,a.resume))
