#!/usr/bin/env python3
"""Replay frozen 5c2de28 without touching current production files or git state."""
import argparse, datetime, hashlib, json, pathlib, shlex, subprocess, tarfile, time
p=argparse.ArgumentParser();p.add_argument('--root',default='.',help='template repository root');p.add_argument('--samples',type=int,default=30);a=p.parse_args()
root=pathlib.Path(a.root).resolve();work=root/'work/performance-baseline/replay';work.mkdir(parents=True,exist_ok=True)
sha='5c2de289bc5744e5c252d782901cdba8b6f62459'
paths=['die/CMakeLists.txt','die/src','die/include','die/tests','die/configs','die/validation','astra-sim-wscllm/extern/helper/json/json.hpp']
archive=work/'source.tar'
with archive.open('wb') as out:subprocess.run(['git','archive','--format=tar',sha,*paths],cwd=root,stdout=out,check=True)
source=work/'source';source.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
commands=[['cmake','-S',str(source/'die'),'-B',str(work/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],['cmake','--build',str(work/'build'),'--target','coarse_service_curve_benchmark','-j2'],[str(work/'build/coarse_service_curve_benchmark'),'--samples',str(a.samples),'--output',str(work/'benchmark.json')]]
records=[]
for i,cmd in enumerate(commands):
    if i==2: print('Starting benchmark: ensure no concurrent compiler/test workloads.',flush=True)
    start=time.monotonic()
    with (work/f'step-{i}.log').open('w') as log:
        log.write('$ '+shlex.join(cmd)+'\n');log.flush();r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    records.append({'command':cmd,'exit_code':r.returncode,'wall_seconds':time.monotonic()-start})
    (work/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    if r.returncode and i<2:raise SystemExit(r.returncode)
print(json.dumps(records,indent=2));print('Retained all results under',work)
