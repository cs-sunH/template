#!/usr/bin/env python3
"""Serial functional observations on the already-bound PIC candidate archive."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
import sys

root=Path.cwd();lab=root/'work/runtime-lab';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tracked=[p for d in ['die/src','die/include','die/configs'] for p in (root/d).rglob('*') if p.is_file()]
before={str(p.relative_to(root)):sha(p) for p in tracked}
archive=lab/'public-build/libcoarse_die.a';source=lab/'asymmetric_credit_driver.cc';exe=lab/'asymmetric_credit_driver'
report={'scope':'same previously assembled PIC scratch candidate archive; no production changes, no performance run',
        'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runner_sha256':sha(Path(__file__)),'platform':platform.platform(),
        'source_before':before,'inputs_sha256':{str(p.relative_to(root)):sha(p) for p in
        [source,lab/'coupled_credit_driver.cc',lab/'candidate/Execution.cc',archive,lab/'asymmetric_credit_oracle.py',root/'die/tests/test_support.hh']},
        'started_unix':time.time(),'steps':[]}
def save(): (lab/'asymmetric-identity.json').write_text(json.dumps(report,indent=2)+'\n')
def step(name,command,timeout):
    row={'name':name,'command':list(map(str,command)),'started_unix':time.time()};report['steps'].append(row);save()
    with (lab/(name+'.stdout.log')).open('w') as out,(lab/(name+'.stderr.log')).open('w') as err:
        try: row['return_code']=subprocess.run(command,cwd=root,stdout=out,stderr=err,timeout=timeout).returncode
        except subprocess.TimeoutExpired:row['status']='timeout';row['timeout_seconds']=timeout
        except OSError as e:row['status']='not_run_oserror';row['error']=str(e)
    row['finished_unix']=time.time()
    row['stdout_sha256']=sha(lab/(name+'.stdout.log'));row['stderr_sha256']=sha(lab/(name+'.stderr.log'));save()
    return row.get('return_code')==0
flags=['g++','-std=c++17','-O3','-DNDEBUG','-I',str(root/'die/tests'),'-I',str(root/'die/include'),
       '-I',str(root/'astra-sim-wscllm/extern/helper'),f'-DCOARSE_DIE_SOURCE_DIR="{root}/die"',
       f'-DCOARSE_DIE_ROOT_DIR="{root}"']
if step('asymmetric-compile',flags+[str(source),str(archive),'-o',str(exe)],120):
    report['executable_sha256']=sha(exe)
    step('asymmetric-driver',[str(exe)],120)
    step('asymmetric-oracle',[sys.executable,str(lab/'asymmetric_credit_oracle.py'),
                          str(lab/'asymmetric-driver.stdout.log'),str(lab/'asymmetric-results.json')],60)
report['source_after']={str(p.relative_to(root)):sha(p) for p in tracked}
report['production_unchanged']=report['source_before']==report['source_after']
report['archive_after_sha256']=sha(archive);report['archive_unchanged']=report['archive_after_sha256']==report['inputs_sha256'][str(archive.relative_to(root))]
report['finished_unix']=time.time();save()
print(json.dumps({'steps':{r['name']:r.get('return_code',r.get('status')) for r in report['steps']},
                 'production_unchanged':report['production_unchanged'],'archive_unchanged':report['archive_unchanged']}))
raise SystemExit(any(r.get('return_code')!=0 for r in report['steps']))
