#!/usr/bin/env python3
"""Serial scratch-only compile/link/regressions; never a production certificate."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

root=Path.cwd()
lab=root/'work/runtime-lab'
lib=root/'work/build-core-release/libcoarse_die.a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
production=[p for folder in ['die/src','die/include','die/configs']
            for p in (root/folder).rglob('*') if p.is_file()]
before={str(p.relative_to(root)):sha(p) for p in production}
report={'scope':'scratch Execution override plus current fresh static library and objects; serial functional checks only; not adopted or performance evidence',
        'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'production_before':before,'fresh_library_sha256':sha(lib),
        'candidate_source_sha256':sha(lab/'candidate/Execution.cc'),
        'started_unix':time.time(),'steps':[]}

def save():
    (lab/'focused-identity.json').write_text(json.dumps(report,indent=2)+'\n')

def step(name,command,inputs,timeout=None):
    row={'name':name,'command':list(map(str,command)),
         'inputs_sha256':{str(p.relative_to(root)):sha(p) for p in inputs},
         'started_unix':time.time()}
    report['steps'].append(row)
    save()
    with (lab/(name+'.stdout.log')).open('w') as out,(lab/(name+'.stderr.log')).open('w') as err:
        try:
            run=subprocess.run(command,cwd=root,stdout=out,stderr=err,timeout=timeout)
            row['return_code']=run.returncode
        except subprocess.TimeoutExpired:
            row['timeout_seconds']=timeout
    row['finished_unix']=time.time()
    row['stdout_sha256']=sha(lab/(name+'.stdout.log'))
    row['stderr_sha256']=sha(lab/(name+'.stderr.log'))
    save()
    return row.get('return_code')==0

flags=['g++','-std=c++17','-O3','-DNDEBUG','-I',str(root/'die/tests'),
       '-I',str(root/'die/src'),'-I',str(root/'die/include'),
       '-I',str(root/'astra-sim-wscllm/extern/helper'),
       f'-DCOARSE_DIE_SOURCE_DIR="{root}/die"',f'-DCOARSE_DIE_ROOT_DIR="{root}"']
candidate=lab/'Execution-subunit.o'
if not step('compile-Execution',flags+['-c',str(lab/'candidate/Execution.cc'),'-o',str(candidate)],
            [lab/'candidate/Execution.cc']):
    raise SystemExit(1)
report['candidate_object_sha256']=sha(candidate)

sources=[root/'work/norm-audit/current-baseline/runtime_credit_slack_test.cc',
         root/'work/norm-audit/current-baseline/synchronous_three_activity_candidate.cc',
         lab/'scaled_credit_snapshot_guard_test.cc']
for source in sources:
    name=source.stem
    obj=lab/(name+'.o')
    if not step('compile-'+name,flags+['-c',str(source),'-o',str(obj)],[source]):
        continue
    exe=lab/name
    if step('link-'+name,['g++',str(obj),str(candidate),str(lib),'-o',str(exe)],
            [obj,candidate,lib]):
        step('run-'+name,[str(exe)],[exe],30)

existing=['runtime_credit_scale_test','runtime_clock_translation_test',
          'runtime_owner_symmetry_test','service_curve_runtime_test',
          'ratio_allocator_test','packing_allocator_test',
          'runtime_semantics_test','runtime_submit_transaction_test']
for name in existing:
    obj=root/f'work/build-core-release/CMakeFiles/coarse_{name}.dir/tests/{name}.cc.o'
    exe=lab/name
    if step('link-'+name,['g++',str(obj),str(candidate),str(lib),'-o',str(exe)],
            [obj,candidate,lib]):
        step('run-'+name,[str(exe)],[exe],60)

report['production_after']={str(p.relative_to(root)):sha(p) for p in production}
report['production_unchanged']=before==report['production_after']
report['fresh_library_after_sha256']=sha(lib)
report['fresh_library_unchanged']=report['fresh_library_sha256']==sha(lib)
report['finished_unix']=time.time()
save()
runs=[row for row in report['steps'] if row['name'].startswith('run-')]
print(json.dumps({'actual_run_count':len(runs),
                  'run_return_codes':{r['name']:r.get('return_code','timeout') for r in runs},
                  'production_unchanged':report['production_unchanged'],
                  'fresh_library_unchanged':report['fresh_library_unchanged']}))
