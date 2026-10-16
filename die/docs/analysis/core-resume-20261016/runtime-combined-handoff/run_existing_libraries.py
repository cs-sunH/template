#!/usr/bin/env python3
"""Compile only the frozen small C++ regression; link three existing archives."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

root=Path.cwd();lab=root/'work/runtime-combined-tests';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fixture=lab/'fixture-source'
for src in (root/'die/include/coarse_die').glob('*.hh'):
    dst=fixture/'include/coarse_die'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
# The original baseline declaration is deliberately used for this three-library
# comparison. B adds only a private overload, no members/public layout changes.
shutil.copy2(lab/'source-AB/die/include/coarse_die/Execution.hh',fixture/'include/coarse_die/Execution.hh')
for relative in ['tests/test_support.hh','configs/example_hardware.json']:
    dst=fixture/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/'die'/relative,dst)
libraries={'baseline':root/'work/build-core-release/libcoarse_die.a',
           'A_subunit':root/'work/runtime-lab/public-build/libcoarse_die.a',
           'B_snapshot':root/'work/snapshot-lab/libcoarse_die.a'}
sources={'baseline':root/'die/docs/analysis/core-resume-20261016/runtime-subunit/baseline/Execution.cc',
         'A_subunit':root/'die/docs/analysis/core-resume-20261016/runtime-subunit/candidate/Execution.cc',
         'B_snapshot':root/'die/docs/analysis/core-resume-20261016/snapshot-import/candidate/die/src/Execution.cc'}
test=lab/'runtime_subunit_credit_test.cc';obj=lab/'runtime_subunit_credit_test.o'
identity={'scope':'small frozen C++ regression only; explicit existing static libraries; no AB compile, CABI or budget claim',
          'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'runner_sha256':sha(Path(__file__)),'test_sha256':sha(test),
          'production_execution_before_sha256':sha(root/'die/src/Execution.cc'),
          'production_header_before_sha256':sha(root/'die/include/coarse_die/Execution.hh'),
          'fixture_sha256':{str(p.relative_to(root)):sha(p) for p in fixture.rglob('*') if p.is_file()},
          'libraries':{k:{'path':str(p.relative_to(root)),'sha256':sha(p),
                            'source_path':str(sources[k].relative_to(root)),'source_sha256':sha(sources[k])} for k,p in libraries.items()},
          'started_unix':time.time(),'steps':[]}
def save():(lab/'existing-libraries-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
def step(name,command):
    row={'name':name,'command':list(map(str,command)),'started_unix':time.time()};identity['steps'].append(row);save()
    with (lab/(name+'.stdout.log')).open('w') as out,(lab/(name+'.stderr.log')).open('w') as err:
        try:row['return_code']=subprocess.run(command,cwd=root,stdout=out,stderr=err,timeout=60).returncode
        except subprocess.TimeoutExpired:row['status']='timeout'
        except OSError as e:row['status']='not_run_oserror';row['error']=str(e)
    row['finished_unix']=time.time();row['stdout_sha256']=sha(lab/(name+'.stdout.log'));row['stderr_sha256']=sha(lab/(name+'.stderr.log'));save()
    return row.get('return_code')==0
flags=['g++','-std=c++17','-O3','-DNDEBUG','-I',str(fixture/'tests'),'-I',str(fixture/'include'),
       '-I',str(root/'astra-sim-wscllm/extern/helper'),f'-DCOARSE_DIE_SOURCE_DIR="{fixture}"',f'-DCOARSE_DIE_ROOT_DIR="{root}"']
if step('compile-small-test',flags+['-c',str(test),'-o',str(obj)]):
    identity['test_object_sha256']=sha(obj)
    for name,library in libraries.items():
        exe=lab/('subunit-'+name)
        if step('link-'+name,['g++',str(obj),str(library),'-o',str(exe)]):
            identity['libraries'][name]['linked_executable_sha256']=sha(exe)
            step('run-'+name,[str(exe)])
            identity['libraries'][name]['after_sha256']=sha(library)
            identity['libraries'][name]['library_unchanged']=sha(library)==identity['libraries'][name]['sha256']
identity['production_execution_after_sha256']=sha(root/'die/src/Execution.cc')
identity['production_header_after_sha256']=sha(root/'die/include/coarse_die/Execution.hh')
identity['finished_unix']=time.time();save()
runs={s['name']:s.get('return_code',s.get('status')) for s in identity['steps'] if s['name'].startswith('run-')}
identity['expected_regression_result']={'run-baseline':1,'run-A_subunit':0,'run-B_snapshot':1}
identity['actual_regression_result']=runs
identity['expected_result_observed']=runs==identity['expected_regression_result'];save()
print(json.dumps({'runs':runs,'expected_result_observed':identity['expected_result_observed'],
                  'test_sha256':identity['test_sha256'],'production_execution_sha256':identity['production_execution_after_sha256']}))
raise SystemExit(not identity['expected_result_observed'])
