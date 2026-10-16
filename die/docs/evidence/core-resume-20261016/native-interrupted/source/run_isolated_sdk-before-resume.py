#!/usr/bin/env python3
"""Plan, or explicitly execute, isolated SDK ABI and actual consumer checks.

Preparation is separate. No production SDK publisher or source edits run here.
A failure is saved as failed; availability and JOINT provider guards stay active.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPOS=('astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint')
CONFIG_COUNT=dict(zip(REPOS,(21,20,23,18,16)))
PY_MODULES={
 'astra-sim-face':['online.test_die_model_adapter','test_die_model_oracle','test_face_scheduler','online.test_graph_batch_builder'],
 'astra-sim-face-LRU':['online.test_die_model_adapter','test_die_model_oracle','test_face_scheduler','online.test_graph_batch_builder'],
 'astra-sim-wscllm':['online.test_die_model_adapter','online.test_graph_batch_builder','test_wsc_llm_scheduler'],
 'astra-sim-wscllm-LRU':['online.test_die_model_adapter','online.test_graph_batch_builder','test_wsc_llm_scheduler'],
 'astra-sim-joint':['online.test_die_model_adapter','online.test_coarse_die_adapter','online.test_coarse_die_workspace','test_face_scheduler'],
}
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--snapshot',type=Path,help='prepare_isolated_sdk output directory; default work/core-resume-sdk')
 p.add_argument('--output',type=Path,help='fresh result directory under work/; default work/core-resume-sdk-validation/<timestamp>')
 p.add_argument('--repo',choices=REPOS,nargs='+',default=list(REPOS))
 p.add_argument('--phases',choices=('abi','python','native'),nargs='+',default=['abi'])
 p.add_argument('--cmake',default=shutil.which('cmake') or 'cmake')
 p.add_argument('--ctest',default=shutil.which('ctest') or 'ctest')
 p.add_argument('--compiler',default=shutil.which('g++') or 'g++')
 p.add_argument('--jobs',type=int,default=2)
 p.add_argument('--execute',action='store_true',help='default only prints the command plan')
 a=p.parse_args();root=Path(__file__).resolve().parents[4]
 snapshot=(a.snapshot or root/'work/core-resume-sdk').resolve()
 out=(a.output or root/'work/core-resume-sdk-validation'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())).resolve()
 stage=snapshot/'stage-root'
 for folder in (snapshot,out):
  if not folder.is_relative_to((root/'work').resolve()) or folder==(root/'work').resolve():p.error('snapshot/output must be dedicated children of repository work/')
 if a.jobs<1:p.error('--jobs must be positive')
 if len(set(a.repo))!=len(a.repo) or len(set(a.phases))!=len(a.phases):p.error('duplicate repositories/phases')
 manifest_path=snapshot/'manifest.json'
 if manifest_path.is_file():
  manifest=json.loads(manifest_path.read_text());snapshot_status='frozen_manifest_found'
 elif a.execute:
  p.error('snapshot not prepared; run prepare_isolated_sdk.py first')
 else:
  snapshot_status='not_prepared_command_plan_only'
  manifest={'stage_root':str(stage),'public_release':{'source_digest':'not_frozen_no_snapshot'},
            'variants':{name:{'build':str(snapshot/'build-abi'/name)} for name in REPOS}}
 if manifest['stage_root']!=str(stage):p.error('snapshot has been moved; prepare a fresh snapshot')
 commands=[]
 def add(label,argv,*,cwd=root,env=None,expected=None,requires=(),new_build=False):
  commands.append({'label':label,'command':list(map(str,argv)),'cwd':str(cwd),
   'environment':env or {},'expected':expected,'requires':list(requires),'new_build':new_build})
 for name in a.repo:
  repo=stage/name;build=Path(manifest['variants'][name]['build'])
  abi_env={'PYTHONDONTWRITEBYTECODE':'1','DIE_MODEL_CABI_PATH':str(build/'lib/libdie_model_cabi.so'),
           'DIE_COARSE_CABI_PATH':str(build/'lib/libcoarse_die_cabi.so')}
  if 'abi' in a.phases:
   c=name+'-abi-configure';b=name+'-abi-build'
   add(c,[a.cmake,'-S',repo/'extern/die_model','-B',build,'-G','Unix Makefiles',
     '-DCMAKE_CXX_COMPILER='+a.compiler,'-DCMAKE_BUILD_TYPE=Release','-DCOARSE_DIE_BUILD_TESTS=OFF'],new_build=True)
   add(b,[a.cmake,'--build',build,'--parallel',a.jobs,'--target','die_model_cabi','coarse_die_cabi'],requires=[c])
   add(name+'-config',[sys.executable,'-B','-m','unittest','discover','-s',repo/'sh_test_mesh/tests',
      '-p','test_config_resolver*.py','-v'],env=abi_env,expected=('unittest',CONFIG_COUNT[name]),requires=[b])
   add(name+'-typed-resolver',[sys.executable,'-B',stage/'work/integration-audit/probe_typed_resolver.py',
      '--library',build/'lib/libcoarse_die_cabi.so','--label','actual-typed','--child',name],env=abi_env,requires=[b])
  if 'python' in a.phases:
   binding=repo/'extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py';g=name+'-python-bindings'
   want=next((row['generated_sha256'] for row in manifest.get('protobuf_generation',[]) if row['repo']==name),'not_frozen_no_snapshot')
   code='import pathlib,hashlib,sys; p=pathlib.Path(sys.argv[1]); assert p.is_file(), str(p); assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2], "frozen binding changed"; print("verified frozen generated protobuf binding")'
   add(g,[sys.executable,'-B','-c',code,binding,want],cwd=repo)
   workload=repo/'sh_test_mesh/workload/llama2_7b_inference'
   env={**abi_env,'PYTHONPATH':os.pathsep.join(map(str,(repo,repo/'sh_test_mesh',workload)))}
   add(name+'-python-consumers',[sys.executable,'-B','-m','unittest','-v',*PY_MODULES[name]],
       cwd=workload,env=env,expected=('unittest_nonempty',None),requires=[g])
   if name in ('astra-sim-face','astra-sim-face-LRU'):
    add(name+'-real-cost-consumer',[sys.executable,'-B',stage/'die/docs/analysis/analytical-curves-partial-20261015/integration/face_cost_abi_probe.py',
      '--repo-root',repo,'--library',build/'lib/libcoarse_die_cabi.so','--expect','new',
      '--output',out/(name+'-real-cost-consumer.json')],env=env)
  if 'native' in a.phases:
   native=snapshot/'build-native'/name
   prefix='AstraSim_Analytical_Congestion_Aware_'
   targets=[prefix+'DieExecutionLifecycleTest',prefix+'DieModelAdapterTest',prefix+'LocalHbmDieMemoryTest']
   if name in ('astra-sim-face','astra-sim-face-LRU'):targets += [prefix+'WorkspaceAdmissionTest']
   elif name in ('astra-sim-wscllm','astra-sim-wscllm-LRU'):targets += ['AstraSim_Analytical_WorkspaceAdmissionTest']
   if name=='astra-sim-joint':
    targets+=['JointPipelineWindowTest','NativeProjectionCApiTest','NativeCoarseProjectionTest',
              'NativeCoarseDieDomainTest',prefix+'CoarseWorkspaceRuntimeTest',prefix+'HbmTailComponentTest','JointPipelineMutationTest']
   c=name+'-native-configure';b=name+'-native-build'
   add(c,[a.cmake,'-S',snapshot/'aggregate','-B',native,'-G','Unix Makefiles',
     '-DASTRA_SIM_ROOT='+str(repo),'-DBUILDTARGET=congestion_aware','-DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON',
     '-DCMAKE_CXX_COMPILER='+a.compiler,'-DCMAKE_BUILD_TYPE=Release'],new_build=True)
   add(b,[a.cmake,'--build',native,'--parallel',a.jobs,'--target',*targets,'coarse_die_cabi','die_model_cabi',
          *(['joint_projection'] if name=='astra-sim-joint' else [])],requires=[c])
   add(name+'-native-consumers',[a.ctest,'--test-dir',native,'--output-on-failure','-R','^('+'|'.join(targets)+')$'],
       expected=('ctest',len(targets)),requires=[b])
   if name=='astra-sim-joint':
    workload=repo/'sh_test_mesh/workload/llama2_7b_inference'
    env={'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(workload),
      'NATIVE_COARSE_PROJECTION_LIBRARY':str(native/'lib/libjoint_projection.so'),
      'DIE_COARSE_CABI_PATH':str(native/'lib/libcoarse_die_cabi.so'),
      'DIE_MODEL_CABI_PATH':str(native/'lib/libdie_model_cabi.so')}
    add(name+'-native-python',[sys.executable,'-B','-m','unittest','joint.test_native_coarse_projection','-v'],
        cwd=workload,env=env,expected=('unittest_nonempty',None),requires=[b])
 if 'python' in a.phases and all(name in a.repo for name in ('astra-sim-face','astra-sim-face-LRU','astra-sim-joint')):
  build=Path(manifest['variants']['astra-sim-face']['build'])
  add('actual-python-availability',[sys.executable,'-B',stage/'die/docs/analysis/consumer-availability-20261016/probe_python_native.py',
      '--oracle-root',stage,'--library',build/'lib/libcoarse_die_cabi.so','--output',out/'actual-python-availability.json'])
 if not a.execute:
  print(json.dumps({'scope':'dry_run_only','snapshot_status':snapshot_status,'source_digest':manifest['public_release']['source_digest'],'commands':commands},indent=2));return 0
 out.mkdir(parents=True,exist_ok=False)
 def check_frozen():
  bad=[rel for rel,want in manifest['frozen_files_sha256'].items() if not (stage/rel).is_file() or SHA(stage/rel)!=want]
  if bad:raise RuntimeError('frozen stage source changed: '+repr(bad[:12]))
 check_frozen();results=[];status={}
 for command in commands:
  label=command['label'];resultdir=out/label;resultdir.mkdir()
  bad=[x for x in command['requires'] if status.get(x)!='passed']
  if bad:
   row={**command,'status':'not_run_dependency_failed','dependencies':bad};status[label]=row['status'];results.append(row)
   (resultdir/'identity.json').write_text(json.dumps(row,indent=2)+'\n');continue
  if command['new_build']:
   build=Path(command['command'][command['command'].index('-B')+1])
   if build.exists():raise RuntimeError('refusing to reuse native/ABI build directory '+str(build))
  env=os.environ.copy();env.update(command['environment']);start=time.time()
  with (resultdir/'run.log').open('w') as log:
   proc=subprocess.run(command['command'],cwd=command['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
  text=(resultdir/'run.log').read_text();expected=command['expected']
  checks={'process_success':proc.returncode==0}
  if expected:
   kind,count=expected;skipped=bool(re.search(r'\bskipped\b|Not Run',text,re.I));checks['zero_skips']=not skipped
   if kind.startswith('unittest'):
    found=re.findall(r'Ran (\d+) tests? in ',text)
    checks['expected_tests']=bool(found) and (int(found[-1])==count if count is not None else int(found[-1])>0)
   elif kind=='ctest':
    found=re.findall(r'(\d+) tests? failed out of (\d+)',text)
    checks['expected_tests']=bool(found) and found[-1]==('0',str(count))
  check_frozen()
  libraries={str(x):SHA(x) for pattern in ('build-abi/*/lib/*.so','build-native/*/lib/*.so') for x in sorted(snapshot.glob(pattern))}
  row={**command,'status':'passed' if all(checks.values()) else 'failed','checks':checks,
    'return_code':proc.returncode,'elapsed_seconds':time.time()-start,'libraries':libraries,
    'source_digest':manifest['public_release']['source_digest'],'snapshot_manifest_sha256':SHA(snapshot/'manifest.json'),
    'runner_sha256':SHA(Path(__file__)),'log_sha256':SHA(resultdir/'run.log')}
  status[label]=row['status'];results.append(row)
  (resultdir/'identity.json').write_text(json.dumps(row,indent=2)+'\n')
  (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
  print(json.dumps({'label':label,'status':row['status'],'return_code':row['return_code']}),flush=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
 # Preserve all actual consumer failures (including R4 and coarse JOINT guards).
 return int(any(row['status']!='passed' for row in results))
if __name__=='__main__':raise SystemExit(main())
