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
from sdk_resume_support import (artifact_snapshot, build_root, cache_validation,
    check_frozen_snapshot, retention, tool_input_snapshot, write_json)

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
 p.add_argument('--resume-from',type=Path,help='retain verified terminal records from an interrupted older result root; output remains fresh')
 p.add_argument('--execute',action='store_true',help='default only prints the command plan')
 a=p.parse_args();root=Path(__file__).resolve().parents[4]
 snapshot=(a.snapshot or root/'work/core-resume-sdk').resolve()
 out=(a.output or root/'work/core-resume-sdk-validation'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())).resolve()
 stage=snapshot/'stage-root'
 resume=a.resume_from.resolve() if a.resume_from else None
 if resume is not None and (not resume.is_dir() or resume==out):p.error('--resume-from must be an existing result directory different from output')
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
   'environment':env or {},'expected':list(expected) if expected else None,'requires':list(requires),'new_build':new_build})
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
   if name=='astra-sim-joint':
    add(name+'-typed-resolver',[sys.executable,'-B',Path(__file__).with_name('probe_coarse_hardware_consumer.py'),
      '--repo-root',repo,'--library',build/'lib/libcoarse_die_cabi.so',
      '--config',repo/'extern/die_model/coarse/configs/example_hardware.json',
      '--output',out/(name+'-typed-hardware-consumer.json')],env=abi_env,requires=[b])
   else:
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
 digest=manifest['public_release']['source_digest']
 manifest_sha=SHA(manifest_path) if manifest_path.is_file() else None
 runner_sha=SHA(Path(__file__))
 support_sha=SHA(Path(__file__).with_name('sdk_resume_support.py'))
 retained={}
 if resume is not None:
  for command in commands:
   decision=retention(resume,command,digest,manifest_sha)
   upstream=[label for label in command['requires']
     if not retained.get(label,{}).get('retain') or retained[label].get('original_status')!='passed']
   if decision.get('retain') and upstream:
    decision.update(retain=False,reason='upstream_not_retained_passed',upstream=upstream)
   retained[command['label']]=decision
 if not a.execute:
  decisions={label:{key:value for key,value in row.items() if key!='old_record'} for label,row in retained.items()}
  print(json.dumps({'scope':'dry_run_only','snapshot_status':snapshot_status,'source_digest':digest,
   'commands':commands,'resume_from':str(resume) if resume else None,'resume_decisions':decisions,
   'actual_frozen_source_hash_check':'required_before_execute; dry-run validates manifest/retention identities only'},indent=2));return 0
 out.mkdir(parents=True,exist_ok=False)
 check_frozen_snapshot(stage,snapshot,manifest)
 results=[];status={}
 for command in commands:
  label=command['label'];resultdir=out/label;resultdir.mkdir()
  decision=retained.get(label,{})
  if decision.get('retain'):
   # Retain failed as failed, passed as passed. Preserve the original record
   # and its runner identity; the new runner is identified separately.
   old=decision['old_record']
   shutil.copy2(Path(decision['source_phase_root'])/'identity.json',resultdir/'retained-original-identity.json')
   shutil.copy2(Path(decision['source_phase_root'])/'run.log',resultdir/'run.log')
   row={**old,'retained_from':{key:value for key,value in decision.items() if key!='old_record'},
     'resumed_by_runner_sha256':runner_sha,'resumed_by_support_sha256':support_sha,
     'retained_at_unix':time.time(),'artifacts_at_retention':artifact_snapshot(command,commands)}
   historical={'status':'not_recorded_by_original_runner; retention snapshot is current, not a historical before/after hash'}
   row.setdefault('artifacts_before',historical);row.setdefault('artifacts_after',historical)
   status[label]=row['status'];results.append(row)
   write_json(resultdir/'identity.json',row);write_json(out/'summary.json',results)
   print(json.dumps({'label':label,'status':row['status'],'action':'retained_verified_original'}),flush=True)
   continue
  bad=[x for x in command['requires'] if status.get(x)!='passed']
  if bad:
   row={**command,'status':'not_run_dependency_failed','dependencies':bad,
     'resume_decision':{key:value for key,value in decision.items() if key!='old_record'}}
   status[label]=row['status'];results.append(row)
   write_json(resultdir/'identity.json',row);write_json(out/'summary.json',results);continue
  if command['new_build']:
   build=build_root(command)
   if build.exists():
    refusal={**command,'status':'not_run_unfamiliar_existing_build','path':str(build),
      'resume_decision':{key:value for key,value in decision.items() if key!='old_record'}}
    write_json(resultdir/'refused.json',refusal)
    raise RuntimeError('refusing to reuse native/ABI build directory without verified retained configure '+str(build))
  check_frozen_snapshot(stage,snapshot,manifest)
  env=os.environ.copy();env.update(command['environment']);start=time.time()
  before=artifact_snapshot(command,commands)
  started={**command,'state':'launch_pending','started_unix':start,'runner_pid':os.getpid(),
   'source_digest':digest,'snapshot_manifest_sha256':manifest_sha,'runner_sha256':runner_sha,
   'support_sha256':support_sha,'tool_inputs_sha256':tool_input_snapshot(command),
   'artifacts_before':before,
   'resume_decision':{key:value for key,value in decision.items() if key!='old_record'}}
  write_json(resultdir/'started.json',started)
  try:
   with (resultdir/'run.log').open('w') as log:
    proc=subprocess.Popen(command['command'],cwd=command['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    started.update(state='process_launched',child_pid=proc.pid)
    write_json(resultdir/'started.json',started)
    returncode=proc.wait()
  except OSError as error:
   with (resultdir/'run.log').open('a') as log:log.write('launch_error: '+str(error)+'\n')
   returncode=127
  text=(resultdir/'run.log').read_text();expected=command['expected']
  checks={'process_success':returncode==0}
  if expected:
   kind,count=expected;skipped=bool(re.search(r'\bskipped\b|Not Run',text,re.I));checks['zero_skips']=not skipped
   if kind.startswith('unittest'):
    found=re.findall(r'Ran (\d+) tests? in ',text)
    checks['expected_tests']=bool(found) and (int(found[-1])==count if count is not None else int(found[-1])>0)
   elif kind=='ctest':
    found=re.findall(r'(\d+) tests? failed out of (\d+)',text)
    checks['expected_tests']=bool(found) and found[-1]==('0',str(count))
  check_frozen_snapshot(stage,snapshot,manifest)
  after=artifact_snapshot(command,commands)
  libraries={str(x):SHA(x) for pattern in ('build-abi/*/lib/*.so','build-native/*/lib/*.so') for x in sorted(snapshot.glob(pattern))}
  row={**command,'status':'passed' if all(checks.values()) else 'failed','checks':checks,
    'return_code':returncode,'elapsed_seconds':time.time()-start,'libraries':libraries,
    'source_digest':digest,'snapshot_manifest_sha256':manifest_sha,
    'runner_sha256':runner_sha,'support_sha256':support_sha,'log_sha256':SHA(resultdir/'run.log'),
    'started_record_sha256':SHA(resultdir/'started.json'),'tool_inputs_sha256':started['tool_inputs_sha256'],
    'artifacts_before':before,'artifacts_after':after,
    'resume_decision':{key:value for key,value in decision.items() if key!='old_record'}}
  status[label]=row['status'];results.append(row)
  write_json(resultdir/'identity.json',row);write_json(out/'summary.json',results)
  print(json.dumps({'label':label,'status':row['status'],'return_code':row['return_code']}),flush=True)
 write_json(out/'summary.json',results)
 # Existing real failures (R4 and JOINT guards) remain failures on resume.
 return int(any(row['status']!='passed' for row in results))
if __name__=='__main__':raise SystemExit(main())
