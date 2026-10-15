import datetime,hashlib,json,os,pathlib,shlex,shutil,subprocess,tarfile,time
root=pathlib.Path('/workspace/template');w=root/'work/allocation-overhead-baseline';e=root/'die/docs/evidence/allocation-overhead-20261016/baseline';sha='eba9e972e0514dddb4913dace1c00c57c3f63076'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=w/'source';assert not source.exists();archive=w/'baseline-sdk.tar';paths=['die/CMakeLists.txt','die/src','die/include','die/configs','die/tests','die/validation','astra-sim-wscllm/extern/helper/json/json.hpp']
with archive.open('wb') as f:subprocess.run(['git','archive','--format=tar',sha,*paths],cwd=root,stdout=f,check=True)
source.mkdir()
with tarfile.open(archive) as tf:tf.extractall(source,filter='data')
script=root/'die/docs/analysis/allocation-overhead-20261016/capture_public_results.py';shutil.copy2(script,e/'capture_public_results.py')
consumer_paths=['die/validation/cabi.py','die/configs/example_hardware.json','die/configs/example_tasks.json']
identity={'baseline_commit':sha,'scope':'before phase4 Compiler/Execution allocation optimizations; actual eight-task public C ABI capture, no performance benchmark or old oracle run','archive_paths':paths,'archive_sha256':h(archive),'source_file_sha256':{str(p.relative_to(source)):h(p) for p in sorted(source.rglob('*')) if p.is_file()},'capture_script_path':str(script),'capture_script_sha256':h(script),'consumer_inputs':{name:{'current_sha256':h(root/name),'archived_sha256':h(source/name),'unchanged_from_frozen_baseline':h(root/name)==h(source/name)} for name in consumer_paths},'prior_capture_attempt':{'status':'failed','reported_cabi_status':9,'reason':'initial script destroyed the model without releasing compile handles','before_output_created':False,'source_of_fact':'parent-agent handoff; no initial before artifact was obtained or reused','fix':'current capture script releases every returned plan handle in finally before destroying model'}}
(e/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
commands=[('configure',['cmake','-S',str(source/'die'),'-B',str(w/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']),('build',['cmake','--build',str(w/'build'),'--target','coarse_die_cabi','-j2']),('capture',['python3',str(script),'--library',str(w/'build/lib/libcoarse_die_cabi.so'),'--output',str(e/'public-before.json')])]
records=[];env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for phase,cmd in commands:
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
 with (e/(phase+'.log')).open('w') as log:
  log.write('$ '+shlex.join(cmd)+'\n');log.flush();p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'phase':phase,'command':cmd,'started_utc':started,'wall_seconds':time.monotonic()-t,'exit_code':p.returncode,'log':phase+'.log'});(e/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
 if phase=='build':
  artifacts=[w/'build/lib/libcoarse_die_cabi.so',w/'build/libcoarse_die.a',w/'build/compile_commands.json']
  identity['build_files']={str(p):{'sha256':h(p),'bytes':p.stat().st_size} for p in artifacts}
  for name,cmd2 in [('compiler',['c++','--version']),('cmake',['cmake','--version']),('library_readelf',['readelf','-n',str(w/'build/lib/libcoarse_die_cabi.so')]),('library_ldd',['ldd',str(w/'build/lib/libcoarse_die_cabi.so')])]:
   q=subprocess.run(cmd2,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);identity[name]={'exit_code':q.returncode,'output':q.stdout}
  (e/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
identity['capture_script_sha256_after']=h(script);identity['capture_script_unchanged_during_run']=identity['capture_script_sha256']==h(script);identity['public_before_sha256']=h(e/'public-before.json');identity['capture_status']='passed_actual_new_baseline_capture'
(e/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps({'baseline_commit':sha,'phases':records,'public_before_sha256':identity['public_before_sha256']},indent=2))
