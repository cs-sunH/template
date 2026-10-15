import datetime,hashlib,json,pathlib,shlex,shutil,subprocess,sys,tarfile,time
root=pathlib.Path('/workspace/template');w=root/'work/elementwise-drain-baseline';e=root/'die/docs/evidence/elementwise-drain-20261016/baseline';sha='a0d327aeeb4a966faab56b426ed88d61e5607478'
paths=['die/CMakeLists.txt','die/src','die/include','die/tests','die/configs','die/validation','astra-sim-wscllm/extern/helper/json/json.hpp']
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=w/'baseline-sdk.tar'
with archive.open('wb') as out:subprocess.run(['git','archive','--format=tar',sha,*paths],cwd=root,stdout=out,check=True)
source=w/'source';source.mkdir()
with tarfile.open(archive) as tf:tf.extractall(source,filter='data')
case_runner=root/'die/docs/analysis/elementwise-drain-20261016/run_extra_cases.py';oracle=root/'die/validation/elementwise_service_oracle.py';frozen_oracle=source/'die/validation/elementwise_service_oracle.py'
shutil.copy2(case_runner,e/'run_extra_cases.py');shutil.copy2(oracle,e/'elementwise_service_oracle.py')
identity={'scope':'fixed a0d327a baseline for nine additional small offline oracle cases only; no hardware calibration or speed benchmark','baseline_commit':sha,'archive_paths':paths,'archive_sha256':h(archive),'source_file_sha256':{str(p.relative_to(source)):h(p) for p in sorted(source.rglob('*')) if p.is_file()},'case_runner_path':str(case_runner),'case_runner_sha256':h(case_runner),'reference_oracle_path':str(oracle),'reference_oracle_sha256':h(oracle),'archived_reference_oracle_sha256':h(frozen_oracle),'same_reference_oracle_as_baseline_commit':h(oracle)==h(frozen_oracle),'driver_source_sha256':h(source/'die/tests/elementwise_oracle_driver.cc')}
(e/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
commands=[('configure',['cmake','-S',str(source/'die'),'-B',str(w/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']),('build',['cmake','--build',str(w/'build'),'--target','coarse_elementwise_oracle_driver','-j2']),('extra-cases',['python3',str(case_runner),'--driver',str(w/'build/coarse_elementwise_oracle_driver'),'--output',str(e/'extra')])]
records=[]
env=__import__('os').environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for label,cmd in commands:
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
 with (e/(label+'.log')).open('w') as log:
  log.write('$ '+shlex.join(cmd)+'\n');log.flush();p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'phase':label,'command':cmd,'started_utc':started,'wall_seconds':time.monotonic()-t,'exit_code':p.returncode,'log':label+'.log'})
 (e/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
 if label=='build':
  files=[w/'build/coarse_elementwise_oracle_driver',w/'build/libcoarse_die.a',w/'build/compile_commands.json',w/'build/CMakeCache.txt']
  identity['build_files']={str(p):{'sha256':h(p),'bytes':p.stat().st_size} for p in files}
  for name,cmd2 in [('driver_readelf',['readelf','-n',str(w/'build/coarse_elementwise_oracle_driver')]),('compiler',['c++','--version']),('cmake',['cmake','--version'])]:
   q=subprocess.run(cmd2,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);identity[name]={'exit_code':q.returncode,'output':q.stdout}
  (e/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps({'baseline_commit':sha,'same_oracle':identity['same_reference_oracle_as_baseline_commit'],'phases':records},indent=2))
