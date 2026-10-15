import datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,time
root=pathlib.Path('/workspace/template');w=root/'work/native-dependency-check/elementwise-preview';e=root/'die/docs/evidence/continue-native-20261015';identity=json.loads((w/'stage-identity.json').read_text());phase=sys.argv[1]
assert identity['sdk_release']['source_digest']=='56e684c09eca0d6d3aa2677f03775219d05611c536b6546de741bd8f24e0f948',identity['sdk_release']['source_digest']
env=os.environ.copy();env.pop('PROTOBUF_FROM_SOURCE',None)
for name,value in identity['environment'].items():
 if name=='PROTOBUF_FROM_SOURCE':continue
 env[name]=value.replace('${LD_LIBRARY_PATH:-}',os.environ.get('LD_LIBRARY_PATH',''))
if phase=='native-build':
 command=['cmake','--build',identity['build_root'],'-j1','--target','NativeCoarseDieDomainTest','NativeCoarseProjectionTest','NativeProjectionCApiTest','JointPipelineWindowTest','joint_projection','die_model_cabi','coarse_die_cabi']
elif phase=='online-build':
 command=['cmake','--build',identity['build_root'],'-j1','--target','AstraSim_Analytical_Congestion_Aware_Online','joint_ledger_cabi']
elif phase=='python-schema':
 joint=pathlib.Path(identity['stage_root'])/'astra-sim-joint';schema=joint/'extern/graph_frontend/chakra/schema/protobuf'
 command=['protoc','--proto_path='+str(schema),'--python_out='+str(schema),str(schema/'et_def.proto')]
elif phase=='config-abi':
 command=['python3','-m','unittest','discover','-s',str(pathlib.Path(identity['stage_root'])/'astra-sim-joint/sh_test_mesh/tests'),'-p','test_config_resolver*.py','-v']
elif phase=='adapter-abi':
 workload=pathlib.Path(identity['stage_root'])/'astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference'
 command=['python3','-m','pytest','-p','no:cacheprovider','-q',str(workload/'online/test_die_model_adapter.py'),str(workload/'online/test_coarse_die_adapter.py'),str(workload/'online/test_coarse_die_workspace.py')]
else:command=identity['commands'][phase]
joint=pathlib.Path(identity['stage_root'])/'astra-sim-joint'
env['PYTHONPATH']=':'.join([str(joint),str(joint/'sh_test_mesh'),identity['environment']['PYTHONPATH']])
started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
stem='elementwise-'+phase
attempt=0
while (e/(stem+('.retry'+str(attempt) if attempt else '')+'.log')).exists():attempt+=1
stem+=('.retry'+str(attempt) if attempt else '')
logpath=e/(stem+'.log')
with logpath.open('w') as log:
 log.write('$ '+shlex.join(command)+'\n');log.flush();p=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
meta={'phase':phase,'command':command,'environment':{name:env.get(name,'unset') for name in identity['environment']},'cwd':str(root),'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-t,'exit_code':p.returncode,'joint_source_commit':identity['joint_source_commit'],'sdk_source_digest':identity['sdk_release']['source_digest'],'log':str(logpath),'log_sha256':hashlib.sha256(logpath.read_bytes()).hexdigest(),'scope':'isolated preview only; no official vendor publish'}
(e/(stem+'.meta.json')).write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2));sys.exit(p.returncode)
