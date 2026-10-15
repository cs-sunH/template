import datetime,hashlib,json,os,pathlib,shlex,subprocess,time
root=pathlib.Path('/workspace/template');w=root/'work/native-dependency-check';e=root/'die/docs/evidence/continue-native-20261015';prefix=pathlib.Path('/workspace/cloud-setup/sysroot/usr');lib=prefix/'lib/x86_64-linux-gnu'
paths=[prefix/'bin/protoc',lib/'libprotobuf.so',lib/'libprotoc.so.32',prefix/'include/google/protobuf/stubs/common.h',w/'probe/et_def.proto']
identity={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'dependencies':{str(x):{'exists':x.exists(),'realpath':str(x.resolve()),'sha256':hashlib.sha256(x.read_bytes()).hexdigest() if x.exists() else None} for x in paths},'legacy_failure':'missing stale work/protobuf-root libprotobuf.so path; not a source compile failure','no_network_install_needed':True,'scope':'dependency smoke only; no current SDK/public model integration claim'}
commands=[['protoc','--version'],['python3','-c','import google.protobuf; print(google.protobuf.__version__)'],['ldd',str(prefix/'bin/protoc')],['ldd',str(lib/'libprotobuf.so')],['cmake','-S',str(w/'probe'),'-B',str(w/'probe-build'),'-DProtobuf_INCLUDE_DIR='+str(prefix/'include'),'-DProtobuf_LIBRARY='+str(lib/'libprotobuf.so'),'-DProtobuf_PROTOC_EXECUTABLE='+str(prefix/'bin/protoc'),'-DProtobuf_PROTOC_LIBRARY='+str(lib/'libprotoc.so.32'),'-DCMAKE_BUILD_TYPE=Debug'],['cmake','--build',str(w/'probe-build'),'--target','protobuf_smoke','-j1'],[str(w/'probe-build/protobuf_smoke')],[str(prefix/'bin/protoc'),'--proto_path='+str(w/'probe'),'--python_out='+str(w/'probe-build'),str(w/'probe/et_def.proto')],['python3','-c','import sys; sys.path.insert(0,sys.argv[1]); import et_def_pb2; x=et_def_pb2.Node(id=42,name="schema-smoke",data_deps=[7]); y=et_def_pb2.Node.FromString(x.SerializeToString()); assert x==y; print("protobuf Python schema roundtrip passed")',str(w/'probe-build')]]
records=[]
for i,cmd in enumerate(commands):
 t=time.monotonic()
 with (e/f'dependency-{i:02d}.log').open('w') as log:
  log.write('$ '+shlex.join(cmd)+'\n');log.flush();run=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 records.append({'command':cmd,'exit_code':run.returncode,'wall_seconds':time.monotonic()-t,'log':f'dependency-{i:02d}.log'})
 if run.returncode:break
identity['commands']=records;identity['status']='passed' if len(records)==len(commands) and all(r['exit_code']==0 for r in records) else 'failed'
for name in ['resolved.txt','protobuf_smoke']:
 p=w/'probe-build'/name
 if p.exists():identity[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'content':p.read_text() if name.endswith('.txt') else None}
(e/'dependency-check.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps({'status':identity['status'],'commands':len(records)},indent=2))
