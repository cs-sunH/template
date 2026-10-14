from pathlib import Path
import subprocess, os, json, hashlib, re
R=next(p for p in Path(__file__).resolve().parents if (p / 'die/src/Execution.cc').is_file()); E=R/'die/docs/evidence/handoff-lifecycle-online-20261014'
cm=R/'work/tooling/cmake/cmake/data/bin/cmake'; env=dict(os.environ)
env.update(PROTOBUF_FROM_SOURCE='True',CMAKE_PREFIX_PATH=str(R/'work/deps/protobuf'))
env['PATH']=str(R/'work/deps/protobuf/bin')+':'+env['PATH']
records=[]
for repo in ['astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-face','astra-sim-face-LRU','astra-sim-joint']:
 b=R/'work/integration-build/build'/repo; out=E/'integration'/repo; out.mkdir(parents=True,exist_ok=True)
 cmd=[str(cm),'--build',str(b),'--target','coarse_die_cabi','die_model_cabi','-j','2']
 with (out/'final-abi-build.log').open('w') as f: p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 e=dict(env,DIE_MODEL_CABI_PATH=str(b/'lib/libdie_model_cabi.so'),DIE_COARSE_CABI_PATH=str(b/'lib/libcoarse_die_cabi.so'))
 cmd2=['python3','-m','unittest','discover','-s',str(R/repo/'sh_test_mesh/tests'),'-p','test_config_resolver*.py','-v']
 with (out/'final-config-tests.log').open('w') as f: q=subprocess.run(cmd2,cwd=R/repo,env=e,stdout=f,stderr=subprocess.STDOUT)
 log=(out/'final-config-tests.log').read_text(); n=re.search(r'Ran (\d+) tests?',log); skip=re.search(r'skipped=(\d+)',log)
 record=dict(repo=repo,build_exit=p.returncode,config_exit=q.returncode,config_tests=int(n.group(1)) if n else None,skipped=int(skip.group(1)) if skip else 0,commands=[cmd,cmd2],binary_sha256={x:hashlib.sha256((b/'lib'/x).read_bytes()).hexdigest() for x in ['libcoarse_die_cabi.so','libdie_model_cabi.so']})
 records.append(record);print(json.dumps(record),flush=True)
 (E/'five-abi-config.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode or q.returncode: raise SystemExit(1)
