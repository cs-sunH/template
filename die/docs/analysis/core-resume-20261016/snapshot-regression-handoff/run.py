#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2];work=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=work/'runtime_snapshot_split_owner_test.cc';fixture=work/'fixtures/snapshot_split_owner.json';baseline_a=root/'work/build-core-release/libcoarse_die.a';baseline_so=root/'work/build-core-release/lib/libcoarse_die_cabi.so';candidate_a=root/'work/snapshot-lab/libcoarse_die.a';candidate_so=root/'work/snapshot-lab/libcoarse_die_cabi.so'
tracked=[source,fixture,root/'die/tests/test_support.hh',root/'die/src/Execution.cc',root/'die/include/coarse_die/Execution.hh',baseline_a,baseline_so,candidate_a,candidate_so]
identity={'scope':'Single new C++ C ABI regression source, baseline-red/candidate-green; unchanged prebuilt libraries from this-session fresh baseline and separately identified B-only mixed-object scratch candidate. No production edits, fresh-final acceptance or budget run.','before':{str(p.relative_to(root)):sha(p)for p in tracked},'steps':[]}
def run(name,cmd):
 start=time.monotonic();result=subprocess.run(list(map(str,cmd)),cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);(work/(name+'.stdout.log')).write_text(result.stdout);(work/(name+'.stderr.log')).write_text(result.stderr);identity['steps'].append({'name':name,'command':list(map(str,cmd)),'exit':result.returncode,'elapsed_s':time.monotonic()-start});return result
obj=work/'runtime_snapshot_split_owner_test.o'
compile=run('compile',['c++','-std=c++17','-O3','-DNDEBUG','-I',root/'die/tests','-I',root/'die/include','-I',root/'astra-sim-wscllm/extern/helper','-DCOARSE_DIE_SOURCE_DIR="'+str(root/'die')+'"','-DCOARSE_DIE_ROOT_DIR="'+str(root)+'"','-DCOARSE_DIE_SPLIT_FIXTURE="'+str(fixture)+'"','-c',source,'-o',obj]);assert compile.returncode==0,compile.stderr
for label,a,so in [('baseline',baseline_a,baseline_so),('candidate',candidate_a,candidate_so)]:
 exe=work/(label+'-runtime-snapshot-split-owner');link=run(label+'-link',['c++',obj,a,so,'-Wl,-rpath,'+str(so.parent),'-o',exe]);assert link.returncode==0,link.stderr
 result=run(label+'-test',[exe]);run(label+'-ldd',['ldd',exe])
 identity[label]={'binary_sha256':sha(exe),'static_library_sha256':sha(a),'shared_library_sha256':sha(so),'test_exit':result.returncode}
identity['after']={str(p.relative_to(root)):sha(p)for p in tracked};identity['source_and_libraries_unchanged']=identity['before']==identity['after'];identity['tool_sha256']=sha(pathlib.Path(__file__));(work/'identity.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n');print(json.dumps({k:identity[k]for k in ['baseline','candidate','source_and_libraries_unchanged']}));assert identity['source_and_libraries_unchanged'];assert identity['baseline']['test_exit']==1;assert identity['candidate']['test_exit']==0
