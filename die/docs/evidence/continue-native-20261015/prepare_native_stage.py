#!/usr/bin/env python3
"""Prepare immutable JOINT integration preview; never publish or build automatically."""
import argparse,hashlib,json,pathlib,shlex,shutil,subprocess,tarfile
p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--sdk',required=True,help='already frozen and validated common die SDK directory');p.add_argument('--source-commit',required=True,help='exact committed JOINT source identity');p.add_argument('--tag',required=True,help='new stage name; existing paths are refused');a=p.parse_args()
root=pathlib.Path(a.root).resolve();sdk=pathlib.Path(a.sdk).resolve()
if not a.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.tag):p.error('tag must be a simple new directory name')
work=root/'work/native-dependency-check'/a.tag
if work.exists():p.error('refusing to replace existing '+str(work))
sha=subprocess.check_output(['git','rev-parse',a.source_commit+'^{commit}'],cwd=root,text=True).strip()
for name in ['CMakeLists.txt','include','src','configs']:
 if not (sdk/name).exists():p.error('missing frozen SDK member: '+name)
work.mkdir(parents=True);archive=work/'joint-source.tar';stage=work/'stage-root';stage.mkdir()
paths=['astra-sim-joint','die/docs/analysis/native-local-projection-20261014/online-reproduction']
with archive.open('wb') as out:subprocess.run(['git','archive','--format=tar',sha,*paths],cwd=root,stdout=out,check=True)
with tarfile.open(archive) as tar:tar.extractall(stage,filter='data')
coarse=stage/'astra-sim-joint/extern/die_model/coarse';shutil.rmtree(coarse);coarse.mkdir()
shutil.copy2(sdk/'CMakeLists.txt',coarse/'CMakeLists.txt')
for name in ['include','src','configs']:shutil.copytree(sdk/name,coarse/name)
files={str(x.relative_to(coarse)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(coarse.rglob('*')) if x.is_file()}
manifest={'schema':'coarse_die_release_v2','source_digest':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'files':files}
(coarse/'RELEASE.json').write_text(json.dumps(manifest,indent=2)+'\n')
prefix=pathlib.Path('/workspace/cloud-setup/sysroot/usr');lib=prefix/'lib/x86_64-linux-gnu';build=work/'build'
for path in [prefix/'include/google/protobuf/message.h',prefix/'bin/protoc',lib/'libprotobuf.so',lib/'libprotoc.so.32']:
 if not path.is_file():raise SystemExit('dependency not available: '+str(path))
aggregate=stage/'die/docs/analysis/native-local-projection-20261014/online-reproduction/aggregate';joint=stage/'astra-sim-joint'
commands={
 'configure':['cmake','-S',str(aggregate),'-B',str(build),'-DREPOSITORY_ROOT='+str(root),'-DASTRA_SIM_ROOT='+str(joint),'-DCMAKE_BUILD_TYPE=Release','-DBUILDTARGET=congestion_aware','-DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON','-DCOARSE_DIE_BUILD_TESTS=OFF','-DCOARSE_DIE_JSON_INCLUDE='+str(joint/'extern/helper'),'-DProtobuf_INCLUDE_DIR='+str(prefix/'include'),'-DProtobuf_LIBRARY='+str(lib/'libprotobuf.so'),'-DProtobuf_PROTOC_EXECUTABLE='+str(prefix/'bin/protoc'),'-DProtobuf_PROTOC_LIBRARY='+str(lib/'libprotoc.so.32')],
 'build':['cmake','--build',str(build),'-j2','--target','NativeCoarseDieDomainTest','NativeCoarseProjectionTest','NativeProjectionCApiTest','JointPipelineWindowTest','joint_projection','AstraSim_Analytical_Congestion_Aware_Online','die_model_cabi','coarse_die_cabi','joint_ledger_cabi'],
 'native_ctest':['ctest','--test-dir',str(build),'--output-on-failure','-R','^(NativeCoarseDieDomainTest|NativeCoarseProjectionTest|NativeProjectionCApiTest|JointPipelineWindowTest)$'],
 'native_python':['python3','-m','unittest','joint.test_native_coarse_projection','-v'],
}
environment={'PROTOBUF_FROM_SOURCE':'unset','LD_LIBRARY_PATH':str(lib)+':'+str(build/'lib')+':${LD_LIBRARY_PATH:-}','NATIVE_COARSE_PROJECTION_LIBRARY':str(build/'lib/libjoint_projection.so'),'DIE_COARSE_CABI_PATH':str(build/'lib/libcoarse_die_cabi.so'),'DIE_MODEL_CABI_PATH':str(build/'lib/libdie_model_cabi.so'),'PYTHONPATH':str(joint/'sh_test_mesh/workload/llama2_7b_inference'),'PYTHONDONTWRITEBYTECODE':'1','JOINT_REPRO_REPO_ROOT':str(root),'ASTRA_SIM_ROOT':str(joint),'BUILD_DIR':str(build),'JOINT_REPRO_WORKDIR':str(work/'online')}
for profile in ['legacy','coarse_intended']:
 commands['online_'+profile]=['bash',str(stage/'die/docs/analysis/native-local-projection-20261014/online-reproduction/run_tiny_online.sh'),profile]
identity={'joint_source_commit':sha,'joint_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'sdk_source_path':str(sdk),'sdk_release':manifest,'stage_root':str(stage),'build_root':str(build),'commands':commands,'environment':environment,'status':'prepared_not_configured_or_built','guard_contract':'coarse_intended must retain its real provider guard; live_world_import_supported is not changed'}
(work/'stage-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
lines=['#!/usr/bin/env bash','set -euo pipefail','source /workspace/cloud-setup/activate.sh','unset PROTOBUF_FROM_SOURCE']
for key,value in environment.items():
 if key=='PROTOBUF_FROM_SOURCE':continue
 if key=='LD_LIBRARY_PATH':lines.append('export LD_LIBRARY_PATH='+shlex.quote(str(lib)+':'+str(build/'lib'))+':"${LD_LIBRARY_PATH:-}"')
 else:lines.append('export '+key+'='+shlex.quote(value))
lines+=['case "${1:?choose configure|build|native_ctest|native_python|online_legacy|online_coarse_intended}" in']
for name,cmd in commands.items():lines.append('  '+name+') exec '+shlex.join(cmd)+' ;;')
lines+=['  *) exit 2 ;;','esac']
runner=work/'run_phase.sh';runner.write_text('\n'.join(lines)+'\n');runner.chmod(0o755)
print(json.dumps({'status':identity['status'],'joint_source_commit':sha,'sdk_digest':manifest['source_digest'],'phase_runner':str(runner),'commands':commands},indent=2))
