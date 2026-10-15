#!/usr/bin/env python3
"""Freeze five consumer trees and the current SDK in a new isolated preview."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import tarfile

p=argparse.ArgumentParser()
p.add_argument('--tag',required=True)
a=p.parse_args()
if not a.tag or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in a.tag):
    p.error('tag must name a new simple workspace directory')
root=pathlib.Path(__file__).resolve().parents[4]
work=root/'work/consumer-availability'/a.tag
if work.exists():p.error('refusing to replace an existing preview')
repos=['astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint']
sdk_paths=['die/CMakeLists.txt','die/include','die/src','die/configs','die/tools/sync_repositories.py']
aggregate='die/docs/analysis/native-local-projection-20261014/online-reproduction/aggregate'
base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
formal_before={repo:sha(root/repo/'extern/die_model/coarse/RELEASE.json') for repo in repos}
work.mkdir(parents=True);stage=work/'stage-root';stage.mkdir();archive=work/'source.tar'
with archive.open('wb') as out:
    subprocess.run(['git','archive','--format=tar',base,*repos,*sdk_paths,aggregate],cwd=root,stdout=out,check=True)
with tarfile.open(archive) as source:source.extractall(stage,filter='data')
modified=subprocess.check_output(['git','diff','--name-only',base,'--',*repos,*sdk_paths],cwd=root,text=True).splitlines()
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*repos,*sdk_paths],cwd=root,text=True).splitlines()
overrides={}
for name in sorted(set(modified+untracked)):
    if any(name.startswith(repo+'/extern/die_model/coarse/') for repo in repos):
        raise RuntimeError('formal vendor has unreviewed changes: '+name)
    source=root/name;target=stage/name
    if not source.is_file():raise RuntimeError('preview refuses an unexplained source deletion: '+name)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);overrides[name]=sha(source)
with (work/'sdk-preview-publish.log').open('w') as log:
    subprocess.run(['python3',str(stage/'die/tools/sync_repositories.py')],cwd=stage,stdout=log,stderr=subprocess.STDOUT,check=True)
formal_after={repo:sha(root/repo/'extern/die_model/coarse/RELEASE.json') for repo in repos}
if formal_before!=formal_after:raise RuntimeError('formal SDK manifests changed during isolated preparation')
prefix=pathlib.Path('/workspace/cloud-setup/sysroot/usr');lib=prefix/'lib/x86_64-linux-gnu'
commands={};releases={}
for repo in repos:
    source=stage/repo;build=work/'build'/repo
    configure=['cmake','-S',str(stage/aggregate),'-B',str(build),'-DREPOSITORY_ROOT='+str(stage),
        '-DASTRA_SIM_ROOT='+str(source),'-DCMAKE_BUILD_TYPE=Release','-DBUILDTARGET=congestion_aware',
        '-DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON','-DCOARSE_DIE_BUILD_TESTS=OFF',
        '-DCOARSE_DIE_JSON_INCLUDE='+str(source/'extern/helper'),'-DProtobuf_INCLUDE_DIR='+str(prefix/'include'),
        '-DProtobuf_LIBRARY='+str(lib/'libprotobuf.so'),'-DProtobuf_PROTOC_EXECUTABLE='+str(prefix/'bin/protoc'),
        '-DProtobuf_PROTOC_LIBRARY='+str(lib/'libprotoc.so.32')]
    targets=['AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest','die_model_cabi','coarse_die_cabi']
    tests=['AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest']
    if repo=='astra-sim-joint':
        targets+=['AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest','NativeCoarseDieDomainTest',
            'NativeCoarseProjectionTest','NativeProjectionCApiTest','JointPipelineWindowTest','joint_projection']
        tests+=targets[3:8]
    commands[repo]={'configure':configure,'build':['cmake','--build',str(build),'--parallel','2','--target',*targets],
        'ctest':['ctest','--test-dir',str(build),'--output-on-failure','-R','^('+'|'.join(tests)+')$']}
    releases[repo]=json.loads((source/'extern/die_model/coarse/RELEASE.json').read_text())
identity={'status':'frozen_not_configured_or_built','base_commit':base,'archive_sha256':sha(archive),
    'overlay_sha256':overrides,'stage_root':str(stage),'formal_manifests_before':formal_before,
    'formal_manifests_after':formal_after,'preview_releases':releases,'commands':commands,
    'guard_contract':'isolated SDK only; formal vendor and JOINT provider/live/sh30 guards remain intact'}
(work/'stage-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps({'status':identity['status'],'stage_root':str(stage),'base_commit':base,
    'overlay_files':len(overrides),'sdk_digests':{repo:value['source_digest'] for repo,value in releases.items()}}))
