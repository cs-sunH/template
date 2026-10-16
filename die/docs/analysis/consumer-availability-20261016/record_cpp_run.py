#!/usr/bin/env python3
"""Record real consumer runs with adapter/SDK/build identities, no credentials."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--root',default=str(pathlib.Path(__file__).resolve().parents[4]))
p.add_argument('--prefix',required=True)
p.add_argument('--build',required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
root=pathlib.Path(a.root).resolve()
prefix=pathlib.Path(a.prefix).resolve();prefix.parent.mkdir(parents=True,exist_ok=True)
build=pathlib.Path(a.build).resolve()
command=a.command[1:] if a.command[:1]==['--'] else a.command
if not command:p.error('command required')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
paths=[root/'die/include/coarse_die/Prediction.hh',root/'die/tests/prediction_availability_test.cc',root/'die/CMakeLists.txt',root/'die/tools/estimate.py',root/'die/tests/estimate_cli_test.py']
sdk={}
for repo in ['astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint']:
    workload=root/repo/'astra-sim/workload'
    paths += [workload/name for name in ['CoarseDiePrediction.hh','CoarseDieRuntime.cc','CoarseDieRuntime.hh','DieModelAdapter.cc','DieModelAdapter.hh']]
    paths += list((workload/'execution_driven').rglob('*.cc'))+list((workload/'execution_driven').rglob('*.hh'))
    coarse=root/repo/'extern/die_model/coarse'
    manifest=json.loads((coarse/'RELEASE.json').read_text())
    sdk[repo]={'manifest':manifest,'actual_files_match_manifest':all((coarse/name).is_file() and sha(coarse/name)==value for name,value in manifest['files'].items())}
    paths += [root/repo/'CMakeLists.txt',root/repo/'astra-sim/network_frontend/analytical/CMakeLists.txt']
artifacts=list(build.rglob('*.a'))+list(build.rglob('*.so'))
artifacts += [x for x in build.rglob('*') if x.is_file() and (x.name.startswith('coarse_') and x.name.endswith('_test') or x.name.endswith('Test'))]
executable=pathlib.Path(command[0]);executable=executable if executable.is_absolute() else root/executable
if executable.is_file():artifacts.append(executable)
cache=build/'CMakeCache.txt'
enclosing_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
git_root=pathlib.Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip()).resolve()
meta={'base_head':enclosing_head if git_root==root else None,
      'observed_enclosing_git_head':enclosing_head,'source_checkout_is_git_root':git_root==root,
      'source_root':str(root),
      'command':command,'source_sha256':{str(x.relative_to(root)):sha(x) for x in sorted(set(paths)) if x.is_file()},
      'installed_sdk':sdk,'binary_sha256':{str(x):sha(x) for x in sorted(set(artifacts)) if x.is_file()},
      'recorder_sha256':sha(pathlib.Path(__file__)),'cmake_cache_sha256':sha(cache) if cache.is_file() else None,'started_unix':time.time()}
freeze_path=root.parent/'stage-identity.json'
if git_root!=root and freeze_path.is_file():
    freeze=json.loads(freeze_path.read_text())
    if pathlib.Path(freeze['stage_root']).resolve()!=root:
        raise RuntimeError('isolation freeze does not identify this source root')
    meta['isolated_archive']={'base_commit':freeze['base_commit'],'freeze_sha256':sha(freeze_path),
        'overlay_matches_frozen_source':all((root/name).is_file() and sha(root/name)==value
            for name,value in freeze['overlay_sha256'].items())}
cpu=pathlib.Path('/sys/fs/cgroup/cpu.stat');meta['cpu_before']=cpu.read_text() if cpu.is_file() else None
with pathlib.Path(str(prefix)+'.log').open('w') as log:
    try:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    except OSError as error:
        log.write(str(error)+"\n");result=subprocess.CompletedProcess(command,127)
meta.update(return_code=result.returncode,finished_unix=time.time(),cpu_after=cpu.read_text() if cpu.is_file() else None)
pathlib.Path(str(prefix)+'-identity.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k not in ['source_sha256','installed_sdk','binary_sha256']}))
raise SystemExit(result.returncode)
