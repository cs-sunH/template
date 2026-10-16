#!/usr/bin/env python3
"""Fresh current legacy Core32 ABI/config verification; never publishes the SDK.

All generated files are confined to a dedicated directory under work/.
Use --scope full for the legacy C++ unit suite; abi builds only die_model_cabi.
The installed coarse SDK is hashed for identity but not built or endorsed here.
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

REPOS = ('astra-sim-face', 'astra-sim-face-LRU', 'astra-sim-wscllm',
         'astra-sim-wscllm-LRU', 'astra-sim-joint')
EXPECTED_CONFIG = dict(zip(REPOS, (21, 20, 23, 18, 16)))
TARGETS = ['die_model_cabi', 'tx8_reference_test', 'hardware_test',
           'spec_parse_test', 'configs_consistency_test', 'workload_test',
           'mapping_test', 'costmodel_test', 'tx8_compile_test',
           'execution_test', 'selection_fuzz_test', 'capi_test']
JOINT_TARGETS = ['internal_slice_test', 'internal_service_progress_test',
                 'k2_timer_window_memo_test', 'k2_plan_negative_cache_test']
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def source_identity(root: Path, repo: Path) -> dict:
    base = repo / 'extern/die_model'
    files = [p for p in base.rglob('*') if p.is_file()
             and 'coarse' not in p.relative_to(base).parts]
    files += [repo / 'sh_test_mesh/config_resolver.py']
    files += list((repo / 'sh_test_mesh/tests').glob('test_config_resolver*.py'))
    files += list((repo / 'sh_test_mesh/hardware').glob('*.json'))
    files += [p for p in (repo / 'astra-sim/workload/execution_driven/tests').glob('*.cc')
              if p.name in ('die_model_adapter_test.cc', 'die_execution_lifecycle_test.cc',
                            'local_hbm_die_memory_test.cc')]
    return {str(p.relative_to(root)): SHA(p) for p in sorted(set(files))}

def coarse_identity(repo: Path) -> dict:
    base = repo / 'extern/die_model/coarse'
    release = json.loads((base / 'RELEASE.json').read_text())
    bad = [rel for rel, want in release['files'].items()
           if not (base/rel).is_file() or SHA(base/rel) != want]
    actual = hashlib.sha256(json.dumps(release['files'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if bad or actual != release['source_digest']:
        raise RuntimeError('installed coarse manifest/content mismatch: ' + repr(bad))
    return {'source_digest': actual, 'file_count': len(release['files']),
            'scope': 'installed historical canonical SDK identity only; not current SDK validation'}

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', choices=REPOS, nargs='+', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--scope', choices=('abi', 'full'), default='full')
    p.add_argument('--phases', choices=('configure','build','unit','config','default'),
                   nargs='+', default=['configure','build','unit','config','default'])
    p.add_argument('--cmake', default=shutil.which('cmake'))
    p.add_argument('--ctest', default=shutil.which('ctest'))
    p.add_argument('--compiler', default=shutil.which('g++'))
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--output-root', type=Path, help='dedicated output root under repository work/; default work/core-resume-validation')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.label):
        p.error('--label must be a simple directory name')
    if not a.cmake or not a.ctest or not a.compiler: p.error('activate the toolchain or pass --cmake --ctest --compiler')
    if a.jobs < 1: p.error('--jobs must be positive')
    if a.scope == 'abi' and 'unit' in a.phases: p.error('--scope abi cannot run unit; use --phases configure build config default')
    if len(set(a.phases)) != len(a.phases): p.error('duplicate phases')
    order = ('configure','build','unit','config','default')
    if sorted(a.phases, key=order.index) != a.phases: p.error('phases must follow configure/build/unit/config/default order')
    root = Path(__file__).resolve().parents[4]
    output_root = (a.output_root or root / 'work/core-resume-validation').resolve()
    if not output_root.is_relative_to((root / 'work').resolve()) or output_root == (root / 'work').resolve():
        p.error('--output-root must be a dedicated child of repository work/')
    output = output_root / a.label
    output.mkdir(parents=True, exist_ok=True)
    overall = []
    for name in a.repo:
        repo = root / name
        run = output / name
        build = run / 'build'
        run.mkdir(exist_ok=True)
        initial = source_identity(root, repo)
        coarse = coarse_identity(repo)
        target = TARGETS + (JOINT_TARGETS if name == 'astra-sim-joint' else [])
        if a.scope == 'abi': target = ['die_model_cabi']
        default_code = '''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from config_resolver import load_hardware_config
h=load_hardware_config(Path(sys.argv[2]),"validation-die-64g")
assert h.die_derived["cores"]==100
assert abs(h.die_derived["pe_peak_int4_tops"]-126.6)<1e-12
assert h.die_node["die-width-mm"]==25 and h.die_node["die-height-mm"]==32
assert h.die_derived["die_geometry_mode"]=="explicit_width_height"
assert h.local_hbm_capacity_bytes==64000000000
print(json.dumps({"status":"passed_authorized_engineering_point","scope":"legacy HBM4x16GB compatibility; not target calibration","summary":h.die_summary}))'''
        commands = {
          'configure': [a.cmake,'-S',str(repo/'extern/die_model'),'-B',str(build),
            '-G','Unix Makefiles','-DCMAKE_CXX_COMPILER='+a.compiler,
            '-DCMAKE_BUILD_TYPE=Release','-DCOARSE_DIE_BUILD_TESTS=OFF'],
          'build': [a.cmake,'--build',str(build),'--parallel',str(a.jobs),'--target']+target,
          'unit': [a.ctest,'--test-dir',str(build),'--output-on-failure'],
          'config': [sys.executable,'-B','-m','unittest','discover','-s',
            str(repo/'sh_test_mesh/tests'),'-p','test_config_resolver*.py','-v'],
          'default': [sys.executable,'-B','-c',default_code,str(repo/'sh_test_mesh'),
            str(repo/'sh_test_mesh/hardware/face_case5_config_c.json')],
        }
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['DIE_MODEL_CABI_PATH'] = str(build/'lib/libdie_model_cabi.so')
        # Intentionally points to an absent file: this runner certifies legacy,
        # while actual typed coarse consumers require isolated current SDK builds.
        env['DIE_COARSE_CABI_PATH'] = str(build/'lib/libcoarse_die_cabi.so')
        for phase in a.phases:
            phase_dir = run/'phases'/phase
            phase_dir.mkdir(parents=True, exist_ok=False)
            if phase == 'configure' and build.exists():
                raise RuntimeError('refusing to reuse build directory: '+str(build))
            if phase in ('config','default') and not (build/'lib/libdie_model_cabi.so').is_file():
                raise RuntimeError('required fresh legacy ABI library is missing')
            command = commands[phase]
            identity = {'repo':name,'phase':phase,'scope':a.scope,
                'command':command,'cwd':str(root),'runner_sha256':SHA(Path(__file__)),
                'source_sha256':initial,'installed_coarse_identity':coarse,
                'started_unix':time.time()}
            with (phase_dir/'run.log').open('w') as log:
                result = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            log_text = (phase_dir/'run.log').read_text()
            tests = re.findall(r'Ran (\d+) tests? in ',log_text)
            ctests = re.findall(r'(\d+) tests? failed out of (\d+)',log_text)
            skipped = bool(re.search(r'\bskipped\b|\*\*\*Skipped|Not Run', log_text, re.I))
            current = source_identity(root,repo)
            unchanged = initial == current and coarse_identity(repo) == coarse
            checks = {'process_success':result.returncode == 0,'source_unchanged':unchanged}
            if phase == 'config':
                checks.update(expected_count=bool(tests) and int(tests[-1]) == EXPECTED_CONFIG[name],zero_skips=not skipped)
            if phase == 'unit':
                checks.update(expected_count=bool(ctests) and ctests[-1] == ('0','15' if name == 'astra-sim-joint' else '11'),zero_skips=not skipped)
            identity.update(return_code=result.returncode,finished_unix=time.time(),
                tests_run=int(tests[-1]) if tests else None,ctest_counts=ctests,
                checks=checks,status='passed' if all(checks.values()) else 'failed',
                log_sha256=SHA(phase_dir/'run.log'),
                libraries={str(x.relative_to(root)):SHA(x) for x in sorted((build/'lib').glob('*.so'))})
            (phase_dir/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
            row = {k:identity[k] for k in ('repo','phase','status','return_code','tests_run','checks')}
            overall.append(row)
            (output/'summary.json').write_text(json.dumps(overall,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if identity['status'] != 'passed':
                print(log_text[-6500:],file=sys.stderr)
                return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
