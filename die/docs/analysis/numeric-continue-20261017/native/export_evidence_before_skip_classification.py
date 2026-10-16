#!/usr/bin/env python3
"""Export only completed fresh native results and text identities, never ELF."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]
WORK = ROOT / 'work/native-resume-20261017'
OUT = ROOT / 'die/docs/evidence/numeric-continue-20261017/native'
REPOS = ('astra-sim-face-LRU', 'astra-sim-wscllm', 'astra-sim-wscllm-LRU')
ALL_REPOS = ('astra-sim-face', *REPOS, 'astra-sim-joint')
CANONICAL = '23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def export():
    summary = json.loads((WORK / 'results/summary.json').read_text())
    expected = {repo + '-' + phase for repo in REPOS
                for phase in ('native-configure', 'native-build', 'native-consumers')}
    assert {row['label'] for row in summary} == expected, 'incomplete phase set'
    assert all((row['status'] in ('passed', 'failed') and
                isinstance(row['return_code'], int)) or
               row['status'] == 'not_run_dependency_failed' for row in summary)
    OUT.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads((WORK / 'sdk/manifest.json').read_text())
    copied = {}

    def copy(source, relative):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == sha(source)
        copied[str(relative)] = {'original_path': str(source), 'sha256': sha(target),
                                 'bytes': target.stat().st_size}

    for path in sorted((WORK / 'results').rglob('*')):
        if path.is_file():
            assert path.suffix in ('.json', '.log'), str(path)
            copy(path, Path('results') / path.relative_to(WORK / 'results'))
    for name in ('manifest.json', 'protobuf-generation.json', 'build-map.json'):
        copy(WORK / 'sdk' / name, Path('snapshot') / name)
    for path in sorted((WORK / 'sdk/protobuf-generation').glob('*.log')):
        copy(path, Path('snapshot/protobuf-generation') / path.name)
    tools = {}
    for name in ('cmake', 'ctest', 'g++', 'protoc', 'python'):
        actual = shutil.which(name)
        assert actual, name
        proc = subprocess.run([actual, '--version'], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tools[name] = {'path': actual, 'resolved_path': str(Path(actual).resolve()),
                       'sha256': sha(actual), 'return_code': proc.returncode,
                       'version_output': proc.stdout}
    script_dir = ROOT / 'die/docs/analysis/core-resume-20261016'
    tools['scripts_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in
        (script_dir / 'prepare_isolated_sdk.py', script_dir / 'run_isolated_sdk.py',
         script_dir / 'sdk_resume_support.py', script_dir / 'aggregate/CMakeLists.txt')}
    tools['activation'] = {'path': '/workspace/cloud-setup/activate.sh',
                           'sha256': sha('/workspace/cloud-setup/activate.sh'),
                           'content': Path('/workspace/cloud-setup/activate.sh').read_text()}
    write(OUT / 'toolchain.json', tools)

    # Identity is from actual live files, not merely the digest label in RELEASE.
    formal = {}
    for repo in ALL_REPOS:
        coarse = ROOT / repo / 'extern/die_model/coarse'
        release = json.loads((coarse / 'RELEASE.json').read_text())
        files = {rel: {'expected': want, 'actual': sha(coarse / rel)}
                 for rel, want in release['files'].items()}
        formal[repo] = {'source_digest': release['source_digest'],
                        'canonical24': release['source_digest'] == CANONICAL and len(files) == 24,
                        'all_files_match': all(v['expected'] == v['actual'] for v in files.values()),
                        'files': files}
    assert all(v['canonical24'] and v['all_files_match'] for v in formal.values())
    write(OUT / 'formal-vendor-check.json', formal)

    public = snapshot['public_release']
    actual_public = {rel: sha(ROOT / 'die' / rel) for rel in public['files']}
    public_equal = actual_public == public['files']
    production_deltas = {rel: {'frozen_ab_sha256': want,
                              'production_sha256_at_export': actual_public[rel]}
                         for rel, want in public['files'].items()
                         if actual_public[rel] != want}
    frozen = snapshot['frozen_files_sha256']
    stage = WORK / 'sdk/stage-root'
    assert all(not p.is_symlink() for p in stage.rglob('*')), 'source symlink'
    mismatches = [rel for rel, want in frozen.items() if sha(stage / rel) != want]
    assert not mismatches
    consumer_files = {rel: want for rel, want in frozen.items()
        if rel.split('/')[0] in ALL_REPOS and '/extern/die_model/coarse/' not in rel
        and not rel.endswith('/et_def_pb2.py')}
    consumer_mismatches = [rel for rel, want in consumer_files.items()
                          if sha(ROOT / rel) != want]
    assert not consumer_mismatches, 'current consumers changed since freeze'
    write(OUT / 'source-verification.json', {
        'current_head_at_export': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'snapshot_source_head': snapshot['source_head'], 'public_digest': public['source_digest'],
        'public_current_files_equal_snapshot': public_equal, 'public_files_sha256': actual_public,
        'production_deltas_at_export': production_deltas,
        'scope_note': 'Native tests are exclusively frozen A+B baseline; later working-tree Runtime candidates are not tested here.',
        'frozen_files_checked': len(frozen), 'frozen_file_mismatches': mismatches,
        'consumer_current_files_checked': len(consumer_files),
        'consumer_current_file_mismatches': consumer_mismatches,
        'consumer_comparison_exclusions': ['isolated new SDK', 'real generated et_def_pb2.py'],
        'no_stage_source_symlinks': True, 'no_builds_or_elf_reused': True})

    cases = {}
    r4_review = {}
    for repo in REPOS:
        build = WORK / 'sdk/build-native' / repo
        for relative in ('CMakeCache.txt', 'Testing/Temporary/LastTest.log',
                         'Testing/Temporary/LastTestsFailed.log', 'CTestTestfile.cmake'):
            path = build / relative
            if path.is_file():
                copy(path, Path('build-text') / repo / relative)
        for path in sorted(build.rglob('link.txt')):
            copy(path, Path('build-text') / repo / path.relative_to(build))
        row = next(r for r in summary if r['label'] == repo + '-native-consumers')
        if row['status'] == 'not_run_dependency_failed':
            cases[repo] = {'native_consumer_status': row['status'],
                           'dependencies': row['dependencies'],
                           'tests': None, 'passed': None, 'failed': None, 'skipped': None}
            continue
        log = (WORK / 'results' / row['label'] / 'run.log').read_text()
        found = re.findall(r'(\d+) tests? failed out of (\d+)', log)
        assert found, repo
        failed, total = map(int, found[-1])
        cases[repo] = {'tests': total, 'passed': total - failed, 'failed': failed,
                       'skipped': bool(re.search(r'\bskipped\b|Not Run', log, re.I)),
                       'lifecycle_r4_guard_failure_observed': 'prediction_not_available' in log,
                       'native_consumer_status': row['status']}
        lifecycle = stage / repo / 'astra-sim/workload/execution_driven/tests/die_execution_lifecycle_test.cc'
        source = lifecycle.read_text()
        fixture = lifecycle.parent / 'testdata/face_emitted_ffn_region.json'
        if repo == 'astra-sim-face-LRU':
            fixture_doc = json.loads(fixture.read_text())
            spec = fixture_doc['emitted_node']['compute']['die_op_spec']
            fixture_identity = {'path': str(fixture), 'sha256': sha(fixture)}
        else:
            raw = re.search(r'face_emitter_ffn_spec_json\(\).*?R"json\((.*?)\)json"', source, re.S)
            assert raw
            spec = json.loads(raw.group(1))
            fixture_identity = {'path': str(lifecycle), 'function': 'face_emitter_ffn_spec_json',
                                'sha256': sha(lifecycle)}
        assert spec['layer_count'] == 2 and spec['weight_passes'] == 2
        r4_review[repo] = {'source_path': str(lifecycle), 'source_sha256': sha(lifecycle),
            'fixture_identity': fixture_identity, 'original_ffn_task': spec,
            'R': spec['layer_count'] * spec['weight_passes'],
            'guard_failure_observed': 'prediction_not_available' in log,
            'not_run_after_abort': 'remaining statements in this lifecycle process; separate CTest tests still ran',
            'guard_and_original_layers_passes_unchanged': True}
    write(OUT / 'r4-lifecycle-review.json', r4_review)
    built_libraries = {repo: {path.name: {'path': str(path), 'sha256': sha(path),
                                        'bytes': path.stat().st_size}
                            for path in sorted((WORK / 'sdk/build-native' / repo / 'lib').glob('*.so'))}
                       for repo in REPOS}
    write(OUT / 'checkpoint-summary.json', {
        'scope': 'fresh frozen A+B baseline native remainder; not later C4r working tree; isolated unpublished SDK; no full business or CPU budget acceptance',
        'public_digest': public['source_digest'], 'source_head_at_freeze': snapshot['source_head'],
        'build_jobs': 2, 'repositories_built_sequentially': True,
        'no_elf_or_build_reuse': True, 'native': cases,
        'fresh_shared_libraries': built_libraries,
        'phases': [{'label': r['label'], 'status': r['status'], 'return_code': r.get('return_code'),
                    'elapsed_seconds': r.get('elapsed_seconds')} for r in summary],
        'not_run': ['FACE/JOINT native re-execution (prior A+B evidence remains separate)',
                    'full five-repository business E2E and CPU<=10% comparison',
                    'public accuracy/performance/calibration', 'SDK publication'],
        'formal_sdk': 'canonical24 unchanged; actual hashes verified',
        'engineering_status': 'NOT ACCEPTED', 'exported_unix': time.time()})
    all_files = {str(p.relative_to(OUT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(OUT.rglob('*')) if p.is_file()}
    write(OUT / 'manifest.json', {'scope': 'text-only; no ELF/object/library/compiler-cache',
                                 'files': all_files, 'copied_original_files': copied,
                                 'exporter_sha256': sha(__file__)})
    print(json.dumps({'exported_files': len(all_files) + 1, 'native': cases,
                      'evidence': str(OUT)}, indent=2))

if __name__ == '__main__':
    export()
