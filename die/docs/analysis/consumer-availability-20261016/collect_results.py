#!/usr/bin/env python3
"""Collect current, source-bound consumer checks without replacing failures."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--require-old-pass', action='store_true')
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
evidence = root / 'die/docs/evidence/consumer-availability-20261016'
output = pathlib.Path(a.output).resolve()
if output.exists(): p.error('fresh output required')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
repos = ['astra-sim-face', 'astra-sim-face-LRU', 'astra-sim-wscllm', 'astra-sim-wscllm-LRU', 'astra-sim-joint']
summary = {'observed_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
           'scope': 'consumer availability, not model accuracy / hardware calibration / five-workload CPU acceptance',
           'matrices': {}, 'old_sdk_current_tests': {}, 'formal_sdk': {}}
for label in ['old-face-joint', 'old-three', 'old-lru-public-seam', 'preview-five']:
    file = evidence / label / 'matrix.json'
    if not file.is_file(): continue
    matrix = json.loads(file.read_text())
    snapshot = output.with_name(output.stem + '-' + label + '-matrix.json')
    if snapshot.exists(): raise RuntimeError('fresh matrix snapshot required')
    snapshot.write_bytes(file.read_bytes())
    summary['matrices'][label] = {'sha256': sha(file), 'snapshot': str(snapshot),
                                  'mode': matrix['mode'], 'runs': matrix['runs']}
    if matrix['mode'] != 'old': continue
    for repo, runs in matrix['runs'].items():
        test = runs.get('ctest')
        if not test: continue
        identity = pathlib.Path(test['identity'])
        meta = json.loads(identity.read_text())
        # Check the actual tested repository, shared policy and its fixture,
        # rather than asserting other repositories were compiled in this run.
        current = {name: digest == sha(root / name)
                   for name, digest in meta['source_sha256'].items()
                   if name.startswith(repo + '/') or name == 'die/include/coarse_die/Prediction.hh'}
        summary['old_sdk_current_tests'][repo] = {
            'matrix': label, 'return_code': test['return_code'],
            'identity_sha256': sha(identity), 'all_tested_sources_still_match': all(current.values()),
            'source_checks': current,
            'sdk_matches_manifest': meta['installed_sdk'][repo]['actual_files_match_manifest']}
        build = pathlib.Path(test['command'][test['command'].index('--test-dir') + 1])
        raw = build / 'Testing/Temporary/LastTest.log'
        if raw.is_file():
            retained = evidence / label / (repo + '-LastTest.log')
            if retained.exists() and retained.read_bytes() != raw.read_bytes():
                raise RuntimeError('refusing to replace an existing raw CTest log: ' + str(retained))
            shutil.copyfile(raw, retained)
            summary['old_sdk_current_tests'][repo]['last_test_log_sha256'] = sha(retained)
for repo in repos:
    sdk = root / repo / 'extern/die_model/coarse'
    manifest = sdk / 'RELEASE.json'
    release = json.loads(manifest.read_text())
    summary['formal_sdk'][repo] = {'digest': release['source_digest'], 'files': len(release['files']),
        'manifest_sha256': sha(manifest), 'all_files_match': all(
            (sdk / name).is_file() and sha(sdk / name) == digest for name, digest in release['files'].items())}
canonical = root / 'die/include/coarse_die/Prediction.hh'
summary['compatibility_headers_match_canonical'] = all(
    (root / repo / 'astra-sim/workload/CoarseDiePrediction.hh').read_bytes() == canonical.read_bytes() for repo in repos)
summary['public_production_matches_9bb'] = all(
    path.read_bytes() == subprocess.check_output(['git', 'show', '9bb8c0ac74ad3b78edb0004d7ebbc61f00942dda:' + str(path.relative_to(root))], cwd=root)
    for path in sorted((root / 'die/src').rglob('*')) if path.is_file())
summary['real_python_native'] = {label: json.loads((evidence / 'python' / ('actual-native-' + label + '.json')).read_text())['status']
    for label in ['red', 'green']}
public = json.loads((evidence / 'cpp/public-47-identity.json').read_text())
earlier = json.loads((evidence / 'cpp/public-46-identity.json').read_text())
libs = lambda value: {pathlib.Path(name).name: digest for name, digest in value['binary_sha256'].items()
                      if name.endswith(('.a', '.so'))}
summary['public_47_return_code'] = public['return_code']
summary['public_46_47_libraries_identical'] = libs(public) == libs(earlier)
summary['public_library_sha256'] = libs(public)
summary['cli_final_seven'] = {label: json.loads((evidence / 'cli' / (label + '-final-seven') / 'identity.json').read_text())
                              for label in ['red', 'green']}
old = summary['old_sdk_current_tests']
summary['old_sdk_five_boundary_status'] = ('passed' if set(old) == set(repos) and all(
    x['return_code'] == 0 and x['all_tested_sources_still_match'] and x['sdk_matches_manifest']
    for x in old.values()) else 'not_passed_or_incomplete')
if a.require_old_pass and summary['old_sdk_five_boundary_status'] != 'passed':
    raise RuntimeError('five old-SDK source-bound C++ suites have not all passed')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ['old_sdk_five_boundary_status',
    'compatibility_headers_match_canonical', 'public_production_matches_9bb', 'real_python_native']}))
