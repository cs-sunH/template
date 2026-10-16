#!/usr/bin/env python3
"""Collect the completed isolated preview and the three updated fixtures.

Keep the original failed standard suites, immutable SDK freeze and older
checkpoint summaries. Supplemental fixtures do not turn R4 rejection into a
business pass. Source checks are against actual bytes, not enclosing Git HEAD.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
evidence = root / 'die/docs/evidence/consumer-availability-20261016'
output = pathlib.Path(a.output).resolve()
if output.exists(): p.error('fresh output required')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())
repos = ['astra-sim-face', 'astra-sim-face-LRU', 'astra-sim-wscllm',
         'astra-sim-wscllm-LRU', 'astra-sim-joint']
freeze_path = evidence / 'preview-preparation/new-sdk-consumer-boundaries-freeze.json'
freeze = read(freeze_path)
stage = pathlib.Path(freeze['stage_root'])
matrix_path = evidence / 'preview-five/matrix.json'
matrix = read(matrix_path)
summary = {
    'observed_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'scope': 'isolated ABI/configuration/admission and fixture compatibility; not formal SDK release, model accuracy, hardware calibration or workload CPU acceptance',
    'collector_sha256': sha(pathlib.Path(__file__)),
    'freeze_sha256': sha(freeze_path), 'archive_base_commit': freeze['base_commit'],
    'overlay_still_matches_freeze': all(sha(stage / name) == value for name, value in freeze['overlay_sha256'].items()),
    'original_preview_matrix_sha256': sha(matrix_path),
    'preview': {}, 'updated_fixtures': {}, 'old_sdk_current_tests': {}, 'formal_sdk': {}}

def source_checks(meta, repo, source):
    selected = {name: (source / name).is_file() and sha(source / name) == value
                for name, value in meta['source_sha256'].items()
                if name.startswith(repo + '/') or name == 'die/include/coarse_die/Prediction.hh'}
    if not selected: raise RuntimeError('missing tested source identity')
    return {'all_tested_sources_still_match': all(selected.values()), 'source_checks': selected}

def retained_ctest(build, repo, suffix):
    raw = build / 'Testing/Temporary/LastTest.log'
    retained = output.with_name(output.stem + '-' + repo + '-' + suffix + '-LastTest.log')
    if retained.exists(): raise RuntimeError('fresh raw-log destination required')
    retained.write_bytes(raw.read_bytes())
    return {'path': str(retained), 'sha256': sha(retained)}

config_labels = ['preview-config-face-corrected', 'preview-config-face-lru',
                 'preview-config-wscllm', 'preview-config-wscllm-lru', 'preview-config-joint']
repeat_labels = ['preview-repeat-face', 'preview-repeat-face-lru',
                 'preview-repeat-wscllm', 'preview-repeat-wscllm-lru', 'preview-repeat-joint']
if not (evidence / repeat_labels[0]).is_dir():
    raise RuntimeError('FACE repeat evidence label must match the actual retained run')
expected_coarse = read(evidence / 'cpp/public-47-identity.json')['binary_sha256']
expected_coarse = next(value for name, value in expected_coarse.items() if name.endswith('/libcoarse_die_cabi.so'))

for repo, config_label, repeat_label in zip(repos, config_labels, repeat_labels):
    runs = matrix['runs'][repo]
    build = pathlib.Path(runs['build']['command'][2])
    config = read(evidence / config_label / 'identity.json')
    repeat = read(evidence / repeat_label / 'probe-identity.json')
    repeat_run = read(evidence / repeat_label / 'run-identity.json')
    test = read(pathlib.Path(runs['ctest']['identity']))
    sdk = stage / repo / 'extern/die_model/coarse'
    release = read(sdk / 'RELEASE.json')
    current_sdk_sources = {name: sha(root / 'die' / name) == value
                           for name, value in release['files'].items()
                           if name != 'CMakeLists.txt'}
    item = summary['preview'][repo] = {
        'configure_return_code': runs['configure']['return_code'],
        'build_return_code': runs['build']['return_code'],
        'original_standard_ctest_return_code': runs['ctest']['return_code'],
        'original_standard_sources': source_checks(test, repo, stage),
        'original_standard_log': retained_ctest(build, repo, 'original-preview'),
        'config': {key: config[key] for key in ['status', 'tests_run', 'skips_mentioned', 'return_code', 'library_sha256']},
        'config_identity_sha256': sha(evidence / config_label / 'identity.json'),
        'config_inputs_still_match': all(sha(pathlib.Path(name)) == value for name, value in config['source_sha256'].items()),
        'coarse_cabi_matches_public_47': config['library_sha256']['DIE_COARSE_CABI_PATH'] == expected_coarse,
        'repeat_probe': {key: repeat[key] for key in ['compile_return_code', 'link_return_code', 'run_return_code']},
        'repeat_probe_identity_sha256': sha(evidence / repeat_label / 'probe-identity.json'),
        'repeat_run_return_code': repeat_run['return_code'],
        'repeat_sources': source_checks(repeat_run, repo, stage),
        'sdk_digest': release['source_digest'], 'sdk_files': len(release['files']),
        'public_source_config_header_checks': current_sdk_sources,
        'all_public_sources_configs_headers_match': all(current_sdk_sources.values()),
        'sdk_files_match_manifest': all(sha(sdk / name) == value for name, value in release['files'].items())}

for repo, label in [('astra-sim-wscllm', 'preview-cancel-sampling-wscllm'),
                    ('astra-sim-wscllm-LRU', 'preview-cancel-sampling-wscllm-lru'),
                    ('astra-sim-joint', 'preview-native-sharing-fixture-corrected-target')]:
    meta = read(evidence / label / 'probe-identity.json')
    run = read(evidence / label / 'run-identity.json')
    fixture = pathlib.Path(meta['fixture_source'])
    log = (evidence / label / 'run.log').read_text()
    summary['updated_fixtures'][repo] = {
        'evidence': label, 'fixture': str(fixture.relative_to(root)),
        'current_fixture_matches_tested': sha(fixture) == meta['input_sha256'][str(fixture)],
        'compile_return_code': meta['compile_return_code'], 'link_return_code': meta['link_return_code'],
        'run_return_code': run['return_code'],
        'failed_assertions': re.findall(r'^.*\[FAIL\].*$', log, re.MULTILINE),
        'r4_prediction_not_available': 'prediction_not_available' in log,
        'native_projection_all_pass': 'NativeCoarseProjectionTest ALL PASS' in log,
        'production_sources': source_checks(run, repo, stage),
        'probe_identity_sha256': sha(evidence / label / 'probe-identity.json')}

# FACE sources were not changed after their original full old-SDK run.
# The two WS lifecycle fixtures and JOINT native fixture have fresh full runs.
old = read(evidence / 'checkpoint-summary-final.json')['old_sdk_current_tests']
for repo in repos:
    if repo in repos[:2]:
        label = old[repo]['matrix']
        run = read(evidence / label / (repo + '-ctest-identity.json'))
        identity = evidence / label / (repo + '-ctest-identity.json')
        last_log = None  # Original raw log remains in the published checkpoint.
    else:
        stem = {'astra-sim-wscllm': 'old-wscllm-test',
                'astra-sim-wscllm-LRU': 'old-wscllm-lru-test',
                'astra-sim-joint': 'old-joint-sharing-test'}[repo]
        identity = evidence / 'cancel-sampling' / (stem + '-identity.json')
        run = read(identity)
        last_log = retained_ctest(root / 'work/consumer-availability' / ('old-' + repo + '-build'), repo, 'updated-old-sdk')
    summary['old_sdk_current_tests'][repo] = {
        'identity': str(identity), 'identity_sha256': sha(identity),
        'return_code': run['return_code'], **source_checks(run, repo, root),
        'sdk_files_match_manifest': run['installed_sdk'][repo]['actual_files_match_manifest'],
        'latest_ctest_raw': last_log}
    sdk = root / repo / 'extern/die_model/coarse'
    release = read(sdk / 'RELEASE.json')
    summary['formal_sdk'][repo] = {'digest': release['source_digest'], 'files': len(release['files']),
        'manifest_sha256': sha(sdk / 'RELEASE.json'),
        'all_files_match': all(sha(sdk / name) == value for name, value in release['files'].items())}

summary['config_tests_passed'] = sum(x['config']['tests_run'] for x in summary['preview'].values()
    if x['config']['status'] == 'passed')
summary['five_repeat_admission_probes_passed'] = all(
    all(x['repeat_probe'][key] == 0 for key in x['repeat_probe']) and x['repeat_run_return_code'] == 0
    and x['repeat_sources']['all_tested_sources_still_match'] for x in summary['preview'].values())
summary['old_sdk_five_boundary_status'] = 'passed' if all(
    x['return_code'] == 0 and x['all_tested_sources_still_match'] and x['sdk_files_match_manifest']
    for x in summary['old_sdk_current_tests'].values()) else 'failed_or_incomplete'
summary['formal_release_status'] = 'not_published'
summary['standard_r4_business_status'] = 'failed_prediction_not_available'
summary['five_workload_cpu_status'] = 'not_run'
summary['engineering_acceptance'] = 'NOT ACCEPTED'
output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ['config_tests_passed', 'five_repeat_admission_probes_passed',
    'old_sdk_five_boundary_status', 'formal_release_status', 'standard_r4_business_status']}))
