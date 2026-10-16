#!/usr/bin/env python3
"""Keep candidate/storage equivalence separate from budget and accuracy claims."""
import argparse
import gzip
import hashlib
import json
import pathlib
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
base = root / 'die/docs/evidence/compiler-ranking-20261016'
old = root / 'die/docs/evidence/hotpath-allocation-20261016'
out = pathlib.Path(a.output)
if out.exists(): p.error('fresh output required')
read = lambda name: json.loads((base / name).read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
test = read('final-public-identity.json')
benchmark_identity = read('benchmark-final-run-identity.json')
before, after = read('api-baseline.json'), read('api-final.json')
refs = read('oracle-final-content-references.json')
public = read('public-candidates-equality.json')
raw = gzip.decompress((root / public['retained_gzip']).read_bytes())
assert hashlib.sha256(raw).hexdigest() == public['raw_sha256']
result = {
    'observed_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'scope': 'Internal GEMM rank diagnostics deferred; every candidate still validated and estimated. No physics/calibration change.',
    'public_tests': {'return_code': test['return_code'], 'tests_run': test['tests_run'],
                     'log': 'final-public.log'},
    'all_production_headers_and_test_sources_still_match': all(sha(root / name) == value
        for name, value in test['source_sha256'].items()
        if name.startswith(('die/src/', 'die/include/', 'die/tests/'))),
    'actual_public_objects': {'cases': len(before['rows']), 'exactly_equal': before['rows'] == after['rows'],
        'baseline_cabi_sha256': before['library_sha256'], 'new_cabi_sha256': after['library_sha256']},
    'public_all_candidate_complete_wire': public,
    'materialization_test': {'tasks': 64, 'complete_wire_comparisons': 1092,
        'overflow_rejections_compared': 4, 'corrupted_rank_key_rejected_without_mutation': True,
        'evidence': 'summary-test-LastTest.log; final-public.log'},
    'fresh_norm_scenarios': 96,
    'fresh_norm_equal_reference_files': len(refs['files']),
    'fresh_norm_retained_content_verified': all(sha(root / row['retained_equal_bytes']) == row['sha256']
                                              for row in refs['files']),
    'phase_diagnostics': {}, 'public_benchmark': read('benchmark-final.json'),
    'accuracy_status': 'unchanged scoped synthetic Norm completion96/first-output160 target10%; original busy54 failed; not hardware calibration',
    'formal_sdk': {}, 'five_new_model_business_cpu': 'not_run_not_published',
    'engineering_acceptance': 'NOT ACCEPTED',
}
prior = json.loads((old / 'candidate-paths-flat.json').read_text())
now = read('candidate-paths-final.json')
result['all_36_candidate_paths_pivots_latencies_equal'] = (
    [c['candidate_paths'] for c in prior['cases']] == [c['candidate_paths'] for c in now['cases']])
for name, report in [('baseline_flat', json.loads((old / 'phase-flat-simplex.json').read_text())),
                     ('final', read('phase-final.json'))]:
    result['phase_diagnostics'][name] = {
        'scope': report['scope'], 'samples': report['samples'],
        'gemm_medians': {str(c['m']): {phase: {metric: statistics.median(row[phase][metric] for row in c['samples'])
            for metric in ['cpp_new_calls', 'cpp_new_bytes', 'wall_ns', 'thread_cpu_ns']}
            for phase in ['enumerate', 'all_candidate_runtime_with_validation', 'actual_full_compile']}
            for c in report['cases']}}
result['public_libraries_still_match_tests'] = all(sha(pathlib.Path(name)) == value
    for name, value in test['binary_after_sha256'].items()
    if pathlib.Path(name).name in ['libcoarse_die.a', 'libcoarse_die_cabi.so'])
result['performance_run_source_matches_tests'] = all(benchmark_identity['source_sha256'][name] == value
    for name, value in test['source_sha256'].items()
    if name.startswith(('die/src/', 'die/include/', 'die/tests/')))
for repo in ['astra-sim-face', 'astra-sim-face-LRU', 'astra-sim-wscllm', 'astra-sim-wscllm-LRU', 'astra-sim-joint']:
    sdk = root / repo / 'extern/die_model/coarse'
    release = json.loads((sdk / 'RELEASE.json').read_text())
    result['formal_sdk'][repo] = {'digest': release['source_digest'], 'files': len(release['files']),
        'all_actual_files_match': all(sha(sdk / name) == value for name, value in release['files'].items())}
assert result['public_tests']['return_code'] == 0 and result['public_tests']['tests_run'] == 49
for key in ['all_production_headers_and_test_sources_still_match', 'all_36_candidate_paths_pivots_latencies_equal',
            'fresh_norm_retained_content_verified', 'public_libraries_still_match_tests', 'performance_run_source_matches_tests']:
    assert result[key], key
assert result['actual_public_objects']['exactly_equal'] and result['fresh_norm_equal_reference_files'] == 128
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ['public_tests', 'actual_public_objects', 'all_36_candidate_paths_pivots_latencies_equal']}))
