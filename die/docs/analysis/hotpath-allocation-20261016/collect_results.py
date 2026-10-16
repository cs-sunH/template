#!/usr/bin/env python3
"""Summarize allocation diagnostics separately from failed budget samples."""
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];base=root/'die/docs/evidence/hotpath-allocation-20261016'
out=pathlib.Path(a.output)
if out.exists():p.error('fresh output required')
read=lambda name:json.loads((base/name).read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
test=read('ratio-reuse-public-identity.json');run=read('ratio-reuse-benchmark-run-identity.json')
result={'observed_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'scope':'equivalent ratio scratch storage reuse; instrumentation is not budget evidence',
    'public_tests':{'return_code':test['return_code'],'tests_run':test['tests_run']},
    'all_production_headers_and_test_sources_still_match':all(sha(root/name)==value
        for name,value in test['source_sha256'].items() if name.startswith(('die/src/','die/include/','die/tests/'))),
    'phase_diagnostics':{},'public_benchmark':read('ratio-reuse-benchmark.json'),
    'baseline_actual_benchmark_reference':'die/docs/evidence/norm-row-accounting-20261016/benchmark.json'}
before=read('api-baseline.json');after=read('api-ratio-reuse.json')
result['actual_public_objects']={'cases':len(before['rows']),'exactly_equal':before['rows']==after['rows'],
    'baseline_cabi_sha256':before['library_sha256'],'new_cabi_sha256':after['library_sha256']}
for name in ['phase-baseline','phase-ratio-reuse']:
    report=read(name+'.json');cases={}
    for case in report['cases']:
        cases[str(case['m'])]={phase:{metric:statistics.median(row[phase][metric] for row in case['samples'])
            for metric in ['wall_ns','thread_cpu_ns','cpp_new_calls','cpp_new_bytes']}
            for phase in ['normalize','enumerate','hardware_copy','candidate_structure_validation_subset',
                          'all_candidate_runtime_with_validation','actual_full_compile']}
    updates={str(owners):{metric:statistics.median(row[metric] for row in report['single_os_updates'] if row['owners']==owners)
        for metric in ['wall_ns','thread_cpu_ns','cpp_new_calls','cpp_new_bytes']} for owners in [1,8,32]}
    result['phase_diagnostics'][name]={'scope':report['scope'],'samples':report['samples'],
        'gemm_medians':cases,'os_update_medians':updates}
refs=read('oracle-content-references.json')
result['fresh_norm_reference_content_all_matches']=all(sha(root/item['retained_equal_bytes'])==item['sha256'] for item in refs['files'])
result['fresh_norm_reference_file_count']=len(refs['files'])
result['fresh_norm_reference_scenarios']=96
result['accuracy_status']='unchanged scoped Norm evidence: completion 96 target / first output 160 target; original busy still 54 failed; not hardware calibration'
result['public_libraries_still_match_tests']=all(sha(pathlib.Path(name))==value for name,value in test['binary_after_sha256'].items()
    if pathlib.Path(name).name in ['libcoarse_die.a','libcoarse_die_cabi.so'])
result['performance_run_current_source_matches_tests']=all(run['source_sha256'][name]==value for name,value in test['source_sha256'].items()
    if name.startswith(('die/src/','die/include/','die/tests/')))
result['formal_sdk']={}
for repo in ['astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint']:
    sdk=root/repo/'extern/die_model/coarse';r=json.loads((sdk/'RELEASE.json').read_text())
    result['formal_sdk'][repo]={'digest':r['source_digest'],'files':len(r['files']),
        'all_actual_files_match':all(sha(sdk/name)==value for name,value in r['files'].items())}
result['five_new_model_business_cpu']='not_run_not_published'
result['engineering_acceptance']='NOT ACCEPTED'
assert result['actual_public_objects']['exactly_equal'] and result['all_production_headers_and_test_sources_still_match']
assert result['public_libraries_still_match_tests'] and result['performance_run_current_source_matches_tests']
assert result['fresh_norm_reference_content_all_matches']
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['public_tests','actual_public_objects',
    'fresh_norm_reference_content_all_matches','public_libraries_still_match_tests']}))
