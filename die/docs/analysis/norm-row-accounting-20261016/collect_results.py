#!/usr/bin/env python3
"""Retain source-bound Norm scope, including failed busy and speed gates."""
import argparse
import collections
import hashlib
import json
import pathlib
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];evidence=root/'die/docs/evidence/norm-row-accounting-20261016'
out=pathlib.Path(a.output).resolve()
if out.exists():p.error('fresh output required')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda name:json.loads((evidence/name).read_text())
mapping=root/'die/src/Mapping.cc'
full=read('public-ctest-final-identity.json');unit=read('row-final-test-identity.json')
benchmark=read('benchmark.json');benchmark_run=read('benchmark-run-identity.json')
oracle=read('oracle-run-final-identity.json')
summary={'observed_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'scope':'Norm one-pass row-resident accounting and bounded curves, not independent hardware calibration or five-consumer release',
    'collector_sha256':sha(pathlib.Path(__file__)),
    'current_mapping_sha256':sha(mapping),'public_full_tests':{'return_code':full['return_code'],'tests_run':full['tests_run']},
    'current_expanded_norm_test':{'return_code':unit['return_code'],'tests_run':unit['tests_run'],
        'current_fixture_matches':unit['source_sha256']['die/tests/norm_row_accounting_test.cc']==sha(root/'die/tests/norm_row_accounting_test.cc')},
    'model_source_matches_test_benchmark_oracle':all(meta['source_sha256']['die/src/Mapping.cc']==sha(mapping)
        for meta in [full,unit,benchmark_run,oracle]),
    'public_full_other_test_sources_still_match':all(sha(root/name)==value
        for name,value in full['source_sha256'].items() if name.startswith('die/tests/')
        and name not in ['die/tests/norm_row_accounting_test.cc','die/tests/service_curve_benchmark.cc']),
    'scope_of_test_changes_after_full_48':'only Norm fixture strengthened (54 odd/streamed cases and per-Core physical slot grids), and separate benchmark gained two Norm cases; unchanged 47 suites reused with source checks',
    'old_model_final_same_test_return_code':read('old-model-final-test-red-identity.json')['return_code'],
    'accuracy':{},'benchmark_status':benchmark['status'],'benchmark_failures':benchmark['failures'],
    'norm_performance':{},'formal_sdk':{}}
for label in ['curves-v1','curves-v2-prefix']:
    counts=collections.Counter();completion=collections.Counter();first=collections.Counter()
    metrics={'completion':[],'first_output':[],'busy':[]};max_events=0
    for path in sorted((evidence/label).glob('*.report.json')):
        report=json.loads(path.read_text())
        assert report['ledger_matches']
        raw=json.loads(path.with_name(path.name.replace('.report.json','.raw.json')).read_text())
        max_events=max(max_events,*[value['event_advances'] for value in raw['runs'].values()])
        for scene in report['scenarios'].values():
            comparison=scene['comparison'];counts[comparison['status']]+=1
            completion['target_10' if comparison['completion_target_10_percent'] else 'within20_unaccepted' if comparison['completion_within_20_percent'] else 'failed']+=1
            for item in comparison['first_output_readiness']:
                first['target_10' if item['target_10_percent'] else 'within20_unaccepted' if item['within_20_percent'] else 'failed']+=1
            metrics['completion']+=comparison['completion'];metrics['first_output']+=comparison['first_output_readiness']
            metrics['busy']+=list(comparison['busy_original_definition'].values())
    summary['accuracy'][label]={'completion_and_original_busy':dict(counts),'completion_scenarios':dict(completion),
        'first_output_observations':dict(first),'max_cpp_event_advances':max_events,
        'error_maxima':{name:{'relative':max(item['relative_error'] for item in values if item['relative_error'] is not None),
            'absolute_s':max(item['absolute_error_s'] for item in values)} for name,values in metrics.items()},
        'calibration':'model_to_model_only_not_hardware','within20_accepted':False}
refs=read('final-content-references.json')
summary['fresh_final_oracle_raw_content_matches_retained_v2']=all(sha(root/item['retained_equal_bytes'])==item['sha256']
    for item in refs['files'])
summary['final_oracle_actual_files']=len(refs['files'])
for item in benchmark['query_fixtures']:
    if not item['case'].startswith('norm_'):continue
    summary['norm_performance'][item['case']]={name:{key:value[key] for key in ['p95_ns','fixed_threshold_ns','passed']}
        for name,value in item['metrics'].items() if name!='cabi_hot_full_estimate_report'}
summary['os_update_p95_ns']={item['case']:item['metric']['p95_ns'] for item in benchmark['runtime_update_fixtures']
    if item['case'].startswith('compiled_os_credit_curve_bulk')}
old=read('unaffected-old-final.json');current=read('unaffected-current-final.json')
summary['unaffected_actual_public_cases']={'count':len(current['rows']),'complete_objects_exactly_equal':old['rows']==current['rows']}
for repo in ['astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint']:
    sdk=root/repo/'extern/die_model/coarse';manifest=json.loads((sdk/'RELEASE.json').read_text())
    summary['formal_sdk'][repo]={'digest':manifest['source_digest'],'files':len(manifest['files']),
        'actual_files_match_manifest':all(sha(sdk/name)==value for name,value in manifest['files'].items())}
# Reconstruct the earlier prototype source from a committed base and its
# retained patch; never alter a tracked source or the immutable SDK preview.
archive=root/'work/norm-row-accounting'/('reconstruct-'+out.stem)
(archive/'die/src').mkdir(parents=True,exist_ok=False)
old_mapping=subprocess.check_output(['git','show','089c7db92a74d29dcd924e4f2a8d25d08eaa1663:die/src/Mapping.cc'],cwd=root)
(archive/'die/src/Mapping.cc').write_bytes(old_mapping)
subprocess.run(['git','apply','--directory='+str(archive.relative_to(root)),
    str(pathlib.Path(__file__).with_name('norm-rows-v1.patch'))],cwd=root,check=True)
summary['v1_patch_reconstructs_actual_tested_source']=sha(archive/'die/src/Mapping.cc')==read('oracle-run-v1-identity.json')['source_sha256']['die/src/Mapping.cc']
summary['public_library_sha256']={pathlib.Path(name).name:value for name,value in full['binary_after_sha256'].items()
    if pathlib.Path(name).name in ['libcoarse_die.a','libcoarse_die_cabi.so']}
summary['current_libraries_still_match_full_tests']=all(sha(pathlib.Path(name))==value for name,value in full['binary_after_sha256'].items()
    if pathlib.Path(name).name in ['libcoarse_die.a','libcoarse_die_cabi.so'])
summary['five_consumer_new_model_tests']='not_run_not_published'
summary['five_workload_cpu_budget']='not_run'
summary['independent_target_measurements']='not_available'
summary['tx8_priors']='unchanged_conditional_reference_not_target_calibration'
summary['engineering_acceptance']='NOT ACCEPTED'
assert summary['model_source_matches_test_benchmark_oracle'] and summary['current_expanded_norm_test']['current_fixture_matches']
assert summary['current_libraries_still_match_full_tests'] and summary['v1_patch_reconstructs_actual_tested_source']
out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:summary[key] for key in ['public_full_tests','current_expanded_norm_test',
    'model_source_matches_test_benchmark_oracle','benchmark_status','unaffected_actual_public_cases']}))
