#!/usr/bin/env python3
"""Summarize the unsuccessful traversal experiment without adopting its model."""
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];base=root/'die/docs/evidence/j32-update-20261016'
out=pathlib.Path(a.output)
if out.exists():p.error('fresh output required')
read=lambda name:json.loads((base/name).read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
prior=root/'die/docs/evidence/compiler-ranking-20261016'
reference=json.loads((prior/'final-public-identity.json').read_text())
restored=read('restore-unchanged-production-build-identity.json')
fused=read('fused-public-identity.json');api=json.loads((prior/'api-final.json').read_text())
budget=read('fused-benchmark.json');refs=read('fused-oracle-content-references.json')
result={'source_base':'5caede8ca81902baf15c7d3228db70f2087302fb',
    'decision':'not_adopted_no_supported_speed_benefit; source patch/raw preserved, production exactly restored',
    'fused_public_tests':{'tests_run':fused['tests_run'],'return_code':fused['return_code']},
    'fused_complete_11_api_objects_equal':api['rows']==read('fused-api.json')['rows'],
    'fused_norm_actual_scenarios':96,'fused_norm_retained_equal_files':len(refs['files']),
    'fused_norm_raw_verified':all(sha(root/row['retained_equal_bytes'])==row['sha256']for row in refs['files']),
    'fused_budget_status':budget['status'],'fused_budget_failures':budget['failures'],
    'fused_budget_raw':'die/docs/evidence/j32-update-20261016/fused-benchmark.json',
    'instrumented_trace_medians':{},
    'current_all_production_headers_and_test_sources_equal_prior_tested':all(sha(root/name)==value
        for name,value in reference['source_sha256'].items()if name.startswith(('die/src/','die/include/','die/tests/'))),
    'current_original_tested_libraries_byte_equal':all(sha(pathlib.Path(name))==value
        for name,value in reference['binary_after_sha256'].items()
        if pathlib.Path(name).name in ['libcoarse_die.a','libcoarse_die_cabi.so']),
    'production_rebuild_return_code':restored['return_code'],
    'current_public_tests_reused':{'tests_run':49,'log':'die/docs/evidence/compiler-ranking-20261016/final-public.log',
        'reason':'All production/header/test contents and actual rebuilt static/CABI bytes equal prior tested version; experiment49 is separately scoped.'},
    'current_cabi_sha256':sha(root/'work/build-die-continue/lib/libcoarse_die_cabi.so'),
    'formal_new_sdk_publication':'not_run','engineering_acceptance':'NOT ACCEPTED'}
for stage in ['baseline','fused']:
    report=read('trace-'+stage+'.json')
    result['instrumented_trace_medians'][stage]={str(j):{k:statistics.median(x['trace'][k]['inclusive_thread_cpu_ns']
        for x in report['rows']if x['owners']==j)for k in ['update','advance','allocate','collect','symmetry','rows']}
        for j in [1,8,32]}
for key in ['fused_complete_11_api_objects_equal','fused_norm_raw_verified',
            'current_all_production_headers_and_test_sources_equal_prior_tested','current_original_tested_libraries_byte_equal']:
    assert result[key],key
assert fused['return_code']==0 and fused['tests_run']==49 and len(refs['files'])==128
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
