#!/usr/bin/env python3
"""Summarize this checkpoint without changing any historical measurement."""
import argparse
import hashlib
import json
import pathlib
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4]
base=root/'die/docs/evidence/repeat-availability-20261016'
read=lambda name:json.loads((base/name).read_text())
out=pathlib.Path(a.output)
if out.exists():p.error('fresh output required')
test=read('green-public-identity.json');budget=read('budget.json')
old=json.loads((root/'die/docs/evidence/norm-long-validation-20261016/api-canonical.json').read_text())
new=read('api-green.json')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
source_matches=all(sha(root/name)==value for name,value in test['source_sha256'].items()
    if name.startswith(('die/src/','die/include/','die/tests/')))
result={'observed_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'scope':'Correct zero-overhead repeated-region availability; physical timing unchanged',
    'public_tests':{'tests':test['tests_run'],'return_code':test['return_code']},
    'current_production_and_test_sources_match':source_matches,
    'default_profile_complete_public_objects':{'count':len(old['rows']),'exactly_equal':old['rows']==new['rows']},
    'zero_overhead_probe':read('probe-comparison.json'),
    'performance':{'status':budget['status'],'failures':budget['failures'],
        'queries':{row['case']:{k:v['p95_ns'] for k,v in row['metrics'].items()} for row in budget['query_fixtures']},
        'single_updates_p95_ns':{row['case']:row['metric']['p95_ns'] for row in budget['runtime_update_fixtures']}},
    'formal_sdk':{},'repeated_region_model':'not_implemented','new_long_row_model':'not_adopted',
    'target_hardware_calibration':'not_available','engineering_acceptance':'NOT ACCEPTED'}
for repo in ['astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint']:
    sdk=root/repo/'extern/die_model/coarse';release=json.loads((sdk/'RELEASE.json').read_text())
    result['formal_sdk'][repo]={'digest':release['source_digest'],'files':len(release['files']),
        'all_actual_files_match':all(sha(sdk/name)==value for name,value in release['files'].items())}
assert test['return_code']==0 and test['tests_run']==50 and source_matches
assert result['default_profile_complete_public_objects']['exactly_equal']
assert all(row['all_actual_files_match'] for row in result['formal_sdk'].values())
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['public_tests','current_production_and_test_sources_match',
    'default_profile_complete_public_objects','engineering_acceptance']}))
