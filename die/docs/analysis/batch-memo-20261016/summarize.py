#!/usr/bin/env python3
"""Compare captured production identities, public objects and all perf metrics."""
import hashlib
import json
import pathlib
import subprocess

root=pathlib.Path(__file__).resolve().parents[4]
evidence=root/'die/docs/evidence/batch-memo-20261016'
read=lambda name:json.loads((evidence/name).read_text())
before,after=read('public-before.json'),read('public-after.json')
base,final=read('baseline-benchmark-identity.json'),read('after-benchmark-identity.json')
production={p:h for p,h in base['source_files'].items() if not p.startswith('die/tests/')}
git_matches={p:hashlib.sha256(subprocess.check_output(['git','show',base['baseline_commit']+':'+p],cwd=root)).hexdigest()==h for p,h in production.items()}
static='work/build-die-continue/libcoarse_die.a'
summary={'scope':'C ABI raw-batch memo optimization; no model/hardware/calibration change',
    'baseline_production_matches_committed_source':all(git_matches.values()),
    'baseline_production_file_checks':git_matches,
    'production_source_changes':[p for p,h in production.items() if final['source_files'].get(p)!=h],
    'static_model_library_byte_identical':base['binaries'][static]==final['binaries'][static],
    'eight_complete_public_objects_equal':before['results']==after['results'],
    'independent_accuracy_rerun':'not_run: static model library byte-identical; eight public outputs actually recaptured',
    'benchmarks':{}}
for label,name in [('before','baseline-benchmark.json'),('after','after-benchmark.json')]:
    value=read(name)
    summary['benchmarks'][label]={'status':value['status'],'failures':value['failures'],
        'query_p95_ns':{x['case']:{k:v['p95_ns'] for k,v in x['metrics'].items()} for x in value['query_fixtures']},
        'update_p95_ns':{x['case']:x['metric']['p95_ns'] for x in value['runtime_update_fixtures']},
        'cache':{x['case']:x['cache'] for x in value['query_fixtures']}}
summary['auxiliary_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    pathlib.Path(__file__),root/'die/docs/analysis/allocation-overhead-20261016/capture_public_results.py',
    root/'die/validation/cabi.py']}
(evidence/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['benchmarks','baseline_production_file_checks','auxiliary_sha256']}))
assert summary['baseline_production_matches_committed_source'] and summary['static_model_library_byte_identical'] and summary['eight_complete_public_objects_equal']
