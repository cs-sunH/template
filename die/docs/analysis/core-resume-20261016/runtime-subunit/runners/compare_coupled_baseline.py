#!/usr/bin/env python3
"""Exact baseline/candidate physical observations; failed calls stay failed."""
from pathlib import Path
import hashlib
import importlib.util
import json

lab=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('physical_summary',lab/'summarize_coupled.py')
# Do not import the summary script (it has an intentional executable top-level).
# Reproduce only its declarative field selection by using the stored observations
# with all coordinate/identity fields removed recursively below.
excluded={'snapshot_credit','snapshot_credit_capacity','credit','credit_delta','credit_slope',
          'plan_fingerprint','hardware_fingerprint','generation','runtime_generation','allocation_updates','internal_events'}
def physical(value):
    if isinstance(value,list):return [physical(x) for x in value]
    if isinstance(value,dict):return {k:physical(v) for k,v in value.items() if k not in excluded}
    return value
def rows(stem):
    return {(r['family'],r['owners'],r['unit_exponent'],r['boundary']):r
            for r in map(json.loads,(lab/(stem+'-driver.stdout.log')).read_text().splitlines())}
base=rows('baseline-coupled');candidate=rows('coupled');result=[]
for key,new in candidate.items():
    old=base[key];equal=[];different=[];missing=[]
    for stage in ['initial','quarter','cancel','completion']:
        if stage not in old or stage not in new:missing.append(stage);continue
        (equal if physical(old[stage])==physical(new[stage]) else different).append(stage)
    result.append(dict(zip(['family','owners','unit_exponent','boundary'],key))|{
        'exactly_equal_samples':equal,'different_samples':different,'missing_in_either':missing,
        'baseline_error':old.get('error'),'candidate_error':new.get('error')})
out={'scope':'physical fields only, exact baseline/candidate JSON comparison; differences are observations, not waivers',
     'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':result}
(lab/'baseline-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'cases':len(result),'ordinary_unit_cases_differ':sum(bool(c['different_samples']) for c in result if c['unit_exponent'] in [0,-10]),
                  'subunit_cases_differ':sum(bool(c['different_samples']) for c in result if c['unit_exponent']==-60)}))
