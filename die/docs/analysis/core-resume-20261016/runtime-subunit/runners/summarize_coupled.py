#!/usr/bin/env python3
"""Compare public physical fields across declared credit coordinates exactly."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json

lab=Path(__file__).resolve().parent
counter_keys=['hbm_busy_ns','hbm_shared_ns','compute_busy_ns','row_busy_ns','row_shared_ns',
              'hbm_served_bytes_by_row_direction','served_by_category']
execution_keys=['state','issue_ns','ready_ns','finish_ns','elapsed_ns','served',
                'hbm_served_bytes','compute_gate_served_work','terminal','cancelled']
activity_keys=['activity_id','role','status','progress','rate','alpha','planned_work',
               'served_work','remaining_work','hbm_payload_bytes','hbm_served_bytes',
               'vec_eqops','vec_served_eqops','int4_ops','int4_served_ops','done']
def physical(sample):
    return {'now_ns':sample['now_ns'],'drained':sample['drained'],'next_event_ns':sample['next_event_ns'],
            'counters':{k:sample['counters'][k] for k in counter_keys},
            'executions':[{**{k:e[k] for k in execution_keys},
                           'activities':[{k:a[k] for k in activity_keys} for a in e['activities']]}
                          for e in sample['executions']]}
report={'scope':'exact physical JSON field invariance; excludes coordinate-dependent credit and Plan identity; no tolerance',
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'groups':[]}
for stem in ['coupled','asymmetric']:
    rows=[json.loads(line) for line in (lab/(stem+'-driver.stdout.log')).read_text().splitlines()]
    groups={}
    for row in rows:groups.setdefault((row['family'],row['owners'],row['boundary']),[]).append(row)
    for key,group in groups.items():
        baseline=next(r for r in group if r['unit_exponent']==0)
        missing=[];differences=[];equal=[]
        for label in ['initial','quarter','half','cancel','completion']:
            present=[r for r in group if label in r]
            if not present:continue
            if len(present)!=3:
                missing.append(label);continue
            a=physical(baseline[label])
            if all(physical(r[label])==a for r in group):equal.append(label)
            else:differences.append(label)
        report['groups'].append({'family':key[0],'owners':key[1],'boundary':key[2],
                                 'exactly_equal_samples':equal,'different_samples':differences,
                                 'incomplete_samples':missing,
                                 'all_three_units_same_snapshot_exception':len({r.get('snapshot_error',r.get('error')) for r in group})==1})
report['group_count']=len(report['groups'])
report['different_sample_count']=sum(len(g['different_samples']) for g in report['groups'])
report['note']='Missing post-import samples remain failed/not_run in the case oracle; this comparison does not waive them.'
(lab/'unit-invariance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['group_count','different_sample_count']}))
