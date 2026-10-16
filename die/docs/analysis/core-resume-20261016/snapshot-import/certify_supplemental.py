#!/usr/bin/env python3
"""Exact binary64 rate feasibility and separate metric discrepancies.

This reports original immutable-oracle failures unchanged. A few ULP residual
is not classified as a relative hardware-model error; no tolerance is added.
"""
import argparse,hashlib,json,pathlib
from fractions import Fraction as F
from decimal import Decimal,localcontext
p=argparse.ArgumentParser();p.add_argument('input',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);args=p.parse_args()
rows=[json.loads(line) for line in args.input.read_text().splitlines()]
def rational(v):
 with localcontext() as c:c.prec=80;decimal=str(Decimal(v.numerator)/Decimal(v.denominator))
 return {'numerator':str(v.numerator),'denominator':str(v.denominator),'decimal':decimal}
def enc(v):return rational(F(v))
results=[]
for row in rows:
 if row['family']!='same_owner_split' or row['owners'] not in [2,8]:continue
 j=row['owners'];unit=F(2)**row['unit_exponent'];case={'owners':j,'unit_exponent':row['unit_exponent'],'certificates':[],'metrics':[],'unavailable_metrics':[]}
 def metric(stage,name,actual,expected,identity=None):
  if actual is None:
   case['unavailable_metrics'].append({'stage':stage,'name':name,'identity':identity,'expected':enc(expected),'actual':None});return
  actual=F(actual);expected=F(expected);error=abs(actual-expected)
  case['metrics'].append({'stage':stage,'name':name,'identity':identity,'actual':enc(actual),'expected':enc(expected),'absolute_error':enc(error),'relative_error':enc(error/abs(expected)) if expected else None})
 for stage,progress,terminal in [('initial',F(0),False),('quarter',F(1,4),False),('completion',F(1),True),('cancel',F(1,4),True)]:
  observed=row[stage];capacities=list(map(F,row['synthetic_rates']));used=[F(0)]*13;private=[];credit=[]
  for index,e in enumerate(observed['executions']):
   for a in e['activities']:
    aid=[index+1,a['activity_id']];rate=F(a['rate']);work=list(map(F,a['planned_work']))
    for resource in range(13):used[resource]+=work[resource]*rate
    definition=next(d for ep in row['plans'][index]['epochs'] for d in ep['activities'] if d['id']==a['activity_id']);private_used=F(definition['private_time_s'])*rate
    private.append({'execution':index+1,'activity':a['activity_id'],'used':enc(private_used),'bound':enc(1),'strict_feasible':private_used<=1})
    metric(stage,'rate',a['rate'],0 if terminal else F(1,j),aid);metric(stage,'progress',a['progress'],progress,aid)
    metric(stage,'alpha',a['alpha'],0 if terminal else F(float(F(1,3))) if index<3 else 1,aid)
    metric(stage,'activity_vec_eqops',a['vec_served_eqops'],progress*F(a['vec_eqops']),aid)
    for b in range(2):
     for direction in range(2):metric(stage,'activity_hbm_bytes',a['hbm_served_bytes'][b][direction],progress*F(a['hbm_payload_bytes'][b][direction]),aid+[b,direction])
   if not e['terminal']:
    for b in range(2):
     value=F(e['snapshot_credit'][b]);capacity=F(e['snapshot_credit_capacity'][b]);tangent=sum((F(a.get('credit_slope',a['credit_delta'])[b])*F(a['rate']) for a in e['activities']),F(0))
     exact_low=value==0;exact_high=value==capacity
     # Report the original Runtime's pre-existing boundary mask too. This is
     # not a newly invented feasibility tolerance; no tangent epsilon is used.
     tolerance=F(2e-10)*max(F(1),capacity);runtime_low=value<=tolerance;runtime_high=value>=capacity-tolerance
     credit.append({'execution':index+1,'buffer':b,'value':enc(value),'capacity':enc(capacity),'net_tangent':enc(tangent),'strict_inventory_feasible':0<=value<=capacity,'at_exact_low':exact_low,'at_exact_high':exact_high,'strict_exact_boundary_tangent_feasible':(not exact_low or tangent>=0)and(not exact_high or tangent<=0),'at_existing_runtime_low':runtime_low,'at_existing_runtime_high':runtime_high,'strict_existing_runtime_tangent_feasible':(not runtime_low or tangent>=0)and(not runtime_high or tangent<=0)})
   if index<3:
    expected=[[unit*progress,F(0)],[unit*(1-progress),unit*progress],[F(0),unit*(1-progress)]][index]
    for b in range(2):metric(stage,'credit_units',e['snapshot_credit'][b],expected[b],[index+1,b])
   if terminal:metric(stage,'finish_ns',e['finish_ns'],j*10**9//4 if stage=='cancel' else j*10**9,index+1)
  counters=observed['counters'];elapsed=j*10**9//4 if stage=='cancel' else counters['now_ns']
  for name in ['compute_busy_ns','hbm_busy_ns']:
   metric(stage,name,counters[name],elapsed)
  for b in range(2):metric(stage,'row_busy_ns',counters['row_busy_ns'][b],elapsed,b)
  metric(stage,'global_vec_eqops',counters['served_by_category']['comp']['vec_eqops'],progress*(1024+3072*(j-1)))
  for b in range(2):
   for d in range(2):metric(stage,'global_hbm_bytes',counters['hbm_served_bytes_by_row_direction'][b][d],progress*1024 if (b,d) in [(0,0),(1,1)] else 0,[b,d])
  physical=[{'resource':r,'used':enc(used[r]),'capacity':enc(capacities[r]),'margin':enc(capacities[r]-used[r]),'strict_feasible':used[r]<=capacities[r]} for r in range(13)]
  case['certificates'].append({'stage':stage,'physical':physical,'private':private,'credit':credit,'strict_feasible':all(r['strict_feasible'] for r in physical+private)and all(r['strict_inventory_feasible']and r['strict_exact_boundary_tangent_feasible']and r['strict_existing_runtime_tangent_feasible']for r in credit)})
 summary={}
 for kind in {m['name']for m in case['metrics']}:
  ms=[m for m in case['metrics']if m['name']==kind];maxabs=max(ms,key=lambda m:F(int(m['absolute_error']['numerator']),int(m['absolute_error']['denominator'])));nonzero=[m for m in ms if m['relative_error'] is not None];maxrel=max(nonzero,key=lambda m:F(int(m['relative_error']['numerator']),int(m['relative_error']['denominator'])))if nonzero else None
  summary[kind]={'max_absolute_error':maxabs['absolute_error'],'max_absolute_location':{k:maxabs[k]for k in ['stage','identity']},'max_relative_error':maxrel['relative_error']if maxrel else None,'max_relative_location':{k:maxrel[k]for k in ['stage','identity']}if maxrel else None}
 case['metric_summary']=summary;case['all_strict_feasible']=all(c['strict_feasible']for c in case['certificates']);results.append(case)
report={'scope':'Six original same-owner split fixtures, B-only archive. Exact Fraction(binary64)13-resource/PRIVATE/current-credit certificates, no new epsilon. Metric discrepancies remain separate from strict original oracle status.','input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'tool_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'cases':results,'all_source_allocations_strict_feasible':all(r['all_strict_feasible']for r in results)}
args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'all_strict_feasible':report['all_source_allocations_strict_feasible'],'cases':[{'owners':r['owners'],'unit':r['unit_exponent'],'feasible':r['all_strict_feasible'],'max_rate_relative':r['metric_summary']['rate']['max_relative_error']['decimal'],'max_busy_relative':r['metric_summary']['compute_busy_ns']['max_relative_error']['decimal'],'max_credit_relative':r['metric_summary']['credit_units']['max_relative_error']['decimal']}for r in results]}))
