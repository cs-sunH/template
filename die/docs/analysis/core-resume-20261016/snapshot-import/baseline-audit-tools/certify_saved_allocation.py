#!/usr/bin/env python3
"""Independent exact-rational feasibility certificate for the captured rates."""
import hashlib,json,pathlib,struct
from fractions import Fraction as F
from decimal import Decimal,getcontext
getcontext().prec=100
root=pathlib.Path(__file__).resolve().parent
s=json.loads((root/'failed-snapshot.json').read_text())
rebuilt=json.loads((root/'result.json').read_text())['diagnostic_rebuilt_without_allocation_marker']['rebuilt_snapshot']
D=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
enc=lambda x:{'numerator':str(x.numerator),'denominator':str(x.denominator),'decimal':str(D(x))}
bits=lambda x:struct.unpack('>Q',struct.pack('>d',x))[0]
active=[];credit_rows=[];rates=[]
for e,f in zip(s['executions'],rebuilt['executions']):
 current=[]
 for a,b in zip(e['activities'],f['activities']):
  if a['done'] or a['epoch_index']!=e['epoch_index']:continue
  definition=next(d for ep in e['plan']['epochs'] for d in ep['activities'] if d['id']==a['activity_id'])
  if not any(definition['work']):continue
  curve=definition.get('credit_curve',[])
  if curve:
   progress=Decimal(a['curve_progress_exact']);j=0
   while j+2<len(curve) and progress>=Decimal.from_float(curve[j+1]['progress']):j+=1
   lo,hi=curve[j:j+2];slopes=[(hi['cumulative'][i]-lo['cumulative'][i])/(hi['progress']-lo['progress']) for i in range(2)]
  else:slopes=definition['credit_delta']
  current.append((a,definition,slopes));active.append((a,definition))
  rates.append({'handle':e['handle'],'owner':e['owner'],'activity_id':a['activity_id'],'saved_rate_hex':a['rate'].hex(),'rebuilt_rate_hex':b['rate'].hex(),'signed_ulp_difference':bits(a['rate'])-bits(b['rate']),'alpha_equal':a['alpha']==b['alpha']})
 for i,(credit,capacity) in enumerate(zip(e['credit'],e['credit_capacity'])):
  tol=2e-10*max(1.0,capacity);low=credit<=tol;high=credit>=capacity-tol
  tangent=sum((F.from_float(slopes[i])*F.from_float(a['rate']) for a,d,slopes in current),F(0))
  credit_rows.append({'handle':e['handle'],'buffer':i,'credit':credit,'capacity':capacity,'at_low':low,'at_high':high,'slope_by_activity':[[a['activity_id'],slopes[i]] for a,d,slopes in current],'net_saved_tangent':enc(tangent),'strict_feasible':(not low or tangent>=0) and(not high or tangent<=0)})
physical=[]
for i,c in enumerate(s['capacities']):
 used=sum((F.from_float(a['rate'])*F.from_float(d['work'][i]) for a,d in active),F(0));margin=F.from_float(c)-used
 physical.append({'resource':i,'used':enc(used),'margin':enc(margin),'strict_feasible':margin>=0})
private=[]
for a,d in active:
 used=F.from_float(a['rate'])*F.from_float(d['private_time_s']);private.append({'activity_id':a['activity_id'],'used':enc(used),'strict_feasible':used<=1})
report={'scope':'Exact rational operations on captured binary64 plan work, capacities, rates and segment derivatives. No feasibility epsilon and no import acceptance inferred.','original_snapshot_sha256':hashlib.sha256((root/'failed-snapshot.json').read_bytes()).hexdigest(),'active_count':len(active),'rates':rates,'physical':physical,'private':private,'current_credit_tangents':credit_rows}
report['all_strict_feasible']=all(r['strict_feasible'] for name in ['physical','private','current_credit_tangents'] for r in report[name])
(root/'exact-feasibility.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'all_strict_feasible':report['all_strict_feasible'],'active_count':len(active),'compute_gate_margin':physical[0]['margin'],'rate_ulp_differences':[r['signed_ulp_difference'] for r in rates]}))
assert report['all_strict_feasible']
