"""Replay newly generated return cells at exact interior/orbit points.

This expands one bounded small-window return for independent state/ledger
checks; the production path never uses this validation expansion.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
from discovery import Aff,Geometry,Owner,Discovery,LEDGERS

R=Path(__file__).resolve().parent
raw=json.loads((R/'first-return-certificate.json').read_text())
g=Geometry(**raw['geometry'])
r=raw['record']
def expr(a,x):return F(a['slope'])*x+F(a['intercept'])
def state_shape(states):
    return [(o.next_load,o.retired,sorted((s,w) for s,(w,p) in o.active.items()),o.loaded,o.computed) for o in states]
expected=[(o['next_load'],o['retired'],sorted((int(s),a['wave']) for s,a in o['active'].items()),o['loaded'],o['computed']) for o in r['end_lifecycle']]
x0=F(r['initial_state']);fixed=F(r['fixed_point'])
points=[x0,fixed,(2*x0+fixed)/3]
records=[]
for point in points:
    d=Discovery(g,offsets=(F(0),F(0)),cap=1024,enable_skip=False)
    d.owners=[]
    for o in r['initial_lifecycle']:
        d.owners.append(Owner(started=True,next_load=o['next_load'],retired=o['retired'],active={int(s):(a['wave'],Aff(b=expr(a['progress'],point))) for s,a in o['active'].items()},loaded=list(o['loaded']),computed=list(o['computed'])))
    while state_shape(d.owners)!=expected:d.record_step()
    for i,o in enumerate(d.owners):
        for s,(wave,p) in o.active.items():assert p.at(point)==expr(r['end_lifecycle'][i]['active'][str(s)]['progress'],point),(i,s)
    for name in LEDGERS:assert d.ledger[name].at(point)==expr(r['ledger'][name],point),name
    nxt=expr(r['map'],point)
    coordinate=r['coordinate'];actual=d.owners[coordinate['owner']].active[coordinate['stage']][1].at(point)
    assert actual==nxt
    assert d.max_in<=2 and d.max_out<=2 and d.max_active<=6
    records.append({'point':str(point),'expanded_events':d.events,'map_and_all_11_ledgers_exact':True,'queue_cursor_membership_exact':True,'max_input_slots':d.max_in,'max_output_slots':d.max_out,'max_active':d.max_active})
result={'status':'passed','scope':'one exact generated return replay, not production or full simulation; independent of archived traces but uses same new geometry/rate helper','certificate_sha256':hashlib.sha256((R/'first-return-certificate.json').read_bytes()).hexdigest(),'records':records}
(R/'return-replay-check.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: 3 exact orbit points, full live queue/cursor state and 11 ledgers; window expansion counts', [r['expanded_events'] for r in records])
