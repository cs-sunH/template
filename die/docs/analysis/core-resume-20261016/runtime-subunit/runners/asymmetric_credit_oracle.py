#!/usr/bin/env python3
"""Hand-derived Fraction oracle with active, unequal signed tangent rates.

At J identical owners: C=1/J by the gate. L,S can reach 2/J on their separate
HBM rows, PRIVATE=1/2. Empty buffers require L>=C>=S, giving (2,1,1)/J;
full buffers require L<=C<=S, giving (1,1,2)/J. Interior gives (2,1,2)/J.
At J/2 seconds the faster activities finish. Remaining nonzero roles progress
at 1/J, finishing all paid work at J seconds. No discrete tile/credit replay.
These powers-of-two rates and sample progress values are exactly representable.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import sys

def check(case):
    errors=[]
    def need(ok,msg):
        if not ok:errors.append(msg)
    need(case['driver_status']=='observed','driver public calls succeed')
    if case['driver_status']!='observed':return errors+[case.get('error','')]
    if 'snapshot_error' in case:errors.append('snapshot import: '+case['snapshot_error'])
    else:
        need(case.get('snapshot_roundtrip_exact') is True,'exact snapshot roundtrip')
        need(case.get('continuation_snapshot_exact') is True,'exact continuation snapshot')
    j=case['owners'];u=F(2)**case['unit_exponent'];b=case['boundary']
    multipliers=[1 if b=='full' else 2,1,1 if b=='empty' else 2]
    initial=0 if b=='empty' else 2*u if b=='full' else u
    for label,t in [('initial',F(0)),('quarter',F(1,4)),('half',F(1,2)),('completion',F(1)),('cancel',F(1,4))]:
        if label not in case:
            errors.append(label+' not observed');continue
        obs=case[label];cancel=label=='cancel';terminal=label in ['completion','cancel']
        progress=[min(F(1),t*m) for m in multipliers]
        need(obs['now_ns']==(j*10**9//4+10**9 if cancel else t*j*10**9),label+' clock')
        need(obs['drained']==terminal,label+' drained')
        need(obs['next_event_ns']==(None if terminal else j*10**9//2 if t<F(1,2) else j*10**9),label+' physical event')
        counters=obs['counters'];elapsed=t*j*10**9
        row_busy=[min(elapsed,F(j*10**9,m)) for m in multipliers[::2]]
        need(F(counters['compute_busy_ns'])==elapsed,label+' gate busy')
        need(list(map(F,counters['row_busy_ns']))==row_busy,label+' directional busy')
        hbm_busy=max(row_busy)
        need(F(counters['hbm_busy_ns'])==hbm_busy,label+' original union HBM busy')
        need(F(counters['hbm_shared_ns'])==(hbm_busy if j>1 else 0),label+' original shared busy')
        need(list(map(F,counters['row_shared_ns']))==([x if j>1 else 0 for x in row_busy]),label+' directional shared busy')
        matrix=[[512*j*progress[0],F(0)],[F(0),512*j*progress[2]]]
        need([[F(x) for x in row] for row in counters['hbm_served_bytes_by_row_direction']]==matrix,label+' global byte ledger')
        need(F(counters['served_by_category']['comp']['vec_eqops'])==1024*j*progress[1],label+' global vector ledger')
        for execution in obs['executions']:
            need(execution['terminal']==terminal,label+' terminal')
            need(execution['cancelled']==cancel,label+' cancelled')
            if terminal:need(execution['finish_ns']==(j*10**9//4 if cancel else j*10**9),label+' completion time')
            actual_rates=[]
            for n,a in enumerate(execution['activities']):
                p=progress[n];need(F(a['progress'])==p,label+' activity progress')
                expected_rate=0 if terminal or p==1 else F(multipliers[n] if t<F(1,2) else 1,j)
                need(F(a['rate'])==expected_rate,label+' activity signed-constrained rate')
                need(a['done']==(p==1),label+' only paid activities done')
                if n==1:need(F(a['vec_served_eqops'])==1024*p,label+' compute EqWork ledger')
                else:
                    row=0 if n==0 else 1;direction=0 if n==0 else 1
                    need(F(a['hbm_served_bytes'][row][direction])==512*p,label+' role byte ledger')
                actual_rates.append(F(a['rate']))
            if not terminal:
                credit=[initial+u*(progress[0]-progress[1]),initial+u*(progress[1]-progress[2])]
                need(list(map(F,execution['snapshot_credit']))==credit,label+' exact physical credit')
                for n,c in enumerate(credit):
                    need(0<=c<=2*u,label+' finite credit feasibility')
                    derivative=u*(actual_rates[n]-actual_rates[n+1])
                    if c==0:need(derivative>=0,label+' empty signed tangent')
                    if c==2*u:need(derivative<=0,label+' full signed tangent')
    return errors

def main():
    cases=[]
    for line in Path(sys.argv[1]).read_text().splitlines():
        row=json.loads(line);failed=check(row)
        cases.append({k:row[k] for k in ['family','owners','unit_exponent','boundary']}|{'status':'failed' if failed else 'passed','failures':failed})
    result={'case_count':len(cases),'passed':sum(c['status']=='passed' for c in cases),
            'failed':sum(c['status']=='failed' for c in cases),'cases':cases}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['case_count','passed','failed']}))
    return result['failed']>0 or len(cases)!=36
if __name__=='__main__':raise SystemExit(main())
