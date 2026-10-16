#!/usr/bin/env python3
"""Exact Fraction constraints for the declared manual fixture, not Runtime code.

All owners have total Q=3072, all COMPUTE_GATE demands are one second,
and there are J owners. The gate bound gives sum r_C <= 1, owner max-min
symmetry gives each owner's gate rate 1/J. For the coupled fixture all three
roles have alpha=1/3 and equal normalized service, so their rates are 1/J;
empty/full signed tangent rows require the same equality. In the split fixture
A's three handles still share total Q=3072. Each B has one activity Q=3072.
In this split fixture a second explicit SRAM_READ bound is essential: each
A activity uses 1024, B uses 3072 and capacity is 3072. This prevents L/S
from receiving extra otherwise-idle HBM service after C and B saturate the gate.
The first attempted fixture omitted this bound and its failed expectations
are preserved separately; those observations do not demonstrate a Runtime bug.
With this common bound, all activities share the same max-min service 1/J.

J is a power of two, so progress/rates and all physical ledger checks below
are exact binary numbers. Only alpha=1/3 is compared to its nearest IEEE double;
there is no fitted tolerance. Credits are checked in declared unit coordinates.
"""
from fractions import Fraction as F
import json
from pathlib import Path
import sys

def check_case(case):
    failures=[]
    def need(ok,label):
        if not ok: failures.append(label)
    need(case['driver_status']=='observed','all public calls must succeed')
    if case['driver_status']!='observed':
        return failures+[case.get('failed_stage','')+': '+case.get('error','')]
    j=case['owners']; split=case['family']=='same_owner_split'
    u=F(2)**case['unit_exponent']; rate=F(1,j)
    expected_hbm_owners=1 if split else j
    q1=j*10**9//4; end=j*10**9
    need(case['snapshot_roundtrip_exact'],'quarter snapshot byte-equivalent JSON')
    need(case['continuation_snapshot_exact'],'restored continuation exactly equal')
    for label,p,clock,terminal in [('initial',F(0),0,False),('quarter',F(1,4),q1,False),
                                 ('completion',F(1),end,True),('cancel',F(1,4),q1+10**9,True)]:
        obs=case[label];need(obs['now_ns']==clock,label+' clock')
        need(obs['drained']==terminal,label+' drained')
        need(obs['next_event_ns']==(None if terminal else end),label+' next event')
        elapsed=q1 if label=='cancel' else clock
        counters=obs['counters']
        for name in ['compute_busy_ns','hbm_busy_ns']:
            need(F(counters[name])==elapsed,label+' '+name)
        need(list(map(F,counters['row_busy_ns']))==[F(elapsed)]*2,label+' both row busy')
        shared=elapsed if expected_hbm_owners>1 else 0
        need(F(counters['hbm_shared_ns'])==shared,label+' hbm shared by owners')
        need(list(map(F,counters['row_shared_ns']))==[F(shared)]*2,label+' row shared by owners')
        hbm_expected=[[p*1024*expected_hbm_owners,F(0)],[F(0),p*1024*expected_hbm_owners]]
        need([[F(x) for x in row] for row in counters['hbm_served_bytes_by_row_direction']]==hbm_expected,
             label+' global physical HBM ledger')
        total_vec=p*(1024*j if not split else 1024+3072*(j-1))
        need(F(counters['served_by_category']['comp']['vec_eqops'])==total_vec,label+' global typed vector ledger')
        sums=[F(0),F(0),F(0),F(0)];hbm_sum=[[F(0)]*2 for _ in range(2)]
        observed_vec=F(0)
        for index,execution in enumerate(obs['executions']):
            activities=execution['activities']
            expected_alpha=F(1,3) if not split or index<3 else F(1)
            need(execution['terminal']==terminal,label+' execution terminal')
            need(execution['cancelled']==(label=='cancel'),label+' cancelled flag')
            if terminal:need(execution['finish_ns']==(q1 if label=='cancel' else end),label+' terminal finish')
            for activity in activities:
                role=activity['role'];r=F(activity['rate'])
                need(F(activity['progress'])==p,label+' '+role+' progress')
                need(r==(0 if terminal else rate),label+' '+role+' exact physical rate')
                need(activity['alpha']==(0 if terminal else float(expected_alpha)),label+' '+role+' owner alpha')
                need(activity['done']==(label=='completion'),label+' '+role+' paid completion')
                need(F(activity['vec_served_eqops'])==p*F(activity['vec_eqops']),label+' per-activity vector ledger')
                observed_vec+=F(activity['vec_served_eqops'])
                for row in range(2):
                    for direction in range(2):
                        served=F(activity['hbm_served_bytes'][row][direction]);hbm_sum[row][direction]+=served
                        need(served==p*F(activity['hbm_payload_bytes'][row][direction]),label+' per-activity byte ledger')
                if not terminal:
                    sums[0]+=F(activity['planned_work'][0])*r
                    sums[1]+=F(activity['planned_work'][4])*r
                    sums[2]+=F(activity['planned_work'][5])*r
                    sums[3]+=F(activity['planned_work'][1])*r
                    need(r<=1,label+' PRIVATE bound')
            if not terminal:
                if not split:
                    initial=F(0) if case['boundary']=='empty' else 2*u if case['boundary']=='full' else u
                    need(list(map(F,execution['snapshot_credit']))==[initial]*2,label+' coupled net credit conserved')
                    signed=[sum(F(a['credit_slope'][b])*F(a['rate']) for a in activities) for b in range(2)]
                    need(signed==[0,0],label+' exact signed tangents both buffers')
                elif index<3:
                    expected=[[u*p,0],[u*(1-p),u*p],[0,u*(1-p)]][index]
                    need(list(map(F,execution['snapshot_credit']))==expected,label+' split declared inventory')
                    for b,value in enumerate(expected):
                        need(0<=value<=2*u,label+' split capacity feasible')
        need(observed_vec==total_vec,label+' sum vector conservation')
        need(hbm_sum==hbm_expected,label+' sum byte conservation')
        if not terminal:
            need(sums[0]==1,label+' gate saturated at exact fair optimum')
            need(sums[1]<=1024 and sums[2]<=1024,label+' both HBM capacity constraints')
            need(sums[3]==(3072 if split else 0),label+' independent shared SRAM constraint')
    return failures

def main():
    rows=[json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    results=[]
    for row in rows:
        failures=check_case(row)
        results.append({key:row[key] for key in ['family','owners','unit_exponent','boundary']}|{
            'status':'failed' if failures else 'passed','failures':failures})
    summary={'scope':'48 manual fluid coupled/split-owner cases, exact Fraction ledger and signed constraints; synthetic only',
             'case_count':len(results),'passed':sum(r['status']=='passed' for r in results),
             'failed':sum(r['status']=='failed' for r in results),'cases':results}
    Path(sys.argv[2]).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['case_count','passed','failed']}))
    return bool(summary['failed']) or len(results)!=48
if __name__=='__main__':raise SystemExit(main())
