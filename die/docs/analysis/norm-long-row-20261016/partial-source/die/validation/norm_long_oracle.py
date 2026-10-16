#!/usr/bin/env python3
"""Offline discrete two-pass Norm; one Core/row, BF16, R1, nonbinding NoC.

Statistics consumes input tiles serially without an output FIFO; the complete
row barrier precedes application. Application has two physical input/output
slots. Work/bytes derive from task facts; no production curve or duration is
used. All owner Q/alpha, resource and PRIVATE allocation uses the independently
tested nonnegative grouped reference. No hardware calibration or fitting.
"""
import argparse
import hashlib
import json
import math
import pathlib
from elementwise_service_oracle import Activity, grouped_rates, compare

def simulate(hidden,layer,capacities,peak,arrivals):
    sizes=[min(256,hidden-begin) for begin in range(0,hidden,256)]
    vectors=2 if layer else 1
    count=len(arrivals);states=[['pending']*len(sizes) for _ in arrivals]
    phases=[0]*count;live=[];now=0.0;finish=[None]*count;first=[None]*count;events=0
    busy={'hbm_busy_s':0.0,'hbm_shared_s':0.0,'compute_busy_s':0.0,
          'row_busy_s':[0.0,0.0],'row_shared_s':[0.0,0.0]}
    def activity(owner,index,role):
        n=sizes[index];phase=phases[owner];work=[0.0]*13
        if role=='load':
            payload=2*n*(1 if phase==0 else 1+vectors)
            work[2]=payload;work[4]=work[5]=payload/2;quantity=payload
        elif role=='store':
            payload=2*n;work[1]=payload;work[4]=work[5]=n;quantity=payload
        else:
            if phase==0:
                work[0]=(4 if layer else 3)*n/peak
                work[1]=6*n;work[2]=4*n+(16 if index==len(sizes)-1 else 0)
            else:
                work[0]=(9 if layer else 7)*n/peak
                work[1]=(10+6*vectors)*n+(16 if index==0 else 0)
                work[2]=(10+4*vectors)*n
            work[3]=work[1]+work[2];quantity=work[3]
        private=max([work[0]]+[work[r]/capacities[r] for r in [1,2,3] if work[r]])
        return Activity(owner,index,role,work,private,quantity)
    while any(value is None for value in finish):
        for owner in range(count):
            if arrivals[owner]>now+1e-13 or finish[owner] is not None:continue
            state=states[owner]
            if all(s=='stored' for s in state):
                if phases[owner]==0:
                    phases[owner]=1;states[owner]=state=['pending']*len(sizes)
                else:finish[owner]=now;continue
            if phases[owner]==1 and not any(a.request==owner and a.role=='store' for a in live):
                candidates=[i for i,s in enumerate(state) if s=='produced']
                if candidates:
                    index=candidates[0];state[index]='storing';live.append(activity(owner,index,'store'))
            if not any(a.request==owner and a.role=='compute' for a in live):
                used=sum(s in ['computing','produced','storing'] for s in state)
                candidates=[i for i,s in enumerate(state) if s=='buffered']
                if candidates and (phases[owner]==0 or used<2):
                    index=candidates[0];state[index]='computing';live.append(activity(owner,index,'compute'))
            if not any(a.request==owner and a.role=='load' for a in live):
                used=sum(s in ['loading','buffered','computing'] for s in state)
                candidates=[i for i,s in enumerate(state) if s=='pending']
                if candidates and used<2:
                    index=candidates[0];state[index]='loading';live.append(activity(owner,index,'load'))
        if all(x is not None for x in finish):break
        rates=grouped_rates(live,capacities) if live else []
        future=[a-now for a in arrivals if a>now+1e-13]
        future += [(1-a.progress)/rate for a,rate in zip(live,rates) if rate>0]
        if not future:raise ArithmeticError('discrete two-pass Norm deadlocked')
        delta=min(future)
        if not math.isfinite(delta) or delta<=0:raise ArithmeticError('nonadvancing oracle event')
        hbm=set();compute=set();rows=[set(),set()]
        for a,rate in zip(live,rates):
            if rate<=0:continue
            a.progress+=rate*delta
            if a.work[0]>0:compute.add(a.request)
            for row in [0,1]:
                if a.work[4+row]>0:rows[row].add(a.request);hbm.add(a.request)
        if hbm:busy['hbm_busy_s']+=delta
        if len(hbm)>=2:busy['hbm_shared_s']+=delta
        if compute:busy['compute_busy_s']+=delta
        for row in [0,1]:
            if rows[row]:busy['row_busy_s'][row]+=delta
            if len(rows[row])>=2:busy['row_shared_s'][row]+=delta
        now+=delta
        for a in live:
            if a.progress>=1-1e-10:
                state=states[a.request]
                if a.role=='load':state[a.block]='buffered'
                elif a.role=='store' or phases[a.request]==0:state[a.block]='stored'
                else:
                    state[a.block]='produced'
                    if first[a.request] is None:first[a.request]=now
        live=[a for a in live if a.progress<1-1e-10];events+=1
        if events>100000:raise ArithmeticError('offline event guard exceeded')
    return {'finish_s':finish,'first_output_ready_s':first,'counters':busy,'discrete_events':events,
            'scope':'one complete row statistics barrier, then discrete application tiles; two input/output slots'}

def report(raw):
    hidden=int(raw['fixture']['hidden']);layer=raw['fixture']['norm']=='layernorm'
    if not 65537<=hidden<=262145 or raw['hardware']['derived']['die']['core_count']!=1:
        raise ValueError('offline one-Core/row BF16 fixture required')
    vectors=2 if layer else 1;s=raw['compiled_plan']['summary']
    expected={'hbm_read_bytes_total':(4+2*vectors)*hidden,'hbm_write_bytes_total':2*hidden,
              'source_vec_ops':(8 if layer else 6)*hidden,'vec_equivalent_ops':(13 if layer else 10)*hidden,
              'sram_read_bytes':(18+6*vectors)*hidden+16,
              'sram_write_bytes':(18+6*vectors)*hidden+16,
              'local_bus_bytes':(30+10*vectors)*hidden+32}
    for key,value in expected.items():
        if s[key]!=value:raise ValueError(f'independent two-pass ledger mismatch: {key} actual={s[key]} expected={value}')
    peak=float(raw['hardware']['derived']['core']['P_vec_per_core_ops_per_s'])
    caps=[float(r['capacity']) for r in raw['hardware']['resources']]
    scenarios={}
    for name,actual in raw['runs'].items():
        ref=simulate(hidden,layer,caps,peak,[int(n)/1e9 for n in actual['arrivals_ns']])
        scenarios[name]={'reference':ref,'comparison':compare(actual,ref)}
    return {'scope':__doc__,'independent_ledger':expected,'ledger_matches':True,'scenarios':scenarios,
            'parameters_fitted':False,'hardware_calibration':'not_available'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raw=pathlib.Path(a.raw);out=pathlib.Path(a.output)
    if out.exists():p.error('fresh output required')
    result=report(json.loads(raw.read_text()));sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    result.update(raw_sha256=sha(raw),oracle_sha256=sha(pathlib.Path(__file__)),
                  grouped_allocator_sha256=sha(pathlib.Path(__file__).with_name('elementwise_service_oracle.py')))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:item['comparison']['status']for name,item in result['scenarios'].items()}))
if __name__=='__main__':main()
