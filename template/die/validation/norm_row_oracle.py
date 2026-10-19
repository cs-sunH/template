#!/usr/bin/env python3
"""Offline discrete rows, cached BF16 gamma/beta, one Core, one region.

Row input/parameter packing, conversion visits and SRAM/LOCAL use are derived
independently of production curves. Reuses the previously tested 13-domain
nonnegative grouped allocator and finite-slot row scheduling engine. Within a
row, statistics/apply are one serial Compute budget, the same declared macro
assumption; this does not validate real SFU/hardware timing or long-row passes.
"""
import argparse
import hashlib
import json
import pathlib
from elementwise_service_oracle import Activity, Block, compare, simulate

def report(raw):
    fixture=raw['fixture'];rows=int(fixture['rows']);h=int(fixture['hidden'])
    if not 0<rows<=4096 or not 0<h<=4096:
        raise ValueError('offline bounded cached-row fixture required')
    if int(raw['hardware']['derived']['die']['core_count'])!=1:
        raise ValueError('this reference covers one Core only')
    parameters=1 if fixture['norm']=='rmsnorm' else 2
    math_ops=6 if parameters==1 else 8
    blocks=[]
    for row in range(rows):
        first=row==0
        read=2*h+(2*parameters*h if first else 0)
        eq=(math_ops+3)*h+(parameters*h if first else 0)
        blocks.append(Block(0,h,read,2*h,eq,(read//2,read//2),(h,h)))
    capacities=[float(x['capacity']) for x in raw['hardware']['resources']]
    peak=float(raw['hardware']['derived']['core']['P_vec_per_core_ops_per_s'])
    def factory(owner,index,role):
        b=blocks[index];work=[0.0]*13
        if role=='load':
            work[2]=b.read_bytes;work[4],work[5]=b.rows_read;quantity=b.read_bytes
        elif role=='store':
            work[1]=b.write_bytes;work[4],work[5]=b.rows_write;quantity=b.write_bytes
        else:
            first=index==0
            work[0]=b.eqops/peak
            # Two raw SRAM X visits, two materialized FP32 input passes,
            # FP32 parameter application, output cast, and first cache cast.
            work[1]=(16+4*parameters)*h+(2*parameters*h if first else 0)
            work[2]=14*h+(4*parameters*h if first else 0)
            work[3]=work[1]+work[2];quantity=work[3]
        private=max([work[0]]+[work[r]/capacities[r] for r in [1,2,3] if work[r]])
        return Activity(owner,index,role,work,private,quantity)
    expected_read=sum(b.read_bytes for b in blocks);expected_write=rows*2*h
    summary=raw['compiled_plan']['summary']
    ledger_ok=(summary['hbm_read_bytes_total']==expected_read and summary['hbm_write_bytes_total']==expected_write
        and summary['vec_equivalent_ops']==sum(b.eqops for b in blocks))
    if not ledger_ok:raise ValueError('production Norm ledger does not match independently derived cached rows')
    scenarios={}
    for name,actual in raw['runs'].items():
        reference=simulate(blocks,[n/1e9 for n in actual['arrivals_ns']],1 if rows==1 else 2,
            capacities,factory,True)
        scenarios[name]={'reference':reference,'comparison':compare(actual,reference)}
    return {'scope':__doc__,'ledger_matches':ledger_ok,'independent_read_bytes':expected_read,
        'independent_write_bytes':expected_write,'scenarios':scenarios,
        'classification_scope':'completion plus original busy; first output reported separately',
        'hardware_calibration':'not_available','parameters_fitted':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raw_path=pathlib.Path(a.raw);out=pathlib.Path(a.output)
    if out.exists():p.error('fresh output required')
    result=report(json.loads(raw_path.read_text()))
    result['raw_sha256']=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    result['oracle_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    result['shared_reference_sha256']=hashlib.sha256(pathlib.Path(__file__).with_name('elementwise_service_oracle.py').read_bytes()).hexdigest()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:item['comparison']['status'] for name,item in result['scenarios'].items()}))
if __name__=='__main__':main()
