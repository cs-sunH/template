#!/usr/bin/env python3
"""Separate approximation error, strict feasibility and machine ledgers.

Does not alter the exact Fraction oracle, its failures or the Runtime. Fractions
decode every binary64 work/rate/capacity exactly. Round-to-nearest binary p
decodes the snapshot's max_digits10 long-double strings using measured compiler
mantissa widths. No tolerance is added to a resource, PRIVATE, credit or ledger.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import subprocess

lab=Path(__file__).resolve().parent
command=['g++','-dM','-E','-x','c++','/dev/null']
macros=subprocess.run(command,capture_output=True,text=True,check=True)
(lab/'floating-macros.stdout.log').write_text(macros.stdout)
(lab/'floating-macros.stderr.log').write_text(macros.stderr)
definitions={line.split()[1]:line.split()[2] for line in macros.stdout.splitlines() if len(line.split())>=3}
p64=int(definitions['__DBL_MANT_DIG__']);pld=int(definitions['__LDBL_MANT_DIG__'])
assert p64==53 and pld==64,'this decoder has only been reviewed for recorded x86 binary64/binary80 fixtures'
def rounded(value,p):
    value=F(value)
    if not value:return value
    if value<0:return -rounded(-value,p)
    k=value.numerator.bit_length()-value.denominator.bit_length()
    if F(2)**k>value:k-=1
    step=F(2)**(k-p+1)
    return round(value/step)*step
def ld(text):return rounded(F(text),pld)
def encoded(value):return {'fraction':str(value),'decimal':float(value)}
def identity(c):return {k:c[k] for k in ['family','owners','unit_exponent','boundary']}
global_metrics={};violations=[];ledger_failures=[];projection_differences=[];cases=[]
def metric(name,actual,expected,where,local):
    actual=F(actual);expected=F(expected);error=abs(actual-expected)
    relative=error/abs(expected) if expected else None
    entry={'absolute':error,'relative':relative,'where':where,'actual':actual,'expected':expected}
    for metrics in [global_metrics,local]:
        stats=metrics.setdefault(name,{'count':0,'max_absolute':None,'max_relative':None,'zero_expected_nonzero':0,'above_10_percent':0})
        stats['count']+=1
        if stats['max_absolute'] is None or error>stats['max_absolute']['absolute']:stats['max_absolute']=entry
        if relative is not None and (stats['max_relative'] is None or relative>stats['max_relative']['relative']):stats['max_relative']=entry
        stats['zero_expected_nonzero']+=expected==0 and actual!=0
        stats['above_10_percent']+=relative is not None and relative>F(1,10)

for stem in ['coupled','asymmetric']:
  for line in (lab/(stem+'-driver.stdout.log')).read_text().splitlines():
    case=json.loads(line);where=identity(case);local={};j=case['owners'];asym=case['family']=='asymmetric_coupled'
    multipliers=[1 if case['boundary']=='full' else 2,1,1 if case['boundary']=='empty' else 2] if asym else [1,1,1]
    observed=[];unobserved=[]
    for label,t in [('initial',F(0)),('quarter',F(1,4)),('half',F(1,2)),('cancel',F(1,4)),('completion',F(1))]:
      if label=='half' and not asym:continue
      if label not in case:unobserved.append(label);continue
      observed.append(label);sample=case[label];terminal=label in ['cancel','completion'];resource=[F(0)]*13
      target_vec=F(0);served_vec_sum=F(0);served_hbm_sum=[[F(0)]*2 for _ in range(2)]
      for n,e in enumerate(sample['executions']):
        epoch=case['plans'][n]['epochs'][0];planned={a['id']:a for a in epoch['activities']}
        for a in e['activities']:
          definition=planned[a['activity_id']];role_index={'load':0,'compute':1,'store':2}[a['role']]
          expected_progress=min(F(1),t*multipliers[role_index]);actual_rate=F(a['rate'])
          expected_rate=F(0) if terminal or expected_progress==1 else F(multipliers[role_index] if t<F(1,2) else 1,j)
          location=where|{'sample':label,'handle':e['handle'],'activity':a['activity_id']}
          metric('rate_per_second',actual_rate,expected_rate,location,local)
          if expected_rate:
            metric('rate_ulp_error',abs(actual_rate-expected_rate)/F(math.ulp(float(expected_rate))),0,location,local)
          metric('progress',a['progress'],expected_progress,location,local)
          metric('activity_vec_eqops',a['vec_served_eqops'],expected_progress*F(a['vec_eqops']),location,local)
          target_vec+=expected_progress*F(a['vec_eqops']);served_vec_sum+=F(a['vec_served_eqops'])
          for row in range(2):
            for direction in range(2):
              served=F(a['hbm_served_bytes'][row][direction]);served_hbm_sum[row][direction]+=served
              metric('activity_hbm_bytes',served,expected_progress*F(a['hbm_payload_bytes'][row][direction]),location|{'row':row,'direction':direction},local)
          if not terminal:
            for r in range(13):resource[r]+=F(a['planned_work'][r])*actual_rate
            private_excess=F(definition['private_time_s'])*actual_rate-1
            if private_excess>0:violations.append(location|{'kind':'PRIVATE','positive_excess':encoded(private_excess)})
          if a['done']:
            if F(a['vec_served_eqops'])!=F(a['vec_eqops']):ledger_failures.append(location|{'kind':'terminal_integer_vec'})
            if a['hbm_served_bytes']!=a['hbm_payload_bytes']:ledger_failures.append(location|{'kind':'terminal_integer_hbm'})
        if not terminal:
          for b,c in enumerate(e['snapshot_credit']):
            capacity=F(e['snapshot_credit_capacity'][b]);credit=F(c)
            if not 0<=credit<=capacity:violations.append(where|{'sample':label,'handle':e['handle'],'kind':'credit_inventory','buffer':b,'credit':encoded(credit),'capacity':encoded(capacity)})
            derivative=sum(F(a.get('credit_slope',a['credit_delta'])[b])*F(a['rate']) for a in e['activities'])
            wrong=-derivative if credit==0 else derivative if credit==capacity else F(0)
            if wrong>0:violations.append(where|{'sample':label,'handle':e['handle'],'kind':'signed_boundary_tangent','buffer':b,'positive_excess':encoded(wrong)})
        if terminal:
          metric('completion_or_cancel_ns',e['finish_ns'],j*10**9//4 if label=='cancel' else j*10**9,where|{'sample':label,'handle':e['handle']},local)
      if not terminal:
        for r,used in enumerate(resource):
          capacity=F(sample['counters']['capacity'][r]);excess=used-capacity
          if excess>0:violations.append(where|{'sample':label,'kind':'resource','resource':r,'used':encoded(used),'capacity':encoded(capacity),'positive_excess':encoded(excess),'relative_excess':encoded(excess/capacity) if capacity else None})
      counters=sample['counters'];elapsed=t*j*10**9
      metric('compute_busy_ns',counters['compute_busy_ns'],elapsed,where|{'sample':label},local)
      row_busy=[min(elapsed,F(j*10**9,m)) for m in multipliers[::2]] if asym else [elapsed]*2
      metric('hbm_busy_ns',counters['hbm_busy_ns'],max(row_busy),where|{'sample':label},local)
      metric('hbm_shared_ns',counters['hbm_shared_ns'],max(row_busy) if j>1 and case['family']!='same_owner_split' else 0,where|{'sample':label},local)
      for row in range(2):metric('row_busy_ns',counters['row_busy_ns'][row],row_busy[row],where|{'sample':label,'row':row},local)
      metric('global_vec_eqops',counters['served_by_category']['comp']['vec_eqops'],target_vec,where|{'sample':label},local)
      # Sum of individually rounded public doubles need not equal a rounded
      # long-double total. Report this display projection separately; it is not
      # the Runtime's accumulation definition. The actual machine ledger is
      # checked from its retained long-double snapshot below.
      if F(counters['served_by_category']['comp']['vec_eqops'])!=rounded(served_vec_sum,p64):projection_differences.append(where|{'sample':label,'kind':'public_vec_sum_binary64'})
      for row in range(2):
        for direction in range(2):
          if F(counters['hbm_served_bytes_by_row_direction'][row][direction])!=rounded(served_hbm_sum[row][direction],p64):projection_differences.append(where|{'sample':label,'kind':'public_hbm_sum_binary64','row':row,'direction':direction})
    # The actual quarter snapshot retains max_digits10 long-double strings.
    # Decode them to the recorded p=64 machine value. For this one stable
    # interval, powers-of-two byte/eqop multipliers are exact; compare without a
    # guard or epsilon and preserve any discrepancy.
    snapshot=case.get('quarter_snapshot')
    if snapshot:
      aggregate=[[F(0)]*2 for _ in range(2)];vec=F(0)
      for n,e in enumerate(snapshot['executions']):
        epoch=case['plans'][n]['epochs'][0];definitions={a['id']:a for a in epoch['activities']}
        execution_hbm=[[F(0)]*2 for _ in range(2)];execution_vec=F(0)
        for a in e['activities']:
          definition=definitions[a['activity_id']];p=ld(a['curve_progress_exact']) if 'curve_progress_exact' in a else F(a['progress'])
          actual_vec=ld(a['vec_served']);execution_vec+=actual_vec
          if actual_vec!=rounded(p*F(definition['vec_eqops']),pld):ledger_failures.append(where|{'kind':'snapshot_activity_vec_machine','handle':e['handle'],'activity':a['activity_id']})
          for row in range(2):
            for direction in range(2):
              value=ld(a['hbm_served'][row][direction]);execution_hbm[row][direction]+=value
              if value!=rounded(p*F(definition['hbm_bytes'][row][direction]),pld):ledger_failures.append(where|{'kind':'snapshot_activity_hbm_machine','handle':e['handle'],'activity':a['activity_id']})
        if ld(e['vec_served'])!=rounded(execution_vec,pld):ledger_failures.append(where|{'kind':'snapshot_execution_vec_machine'})
        vec+=execution_vec
        for row in range(2):
          for direction in range(2):
            if ld(e['hbm_served'][row][direction])!=rounded(execution_hbm[row][direction],pld):ledger_failures.append(where|{'kind':'snapshot_execution_hbm_machine'})
            aggregate[row][direction]+=execution_hbm[row][direction]
      for row in range(2):
        for direction in range(2):
          if ld(snapshot['stats']['hbm_served'][row][direction])!=rounded(aggregate[row][direction],pld):ledger_failures.append(where|{'kind':'snapshot_global_hbm_machine'})
      # These fixtures' nonzero vector totals are >=256. The 18 digits after
      # the decimal in decimal_measure suffice to recover their binary80 value.
      actual_vec=ld(snapshot['stats']['served_by_category']['comp']['vec_eqops_exact_decimal'])
      if actual_vec!=rounded(vec,pld):ledger_failures.append(where|{'kind':'snapshot_global_vec_machine'})
      public_vec=F(case['quarter']['counters']['served_by_category']['comp']['vec_eqops'])
      if public_vec!=rounded(actual_vec,p64):ledger_failures.append(where|{'kind':'global_vec_public_cast'})
    cases.append(where|{'observed_samples':observed,'not_observed_samples':unobserved,'metrics':local,'original_driver_status':case['driver_status'],'original_error':case.get('error',case.get('snapshot_error'))})

def wire(value):
    if isinstance(value,F):return encoded(value)
    if isinstance(value,dict):return {k:wire(v) for k,v in value.items()}
    if isinstance(value,list):return [wire(v) for v in value]
    return value
report={'scope':'observed synthetic error and exact binary64 feasibility, separate from unchanged exact Fraction case verdicts',
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'compiler_macro_command':command,'macro_log_sha256':hashlib.sha256((lab/'floating-macros.stdout.log').read_bytes()).hexdigest(),
        'binary64_significand_bits':p64,'long_double_significand_bits':pld,
        'global_metrics':global_metrics,'strict_feasibility_violations':violations,'machine_ledger_failures':ledger_failures,
        'public_projection_nonadditivity':projection_differences,'cases':cases,
        'interpretation':'Small relative prediction error does not waive strict feasibility residuals or the snapshot rejection. Missing continuation/cancel/completion remain not_run for the six rejected imports. No target hardware calibration claim.'}
(lab/'layered-assessment.json').write_text(json.dumps(wire(report),indent=2)+'\n')
print(json.dumps({'cases':len(cases),'strict_feasibility_violation_count':len(violations),'machine_ledger_failure_count':len(ledger_failures),
                  'above_10_percent_metric_count':sum(v['above_10_percent'] for k,v in global_metrics.items() if k!='rate_ulp_error')}))
