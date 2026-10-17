#!/usr/bin/env python3
"""Fixed128 dyadic arithmetic against unchanged frozen Fraction inputs."""
import hashlib,importlib.util,json,platform,statistics,subprocess,time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OLD=ROOT/'work/norm-affine-evaluator-20261017'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('frozen_fraction_oracle',HERE/'oracle_base.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
F=oracle.F;decimal=oracle.decimal;from_hex=oracle.from_hex
def sign(q):return(q>0)-(q<0)
def term_value(t):return(-1 if t['negative']else 1)*F(int(t['mantissa_hex'],16))*F(2)**t['exponent']
def directed(q):
 nearest=oracle.rounded_ld(abs(q))if q>=0 else-oracle.rounded_ld(-q)
 def neighbor(x,d):
  if x>=0:return oracle.neighboring_ld(x,d)
  return-oracle.neighboring_ld(-x,-d)
 return(neighbor(nearest,-1)if nearest>q else nearest,neighbor(nearest,1)if nearest<q else nearest)
def verify_result(result,reference):
 if not result['supported']:return None
 magnitude=F(int(result['magnitude_hex'],16))*F(2)**result['exponent']
 exact=result['sign']*magnitude;assert exact==reference,(exact,reference)
 assert result['sign']==sign(reference)
 summed=(F(int(result['positive_group_hex'],16))-int(result['negative_group_hex'],16))*F(2)**result['base_exponent'];assert summed==reference
 lo,hi=map(from_hex,[result['lower_hex'],result['upper_hex']]);assert lo<=reference<=hi
 assert(lo,hi)==directed(reference),(lo,hi,directed(reference))
 return lo,hi
def main():
 if(HERE/'identity.json').exists():raise RuntimeError('refuse overwrite completed checkpoint')
 assert sha(HERE/'input.json')==sha(OLD/'input.json')=='000cebcb1a2d8027bc8bb353f04a95076929ea53320333a0898f1086a43e4700';assert sha(HERE/'oracle_base.py')==sha(OLD/'run.py')
 data=json.loads((HERE/'input.json').read_text());assert len(data['cases'])==1952
 meta={'scope':'fixed unsigned128 dyadic arithmetic only; not Runtime integration, snapshot context, or API budget','input_sha256':sha(HERE/'input.json'),'source_sha256':sha(HERE/'evaluator.cc'),'core_header_sha256':sha(HERE/'Dyadic128.hh'),'oracle_sha256':sha(HERE/'oracle_base.py'),'runner_sha256':sha(Path(__file__)),'json_header_sha256':sha(ROOT/'astra-sim-wscllm/extern/helper/json/json.hpp'),'commands':[],'host':platform.platform()}
 exe=HERE/'evaluator';commands=[('compiler_version',['c++','--version']),('compile',['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-I'+str(ROOT/'astra-sim-wscllm/extern/helper'),str(HERE/'evaluator.cc'),'-o',str(exe)]),('run',[str(exe),str(HERE/'input.json'),str(HERE/'output.jsonl')])]
 for label,cmd in commands:
  start=time.time();r=subprocess.run(cmd,text=True,capture_output=True,cwd=ROOT,timeout=120);meta['commands'].append({'label':label,'command':cmd,'returncode':r.returncode,'elapsed_s':time.time()-start});(HERE/(label+'.stdout.log')).write_text(r.stdout);(HERE/(label+'.stderr.log')).write_text(r.stderr)
  if r.returncode:raise RuntimeError((label,r.returncode))
 records=[json.loads(x)for x in(HERE/'output.jsonl').read_text().splitlines()];abi=records.pop(0);assert abi['supported']
 benchmark=records.pop();controls=records.pop();assert benchmark['kind']=='benchmark'and controls['kind']=='domain_controls';assert len(records)==1952
 groups=defaultdict(lambda:defaultdict(lambda:{'count':0,'supported':0,'unsupported':0,'reasons':Counter(),'exact_sign_errors':0,'exact_integer_identity_errors':0,'interval_errors':0,'negative_balance_retained':0,'interval_strictly_positive':0,'interval_strictly_negative':0,'interval_exact_zero':0,'interval_sign_uncertain':0,'point_interval_exact':0,'maximum_interval_width':'0'}));cases=[];negative_q=negative_h=0
 for source,result in zip(data['cases'],records):
  assert source['id']==result['id'];I,H=map(lambda s:F(float(s)),[source['initial'],source['capacity']]);d=list(map(lambda s:F(float(s)),source['coefficients']));p=[oracle.rounded_ld(F(x))for x in source['progress']]
  if'neighbor'in source:
   k=source['neighbor']['index'];p[k]=oracle.neighboring_ld(p[k],source['neighbor']['direction'])
  assert p==list(map(from_hex,result['progress_hex']))
  q=I+sum(di*pi for di,pi in zip(d,p));negative_q+=q<0;negative_h+=H-q<0
  row={'id':source['id'],'group':source['group'],'label':source['label'],'source':source['source'],'channel':source['channel'],'reference_q':decimal(q),'reference_h':decimal(H-q),'results':{}}
  for side,reference in [('q',q),('h',H-q)]:
   actual=result[side];metric=groups[source['group']][side];metric['count']+=1
   if actual['terms']:
    expected_terms=([I]if side=='q'else[H,-I])+[(1 if side=='q'else-1)*di*pi for di,pi in zip(d,p)];assert list(map(term_value,actual['terms']))==expected_terms
   if actual['supported']:
    lo,hi=verify_result(actual,reference);metric['supported']+=1;metric['negative_balance_retained']+=reference<0;metric['interval_strictly_positive']+=lo>0;metric['interval_strictly_negative']+=hi<0;metric['interval_exact_zero']+=lo==hi==0;metric['interval_sign_uncertain']+=lo<=0<=hi and lo!=hi;metric['point_interval_exact']+=lo==hi
    if hi-lo>F(metric['maximum_interval_width']):metric['maximum_interval_width']=decimal(hi-lo)
    row['results'][side]={'supported':True,'sign':actual['sign'],'lower':decimal(lo),'upper':decimal(hi),'magnitude_hex':actual['magnitude_hex'],'exponent':actual['exponent']}
   else:metric['unsupported']+=1;metric['reasons'][actual['reason']]+=1;row['results'][side]={'supported':False,'reason':actual['reason']}
  cases.append(row)
 expected_reasons={'span_129':'alignment_exceeds_128_bits','mantissa_plus_shift_129':'alignment_exceeds_128_bits','positive_accumulator_overflow':'positive_accumulator_overflow','negative_accumulator_overflow':'negative_accumulator_overflow','output_overflow':'result_or_bound_overflow','output_subnormal':'result_or_bound_subnormal','progress_subnormal_rejected':'nonzero_subnormal_or_invalid_progress','coefficient_subnormal_rejected':'nonfinite_or_subnormal_coefficient'}
 expected_supported={'maximum_128_integer':F(2**128-1),'zero_cancel':F(0),'negative_rounding_flip':-F(2**64+1,2**64),'zero_terms_supported':F(0),'progress_one_supported':F(0)}
 checked=[]
 for control in controls['cases']:
  name=control['name'];result=control['result']
  if name in expected_reasons:assert not result['supported']and result['reason']==expected_reasons[name]
  else:assert name in expected_supported and result['supported'];verify_result(result,expected_supported[name])
  checked.append({'name':name,'passed':True,'supported':result['supported'],'reason':result['reason']})
 cost={'scope':benchmark['scope'],'eligible_samples':benchmark['eligible_samples'],'batches':benchmark['batches'],'rounds_per_batch':benchmark['rounds_per_batch'],'methods':{name:{'batch_average_ns_per_scalar':times,'median_batch_average_ns':statistics.median(times),'min_batch_average_ns':min(times),'max_batch_average_ns':max(times)}for name,times in benchmark['methods'].items()}}
 report={'scope':meta['scope'],'abi':abi,'case_count':len(cases),'input_identical_to_v1':True,'negative_true_inventory_case_count':negative_q,'negative_true_headroom_case_count':negative_h,'groups':{g:dict(v)for g,v in groups.items()},'domain_controls':checked,'benchmark':cost,'proof_scope':'exact algebraic dyadic identity plus pre-shift bit-width and unsigned group-overflow checks; directed LD bounds constructed from leading64/remainder with normal/finite exponent checks; sample comparisons independently verify implementation but do not replace proof or integration/snapshot gates','cases':cases}
 (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');(HERE/'summary.json').write_text(json.dumps({k:v for k,v in report.items()if k!='cases'},indent=2)+'\n');(HERE/'benchmark.json').write_text(json.dumps(benchmark,indent=2)+'\n');(HERE/'domain-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
 meta['executable_sha256']=sha(exe);meta['outputs_sha256']={p.name:sha(p)for p in HERE.iterdir()if p.is_file()and p.suffix in ['.json','.jsonl','.log']and p.name!='identity.json'};(HERE/'identity.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='cases'},indent=2))
if __name__=='__main__':main()
