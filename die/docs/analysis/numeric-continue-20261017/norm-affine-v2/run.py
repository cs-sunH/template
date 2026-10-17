#!/usr/bin/env python3
"""Frozen-input FMA checkpoint. No Runtime/Plan compiler build or execution."""
import hashlib,importlib.util,json,platform,statistics,subprocess,time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OLD=ROOT/'work/norm-affine-evaluator-20261017'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('frozen_fraction_oracle',HERE/'oracle_base.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
F=oracle.F;value=oracle.value;decimal=oracle.decimal
def rounded_signed(q):return oracle.rounded_ld(q)if q>=0 else-oracle.rounded_ld(-q)
def sign(q):return(q>0)-(q<0)
def main():
 if(HERE/'identity.json').exists():raise RuntimeError('refuse overwrite completed checkpoint')
 data=json.loads((HERE/'input.json').read_text());old_input_sha=sha(OLD/'input.json');assert sha(HERE/'input.json')==old_input_sha=='000cebcb1a2d8027bc8bb353f04a95076929ea53320333a0898f1086a43e4700';assert len(data['cases'])==1952
 assert sha(HERE/'oracle_base.py')==sha(OLD/'run.py'),'preserve independent Fraction oracle unchanged'
 meta={'scope':'standalone fixed-term FMA arithmetic at frozen represented progress; not integration, snapshot context, Runtime speed or hardware accuracy','input_sha256':sha(HERE/'input.json'),'frozen_v1_input_sha256':old_input_sha,'frozen_v1_report_sha256':sha(OLD/'report.json'),'frozen_v1_output_sha256':sha(OLD/'output.jsonl'),'oracle_sha256':sha(HERE/'oracle_base.py'),'runner_sha256':sha(Path(__file__)),'source_sha256':sha(HERE/'evaluator.cc'),'json_header_sha256':sha(ROOT/'astra-sim-wscllm/extern/helper/json/json.hpp'),'commands':[],'host':platform.platform()}
 exe=HERE/'evaluator';commands=[('compiler_version',['c++','--version']),('compile',['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-I'+str(ROOT/'astra-sim-wscllm/extern/helper'),str(HERE/'evaluator.cc'),'-o',str(exe)]),('run',[str(exe),str(HERE/'input.json'),str(HERE/'output.jsonl')])]
 for label,cmd in commands:
  start=time.time();r=subprocess.run(cmd,text=True,capture_output=True,cwd=ROOT,timeout=120);meta['commands'].append({'label':label,'command':cmd,'returncode':r.returncode,'elapsed_s':time.time()-start});(HERE/(label+'.stdout.log')).write_text(r.stdout);(HERE/(label+'.stderr.log')).write_text(r.stderr)
  if r.returncode:raise RuntimeError((label,r.returncode))
 rows=[json.loads(x)for x in(HERE/'output.jsonl').read_text().splitlines()];abi=rows.pop(0);assert abi['long_double_digits']==64
 benchmark=rows.pop();assert benchmark['kind']=='benchmark'
 # Re-evaluate every unchanged v1 field with its original independent oracle,
 # and require bit-for-bit agreement with all old expression outputs.
 old_rows=[json.loads(x)for x in(OLD/'output.jsonl').read_text().splitlines()][1:]
 assert len(rows)==len(old_rows)==1952
 for current,old in zip(rows,old_rows):
  assert{key:current[key]for key in old}==old
 filtered=HERE/'unchanged-expressions.jsonl';filtered.write_text('\n'.join(json.dumps(x)for x in[abi]+rows)+'\n');oracle.HERE=HERE;old_report=oracle.analyze(data,filtered);(HERE/'report.json').rename(HERE/'v1-oracle-recheck.json')
 assert{key:old_report[key]for key in['groups','structural_anchor_compared_with_direct','negative_true_inventory_case_count','negative_true_headroom_case_count','given_progress_outside_capacity_case_count']}=={key:json.loads((OLD/'report.json').read_text())[key]for key in['groups','structural_anchor_compared_with_direct','negative_true_inventory_case_count','negative_true_headroom_case_count','given_progress_outside_capacity_case_count']}
 groups=defaultdict(lambda:defaultdict(lambda:{'count':0,'normal_domain_count':0,'unsupported_underflow_domain_count':0,'nonfinite_count':0,'nonzero_errors':0,'normal_nonzero_errors':0,'max_absolute_error':'0','max_absolute_error_case':None,'normal_max_absolute_error':'0','normal_max_absolute_error_case':None,'point_sign_errors':0,'normal_point_sign_errors':0,'incorrect_rounding_count':0,'normal_incorrect_rounding_count':0,'product_decomposition_errors':0,'normal_product_decomposition_errors':0,'interval_excludes_reference':0,'interval_uncertain':0,'interval_certified_positive':0,'interval_certified_negative':0,'interval_exact_zero':0,'old_wrong_sign_corrected':0,'old_correct_sign_regressed':0}))
 cases=[];comparisons=defaultdict(lambda:defaultdict(int))
 u=F(1,2**64);gamma8=8*u/(1-8*u)
 for source,current,previous in zip(data['cases'],rows,old_report['cases']):
  assert source['id']==current['id']==previous['id'];I,H=map(value,[current['initial'],current['capacity']]);d=list(map(value,current['coefficients']));p=list(map(value,current['progress']));q=I+sum(di*pi for di,pi in zip(d,p));row={'id':source['id'],'group':source['group'],'label':source['label'],'source':source['source'],'channel':source['channel'],'reference_q':decimal(q),'reference_h':decimal(H-q),'results':{}}
  for side,reference in [('q',q),('h',H-q)]:
   result=current['fma_dot_'+side];actual=value(result['value']);normal=result['normal_domain'];metric=groups[source['group']][side];metric['count']+=1;metric['normal_domain_count']+=normal;metric['unsupported_underflow_domain_count']+=not normal;metric['nonfinite_count']+=not result['finite'];assert result['finite']
   exact_products=[(1 if side=='q'else-1)*di*pi for di,pi in zip(d,p)]
   product_errors=sum(value(pair['high'])+value(pair['low'])!=exact for pair,exact in zip(result['products'],exact_products));metric['product_decomposition_errors']+=product_errors;metric['normal_product_decomposition_errors']+=product_errors if normal else 0
   error=abs(actual-reference);metric['nonzero_errors']+=bool(error);metric['normal_nonzero_errors']+=normal and bool(error)
   for prefix in ['', 'normal_']if normal else ['']:
    if error>F(metric[prefix+'max_absolute_error']):metric[prefix+'max_absolute_error']=decimal(error);metric[prefix+'max_absolute_error_case']=source['id']
   wrong=sign(actual)!=sign(reference);rounding_wrong=actual!=rounded_signed(reference);metric['point_sign_errors']+=wrong;metric['normal_point_sign_errors']+=normal and wrong;metric['incorrect_rounding_count']+=rounding_wrong;metric['normal_incorrect_rounding_count']+=normal and rounding_wrong
   old_direct=value(current['direct_'+side]['plain']);old_anchor=value(current['anchors'][current[side+'_pivot']][side]['compensated']);old_wrong=sign(old_anchor)!=sign(reference);metric['old_wrong_sign_corrected']+=old_wrong and not wrong;metric['old_correct_sign_regressed']+=not old_wrong and wrong
   # Preserve v1's gamma8/triangle-transfer mechanism without narrowing it to
   # manufacture a signed-boundary pass for the FMA method.
   base_terms=[I]+[di*pi for di,pi in zip(d,p)]if side=='q'else[H-I]+[-di*pi for di,pi in zip(d,p)]
   radius=gamma8*sum(map(abs,base_terms))+abs(actual-old_direct);lo,hi=actual-radius,actual+radius
   eligible=normal and previous['normal_error_bound_eligible']
   if eligible:
    metric['interval_excludes_reference']+=not lo<=reference<=hi;metric['interval_uncertain']+=lo<=0<=hi and radius>0;metric['interval_certified_positive']+=lo>0;metric['interval_certified_negative']+=hi<0;metric['interval_exact_zero']+=lo==hi==0
   comparisons[source['group']][('improved'if error<abs(old_direct-reference)else'worse'if error>abs(old_direct-reference)else'equal')+':'+side]+=1
   row['results'][side]={'value':decimal(actual),'absolute_error':decimal(error),'normal_domain':normal,'point_sign_error':wrong,'incorrect_rounding':rounding_wrong,'product_decomposition_errors':product_errors,'v1_anchor_value':decimal(old_anchor),'v1_direct_value':decimal(old_direct),'interval_graded':eligible,'radius':decimal(radius),'lower':decimal(lo),'upper':decimal(hi)}
  cases.append(row)
 cost={'scope':benchmark['scope'],'eligible_samples':benchmark['eligible_samples'],'batches':benchmark['batches'],'rounds_per_batch':benchmark['rounds_per_batch'],'methods':{name:{'batch_average_ns_per_scalar_dot':times,'median_batch_average_ns':statistics.median(times),'min_batch_average_ns':min(times),'max_batch_average_ns':max(times)}for name,times in benchmark['methods'].items()}}
 report={'scope':meta['scope'],'abi':abi,'case_count':len(cases),'input_frozen_and_v1_outputs_unchanged':True,'negative_true_inventory_case_count':sum(F(c['reference_q'])<0 for c in cases),'negative_true_headroom_case_count':sum(F(c['reference_h'])<0 for c in cases),'groups':{g:dict(v)for g,v in groups.items()},'fma_compared_with_direct':dict(comparisons),'interval_policy':'exactly v1 gamma8 direct-expression bound plus triangle transfer; no tighter FMA bound is asserted; unsupported underflow domain ungraded','benchmark':cost,'cases':cases}
 (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');(HERE/'summary.json').write_text(json.dumps({k:v for k,v in report.items()if k!='cases'},indent=2)+'\n');(HERE/'benchmark.json').write_text(json.dumps(benchmark,indent=2)+'\n')
 meta['executable_sha256']=sha(exe);meta['outputs_sha256']={p.name:sha(p)for p in HERE.iterdir()if p.is_file()and p.suffix in ['.json','.jsonl','.log']and p.name!='identity.json'};(HERE/'identity.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='cases'},indent=2))
if __name__=='__main__':main()
