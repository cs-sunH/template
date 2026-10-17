#!/usr/bin/env python3
"""Standalone affine arithmetic checkpoint; no Runtime build or execution."""
import hashlib,json,platform,subprocess,time
from collections import defaultdict
from decimal import Decimal,localcontext
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD=ROOT/'work/norm-runtime-current';SOURCE=HERE/'evaluator.cc'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def decimal(q):
 with localcontext() as ctx:
  ctx.prec=110
  return str(Decimal(q.numerator)/Decimal(q.denominator))
def from_hex(s):
 sign=-1 if s.startswith('-') else 1;s=s.lstrip('+-');base,exponent=s.lower().split('p');exponent=int(exponent);base=base[2:]
 whole,_,frac=base.partition('.');numerator=int(whole+frac,16);power=exponent-4*len(frac)
 return sign*F(numerator)*(F(2)**power)
def value(x):return from_hex(x['hex'])
def coefficient(x):return repr(float(x))
def load(path):return json.loads(path.read_text())
def pow2(n):return F(2)**n
def binary_exponent(q):
 e=q.numerator.bit_length()-q.denominator.bit_length()
 while q<pow2(e):e-=1
 while q>=pow2(e+1):e+=1
 return e
def rounded_ld(q):
 if q==0:return q
 assert q>0
 spacing=pow2(max(binary_exponent(q)-63,-16445));scaled=q/spacing;n,r=divmod(scaled.numerator,scaled.denominator)
 if 2*r>scaled.denominator or 2*r==scaled.denominator and n%2:n+=1
 return n*spacing
def neighboring_ld(q,direction):
 if q==0:return direction*pow2(-16445)
 e=binary_exponent(q)
 spacing=pow2(max(e-63-(direction<0 and q==pow2(e)),-16445))
 return q+direction*spacing

def prepare():
 cases=[];sources={};contract_path=OLD/'norm-seven-epoch-contract.json';contract=load(contract_path);sources[str(contract_path.relative_to(ROOT))]=sha(contract_path)
 def emit(group,label,epoch,progress,source,neighbor=None):
  acts=epoch['activities'];assert 1<=len(acts)<=3
  for a in acts:
   assert a['credit_curve']==[{'progress':0.0,'cumulative':[0.0,0.0]},{'progress':1.0,'cumulative':a['credit_delta']}]
  for b in range(2):
   row={'id':len(cases),'group':group,'label':label,'source':source,'channel':b,'initial':coefficient(epoch['credit_initial'][b]),'capacity':coefficient(epoch['credit_capacity'][b]),'coefficients':[coefficient(a['credit_delta'][b]) for a in acts],'progress':[str(progress[a['id']]) for a in acts],'activity_ids':[a['id']for a in acts]}
   if neighbor is not None:row['neighbor']=neighbor
   cases.append(row)
 log=OLD/'run-interior-internal-diagnostic/run.stderr.log';sources[str(log.relative_to(ROOT))]=sha(log)
 for line_no,line in enumerate(log.read_text().splitlines(),1):
  if not line.startswith('norm_recompute '):continue
  record=json.loads(line.split(' ',1)[1]);emit('saved_internal',f'line{line_no}:h{record["handle"]}:e{record["epoch"]}',contract['epochs'][record['epoch']],{a['id']:a['progress']for a in record['activities']},str(log.relative_to(ROOT))+':'+str(line_no))
 for label in ['run-combined-ab','run-interior-only','run-physical-enclosure']:
  for path in sorted((OLD/label/'trace').glob('event-*.json')):
   snapshot=load(path);sources[str(path.relative_to(ROOT))]=sha(path)
   for e in snapshot['executions']:
    if e['epoch_index']>=len(e['plan']['epochs']):continue
    epoch=e['plan']['epochs'][e['epoch_index']];progress={a['activity_id']:a.get('curve_progress_exact',str(a['progress']))for a in e['activities']if a['epoch_index']==e['epoch_index']}
    emit('saved_public',f'{label}:{path.stem}:h{e["handle"]}:e{e["epoch_index"]}',epoch,progress,str(path.relative_to(ROOT)))
 builder=OLD/'NormLongRow.hh';text=builder.read_text();sources[str(builder.relative_to(ROOT))]=sha(builder)
 for required in ['tiles=ceildiv(hidden,tile)','remaining_load.credit={{static_cast<long double>(tiles-3),0}}','middle_apply.credit={{-static_cast<long double>(tiles-2),static_cast<long double>(tiles-3)}}','early_store.credit={{0,-static_cast<long double>(tiles-2)}}']:
  assert required in text,required
 # These are actual frozen builder coefficient formulas, not newly compiled
 # Plans. Hardware/shape timing is not evaluated by this experiment.
 coefficient_sets=[]
 for hidden in [513,768,1024,65536,65537,131072,131073]:
  tiles=(hidden+255)//256
  for epoch_no in [1,4]:
   epoch=json.loads(json.dumps(contract['epochs'][epoch_no]));epoch['name']=f'source_formula_h{hidden}_epoch{epoch_no}'
   ds=[[hidden-256,0],[-hidden,0]] if epoch_no==1 else [[tiles-3,0],[-(tiles-2),tiles-3],[0,-(tiles-2)]]
   for a,d in zip(epoch['activities'],ds):a['credit_delta']=d;a['credit_curve']=[{'progress':0.0,'cumulative':[0.0,0.0]},{'progress':1.0,'cumulative':d}]
   if hidden==65537:assert ds==[a['credit_delta']for a in contract['epochs'][epoch_no]['activities']]
   coefficient_sets.append({'hidden':hidden,'tiles':tiles,'tail':hidden-(tiles-1)*256,'epoch':epoch_no,'coefficients':ds,'provenance':'frozen current-Core long-row builder formula; no newly compiled Plan'})
   # Solve q=0 and q=capacity exactly, then inspect adjacent represented
   # progress. All invalid/out-of-domain solutions are excluded before emission.
   for b in range(2):
    d=[F(a['credit_delta'][b])for a in epoch['activities']];nonzero=[i for i,x in enumerate(d)if x]
    if not nonzero:continue
    pivot=min(nonzero,key=lambda i:d[i]);others=[i for i in nonzero if i!=pivot]
    for seed in [F(0),F(1,4),F(1,2),F(3,4),F(1)]:
     for boundary in [F(0),F(epoch['credit_capacity'][b])]:
      p=[seed for _ in d];p[pivot]=(boundary-F(epoch['credit_initial'][b])-sum(d[i]*p[i]for i in others))/d[pivot]
      if not all(0<=v<=1 for v in p):continue
      progress={a['id']:decimal(x)for a,x in zip(epoch['activities'],p)};name=f'h{hidden}:e{epoch_no}:b{b}:seed{seed}:q{boundary}'
      emit('source_formula_boundary_neighbors',name,epoch,progress,str(builder.relative_to(ROOT)))
      for index in nonzero:
       for direction in [-1,1]:
        if p[index]==0 and direction<0 or p[index]==1 and direction>0:continue
        emit('source_formula_boundary_neighbors',name+f':p{index}:{direction:+}',epoch,progress,str(builder.relative_to(ROOT)),{'index':index,'direction':direction})
 # The independent dyadic capacity1 ledger is included as a direct slack
 # control, without invoking its Runtime driver or redoing old 456 cases.
 for filling in [False,True]:
  gap=F(1,2**36);delta=2*gap
  for p in [F(0),F(1,4),F(1,2),F(3,4),F(1)]:
   e={'activities':[{'id':1,'credit_delta':[float(delta if filling else -delta),0]}],'credit_initial':[float(1-gap if filling else gap),0],'credit_capacity':[1,0]}
   for a in e['activities']:a['credit_curve']=[{'progress':0.0,'cumulative':[0.0,0.0]},{'progress':1.0,'cumulative':a['credit_delta']}]
   emit('dyadic_capacity1',f'filling{filling}:p{p}',e,{1:decimal(p)},'exact dyadic control: gap2^-36 delta2^-35')
 input_data={'scope':'standalone affine given-progress arithmetic; not integration or snapshot validation','sources_sha256':sources,'coefficient_sets':coefficient_sets,'cases':cases}
 (HERE/'input.json').write_text(json.dumps(input_data,indent=2)+'\n');return input_data

def analyze(data,path):
 records=[json.loads(line)for line in path.read_text().splitlines()];abi=records.pop(0);assert abi['long_double_digits']==64
 u=F(1,2**64);gamma8=8*u/(1-8*u)
 summary=defaultdict(lambda:defaultdict(lambda:{'count':0,'nonzero_errors':0,'max_absolute_error':'0','max_absolute_error_case':None,'sign_disagreements':0,'bounds_excluded_reference':0,'bounds_uncertain':0,'certified_negative':0,'ungraded_subnormal_bound_cases':0}))
 detailed=[];pairs=defaultdict(lambda:defaultdict(int));input_rows={r['id']:r for r in data['cases']}
 for result in records:
  source=input_rows[result['id']];group=source['group']
  if 'error'in result:raise RuntimeError((source['id'],result['error']))
  I,H,E=map(value,[result['initial'],result['capacity'],result['endpoint']]);d=list(map(value,result['coefficients']));p=list(map(value,result['progress']));n=len(d)
  assert I==F(float(source['initial'])) and H==F(float(source['capacity']))
  assert d==list(map(lambda x:F(float(x)),source['coefficients']))
  expected=[rounded_ld(F(x))for x in source['progress']]
  if'neighbor'in source:
   k=source['neighbor']['index'];expected[k]=neighboring_ld(expected[k],source['neighbor']['direction'])
  assert p==expected,(source['id'],p,expected)
  assert E==I+sum(d);assert value(result['head_initial'])==H-I;assert value(result['head_endpoint'])==H-E
  eligible=all(x==0 or abs(x)>=pow2(-16382)for x in p)
  exact_q=I+sum(di*pi for di,pi in zip(d,p));exact={'q':exact_q,'h':H-exact_q}
  row={'id':source['id'],'group':group,'label':source['label'],'source':source['source'],'channel':source['channel'],'reference_q':decimal(exact['q']),'reference_h':decimal(exact['h']),'reference_outside_capacity':exact['q']<0 or exact['h']<0,'normal_error_bound_eligible':eligible,'results':{}}
  for side in ['q','h']:
   base_terms=[I]+[di*pi for di,pi in zip(d,p)]if side=='q'else[H-I]+[-di*pi for di,pi in zip(d,p)]
   base_plain=value(result['direct_'+side]['plain']);base_radius=gamma8*sum(map(abs,base_terms))
   j=result[side+'_pivot'];anchor=result['anchors'][j][side]
   terms=[I*(1-p[j]),E*p[j]]if side=='q'else[(H-I)*(1-p[j]),(H-E)*p[j]]
   terms += [(1 if side=='q'else -1)*d[i]*(p[i]-p[j])for i in range(n)if i!=j]
   anchor_plain=value(anchor['plain']);anchor_radius=gamma8*sum(map(abs,terms));assert sum(terms)==exact[side]
   methods={'direct_plain':(base_plain,base_radius),'direct_compensated':(value(result['direct_'+side]['compensated']),base_radius+abs(value(result['direct_'+side]['compensated'])-base_plain)),'structural_anchor_plain':(anchor_plain,anchor_radius),'structural_anchor_compensated':(value(anchor['compensated']),anchor_radius+abs(value(anchor['compensated'])-anchor_plain))}
   errors={}
   for method,(actual,radius)in methods.items():
    error=abs(actual-exact[side]);errors[method]=error;metric=summary[group][method+':'+side];metric['count']+=1;metric['nonzero_errors']+=bool(error)
    if error>F(metric['max_absolute_error']):metric['max_absolute_error']=decimal(error);metric['max_absolute_error_case']=source['id']
    sign=lambda x:(x>0)-(x<0)
    metric['sign_disagreements']+=sign(actual)!=sign(exact[side]);metric['ungraded_subnormal_bound_cases']+=not eligible
    if eligible:
     metric['bounds_excluded_reference']+=not(actual-radius<=exact[side]<=actual+radius)
     metric['bounds_uncertain']+=actual-radius<=0<=actual+radius and radius>0;metric['certified_negative']+=actual+radius<0
    row['results'][method+':'+side]={'value':decimal(actual),'absolute_error':decimal(error),'radius':decimal(radius),'lower':decimal(actual-radius),'upper':decimal(actual+radius)}
   new,old=errors['structural_anchor_compensated'],errors['direct_plain'];pairs[group][('improved'if new<old else'worse'if new>old else'equal')+':'+side]+=1
   if old and not new:pairs[group]['became_exact:'+side]+=1
  detailed.append(row)
 report={'scope':data['scope'],'abi':abi,'case_count':len(detailed),'groups':{g:dict(v)for g,v in summary.items()},'structural_anchor_compared_with_direct':dict(pairs),'negative_true_inventory_case_count':sum(F(r['reference_q'])<0 for r in detailed),'negative_true_headroom_case_count':sum(F(r['reference_h'])<0 for r in detailed),'given_progress_outside_capacity_case_count':sum(r['reference_outside_capacity']for r in detailed),'bounds':'gamma8 times exact input term L1 for plain expressions, normal round-to-nearest arithmetic and certified exact constants; compensated bound transferred by triangle inequality from plain, not a sharper compensated theorem','cases':detailed}
 (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 if(HERE/'identity.json').exists():raise RuntimeError('refuse overwrite completed checkpoint')
 data=prepare();exe=HERE/'evaluator';command=['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-I'+str(ROOT/'astra-sim-wscllm/extern/helper'),str(SOURCE),'-o',str(exe)]
 meta={'scope':data['scope'],'sources_sha256':data['sources_sha256'],'evaluator_sha256':sha(SOURCE),'runner_sha256':sha(Path(__file__)),'input_sha256':sha(HERE/'input.json'),'commands':[],'host':platform.platform()}
 for label,cmd in [('compiler_version',['c++','--version']),('compile',command),('run',[str(exe),str(HERE/'input.json'),str(HERE/'output.jsonl')])]:
  before=time.time();r=subprocess.run(cmd,capture_output=True,text=True,timeout=90,cwd=ROOT);meta['commands'].append({'label':label,'command':cmd,'returncode':r.returncode,'elapsed_s':time.time()-before});(HERE/(label+'.stdout.log')).write_text(r.stdout);(HERE/(label+'.stderr.log')).write_text(r.stderr)
  if r.returncode:raise RuntimeError((label,r.returncode))
 report=analyze(data,HERE/'output.jsonl');meta['executable_sha256']=sha(exe);meta['outputs_sha256']={p.name:sha(p)for p in HERE.iterdir()if p.is_file()and p.suffix in ['.json','.jsonl','.log']and p.name!='identity.json'};(HERE/'identity.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items()if k!='cases'},indent=2))
if __name__=='__main__':main()
