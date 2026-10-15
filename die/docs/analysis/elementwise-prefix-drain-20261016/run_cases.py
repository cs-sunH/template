#!/usr/bin/env python3
"""Offline longer elementwise cases; production remains the compiled Runtime."""
import argparse,json,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--driver',required=True);p.add_argument('--output',required=True);p.add_argument('--elements',type=int,nargs='+',default=[513,768,1025,4097]);p.add_argument('--profiles',nargs='+',default=['balanced','memory','compute','local']);p.add_argument('--ops',nargs='+',default=['residual']);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);rows=[]
for op in a.ops:
 for n in a.elements:
  for profile in a.profiles:
   name=f'{op}-e{n}-i1-{profile}-bf16-fp32';cmd=[str(pathlib.Path(a.driver).resolve()),'--op',op,'--elements',str(n),'--executions','1','--profile',profile,'--input-dtype','bf16','--output-dtype','fp32']
   with (out/(name+'.raw.json')).open('w') as output,(out/(name+'.stderr.txt')).open('w') as error:r=subprocess.run(cmd,stdout=output,stderr=error)
   row={'case':name,'driver_command':cmd,'driver_return_code':r.returncode}
   if r.returncode==0:
    cmd=[sys.executable,str(root/'die/validation/elementwise_service_oracle.py'),'--raw',str(out/(name+'.raw.json')),'--output',str(out/(name+'.report.json'))]
    with (out/(name+'.report.log')).open('w') as log:c=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    row.update(report_command=cmd,report_return_code=c.returncode)
   rows.append(row)
(out/'suite.json').write_text(json.dumps({'scope':'single-Core R1 longer cohor ts; synthetic independent reference, not hardware calibration','cases':rows},indent=2)+'\n')
print(json.dumps({'cases':len(rows),'driver_and_report_exited_zero':all(r['driver_return_code']==0 and r.get('report_return_code')==0 for r in rows),'accuracy':'see individual metrics; successful process is not acceptance'}))
