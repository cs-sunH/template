#!/usr/bin/env python3
"""Small offline full/tail cases; no production simulation replay."""
import argparse, json, pathlib, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('--driver',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
rows=[]
for elements in [321,511,512]:
 for profile in ['balanced','memory','compute']:
  name=f'residual-e{elements}-i1-{profile}-bf16-fp32'
  cmd=[str(pathlib.Path(a.driver).resolve()),'--op','residual','--elements',str(elements),'--executions','1','--profile',profile,'--input-dtype','bf16','--output-dtype','fp32']
  with (out/(name+'.raw.json')).open('w') as output,(out/(name+'.stderr.txt')).open('w') as error:r=subprocess.run(cmd,stdout=output,stderr=error)
  row={'case':name,'driver_command':cmd,'driver_return_code':r.returncode}
  if r.returncode==0:
   cmd=[sys.executable,str(root/'die/validation/elementwise_service_oracle.py'),'--raw',str(out/(name+'.raw.json')),'--output',str(out/(name+'.report.json'))]
   with (out/(name+'.report.log')).open('w') as log:c=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
   row.update(report_command=cmd,report_return_code=c.returncode)
  rows.append(row)
(out/'suite.json').write_text(json.dumps({'scope':'9 additional full/tail single-Core cases, not hardware calibration','cases':rows},indent=2)+'\n')
