#!/usr/bin/env python3
"""Run actual C++ Norm curves and the offline row oracle; keep every case."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--driver',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];out=pathlib.Path(a.output).resolve()
out.mkdir(parents=True,exist_ok=False);driver=pathlib.Path(a.driver).resolve()
cases=[(rows,'rmsnorm',profile) for rows in [1,2,3,5,17,65,257]
       for profile in ['balanced','memory','compute','local']]
cases += [(rows,'layernorm',profile) for rows in [2,17] for profile in ['balanced','memory']]
results=[]
for rows,norm,profile in cases:
    label=f'{norm}-rows{rows}-{profile}'
    cmd=[str(driver),'--rows',str(rows),'--hidden','257','--norm',norm,'--profile',profile]
    with (out/(label+'.raw.json')).open('w') as output,(out/(label+'.stderr.txt')).open('w') as error:
        r=subprocess.run(cmd,stdout=output,stderr=error)
    row={'case':label,'driver_command':cmd,'driver_return_code':r.returncode}
    if r.returncode==0:
        cmd=[sys.executable,str(root/'die/validation/norm_row_oracle.py'),'--raw',str(out/(label+'.raw.json')),
             '--output',str(out/(label+'.report.json'))]
        with (out/(label+'.report.log')).open('w') as log:c=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        row.update(report_command=cmd,report_return_code=c.returncode)
        if c.returncode==0:
            report=json.loads((out/(label+'.report.json')).read_text())
            row['scenarios']={name:item['comparison']['status'] for name,item in report['scenarios'].items()}
    results.append(row)
meta={'scope':'32 small offline cases/96 scenarios, cached BF16 row model, no calibration',
      'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),
      'runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'cases':results}
(out/'suite.json').write_text(json.dumps(meta,indent=2)+'\n')
totals={}
for row in results:
    for status in row.get('scenarios',{}).values():totals[status]=totals.get(status,0)+1
print(json.dumps({'process_failures':sum(row['driver_return_code']!=0 or row.get('report_return_code')!=0 for row in results),
                  'completion_and_busy':totals}))
