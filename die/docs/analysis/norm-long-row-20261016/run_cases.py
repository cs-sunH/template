#!/usr/bin/env python3
"""Fresh actual two-pass driver runs plus independent discrete tile reference."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--driver',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
driver=pathlib.Path(a.driver).resolve();cases=[]
fixtures=[(65537,norm,profile)for norm in ['rmsnorm','layernorm']for profile in ['balanced','memory','compute','local']]
fixtures += [(131073,norm,profile)for norm in ['rmsnorm','layernorm']for profile in ['balanced','memory']]
for h,norm,profile in fixtures:
    label=f'{norm}-h{h}-{profile}';cmd=[str(driver),'--hidden',str(h),'--norm',norm,'--profile',profile]
    with (out/(label+'.raw.json')).open('w')as log,(out/(label+'.stderr.txt')).open('w')as err:
        r=subprocess.run(cmd,stdout=log,stderr=err)
    row={'case':label,'driver_command':cmd,'driver_return_code':r.returncode}
    if r.returncode==0:
        cmd=[sys.executable,str(root/'die/validation/norm_long_oracle.py'),'--raw',str(out/(label+'.raw.json')),
             '--output',str(out/(label+'.report.json'))]
        with (out/(label+'.report.log')).open('w')as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        row.update(report_command=cmd,report_return_code=r.returncode)
        if r.returncode==0:
            report=json.loads((out/(label+'.report.json')).read_text())
            row['scenarios']={name:item['comparison']['status']for name,item in report['scenarios'].items()}
    cases.append(row)
summary={'scope':'12 actual BF16 one Core/row R1 two-pass tasks / 36 synthetic scenes; no fitting/hardware calibration',
         'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),
         'runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'cases':cases}
(out/'suite.json').write_text(json.dumps(summary,indent=2)+'\n')
totals={}
for row in cases:
    for status in row.get('scenarios',{}).values():totals[status]=totals.get(status,0)+1
failed=sum(x['driver_return_code']!=0 or x.get('report_return_code')!=0 for x in cases)
print(json.dumps({'process_failures':failed,'completion_and_original_busy':totals}))
raise SystemExit(bool(failed))
