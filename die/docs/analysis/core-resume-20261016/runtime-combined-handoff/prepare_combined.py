#!/usr/bin/env python3
"""Apply only saved source patches under this scratch directory, in both orders."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess

root=Path.cwd();lab=root/'work/runtime-combined-tests'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
patches={'A':root/'die/docs/analysis/core-resume-20261016/runtime-subunit/subunit-curved-credit.patch',
         'B':root/'die/docs/analysis/core-resume-20261016/snapshot-import/import-redundant-row-witness.patch'}
inputs=[root/'die/src/Execution.cc',root/'die/include/coarse_die/Execution.hh',*patches.values()]
report={'scope':'scratch patch-order/conflict inspection only; no production edits',
        'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runner_sha256':sha(Path(__file__)),'inputs_sha256':{str(p.relative_to(root)):sha(p) for p in inputs},'steps':[]}
for order in ['AB','BA']:
    target=lab/('source-'+order)
    for relative in ['die/src/Execution.cc','die/include/coarse_die/Execution.hh']:
        dst=target/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/relative,dst)
    for name in order:
        for check in [True,False]:
            command=['git','apply']+(['--check'] if check else [])+['--directory='+str(target.relative_to(root)),str(patches[name])]
            run=subprocess.run(command,cwd=root,capture_output=True,text=True)
            report['steps'].append({'order':order,'patch':name,'check':check,'command':command,'return_code':run.returncode,
                                    'stdout':run.stdout,'stderr':run.stderr})
            if run.returncode:break
        if run.returncode:break
report['AB_BA_source_equal']=(lab/'source-AB/die/src/Execution.cc').read_bytes()==(lab/'source-BA/die/src/Execution.cc').read_bytes()
report['AB_BA_header_equal']=(lab/'source-AB/die/include/coarse_die/Execution.hh').read_bytes()==(lab/'source-BA/die/include/coarse_die/Execution.hh').read_bytes()
report['production_unchanged']=all(sha(p)==report['inputs_sha256'][str(p.relative_to(root))] for p in inputs[:2])
if all(s['return_code']==0 for s in report['steps']):
    text=''
    for relative in ['die/src/Execution.cc','die/include/coarse_die/Execution.hh']:
        a=(root/relative).read_text().splitlines(True);b=(lab/'source-AB'/relative).read_text().splitlines(True)
        text+=''.join(difflib.unified_diff(a,b,fromfile='a/'+relative,tofile='b/'+relative))
    (lab/'combined-A-B.patch').write_text(text)
(lab/'prepare-identity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'patch_steps':len(report['steps']),'all_apply_steps_passed':all(s['return_code']==0 for s in report['steps']),
                  'both_orders_source_equal':report['AB_BA_source_equal'],'both_orders_header_equal':report['AB_BA_header_equal'],
                  'production_unchanged':report['production_unchanged']}))
