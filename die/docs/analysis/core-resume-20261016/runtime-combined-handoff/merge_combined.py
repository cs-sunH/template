#!/usr/bin/env python3
"""Source-only three-way combination of independently archived A and B."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess

root=Path.cwd();lab=root/'work/runtime-combined-tests';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=root/'die/src/Execution.cc';a=lab/'source-AB/die/src/Execution.cc';b=lab/'source-BA/die/src/Execution.cc'
assert sha(base)=='3c9d58a0ed8c6b64450b86cac1059b3a1dba7b604caf67b31f00034c673e3043'
assert sha(a)=='66f34d699d79b3c6dcffc5af04002799f3b1bfa4d4dcb569165825fc6a3f655a'
assert sha(b)=='81c1152784d2f7a63d6dd6e41c48b72b419c578e4564aa8004ffa10549af8bb4'
report={'scope':'source-only combination, no compile or production edits',
        'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runner_sha256':sha(Path(__file__)),'input_sha256':{str(p.relative_to(root)):sha(p) for p in [base,a,b]},'steps':[]}
results=[]
for left,right in [(a,b),(b,a)]:
    command=['git','merge-file','--diff3','--stdout',str(left),str(base),str(right)]
    run=subprocess.run(command,cwd=root,capture_output=True,text=True)
    report['steps'].append({'command':command,'return_code':run.returncode,'stderr':run.stderr})
    assert run.returncode==0 and '<<<<<<<' not in run.stdout
    results.append(run.stdout)
assert results[0]==results[1],'different merge order produced different source'
target=lab/'combined'
for relative in ['die/src/Execution.cc','die/include/coarse_die/Execution.hh']:
    dst=target/relative;dst.parent.mkdir(parents=True,exist_ok=True)
    if relative.endswith('.cc'):dst.write_text(results[0])
    else:shutil.copy2(lab/'source-BA'/relative,dst)
text=''
for relative in ['die/src/Execution.cc','die/include/coarse_die/Execution.hh']:
    text+=''.join(difflib.unified_diff((root/relative).read_text().splitlines(True),
               (target/relative).read_text().splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
(lab/'combined-A-B.patch').write_text(text)
report['output_sha256']={str(p.relative_to(root)):sha(p) for p in
    [target/'die/src/Execution.cc',target/'die/include/coarse_die/Execution.hh',lab/'combined-A-B.patch']}
source=results[0]
checks={'A_all_14_tolerance_calls_preserved':source.count('credit_boundary_tolerance(')==a.read_text().count('credit_boundary_tolerance('),
        'A_helper_both_templates_preserved':source[source.index('// Curved credit coordinates'):source.index('// Curved Plans retain')]==a.read_text()[a.read_text().index('// Curved credit coordinates'):a.read_text().index('// Curved Plans retain')],
        'B_private_overload_preserved':'void Runtime::allocate(bool redundant_initial_boundaries,' in source,
        'B_initial_witness_rules_preserved':'epoch.credit_initial[buffer] == 0.0' in source and 'epoch.credit_initial[buffer] == epoch.credit_capacity[buffer]' in source,
        'B_shared_pivot_budget_preserved':'runtime->allocate(false, &import_pivots);' in source and 'runtime->allocate(true, &import_pivots);' in source and 'std::size_t pivot_count = prior_import_pivots;' in source,
        'original_same_or_one_ulp_retained':'same_or_one_ulp(saved.rate, saved.activity->rate)' in source,
        'no_merge_markers':'<<<<<<<' not in source}
assert all(checks.values());report['structural_checks']=checks
report['merge_orders_byte_equal']=True
report['production_unchanged']=sha(base)==report['input_sha256'][str(base.relative_to(root))]
(lab/'merge-identity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'merge_orders_equal':True,'checks':checks,'output_sha256':report['output_sha256'],'production_unchanged':report['production_unchanged']}))
