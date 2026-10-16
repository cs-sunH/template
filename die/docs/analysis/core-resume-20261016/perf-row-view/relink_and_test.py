from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import time

root=Path.cwd();base=root/'work/core-perf-lab/build';source=root/'work/core-perf-row-lab';build=source/'build';out=root/'work/perf-row-view'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'scope':'seven affected current-session relink regressions, not a fresh CMake build or full public51','steps':[],'source_sha256':{str(p.relative_to(source)):sha(p) for p in (source/'die').rglob('*') if p.is_file()}}
def run(label,command,inputs):
    row={'label':label,'command':command,'input_sha256':{str(p):sha(p) for p in inputs},'started_unix':time.time()}
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=root)
    row.update(return_code=r.returncode,finished_unix=time.time(),log_sha256=sha(out/(label+'.log')))
    report['steps'].append(row);(out/'affected-identity.json').write_text(json.dumps(report,indent=2)+'\n')
    if r.returncode:raise SystemExit(r.returncode)
targets=['coarse_owner_lp_oracle_driver','coarse_runtime_owner_symmetry_test','coarse_compiler_summary_test','coarse_runtime_semantics_test','coarse_service_curve_runtime_test','coarse_ratio_allocator_test','coarse_service_curve_benchmark']
for target in targets:
    link=base/f'CMakeFiles/{target}.dir/link.txt';tokens=shlex.split(link.read_text());argv=[];inputs=[link,build/'libcoarse_die.a',build/'lib/libcoarse_die_cabi.so'];next_output=False
    for token in tokens:
        if token.startswith('-Wl,--dependency-file='):continue
        if next_output:argv.append(str(build/token));next_output=False
        elif token=='-o':argv.append(token);next_output=True
        elif token=='libcoarse_die.a':argv.append(str(build/token))
        elif token=='lib/libcoarse_die_cabi.so':argv.append(str(build/token))
        elif token.startswith('CMakeFiles/') and token.endswith('.o'):argv.append(str(base/token));inputs.append(base/token)
        else:argv.append(token.replace(str(base/'lib'),str(build/'lib')))
    run('link-'+target,argv,inputs)
    run('ldd-'+target,['ldd',str(build/target)],[build/target])
    loaded=(out/('ldd-'+target+'.log')).read_text();assert str(base/'lib/libcoarse_die_cabi.so') not in loaded
    if 'libcoarse_die_cabi.so' in loaded:assert str(build/'lib/libcoarse_die_cabi.so') in loaded
text=(base/'CTestTestfile.cmake').read_text().replace(str(base),str(build)).replace(str(root/'work/core-perf-lab/die'),str(source/'die'))
(build/'CTestTestfile.cmake').write_text(text)
pattern='^(coarse_owner_lp_oracle_test|coarse_runtime_owner_symmetry_test|coarse_compiler_summary_test|coarse_runtime_semantics_test|coarse_service_curve_runtime_test|coarse_ratio_allocator_test|coarse_packing_allocator_test)$'
run('affected-ctest',['ctest','--test-dir',str(build),'-R',pattern,'--output-on-failure'],[build/'CTestTestfile.cmake',build/'libcoarse_die.a',build/'lib/libcoarse_die_cabi.so'])
report['source_unchanged']=all(sha(source/rel)==value for rel,value in report['source_sha256'].items());assert report['source_unchanged']
report['binary_sha256']={str(p.relative_to(build)):sha(p) for p in list(build.glob('coarse_*'))+[build/'libcoarse_die.a',build/'lib/libcoarse_die_cabi.so'] if p.is_file()}
(out/'affected-identity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'scope':report['scope'],'source_unchanged':True,'step_return_codes':[r['return_code'] for r in report['steps']]}))
