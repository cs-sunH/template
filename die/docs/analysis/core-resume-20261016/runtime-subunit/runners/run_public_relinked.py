#!/usr/bin/env python3
"""Bind scratch Runtime into fresh-session objects, archive, C ABI and CTest.

This is explicitly a relink regression, not a final clean-source CMake build.
"""
from pathlib import Path
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time

root=Path.cwd()
lab=root/'work/runtime-lab'
baseline=root/'work/build-core-release'
build=lab/'public-build'
build.mkdir(exist_ok=True)
(build/'lib').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
public_identity=root/'die/docs/evidence/core-resume-20261016/public-51-identity.json'
public=json.loads(public_identity.read_text())
manifest=json.loads((lab/'manifest.json').read_text())
assert sha(root/'die/src/Execution.cc')==manifest['baseline_sha256']
assert sha(lab/'candidate/Execution.cc')==manifest['candidate_sha256']
assert sha(baseline/'libcoarse_die.a')==public['binary_after_sha256'][str(baseline/'libcoarse_die.a')]

sources=[p for folder in ['die/src','die/include','die/configs','die/tests','die/validation','die/tools']
         for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
report={'scope':'scratch candidate Runtime from current-source Execution, unchanged current-session fresh objects; independently assembled static/CABI libraries and relinked public tests; not a final clean CMake source build or adoption',
        'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'started_unix':time.time(),'public51_identity_sha256':sha(public_identity),
        'candidate_source_sha256':sha(lab/'candidate/Execution.cc'),
        'source_before':{str(p.relative_to(root)):sha(p) for p in sources},
        'baseline_library_sha256':sha(baseline/'libcoarse_die.a'),'steps':[]}

def save():
    (lab/'public-relinked-identity.json').write_text(json.dumps(report,indent=2)+'\n')

def run(name,command,inputs=(),timeout=None):
    entry={'name':name,'command':list(map(str,command)),
           'inputs_sha256':{str(p.relative_to(root)):sha(p) for p in inputs},
           'started_unix':time.time()}
    report['steps'].append(entry)
    save()
    with (lab/(name+'.stdout.log')).open('w') as out,(lab/(name+'.stderr.log')).open('w') as err:
        try:
            result=subprocess.run(command,cwd=root,stdout=out,stderr=err,timeout=timeout)
            entry['return_code']=result.returncode
        except subprocess.TimeoutExpired:
            entry['timeout_seconds']=timeout
        except OSError as error:
            entry['return_code']=127
            entry['launch_error']=str(error)
            err.write(str(error)+'\n')
    entry['finished_unix']=time.time()
    entry['stdout_sha256']=sha(lab/(name+'.stdout.log'))
    entry['stderr_sha256']=sha(lab/(name+'.stderr.log'))
    save()
    if entry.get('return_code')!=0:
        raise RuntimeError('actual failure retained: '+name)

obj=build/'Execution.cc.o'
command=['g++','-std=gnu++17','-O3','-DNDEBUG','-fPIC','-Wall','-Wextra','-Wpedantic',
         '-I',str(root/'die/src'),'-I',str(root/'die/include'),
         '-I',str(root/'astra-sim-wscllm/extern/helper'),
         '-c',str(lab/'candidate/Execution.cc'),'-o',str(obj)]
run('public-compile-Execution-PIC',command,[lab/'candidate/Execution.cc'])
report['candidate_object_sha256']=sha(obj)
candidate_library=build/'libcoarse_die.a'
candidate_library.write_bytes((baseline/'libcoarse_die.a').read_bytes())
original_members=subprocess.check_output(['ar','t',str(candidate_library)],text=True).splitlines()
assert original_members.count('Execution.cc.o')==1
run('public-archive-remove-Execution',['ar','d',str(candidate_library),'Execution.cc.o'],[candidate_library])
assert 'Execution.cc.o' not in subprocess.check_output(['ar','t',str(candidate_library)],text=True).splitlines()
run('public-archive-add-Execution',['ar','r',str(candidate_library),str(obj)],[obj,candidate_library])
run('public-archive-index',['ar','s',str(candidate_library)],[candidate_library])
members=subprocess.check_output(['ar','t',str(candidate_library)],text=True).splitlines()
assert members.count('Execution.cc.o')==1 and sorted(members)==sorted(original_members)
member_hash=lambda archive,name:hashlib.sha256(subprocess.check_output(['ar','p',str(archive),name])).hexdigest()
report['archive_members']={name:member_hash(candidate_library,name) for name in members}
assert report['archive_members']['Execution.cc.o']==sha(obj)
assert all(report['archive_members'][name]==member_hash(baseline/'libcoarse_die.a',name)
           for name in members if name!='Execution.cc.o')
report['candidate_library_sha256']=sha(candidate_library)

testfile=(baseline/'CTestTestfile.cmake').read_text()
targets=sorted(set(re.findall(re.escape(str(baseline))+r'/(coarse_[^"/ ]+)',testfile)))
targets=['coarse_die_cabi']+targets
report['targets']=targets
report['linked_artifacts']={}
for target in targets:
    linkfile=baseline/f'CMakeFiles/{target}.dir/link.txt'
    command=shlex.split(linkfile.read_text())
    rewritten=[]
    inputs=[linkfile,candidate_library]
    output=None
    output_next=False
    for token in command:
        if token.startswith('-Wl,--dependency-file='):
            continue
        if output_next:
            output=build/token
            rewritten.append(str(output))
            output_next=False
            continue
        if token=='-o':
            rewritten.append(token)
            output_next=True
        elif token=='libcoarse_die.a':
            rewritten.append(str(candidate_library))
        elif token=='lib/libcoarse_die_cabi.so':
            rewritten.append(str(build/token))
            inputs.append(build/token)
        elif token.startswith('CMakeFiles/') and token.endswith('.o'):
            rewritten.append(str(baseline/token))
            inputs.append(baseline/token)
        else:
            rewritten.append(token.replace(str(baseline/'lib'),str(build/'lib')))
    assert output is not None
    run('public-link-'+target,rewritten,inputs)
    report['linked_artifacts'][target]={'path':str(output.relative_to(root)),'sha256':sha(output)}
    if target!='coarse_die_cabi':
        run('public-ldd-'+target,['ldd',str(output)],[output])
        loaded=(lab/('public-ldd-'+target+'.stdout.log')).read_text()
        assert str(baseline/'lib/libcoarse_die_cabi.so') not in loaded
        if 'libcoarse_die_cabi.so' in loaded:
            assert str(build/'lib/libcoarse_die_cabi.so') in loaded
    save()

testfile=testfile.replace(str(baseline),str(build))
(build/'CTestTestfile.cmake').write_text(testfile)
report['ctest_file_sha256']=sha(build/'CTestTestfile.cmake')
report['registered_tests']=len(re.findall(r'^add_test\(',testfile,re.M))
assert report['registered_tests']==51
save()
ctest=shutil.which('ctest')
if not ctest:
    cache=(baseline/'CMakeCache.txt').read_text()
    configured=re.search(r'^CMAKE_COMMAND:INTERNAL=(.+)$',cache,re.M)
    if configured:
        possible=Path(configured.group(1)).with_name('ctest')
        if possible.is_file():
            ctest=str(possible)
if not ctest:
    raise RuntimeError('ctest unavailable; no tests executed')
run('public-ctest51',[ctest,'--test-dir',str(build),'-j','1','--output-on-failure'],
    [build/'CTestTestfile.cmake',candidate_library,build/'lib/libcoarse_die_cabi.so'],300)
report['source_after']={str(p.relative_to(root)):sha(p) for p in sources}
report['production_unchanged']=report['source_before']==report['source_after']
report['baseline_library_unchanged']=report['baseline_library_sha256']==sha(baseline/'libcoarse_die.a')
report['finished_unix']=time.time()
save()
print(json.dumps({'registered_tests':report['registered_tests'],
                  'candidate_library_sha256':report['candidate_library_sha256'],
                  'candidate_cabi_sha256':sha(build/'lib/libcoarse_die_cabi.so'),
                  'production_unchanged':report['production_unchanged'],
                  'baseline_library_unchanged':report['baseline_library_unchanged']}))
