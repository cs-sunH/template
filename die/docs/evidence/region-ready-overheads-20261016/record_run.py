#!/usr/bin/env python3
"""Bind a real command and complete raw output to current source/binary SHA256s."""
import argparse, hashlib, json, pathlib, subprocess, time
p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--prefix',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
root=pathlib.Path(a.root).resolve();prefix=pathlib.Path(a.prefix).resolve();prefix.parent.mkdir(parents=True,exist_ok=True)
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd:p.error('command required')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
files=[]
for folder in ['die/src','die/include','die/configs','die/tests']:
 files.extend(f for f in (root/folder).rglob('*') if f.is_file() and f.suffix in {'.hh','.h','.cc','.json','.py'})
files.append(root/'die/CMakeLists.txt')
binaries=[root/'work/build-die-continue'/n for n in ['libcoarse_die.a','lib/libcoarse_die_cabi.so','coarse_service_curve_benchmark','coarse_region_overhead_test']]
binaries.extend(pathlib.Path(c) for c in cmd if pathlib.Path(c).is_absolute() and pathlib.Path(c).is_file() and pathlib.Path(c).suffix not in {'.json','.log'})
meta={'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_files':{str(f.relative_to(root)):sha(f) for f in sorted(set(files))},'binaries':{str(f.relative_to(root)):sha(f) for f in sorted(set(binaries)) if f.is_file()},'command':cmd,'started_unix':time.time()}
cpu=pathlib.Path('/sys/fs/cgroup/cpu.stat');meta['cpu_before']=cpu.read_text() if cpu.exists() else None
with pathlib.Path(str(prefix)+'.log').open('w') as log:
 try:
  r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 except OSError as error:
  log.write(str(error)+"\n");r=subprocess.CompletedProcess(cmd,127)
meta.update(return_code=r.returncode,finished_unix=time.time(),cpu_after=cpu.read_text() if cpu.exists() else None)
pathlib.Path(str(prefix)+'-identity.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k not in {'source_files','binaries'}}));raise SystemExit(r.returncode)
