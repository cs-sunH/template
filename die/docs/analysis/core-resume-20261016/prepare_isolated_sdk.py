#!/usr/bin/env python3
"""Copy tracked five-repository consumers and freeze current SDK under work/."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPOS=('astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint')
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,help='default: work/core-resume-sdk')
 p.add_argument('--expected-digest')
 p.add_argument('--protoc',default=shutil.which('protoc') or 'protoc',help='generate real Python bindings inside isolated copies before freezing')
 a=p.parse_args()
 root=Path(__file__).resolve().parents[4]
 out=(a.output or root/'work/core-resume-sdk').resolve()
 if not out.is_relative_to((root/'work').resolve()) or out==(root/'work').resolve():p.error('--output must be a dedicated child of repository work/')
 source=root/'die'
 files=[source/'CMakeLists.txt']
 for name in ('include','src','configs'):files+=sorted(x for x in (source/name).rglob('*') if x.is_file())
 entries={str(x.relative_to(source)):SHA(x) for x in files}
 digest=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if a.expected_digest and a.expected_digest!=digest:p.error('current public source digest differs from expected identity')
 release={'schema':'coarse_die_release_v2','source_digest':digest,'files':entries}
 out.mkdir(parents=True,exist_ok=False)
 stage=out/'stage-root';stage.mkdir()
 variants={}
 tracked=subprocess.check_output(['git','ls-files','-z','--',*REPOS],cwd=root).decode().split('\0')
 tracked_set=set(filter(None,tracked))
 for rel in filter(None,tracked):
  original=root/rel
  if not original.is_file():continue
  destination=stage/rel;destination.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(original,destination)
 for name in REPOS:
  old=stage/name/'extern/die_model/coarse'
  old_release=json.loads((old/'RELEASE.json').read_text())
  # This deletion is only in the newly-created isolated copy, never a vendor tree.
  shutil.rmtree(old);old.mkdir()
  for rel in entries:
   dest=old/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,dest)
  (old/'RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
  variants[name]={'installed_original_sdk_digest':old_release['source_digest'],
                  'staged_source_root':str(stage/name),'build':str(out/'build-abi'/name)}
 # Real protoc generation occurs before the manifest is frozen. Some variants
 # track et_def_pb2.py, so a later overwrite would invalidate source identity.
 protoc_version=subprocess.check_output([a.protoc,'--version'],text=True).strip()
 protobuf=[]
 proto_rel=Path('extern/graph_frontend/chakra/schema/protobuf/et_def.proto')
 generated_rel=proto_rel.with_name('et_def_pb2.py')
 log_dir=out/'protobuf-generation';log_dir.mkdir()
 for name in REPOS:
  repo=stage/name;generated=repo/generated_rel;original=root/name/generated_rel
  row={'repo':name,'source_proto_sha256':SHA(repo/proto_rel),
    'tracked_original':str(Path(name)/generated_rel) in tracked_set,
    'tracked_original_sha256':SHA(original) if original.is_file() else None,
    'copied_before_generation_sha256':SHA(generated) if generated.is_file() else None,
    'protoc_version':protoc_version,
    'command':[a.protoc,'--proto_path='+str(repo),'--python_out='+str(repo),str(proto_rel)],'cwd':str(repo)}
  proc=subprocess.run(row['command'],cwd=repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  log=log_dir/(name+'.log');log.write_text(proc.stdout)
  row.update(return_code=proc.returncode,log_sha256=SHA(log),
             generated_sha256=SHA(generated) if generated.is_file() else None,
             generated_path=str(generated.relative_to(stage)))
  protobuf.append(row)
  (out/'protobuf-generation.json').write_text(json.dumps(protobuf,indent=2)+'\n')
  if proc.returncode or not generated.is_file():
   raise RuntimeError('isolated protoc generation failed; partial preparation/logs preserved for '+name)
 # Frozen public input/source for the portable actual-response availability probe.
 for rel in entries:
  dest=stage/'die'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,dest)
 probe=Path('die/docs/analysis/consumer-availability-20261016/probe_python_native.py')
 dest=stage/probe;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/probe,dest)
 cost_probe=Path('die/docs/analysis/analytical-curves-partial-20261015/integration/face_cost_abi_probe.py')
 dest=stage/cost_probe;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/cost_probe,dest)
 typed=stage/'work/integration-audit/probe_typed_resolver.py';typed.parent.mkdir(parents=True)
 shutil.copy2(Path(__file__).with_name('typed_resolver_probe_historical.py'),typed)
 aggregate=out/'aggregate';aggregate.mkdir()
 shutil.copy2(Path(__file__).with_name('aggregate')/'CMakeLists.txt',aggregate/'CMakeLists.txt')
 frozen={str(x.relative_to(stage)):SHA(x) for x in sorted(stage.rglob('*')) if x.is_file()}
 manifest={'scope':'isolated candidate SDK/consumer source only; not published, not engineering accepted',
  'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
  'public_release':release,'stage_root':str(stage),'variants':variants,'frozen_files_sha256':frozen,
  'protobuf_generation':protobuf,'prepare_sha256':SHA(Path(__file__)),
  'helper_source_sha256':{'typed_resolver_probe_historical.py':SHA(Path(__file__).with_name('typed_resolver_probe_historical.py')),
                         'aggregate/CMakeLists.txt':SHA(Path(__file__).with_name('aggregate')/'CMakeLists.txt'),
                         str(probe):SHA(root/probe),str(cost_probe):SHA(root/cost_probe)}}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (out/'build-map.json').write_text(json.dumps({name:info['build'] for name,info in variants.items()},indent=2)+'\n')
 print(json.dumps({'status':'frozen_not_built','source_digest':digest,'files':len(entries),
                   'frozen_files':len(frozen),'manifest':str(out/'manifest.json')}))
if __name__=='__main__':main()
