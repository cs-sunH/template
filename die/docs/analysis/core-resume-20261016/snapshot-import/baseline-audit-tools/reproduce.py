#!/usr/bin/env python3
import ctypes,gzip,hashlib,json,pathlib,sys,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary,OwnerV2
out=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
libpath=root/'work/build-core-release/lib/libcoarse_die_cabi.so'
tracked=[root/'die/src/Execution.cc',root/'die/src/PackingAllocation.hh',root/'die/src/Plan.cc',root/'die/src/Capi.cc',root/'die/include/coarse_die/Execution.hh',libpath]
before={str(p.relative_to(root)):sha(p) for p in tracked}
input=json.loads((out/'inputs.json').read_text()); snapshot=json.loads((out/'failed-snapshot.json').read_text())
lib=CoarseDieLibrary(libpath);model=lib.create_model(input['config'])
def import_snapshot(value):
 h=ctypes.c_uint64(0)
 code=lib.lib.die_coarse_import_snapshot_v2(model,json.dumps(value,separators=(',',':')).encode(),ctypes.byref(h))
 row={'code':code,'returned_handle':h.value,'error':lib._last_error() if code else None}
 if not code:
  row['rebuilt_snapshot']=lib.export_snapshot(h.value);lib.advance_to(h.value,10**18);lib.destroy_runtime(h.value)
 return row
result={'direct_original_import':import_snapshot(snapshot)}
# Diagnostic only: omitted marker means discard saved allocation per legacy path.
# This intentionally edited copy is not a successful original-snapshot restore.
legacy=json.loads(json.dumps(snapshot));legacy.pop('allocation_state')
result['diagnostic_rebuilt_without_allocation_marker']=import_snapshot(legacy)
compiled=lib.compile_batch(model,input['tasks'])
runtime=lib.create_runtime(model);handles=[]
for i in range(8):
 owner=i//2+1
 handles.append(lib.submit_plan(runtime,compiled['results'][i%2]['plan_handle'],OwnerV2.make(i+1,owner,owner),0))
result['fresh_replay_rows']=[]
for _ in range(2):
 tick=lib.next_event(runtime);lib.advance_to(runtime,tick)
 result['fresh_replay_rows'].append({'tick':tick,'snapshot':lib.export_snapshot(runtime),'progress':[lib.query_progress(h) for h in handles]})
actual=lib.export_snapshot(runtime)
result['fresh_replay_matches_captured_snapshot']=actual==snapshot
result['fresh_replay_original_import']=import_snapshot(actual)
fork=ctypes.c_uint64(0);code=lib.lib.die_coarse_fork_runtime_v2(runtime,ctypes.byref(fork))
result['fresh_replay_fork']={'code':code,'returned_handle':fork.value,'error':lib._last_error() if code else None}
result['parent_snapshot_unchanged']=lib.export_snapshot(runtime)==actual
if not code:lib.destroy_runtime(fork.value)
lib.advance_to(runtime,10**18);lib.destroy_runtime(runtime)
for p in compiled['results']:lib.release_plan(p['plan_handle'])
lib.destroy_model(model)
result['identity']={'command':[sys.executable,*sys.argv],'source_root':str(root),'tool_sha256':sha(pathlib.Path(__file__)),'inputs_sha256':sha(out/'inputs.json'),'original_snapshot_sha256':sha(out/'failed-snapshot.json'),'original_api_gzip_sha256':sha(root/'die/docs/evidence/core-resume-20261016/perf-candidate/api-baseline.json.gz'),'before':before,'after':{str(p.relative_to(root)):sha(p) for p in tracked}}
result['identity']['source_and_library_unchanged']=result['identity']['before']==result['identity']['after']
(out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['fresh_replay_rows','identity','diagnostic_rebuilt_without_allocation_marker']},indent=2))
assert result['fresh_replay_matches_captured_snapshot']
assert result['direct_original_import']['code']==1 and result['fresh_replay_original_import']['code']==1 and result['fresh_replay_fork']['code']==1
assert result['parent_snapshot_unchanged'] and result['identity']['source_and_library_unchanged']
