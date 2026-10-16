#!/usr/bin/env python3
import ctypes,hashlib,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary,OwnerV2
out=pathlib.Path(__file__).resolve().parent
inputs=json.loads((out/'inputs.json').read_text());library=root/'work/build-core-release/lib/libcoarse_die_cabi.so'
lib=CoarseDieLibrary(library);model=lib.create_model(inputs['config']);plans=lib.compile_batch(model,inputs['tasks'])
result=[]
for count in [2,4,6,8]:
 runtime=lib.create_runtime(model)
 for i in range(count):
  owner=i//2+1;lib.submit_plan(runtime,plans['results'][i%2]['plan_handle'],OwnerV2.make(i+1,owner,owner),0)
 ticks=[]
 for _ in range(2):
  tick=lib.next_event(runtime);ticks.append(tick);lib.advance_to(runtime,tick)
 s=lib.export_snapshot(runtime);f=ctypes.c_uint64();code=lib.lib.die_coarse_fork_runtime_v2(runtime,ctypes.byref(f))
 row={'execution_count':count,'event_ticks':ticks,'snapshot':s,'fork_code':code,'error':lib._last_error() if code else None,'parent_unchanged':lib.export_snapshot(runtime)==s};result.append(row)
 if not code:lib.advance_to(f.value,10**18);lib.destroy_runtime(f.value)
 lib.advance_to(runtime,10**18);lib.destroy_runtime(runtime)
for p in plans['results']:lib.release_plan(p['plan_handle'])
lib.destroy_model(model)
(out/'counts.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='snapshot'} for r in result],indent=2))
