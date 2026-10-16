#!/usr/bin/env python3
import argparse,ctypes,copy,hashlib,json,pathlib,sys
parser=argparse.ArgumentParser()
parser.add_argument('--repo-root',type=pathlib.Path,default=pathlib.Path.cwd())
parser.add_argument('--library',type=pathlib.Path)
parser.add_argument('--baseline-library',type=pathlib.Path)
parser.add_argument('--fixture-dir',type=pathlib.Path)
parser.add_argument('--output',type=pathlib.Path)
args=parser.parse_args();root=args.repo_root.resolve();lab=(args.output or root/'work/snapshot-lab').resolve();lab.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary,OwnerV2
fixture=(args.fixture_dir or root/'work/snapshot-audit').resolve();inputs=json.loads((fixture/'inputs.json').read_text());config=inputs['config'];tasks=inputs['tasks'];frozen=json.loads((fixture/'failed-snapshot.json').read_text())
old=CoarseDieLibrary(args.baseline_library or root/'work/build-core-release/lib/libcoarse_die_cabi.so');new=CoarseDieLibrary(args.library or root/'work/snapshot-lab/libcoarse_die_cabi.so')
models=[lib.create_model(config) for lib in [old,new]];compiled=[lib.compile_batch(model,tasks) for lib,model in zip([old,new],models)]
def im(s):
 h=ctypes.c_uint64();code=new.lib.die_coarse_import_snapshot_v2(models[1],json.dumps(s,separators=(',',':')).encode(),ctypes.byref(h));return code,h.value,new._last_error() if code else None
def replay(lib,model,plans,count):
 r=lib.create_runtime(model)
 for i in range(count):
  o=i//2+1;lib.submit_plan(r,plans['results'][i%2]['plan_handle'],OwnerV2.make(i+1,o,o),0)
 ticks=[]
 for _ in range(2):
  tick=lib.next_event(r);ticks.append(tick);lib.advance_to(r,tick)
 return r,ticks
report={'scope':'scratch import-only witness; baseline unchanged objects reused, not a final fresh build or performance acceptance','counts':[]}
for count in [2,4,6,8]:
 oldr,ticks=replay(old,models[0],compiled[0],count);newr,newticks=replay(new,models[1],compiled[1],count)
 source=old.export_snapshot(oldr);assert source==new.export_snapshot(newr)
 if count==8:assert source==frozen
 code,imported,error=im(source);assert code==0,error;assert new.export_snapshot(imported)==source
 fork=ctypes.c_uint64();code=new.lib.die_coarse_fork_runtime_v2(newr,ctypes.byref(fork));assert code==0,new._last_error();assert new.export_snapshot(fork.value)==source
 before=new.export_snapshot(newr);rejects=[]
 # Independent unchanged-source continuation with no future external controls.
 plain_old,plain_ticks=replay(old,models[0],compiled[0],count)
 plain_code,plain_new,plain_error=im(source);assert plain_code==0,plain_error
 plain_samples=[]
 for step in range(64):
  tick=old.next_event(plain_old);assert new.next_event(plain_new)==tick
  if tick is None:break
  old.advance_to(plain_old,tick);new.advance_to(plain_new,tick)
  assert old.export_snapshot(plain_old)==new.export_snapshot(plain_new)
  plain_samples.append({'tick':tick,'snapshot':old.export_snapshot(plain_old)})
 else:raise AssertionError('plain bounded continuation exceeded')
 old.destroy_runtime(plain_old);new.destroy_runtime(plain_new)
 mutations=[('rate_1percent',lambda s:s['executions'][0]['activities'][1].__setitem__('rate',s['executions'][0]['activities'][1]['rate']*1.01)),('rate_zero',lambda s:s['executions'][0]['activities'][1].__setitem__('rate',0.0)),('alpha_1percent',lambda s:s['executions'][0]['activities'][1].__setitem__('alpha',s['executions'][0]['activities'][1]['alpha']*.99)),('inactive_rate',lambda s:s['executions'][0]['activities'][0].__setitem__('rate',1.0)),('credit',lambda s:s['executions'][0]['credit'].__setitem__(0,s['executions'][0]['credit'][0]+.01)),('capacity_above_hardware',lambda s:s['capacities'].__setitem__(0,2.0)),('fingerprint',lambda s:s.__setitem__('hardware_fingerprint','invalid'))]
 if count==8:
  for name,mutate in mutations:
   bad=copy.deepcopy(source);mutate(bad);c,h,e=im(bad);assert c and h==0,(name,c,h);assert new.export_snapshot(newr)==before
   rejects.append({'name':name,'code':c,'error':e,'parent_unchanged':True})
 worlds=[(old,oldr),(new,newr),(new,imported),(new,fork.value)];samples=[]
 for lib,r in worlds:
  fn=lib.lib.die_coarse_list_executions_v2;fn.argtypes=[ctypes.c_uint64];fn.restype=ctypes.c_void_p
  listed=lib._take_json(fn(r),'list_executions')['executions'];last=next(e['execution_handle'] for e in listed if e['local_handle']==count)
  lib._check(lib.lib.die_coarse_cancel_execution_v2(last,ticks[-1]),'cancel')
  lib._check(lib.lib.die_coarse_update_capacity_v2(r,4,source['capacities'][4]*.75,ticks[-1]),'capacity')
 assert all(lib.export_snapshot(r)==old.export_snapshot(oldr) for lib,r in worlds)
 for step in range(64):
  nexts=[lib.next_event(r) for lib,r in worlds];assert all(t==nexts[0] for t in nexts)
  if nexts[0] is None:break
  for lib,r in worlds:lib.advance_to(r,nexts[0])
  snaps=[lib.export_snapshot(r) for lib,r in worlds];assert all(s==snaps[0] for s in snaps)
  samples.append({'tick':nexts[0],'snapshot':snaps[0]})
 else:raise AssertionError('bounded continuation exceeded')
 report['counts'].append({'count':count,'ticks':ticks,'source_equal':True,'original_snapshot_import_equal':True,'fork_equal':True,'continuation_samples':samples,'final_snapshot':old.export_snapshot(oldr),'negative_cases':rejects,'uncontrolled_continuation_samples':plain_samples})
 for lib,r in worlds:lib.destroy_runtime(r)
for lib,model,plans in zip([old,new],models,compiled):
 for p in plans['results']:lib.release_plan(p['plan_handle'])
 lib.destroy_model(model)
report['tool_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
(lab/'probe-result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'counts':[r['count'] for r in report['counts']],'exact_source_import_fork_continuation':True,'negative_cases':sum(len(r['negative_cases']) for r in report['counts'])}))
