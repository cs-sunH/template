#!/usr/bin/env python3
"""Public Runtime pause probe; independently infer per-Core input slot usage."""
import argparse,copy,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary,OwnerV2
cfg=json.loads((root/'die/configs/example_hardware_explicit.json').read_text());cfg['architecture'].update(core_columns_nx=2,core_rows_ny=1)
task={'op_kind':'residual','tensor_shapes':{'m':1,'hidden':1792},'storage_dtype':'bf16','execution_profile':'fp32_vec_matrix_assumed_v1','layer_instances':1,'weight_passes':1}
lib=CoarseDieLibrary(a.library);m=lib.create_model(cfg);compiled=lib.compile_batch(m,[task]);plan=compiled['results'][0]['plan_handle'];r=lib.create_runtime(m);e=lib.submit_plan(r,plan,OwnerV2.make(1,1,1));trace=[]
try:
 for _ in range(20):
  q=lib.query_progress(e);trace.append(q)
  if q['epoch_index']>=2:break
  tick=lib.next_event(r)
  if tick is None:raise RuntimeError('no event before first bulk')
  lib.advance_to(r,tick)
 else:raise RuntimeError('prefix did not finish within bounded events')
 now=int(lib.query_counters(r)['now_ns']);lib._check(lib.lib.die_coarse_update_capacity_v2(r,0,0.0,now),'pause compute gate')
 lib.advance_to(r,now+1000000000);q=lib.query_progress(e);trace.append(q)
 stages={v['role']:v for v in q['activities'] if v['epoch_index']==2}
 load=float(stages['load']['progress']);compute=float(stages['compute']['progress'])
 # Old v1: remaining five blocks RR-map as [Core0,Core1,Core0,Core1,Core0].
 # First cohort was fully consumed. Each residual block has 2*256*2 input B.
 occupancy=[3*(load-compute),2*(load-compute)];allocated=2*2*256*2
 data={'scope':'fixed two-input-slot template, not total SRAM occupancy or hardware calibration','library_sha256':hashlib.sha256(lib.path.read_bytes()).hexdigest(),'hardware':cfg,'task':task,'compiled_response':compiled,'event_trace':trace,'counters':lib.query_counters(r),'bulk_load_progress':load,'bulk_compute_progress':compute,'per_core_input_blocks':occupancy,'per_core_input_bytes':[n*2*256*2 for n in occupancy],'allocated_input_bytes_per_core':allocated,'fixed_input_slots_violated':max(occupancy)>2+1e-10}
 pathlib.Path(a.output).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:data[k] for k in ['bulk_load_progress','bulk_compute_progress','per_core_input_blocks','per_core_input_bytes','allocated_input_bytes_per_core','fixed_input_slots_violated']}))
finally:
 now=int(lib.query_counters(r)['now_ns']);lib._check(lib.lib.die_coarse_cancel_execution_v2(e,now),'cancel');lib.destroy_execution(e);lib.destroy_runtime(r);lib.release_plan(plan);lib.destroy_model(m)
