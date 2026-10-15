#!/usr/bin/env python3
"""Actual public C ABI pause probe with independent per-Core RR occupancy."""
import argparse
import hashlib
import json
import pathlib
import sys

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / 'die/validation'))
from cabi import CoarseDieLibrary, OwnerV2

cfg = json.loads((root / 'die/configs/example_hardware_explicit.json').read_text())
cfg['architecture'].update(core_columns_nx=2, core_rows_ny=1)
task = {'op_kind': 'residual', 'tensor_shapes': {'m': 1, 'hidden': 1792},
        'storage_dtype': 'bf16', 'execution_profile': 'fp32_vec_matrix_assumed_v1',
        'layer_instances': 1, 'weight_passes': 1}
lib = CoarseDieLibrary(a.library)
model = lib.create_model(cfg)
compiled = lib.compile_batch(model, [task])
plan = compiled['results'][0]['plan_handle']
runtime = lib.create_runtime(model)
execution = lib.submit_plan(runtime, plan, OwnerV2.make(1, 1, 1))
trace = []
try:
    for _ in range(20):
        q = lib.query_progress(execution)
        trace.append(q)
        if q['epoch_index'] >= 2:
            break
        tick = lib.next_event(runtime)
        if tick is None:
            raise RuntimeError('no prefix event')
        lib.advance_to(runtime, tick)
    else:
        raise RuntimeError('prefix exceeded fixed event bound')
    now = int(lib.query_counters(runtime)['now_ns'])
    lib._check(lib.lib.die_coarse_update_capacity_v2(runtime, 0, 0.0, now), 'pause compute')
    lib.advance_to(runtime, now + 1000000000)
    q = lib.query_progress(execution)
    trace.append(q)
    progress = {v['role']: float(v['progress']) for v in q['activities'] if v['epoch_index'] == 2}
    load, compute, store = (progress[k] for k in ['load', 'compute', 'store'])
    assignment = [block % 2 for block in range(7)]
    last_begin = 6
    occupancies = []
    for core in range(2):
        load_blocks = assignment[4:].count(core)
        compute_blocks = assignment[2:last_begin].count(core)
        store_blocks = assignment[:last_begin - 2].count(core)
        inputs = 1 + load_blocks * load - compute_blocks * compute
        visible = 1 + (compute_blocks - 1) * compute - store_blocks * store
        outputs = visible + compute  # penultimate is held until the middle terminal
        occupancies.append({'core': core, 'input_blocks': inputs, 'input_bytes': inputs * 1024,
                            'visible_output_blocks': visible, 'output_blocks': outputs})
    passed = all(-1e-10 <= x['input_blocks'] <= 2 + 1e-10 and
                 -1e-10 <= x['visible_output_blocks'] and x['output_blocks'] <= 2 + 1e-10 for x in occupancies)
    data = {'scope': 'fixed slots and actual public pause; not latency or hardware calibration',
            'library_sha256': hashlib.sha256(lib.path.read_bytes()).hexdigest(),
            'hardware': cfg, 'task': task, 'compiled_response': compiled,
            'event_trace': trace, 'counters': lib.query_counters(runtime),
            'bulk_progress': progress, 'per_core': occupancies, 'slot_bounds_passed': passed}
    pathlib.Path(a.output).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'bulk_progress': progress, 'per_core': occupancies, 'slot_bounds_passed': passed}))
    if not passed:
        raise RuntimeError('physical slot bounds failed')
finally:
    now = int(lib.query_counters(runtime)['now_ns'])
    lib._check(lib.lib.die_coarse_cancel_execution_v2(execution, now), 'cancel')
    lib.destroy_execution(execution)
    lib.destroy_runtime(runtime)
    lib.release_plan(plan)
    lib.destroy_model(model)
