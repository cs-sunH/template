#!/usr/bin/env python3
"""Capture public outputs and bounded live traces for a storage-only comparison.

This is an offline equivalence check, not an accuracy oracle or a CPU budget.
Run separately for each freshly built library, then compare the `results` field.
"""
import argparse
import ctypes
import gzip
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source, library, output = a.source_root.resolve(), a.library.resolve(), a.output.resolve()
if output.exists(): p.error('fresh output required')
sys.path.insert(0, str(source / 'die/validation'))
from cabi import CoarseDieLibrary, OwnerV2

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = sorted(path for name in ['src', 'include', 'configs', 'validation']
                for path in (source / 'die' / name).rglob('*')
                if path.is_file() and '__pycache__' not in path.parts)
identity = {str(path.relative_to(source)): sha(path) for path in inputs}
library_before = sha(library)
lib = CoarseDieLibrary(library)
for name in ['die_coarse_estimate_cost_batch_v2', 'die_coarse_workspace_envelope_v2',
             'die_coarse_workspace_capacity_bound_v2']:
    function = getattr(lib.lib, name)
    function.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    function.restype = ctypes.c_void_p

def call(name, model, payload):
    try:
        return lib._take_json(getattr(lib.lib, name)(model, json.dumps(payload).encode()), name)
    except RuntimeError as error:
        return {'cabi_error': str(error)}

def state(runtime, handles):
    return {'next_event': lib.next_event(runtime),
            'progress': [lib.query_progress(handle) for handle in handles],
            'counters': lib.query_counters(runtime), 'snapshot': lib.export_snapshot(runtime)}

def trace(model, compiled, count, mode):
    runtime = lib.create_runtime(model)
    handles = []
    rows = []
    now = 0
    for i in range(count):
        # Different physical plans, same fair owner in adjacent pairs in the
        # split case. The mixed case retains distinct owners and real shapes.
        index = (i % 3) if mode == 'mixed' else (i % 2) if mode == 'split' else 0
        owner_id = i // 2 + 1 if mode == 'split' else i + 1
        handles.append(lib.submit_plan(runtime, compiled[index]['plan_handle'],
            OwnerV2.make(i + 1, owner_id, owner_id), 0))
    rows.append(state(runtime, handles))
    for step in range(64):
        tick = lib.next_event(runtime)
        if tick is None: break
        if tick < now: raise AssertionError('event moved backwards')
        now = tick
        lib.advance_to(runtime, now)
        rows.append(state(runtime, handles))
        if step == 1:
            fork = ctypes.c_uint64()
            before = lib.export_snapshot(runtime)
            code = lib.lib.die_coarse_fork_runtime_v2(runtime, ctypes.byref(fork))
            if code:
                # Preserve a baseline numerical rejection as an observed
                # failed operation. Equivalence does not reclassify it as a
                # successful snapshot. The live parent must stay unchanged.
                rows.append({'fork_failed': {'status_code': code, 'error': lib._last_error()}})
            else:
                rows.append({'fork_snapshot': lib.export_snapshot(fork.value)})
                lib.advance_to(fork.value, 10**18)
                rows.append({'fork_final_counters': lib.query_counters(fork.value)})
                lib.destroy_runtime(fork.value)
            if lib.export_snapshot(runtime) != before: raise AssertionError('fork mutated parent')
            if count > 1:
                lib._check(lib.lib.die_coarse_cancel_execution_v2(handles[-1], now), 'cancel')
                capacity = lib.query_counters(runtime)['capacity'][4]
                lib._check(lib.lib.die_coarse_update_capacity_v2(runtime, 4, capacity * .75, now),
                           'update_capacity')
                rows.append({'after_cancel_and_capacity': state(runtime, handles)})
    else:
        raise AssertionError('bounded trace exceeded 64 public events')
    if lib.next_event(runtime) is not None: raise AssertionError('trace not drained')
    lib.destroy_runtime(runtime)
    return {'owners': count, 'mode': mode, 'rows': rows}

tasks = json.loads((source / 'die/configs/example_tasks.json').read_text())['specs']
tasks += [{'op_kind': 'norm', 'tensor_shapes': {'m': m, 'hidden': hidden},
           'storage_dtype': 'bf16', 'layer_instances': repeats}
          for m, hidden, repeats in [(8, 4096, 1), (128, 19569, 1), (5, 257, 3)]]
for shape in [(1, 1, 1), (7, 5, 9), (128, 128, 128), (65, 129, 257)]:
    for dtype in ['int4', 'bf16']:
        for repeats in [1, 3]:
            tasks.append({'op_kind': 'gemm', 'tensor_shapes': dict(zip(['m', 'n', 'k'], shape)),
                          'storage_dtype': dtype, 'layer_instances': repeats, 'weight_passes': 1})
tasks += [{'op_kind': 'residual', 'tensor_shapes': {'m': 1, 'hidden': 257},
           'storage_dtype': 'bf16', 'layer_instances': 3},
          {'op_kind': 'gemm', 'tensor_shapes': {'m': 1, 'n': 1, 'k': 1}, 'storage_dtype': 'bad'}]
results = []
for config_name in ['example_hardware_explicit.json', 'tx8_compute_matched_hardware.json']:
    config = json.loads((source / 'die/configs' / config_name).read_text())
    model = lib.create_model(config)
    compiled = lib.compile_batch(model, tasks)
    row = {'config_name': config_name, 'hardware': lib.hardware_summary(model),
           'tasks': tasks, 'compile': compiled, 'estimate': lib.estimate_batch(model, tasks),
           'hot_compile': lib.compile_batch(model, tasks),
           'cost': call('die_coarse_estimate_cost_batch_v2', model, {'specs': tasks}),
           'envelopes': [call('die_coarse_workspace_envelope_v2', model, task) for task in tasks],
           'capacity_bounds': [call('die_coarse_workspace_capacity_bound_v2', model, task) for task in tasks]}
    if sum(item is None for item in compiled['results']) != 1:
        raise AssertionError('expected exactly the invalid-dtype compile failure')
    row['workspace_error_counts'] = {name: sum('cabi_error' in item for item in row[name])
                                     for name in ['envelopes', 'capacity_bounds']}
    # R=1 real GEMM, Norm and residual plans; repeated diagnostic bodies above
    # retain their unavailable status and are not substituted into this trace.
    picks = [compiled['results'][index] for index in [0, 1, 4]]
    if any(item is None for item in picks): raise AssertionError('trace plan missing')
    row['live_traces'] = [trace(model, picks, count, mode)
                         for count, mode in [(1, 'same'), (8, 'same'), (32, 'mixed'), (8, 'split')]]
    # Both compile calls own their returned handles even on cache hits.
    for response in [compiled, row['hot_compile']]:
        for item in response['results']:
            if item is not None: lib.release_plan(item['plan_handle'])
    lib.destroy_model(model)
    results.append(row)
unchanged = all(path.is_file() and sha(path) == identity[str(path.relative_to(source))] for path in inputs)
report = {'scope': 'exact public output/live trace equivalence only; no accuracy/performance acceptance',
          'source_root': str(source), 'library': str(library), 'library_sha256': library_before,
          'tool_sha256': sha(Path(__file__)), 'source_sha256': identity, 'source_unchanged': unchanged,
          'library_unchanged': sha(library) == library_before, 'results': results}
output.parent.mkdir(parents=True, exist_ok=True)
with gzip.GzipFile(filename=str(output), mode='wb', mtime=0) as stream:
    stream.write((json.dumps(report, sort_keys=True, separators=(',', ':')) + '\n').encode())
print(json.dumps({'configs': len(results), 'tasks_per_config': len(tasks), 'trace_cases': 8,
                  'source_unchanged': unchanged, 'library_unchanged': report['library_unchanged'],
                  'output_sha256': sha(output)}))
if not unchanged or not report['library_unchanged']: raise SystemExit(1)
