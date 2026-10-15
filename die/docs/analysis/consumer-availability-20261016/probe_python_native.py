#!/usr/bin/env python3
"""Exercise actual new C ABI responses through the three Python consumers."""
import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import pathlib

p = argparse.ArgumentParser()
p.add_argument('--oracle-root', required=True)
p.add_argument('--library', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
oracle_root = pathlib.Path(a.oracle_root).resolve()
library = pathlib.Path(a.library).resolve()
output = pathlib.Path(a.output).resolve()
if output.exists(): p.error('refusing to replace existing evidence')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
hardware = root / 'die/configs/example_hardware.json'
emitted_file = root / 'astra-sim-face/astra-sim/workload/execution_driven/tests/testdata/face_emitted_ffn_region.json'
emitted = json.loads(emitted_file.read_text())['emitted_node']['compute']['die_op_spec']
one = {'schema': 'coarse_die_task_v2', 'op_kind': 'residual',
       'tensor_shapes': {'m': 1, 'hidden': 257}, 'storage_dtype': 'bf16',
       'layer_instances': 1, 'weight_passes': 1}
repeat = copy.deepcopy(one); repeat['layer_instances'] = 3
tasks = [one, repeat, emitted, {'op_kind': 'invalid_operation_for_availability_probe'}]
report = {'scope': 'real C ABI / consumer availability; not hardware accuracy or performance',
          'library': str(library), 'library_sha256': sha(library),
          'hardware_sha256': sha(hardware), 'emitted_fixture_sha256': sha(emitted_file),
          'probe_sha256': sha(pathlib.Path(__file__)), 'tasks': tasks, 'consumers': {}}
for repo in ['astra-sim-face', 'astra-sim-face-LRU', 'astra-sim-joint']:
    path = oracle_root / repo / 'sh_test_mesh/workload/llama2_7b_inference/online/die_model_oracle.py'
    spec = importlib.util.spec_from_file_location('availability_' + repo.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    record = report['consumers'][repo] = {'oracle_sha256': sha(path), 'runs': {}, 'checks': {}}
    with module.DieModelOracle(lib_path=str(library), config_json_path=str(hardware)) as oracle:
        def raw(endpoint):
            pointer = getattr(oracle._lib, endpoint)(oracle._model, json.dumps({'specs': tasks}).encode())
            if not pointer: raise RuntimeError(oracle._last_error())
            try: return json.loads(ctypes.string_at(pointer))
            finally: oracle._lib.die_coarse_free_result_v2(pointer)
        native = record['raw_full'] = raw('die_coarse_estimate_batch_v2')
        record['checks']['native_has_numeric_unavailable_bodies'] = all(
            native['results'][i]['status'] == 'not_available' and
            isinstance(native['results'][i]['estimated_latency_ns'], int)
            for i in [1, 2])
        record['checks']['native_valid_and_error_siblings'] = (
            native['results'][0]['status'] == 'ok_assumed' and
            native['results'][3] is None and native['errors'][3] is not None)
        calls = [('full', oracle.estimate)]
        if hasattr(oracle, 'estimate_cost'):
            record['raw_cost'] = raw('die_coarse_estimate_cost_batch_v2')
            calls.append(('cost', oracle.estimate_cost))
        for name, call in calls:
            rows = call(tasks)
            errors = list(oracle.last_errors)
            record['runs'][name] = {'results': rows, 'errors': errors}
            record['checks'][name + '_guards_real_responses'] = (
                rows[0] is not None and all(rows[i] is None for i in [1, 2, 3]) and
                errors[0] is None and all('not_available' in (errors[i] or '') for i in [1, 2])
                and errors[3] is not None)
        if hasattr(oracle, 'estimate_cost'):
            oracle.supports_estimate_cost = False
            rows = oracle.estimate_cost(tasks)
            errors = list(oracle.last_errors)
            record['runs']['optional_cost_full_fallback'] = {'results': rows, 'errors': errors}
            record['checks']['optional_full_still_rejects'] = (
                rows[0] is not None and all(rows[i] is None for i in [1, 2, 3]) and
                all('not_available' in (errors[i] or '') for i in [1, 2]))
        try:
            record['estimate_one_result'] = oracle.estimate_one(repeat)
            rejected = False
        except RuntimeError as error:
            record['estimate_one_error'] = str(error)
            rejected = 'not_available' in str(error)
        record['checks']['estimate_one_rejects_real_repeat'] = rejected
    record['status'] = 'passed' if all(record['checks'].values()) else 'failed'
report['status'] = 'passed' if all(x['status'] == 'passed' for x in report['consumers'].values()) else 'failed'
report['public_source_sha256'] = {str(x.relative_to(root)): sha(x)
    for base in ['die/src', 'die/include', 'die/configs']
    for x in sorted((root / base).rglob('*')) if x.is_file()}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'consumers':
    {k: {'status': v['status'], 'checks': v['checks']} for k, v in report['consumers'].items()}}))
raise SystemExit(report['status'] != 'passed')
