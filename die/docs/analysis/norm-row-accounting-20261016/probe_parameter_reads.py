#!/usr/bin/env python3
"""Observe public Norm traffic against independent mandatory byte bounds."""
import argparse
import copy
import hashlib
import json
import pathlib
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / 'die/validation'))
from cabi import CoarseDieLibrary

output = pathlib.Path(a.output)
if output.exists(): p.error('fresh output required')
library = CoarseDieLibrary(a.library)
config_path = root / 'die/configs/example_hardware_explicit.json'
base = json.loads(config_path.read_text())
rows = []
for cores, m, hidden, norm in [(1,8,4096,'rmsnorm'), (4,8,4096,'rmsnorm'),
    (8,8,4096,'rmsnorm'), (4,8,4096,'layernorm'), (4,8,131073,'rmsnorm'),
    (4,8,131073,'layernorm')]:
    config = copy.deepcopy(base)
    config['architecture']['core_columns_nx'] = cores
    config['architecture']['core_rows_ny'] = 1
    task = {'op_kind': 'norm', 'tensor_shapes': {'m': m, 'hidden': hidden, 'norm_type': norm},
            'storage_dtype': 'bf16'}
    model = library.create_model(config)
    try:
        hardware = library.hardware_summary(model)
        estimate = library.estimate_batch(model, [task])
    finally:
        library.destroy_model(model)
    result = estimate['results'][0]
    if result is None:
        rows.append({'config':config,'task':task,'hardware_summary':hardware,'estimate':estimate,
                     'prediction':'not_available_compile_error'})
        print(cores,m,hidden,norm,'compile_error',estimate['errors'])
        continue
    summary = result  # estimate_batch exposes the summary fields directly.
    parameter_vectors = 1 if norm == 'rmsnorm' else 2
    x = m * hidden * 2
    # Independent lower bound: even a perfect fitting cache cannot share
    # parameters for free between row-owning consumer Cores.
    minimum_parameter_reads = min(cores, m) * parameter_vectors * hidden * 2
    minimum_total_read = x + minimum_parameter_reads
    rows.append({'config': config, 'task': task, 'hardware_summary': hardware,
        'estimate': estimate, 'minimum_read_bytes': minimum_total_read,
        'minimum_parameter_read_bytes': minimum_parameter_reads,
        'observed_read_bytes': summary['hbm_read_bytes_total'],
        'mandatory_parameter_bound_met': summary['hbm_read_bytes_total'] >= minimum_total_read})
    print(cores, m, hidden, norm, summary['hbm_read_bytes_total'], minimum_total_read,
          rows[-1]['mandatory_parameter_bound_met'])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'scope': 'workload byte lower bound, not timing or hardware calibration',
    'observed_git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root,text=True).strip(),
    'library': str(library.path), 'library_sha256': sha(library.path),
    'probe_sha256': sha(pathlib.Path(__file__)), 'config_sha256': sha(config_path),
    'mapping_sha256': sha(root / 'die/src/Mapping.cc'), 'rows': rows}, indent=2) + '\n')
