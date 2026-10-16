#!/usr/bin/env python3
"""Compare preserved resource rows across source identities; no hardware search.

The older source is read from Git, never restored into production. Its retired
selector is used only as a historical comparison input. All points below are
explicit verification fixtures, not architecture recommendations.
"""
import argparse
import copy
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--before', type=Path, required=True)
p.add_argument('--after', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[4]
base = json.loads((root / 'die/configs/example_hardware_explicit.json').read_text())
old = json.loads(subprocess.check_output(['git', 'show', '41f1191:die/configs/example_hardware_explicit.json'], cwd=root))
configs = []
for bits in [64, 128, 256, 512]:
    for sram in [.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]:
        for nx, ny in [(1, 1), (3, 2), (2, 3)]:
            for topology in ['distributed_ideal_v2', 'edge_single_word_v2']:
                config = copy.deepcopy(base)
                config['architecture'].update(sram_mib_per_core=sram, noc_width_bits=bits,
                                               core_columns_nx=nx, core_rows_ny=ny)
                config['performance_profile']['timing_topology_profile'] = topology
                configs.append(config)
old_configs = []
for config in configs:
    historical = copy.deepcopy(config)
    for key in ['source_revision', 'source_constants', 'core_resource_table']:
        historical[key] = old[key]
    historical['architecture']['core_type'] = 'small'
    old_configs.append(historical)

def run(executable, inputs):
    result = subprocess.run([str(executable.resolve())], input=json.dumps(inputs),
                            text=True, capture_output=True, check=True, cwd=root)
    return json.loads(result.stdout), result.stderr

before, before_stderr = run(a.before, old_configs)
after, after_stderr = run(a.after, configs)
assert len(before) == len(after) == len(configs) == 192
failures = []
for index, (b, c) in enumerate(zip(before, after)):
    b = copy.deepcopy(b)
    c = copy.deepcopy(c)
    b.pop('fingerprint'); c.pop('fingerprint')
    b['derived']['core'].pop('core_type')
    for point in [b, c]:
        point['derived'].pop('source_revision')
        point['derived'].pop('core_resource_table')
    if b != c:
        failures.append(index)
a.output.mkdir(parents=True, exist_ok=False)
raw = {'scope': 'same explicit hardware, preserved resource rows, changed source identity',
       'configs_before': old_configs, 'configs_after': configs,
       'results_before': before, 'results_after': after,
       'stderr_before': before_stderr, 'stderr_after': after_stderr}
blob = json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2).encode()
with (a.output / 'paired-resources.json.gz').open('wb') as f:
    f.write(gzip.compress(blob, mtime=0))
summary = {'status': 'passed' if not failures else 'failed', 'cases': len(configs),
           'rows': 32, 'grids': 3, 'topology_profiles': 2,
           'differences_excluded': ['retired selector', 'source_revision', 'core_resource_table',
                                    'hardware fingerprint'],
           'numeric_resources_geometry_and_summary_identical': not failures,
           'mismatched_case_indices': failures,
           'raw_uncompressed_sha256': hashlib.sha256(blob).hexdigest(),
           'binary_sha256': {str(e): hashlib.sha256(e.read_bytes()).hexdigest()
                             for e in [a.before, a.after]}}
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
raise SystemExit(bool(failures))
