#!/usr/bin/env python3
"""Run the final hardware resolver at 192 explicit verification points.

Audit Core counts directly against the authoritative DOCX and compare the full
numeric output with the immutable first migration run. The archived output is
a regression reference, not evidence that the current executable was run.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location('docx_audit', root / 'die/tools/check_core_source.py')
    docx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(docx)
    docx.main()
    rows = docx.read_rows(root / '架构建模_单Core版_严格复核修订.docx')
    base = json.loads((root / 'die/configs/example_hardware_explicit.json').read_text())
    configs = []
    for row in rows:
        for nx, ny in [(1, 1), (3, 2), (2, 3)]:
            for topology in ['distributed_ideal_v2', 'edge_single_word_v2']:
                config = copy.deepcopy(base)
                config['architecture'].update(sram_mib_per_core=row['sram_mib_per_core'],
                    noc_width_bits=row['noc_width_bits'], core_columns_nx=nx, core_rows_ny=ny)
                config['performance_profile']['timing_topology_profile'] = topology
                configs.append(config)
    result = subprocess.run([str(a.probe.resolve())], cwd=root, input=json.dumps(configs),
                            text=True, capture_output=True, check=True)
    actual = json.loads(result.stdout)
    reference_path = root / 'die/docs/evidence/single-core-20261016/paired-resources/paired-resources.json.gz'
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(gzip.decompress(reference_bytes))['results_after']
    assert len(actual) == len(reference) == len(configs) == 192
    failures = []
    for i, (config, value, old) in enumerate(zip(configs, actual, reference)):
        arch, derived = config['architecture'], value['derived']
        row, core, die = rows[i // 6], derived['core'], derived['die']
        count = arch['core_columns_nx'] * arch['core_rows_ny']
        if core['N_pe'] != row['n_pe'] or core['N_vec'] != row['n_vec']:
            failures.append({'index': i, 'check': 'DOCX installed resources'})
        if 'core_type' in core or core['N_core'] != count:
            failures.append({'index': i, 'check': 'single Core and grid count'})
        edge = row['sram_mib_per_core'] * .40662 / .25
        expected = [(core['L0_mm'], edge),
                    (core['sram_bytes_per_core'], row['sram_mib_per_core'] * 2**20),
                    (die['minimum_width_mm'], arch['core_columns_nx'] * edge +
                     (arch['core_columns_nx'] - 1) * arch.get('core_pitch_x_mm', 0) + arch['die_extra_width_mm']),
                    (die['minimum_height_mm'], arch['core_rows_ny'] * edge +
                     (arch['core_rows_ny'] - 1) * arch.get('core_pitch_y_mm', 0) + arch['die_extra_height_mm'])]
        if not all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in expected):
            failures.append({'index': i, 'check': 'DOCX SRAM and explicit geometry formulas'})
        without_fingerprint = lambda x: {k: v for k, v in x.items() if k != 'fingerprint'}
        if without_fingerprint(value) != without_fingerprint(old):
            failures.append({'index': i, 'check': 'full archived numeric regression'})
    a.output.mkdir(parents=True, exist_ok=False)
    raw = json.dumps({'configs': configs, 'results': actual, 'stderr': result.stderr},
                     sort_keys=True, indent=2).encode()
    (a.output / 'raw.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    summary = {'status': 'failed' if failures else 'passed', 'cases': 192, 'docx_rows': 32,
        'grids': 3, 'topology_profiles': 2, 'failures': failures,
        'excluded_from_archived_comparison': ['fingerprint: final source constant mirror changed'],
        'reference_path': str(reference_path.relative_to(root)),
        'reference_sha256': hashlib.sha256(reference_bytes).hexdigest(),
        'probe_sha256': hashlib.sha256(a.probe.read_bytes()).hexdigest(),
        'raw_uncompressed_sha256': hashlib.sha256(raw).hexdigest()}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
