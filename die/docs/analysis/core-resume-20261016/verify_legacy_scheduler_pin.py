#!/usr/bin/env python3
"""Record the seven actual legacy ABI terms behind the migrated FACE pin.

This audits the authorized Core32 engineering fixture. It is software
regression evidence, not a target hardware measurement or calibration.
"""
import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo, library = a.repo_root.resolve(), a.library.resolve()
    if a.output.exists():
        p.error('fresh output required')
    os.environ['DIE_MODEL_CABI_PATH'] = str(library)
    sys.dont_write_bytecode = True
    workload = repo / 'sh_test_mesh/workload/llama2_7b_inference'
    sys.path[:0] = [str(workload), str(repo), str(repo / 'sh_test_mesh')]
    from face_scheduler import (FaceHardware, build_die_estimate_specs,
                                estimate_iteration_time_ns)
    from online.die_model_oracle import DieModelOracle
    from test_face_scheduler import FACE_CASE5_DIE_CONFIG, DieModelEstimateTests

    hardware = FaceHardware(3, 2, 100, 1.0, 2.0, 1.0, 0, 0)
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / 'engineering-point.json'
        config.write_text(json.dumps(FACE_CASE5_DIE_CONFIG))
        with DieModelOracle(lib_path=str(library), config_json_path=str(config)) as oracle:
            assert not oracle.coarse
            summary = oracle.hardware_summary()
            assert summary['derived']['cores'] == 100
            assert math.isclose(summary['derived']['pe_peak_int4_tops'], 126.6, rel_tol=1e-12)
            specs, labels = build_die_estimate_specs(DieModelEstimateTests.TOY,
                instance_size=2, p_chunk=8, d_batch=1, d_token=257,
                config_hash=oracle.config_hash)
            results = oracle.estimate(specs)
            assert all(row is not None for row in results), oracle.last_errors
            terms = {label: Decimal(str(row['total_seconds']))
                     for label, row in zip(labels, results)}
            linear = sum((terms[label] for label in
                ('norm', 'qkv_rope', 'o_proj', 'ffn_region', 'residual')), Decimal(0))
            attention = max(terms['attention_prefill'], terms['attention_decode'])
            decimal_ns = (linear + attention) * 10**9
            independent_ceil = int(decimal_ns.to_integral_value(rounding=ROUND_CEILING))
            actual = estimate_iteration_time_ns(hardware, DieModelEstimateTests.TOY,
                instance_size=2, p_chunk=8, d_batch=1, d_token=257, cost_oracle=oracle)
            assert independent_ceil == actual == 6192
            report = {'status': 'passed',
                'scope': 'actual legacy Core32 seven-term FACE fixture; not hardware calibration',
                'hardware_input': FACE_CASE5_DIE_CONFIG, 'hardware_summary': summary,
                'labels': labels, 'specs': specs, 'results': results,
                'seconds_by_term_decimal': {k: str(v) for k, v in terms.items()},
                'linear_seconds_decimal': str(linear),
                'attention_seconds_decimal': str(attention),
                'total_ns_before_ceil_decimal': str(decimal_ns),
                'independent_decimal_ceil_ns': independent_ceil,
                'scheduler_result_ns': actual,
                'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__), workload / 'face_scheduler.py',
                                 workload / 'test_face_scheduler.py']},
                'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in
        ['status', 'total_ns_before_ceil_decimal', 'independent_decimal_ceil_ns', 'scheduler_result_ns']}))


if __name__ == '__main__':
    main()
