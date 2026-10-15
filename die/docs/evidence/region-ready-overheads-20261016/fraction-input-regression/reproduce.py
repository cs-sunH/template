#!/usr/bin/env python3
"""Read-only C ABI correctness probe. No benchmarks or production file edits."""
import argparse
import ctypes
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    repo, source_lib, out = args.repo.resolve(), args.library.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'library-under-test.so'
    shutil.copy2(source_lib, frozen)
    library_sha = sha(frozen)
    paths = ['die/src/Mapping.cc', 'die/src/Compiler.cc', 'die/src/CompilerInternal.hh',
             'die/src/Execution.cc', 'die/src/Plan.cc', 'die/src/EquivalentWork.hh',
             'die/src/Capi.cc', 'die/include/coarse_die/Plan.hh', 'die/include/coarse_die/Execution.hh']
    sources = {}
    for relative in paths:
        source = repo / relative
        target = out / 'observed-source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[relative] = {'sha256': sha(target), 'source_mtime_ns': source.stat().st_mtime_ns}
    def git(*values):
        return subprocess.check_output(['git', *values], cwd=repo, text=True)
    (out / 'observed-source.patch').write_text(git('diff', '--', *paths))
    lib = ctypes.CDLL(str(frozen))
    lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.die_coarse_create_model_v2.restype = ctypes.c_int
    lib.die_coarse_compile_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    lib.die_coarse_compile_batch_v2.restype = ctypes.c_void_p
    lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
    lib.die_coarse_free_result_v2.restype = None
    lib.die_coarse_last_error_v2.restype = ctypes.c_void_p
    lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
    lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
    task = {'op_kind': 'residual', 'tensor_shapes': {'m': 1, 'hidden': 1},
            'storage_dtype': 'bf16', 'layer_instances': 1, 'weight_passes': 1}
    entries = []
    for name, launch, setup in [('control-half-ns', 0.5, 0.0),
                                ('fraction-sum-regression', 0.5, math.ldexp(1.0, -65)),
                                ('control-three-quarter-ns', 0.25, 0.5)]:
        config = json.loads((repo / 'die/configs/example_hardware_explicit.json').read_text())
        config['architecture'].update(core_columns_nx=1, core_rows_ny=1)
        config['performance_profile'].update(launch_ns_per_macro=launch, memory_setup_ns_per_macro=setup)
        model = ctypes.c_uint64()
        created = lib.die_coarse_create_model_v2(json.dumps(config).encode(), ctypes.byref(model))
        entry = {'case': name, 'launch_ns': launch, 'setup_ns': setup,
                 'launch_hex': launch.hex(), 'setup_hex': setup.hex(), 'model_create_status': created}
        exact = Fraction.from_float(launch) + Fraction.from_float(setup)
        entry['exact_required_delay_expression'] = str(exact)
        entry['expected_integer_ready_delay_ns'] = -(-exact.numerator // exact.denominator)
        (out / f'{name}.config.json').write_text(json.dumps(config, indent=2) + '\n')
        if created == 0:
            raw_ptr = lib.die_coarse_compile_batch_v2(model.value, json.dumps({'specs': [task]}).encode())
            if raw_ptr:
                raw = ctypes.string_at(raw_ptr).decode()
                lib.die_coarse_free_result_v2(raw_ptr)
            else:
                error_ptr = lib.die_coarse_last_error_v2()
                raw = ctypes.string_at(error_ptr).decode() if error_ptr else 'null'
            (out / f'{name}.raw.json').write_text(raw + '\n')
            entry['actual_compile_response'] = json.loads(raw)
            entry['model_destroy_status'] = lib.die_coarse_destroy_model_v2(model.value)
        entries.append(entry)
    result = {'scope': 'correctness/input-domain regression; no timing or performance measurement',
              'stage': args.stage, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'git_head': git('rev-parse', 'HEAD').strip(),
              'git_worktree_status': git('status', '--short'),
              'public_library_source_path': str(source_lib), 'frozen_public_library_sha256': library_sha,
              'source_library_sha256_after_probe': sha(source_lib),
              'frozen_library_sha256_after_probe': sha(frozen),
              'source_identity_qualification': 'Sources copied at rerun; this probe does not rebuild or independently attest build provenance. Stage explicitly follows the observed current source, rather than attributing this rerun to the initial baseline library.',
              'observed_source': sources,
              'sources_unchanged_during_probe': all(sha(repo / p) == identity['sha256'] for p, identity in sources.items()),
              'task': task, 'cases': entries}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'stage': args.stage, 'library_sha256': library_sha,
                      'sources_unchanged_during_probe': result['sources_unchanged_during_probe'],
                      'cases': [{'case': e['case'], 'model_create_status': e['model_create_status'],
                                 'expected_integer_ready_delay_ns': e['expected_integer_ready_delay_ns'],
                                 'errors': e.get('actual_compile_response', {}).get('errors')} for e in entries]}, indent=2))


if __name__ == '__main__':
    main()
