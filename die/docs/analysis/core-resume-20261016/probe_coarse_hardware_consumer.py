#!/usr/bin/env python3
"""Check each variant's existing Core32 hardware consumer through its own ABI.

Four variants author typed hardware through config_resolver. JOINT's existing
coarse entry is its online oracle; its authoring resolver remains legacy-only.
No new production API or fallback is introduced by this verification tool.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('fresh output required')
    repo, library = a.repo_root.resolve(), a.library.resolve()
    descriptor = json.loads(a.config.read_text())
    os.environ['DIE_COARSE_CABI_PATH'] = str(library)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(repo / 'sh_test_mesh'),
                   str(repo / 'sh_test_mesh/workload/llama2_7b_inference')]
    joint = repo.name == 'astra-sim-joint'
    if joint:
        from online.die_model_oracle import DieModelOracle
        consumer = repo / 'sh_test_mesh/workload/llama2_7b_inference/online/die_model_oracle.py'
        def resolve(config):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'current.json'
                path.write_text(json.dumps(config))
                with DieModelOracle(lib_path=str(library), config_json_path=str(path)) as oracle:
                    assert oracle.coarse
                    return oracle.hardware_summary()
        entry = 'online.die_model_oracle.DieModelOracle'
    else:
        from config_resolver import derive_coarse_die_hardware, load_hardware_config
        consumer = repo / 'sh_test_mesh/config_resolver.py'
        resolve = derive_coarse_die_hardware
        entry = 'config_resolver.derive_coarse_die_hardware/load_hardware_config'
    summary = resolve(descriptor)
    derived = summary['derived']
    assert derived['core_resource_table']['id'] == 'core32_v1'
    assert derived['core']['N_core'] == 88
    assert (derived['core']['N_vec'], derived['core']['N_pe']) == (2, 40)
    assert 'core_type' not in derived['core']
    if not joint:
        authored = json.loads((repo / 'sh_test_mesh/hardware/face_case5_config_c.json').read_text())
        authored.pop('die')
        authored['coarse-die'] = descriptor
        capacity = int(derived['hardware_summary']['hbm_capacity_bytes'])
        authored['local-hbm']['capacity-profiles'] = {'explicit-fixture':
            {'bytes': capacity, 'label': 'DOCX verification fixture', 'note': 'not a hardware recommendation'}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'authored.json'
            path.write_text(json.dumps(authored))
            hardware = load_hardware_config(path, 'explicit-fixture')
            assert hardware.local_hbm_capacity_bytes == capacity
            assert hardware.die_profile == 'coarse_die_v2'
    rejected = []
    for value in ['small', 'large', 'core', None, 0, True, [], {}]:
        bad = copy.deepcopy(descriptor)
        bad['architecture']['core_type'] = value
        try:
            resolve(bad)
        except (ValueError, RuntimeError) as error:
            assert 'architecture.core_type' in str(error), str(error)
            rejected.append(value)
        else:
            raise AssertionError('obsolete Core selector accepted')
    report = {'status': 'passed', 'repo': repo.name, 'actual_consumer_entry': entry,
        'scope': 'actual existing hardware consumer/ABI; not full business or timing acceptance',
        'joint_authoring_coarse_node': 'not_implemented_existing_legacy_resolver' if joint else 'not_applicable',
        'summary': summary, 'obsolete_selector_values_rejected': rejected,
        'consumer_sha256': hashlib.sha256(consumer.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(a.config.read_bytes()).hexdigest(),
        'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'repo', 'actual_consumer_entry']}))


if __name__ == '__main__': main()
