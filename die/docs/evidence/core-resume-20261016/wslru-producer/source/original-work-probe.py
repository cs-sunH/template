#!/usr/bin/env python3
"""Verify isolated WSLRU producer wire through the existing FACE ctypes client.

This is a verification bridge, not a WSLRU production client or integration.
The original optional test remains skipped. Its five original R4 inputs are
unchanged and checked with the fresh WSLRU legacy ABI. Coarse R1 controls and
coarse original-R4 unavailability are reported as distinct scopes.
"""
from __future__ import annotations
import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
WIRE_SHA = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    snapshot, output = args.snapshot.resolve(), args.output.resolve()
    if not output.is_relative_to(root/'work') or output == root/'work':
        parser.error('output must be a dedicated directory under repository work/')
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'test-only existing producer/protocol/client bridge; not a new production WSLRU client, full Online, timing accuracy or CPU-budget acceptance',
        'command': [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
        'cwd': str(Path.cwd()), 'started_unix': time.time(),
        'probe_sha256': SHA(Path(__file__)), 'status': 'running', 'checks': {}}
    checks = report['checks']
    try:
        stage = snapshot/'stage-root'
        ws = stage/'astra-sim-wscllm-LRU'
        face = stage/'astra-sim-face'
        workload = ws/'sh_test_mesh/workload/llama2_7b_inference'
        producer_path = workload/'online/die_model_adapter.py'
        fixture_path = workload/'online/test_die_model_adapter.py'
        client_path = face/'sh_test_mesh/workload/llama2_7b_inference/online/die_model_oracle.py'
        coarse_config = ws/'extern/die_model/coarse/configs/example_hardware.json'
        manifest_path = snapshot/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        builds = json.loads((snapshot/'build-map.json').read_text())
        build = Path(builds['astra-sim-wscllm-LRU'])
        legacy_lib = build/'lib/libdie_model_cabi.so'
        coarse_lib = build/'lib/libcoarse_die_cabi.so'
        build_identity_path = root/'work/core-resume-sdk-validation/abi/astra-sim-wscllm-LRU-abi-build/identity.json'
        build_identity = json.loads(build_identity_path.read_text())
        source_files = [producer_path, fixture_path, client_path, coarse_config]
        source_identity = {str(path.relative_to(stage)): SHA(path) for path in source_files}
        checks['four_relevant_sources_match_frozen_snapshot'] = all(manifest['frozen_files_sha256'].get(rel) == sha for rel, sha in source_identity.items())
        checks['fresh_wslru_abi_build_was_successful'] = build_identity['status'] == 'passed'
        library_sha = {str(path): SHA(path) for path in (legacy_lib, coarse_lib)}
        checks['both_libraries_match_fresh_wslru_build_identity'] = all(build_identity['libraries'].get(path) == sha for path, sha in library_sha.items())
        checks['coarse_release_matches_build_identity'] = manifest['public_release']['source_digest'] == build_identity['source_digest']
        checks['original_optional_client_still_absent'] = not (workload/'online/die_model_oracle.py').exists()
        report['identity'] = {'snapshot_manifest': str(manifest_path), 'snapshot_manifest_sha256': SHA(manifest_path),
            'public_release_source_digest': manifest['public_release']['source_digest'],
            'sources_sha256': source_identity, 'libraries_sha256': library_sha,
            'fresh_abi_build_identity': str(build_identity_path), 'fresh_abi_build_identity_sha256': SHA(build_identity_path),
            'actual_producer_path': str(producer_path), 'borrowed_existing_protocol_client_path': str(client_path),
            'coarse_config_path': str(coarse_config)}
        if not all(checks.values()):
            raise AssertionError('source/library prerequisites failed: '+repr(checks))
        # Import the real isolated WSLRU test fixture, which imports its actual
        # DieModelContext and build_die_spec. Do not inject a client into online/.
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(workload))
        fixture = load_module('wslru_original_adapter_fixture', fixture_path)
        client = load_module('existing_face_protocol_client', client_path)
        producer = sys.modules[fixture.build_die_spec.__module__]
        checks['actual_producer_module_loaded'] = Path(producer.__file__).resolve() == producer_path.resolve()
        tree = ast.parse(fixture_path.read_text())
        test_class = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='BuildSpecTests')
        setup = next(node for node in test_class.body if isinstance(node,ast.FunctionDef) and node.name=='setUp')
        write = next(node for node in ast.walk(setup) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text')
        config_value = ast.literal_eval(write.args[0].args[0])
        test = next(node for node in test_class.body if isinstance(node,ast.FunctionDef) and node.name=='test_spec_json_round_trip_through_cost_library')
        comprehension = next(node for node in ast.walk(test) if isinstance(node,ast.ListComp))
        op_kinds = ast.literal_eval(comprehension.generators[0].iter)
        original_counts = {keyword.arg: ast.literal_eval(keyword.value) for keyword in comprehension.elt.keywords}
        checks['exact_original_five_op_r4_fixture'] = (op_kinds == ('norm','qkv_rope','attention_region','ffn_region','logits')
            and original_counts == {'layer_count':2,'weight_passes':2})
        config_path = output/'legacy_original_optional_fixture.json'
        config_path.write_text(json.dumps(config_value,indent=2)+'\n')
        context = fixture._make_context()
        context.die_config_path = config_path
        original_r4 = [fixture.build_die_spec(op, context, context.span_records, **original_counts) for op in op_kinds]
        r1_control = [fixture.build_die_spec(op, context, context.span_records, layer_count=1,weight_passes=1) for op in op_kinds]
        original_wire_sha = WIRE_SHA(original_r4)
        checks['original_repeat_counts_and_spans_preserved'] = all(
            spec['layer_count']==2 and spec['weight_passes']==2
            and spec['phase']=='decode' and spec['tp_slice']['tensor_parallel']==4
            and spec['tp_slice']['tensor_parallel_rank']==1
            and [(s['q_tokens'],s['kv_tokens']) for s in spec['spans']]==[(1,41),(1,42)] for spec in original_r4)
        report['original_optional_test'] = {'qualified_name':'online.test_die_model_adapter.BuildSpecTests.test_spec_json_round_trip_through_cost_library',
            'original_state':'still_skipped_die_model_oracle_unavailable; no source/client added',
            'fixture_source_line':write.lineno,'case_source_line':test.lineno,
            'op_kinds':op_kinds,'layer_count':2,'weight_passes':2,'R':4,
            'original_inputs':original_r4,'original_inputs_canonical_sha256':original_wire_sha,
            'legacy_fixture_config':config_value,'legacy_fixture_config_sha256':SHA(config_path)}
        prior_log = root/'work/core-resume-sdk-validation/python-three/astra-sim-wscllm-LRU-python-consumers/run.log'
        report['original_optional_test'].update(prior_suite_log=str(prior_log),prior_suite_log_sha256=SHA(prior_log),
            prior_suite_summary='63 tests: 62 passed, 1 original optional skipped; not reclassified by this bridge')
        with client.DieModelOracle(lib_path=str(legacy_lib), config_json_path=str(config_path)) as oracle:
            results = oracle.estimate(original_r4)
            errors = list(oracle.last_errors)
            checks['legacy_original_r4_all_five_results'] = not oracle.coarse and all(x is not None for x in results) and errors==[None]*5
            report['legacy_original_r4_protocol_round_trip'] = {'scope':'unchanged original optional test five R4 wires through WSLRU fresh legacy ABI using existing FACE client',
                'client_coarse':oracle.coarse,'hardware_summary':oracle.hardware_summary(),'results':results,'errors':errors}
        with client.DieModelOracle(lib_path=str(coarse_lib), config_json_path=str(coarse_config)) as oracle:
            def raw(specs, cost_only=False):
                # Test-only inspection uses this existing client's already-declared
                # ABI/model/free routine to observe status before its guard filters.
                method = (oracle._lib.die_coarse_estimate_cost_batch_v2 if cost_only else oracle._lib.die_coarse_estimate_batch_v2)
                pointer = method(oracle._model,json.dumps({'specs':specs}).encode())
                if not pointer: raise RuntimeError(oracle._last_error())
                try:return json.loads(ctypes.string_at(pointer))
                finally:oracle._lib.die_coarse_free_result_v2(pointer)
            coarse = report['coarse_separate_controls_and_guard'] = {'scope':'same actual WSLRU producer legacy schema normalized by current public coarse compiler; R1 control is separate from original R4',
                'hardware_summary':oracle.hardware_summary(),'R1_inputs':r1_control,'R4_inputs_canonical_sha256':WIRE_SHA(original_r4)}
            raw_r1, raw_r4 = raw(r1_control),raw(original_r4)
            coarse.update(raw_full_R1=raw_r1,raw_full_original_R4=raw_r4)
            checks['coarse_r1_raw_all_ok_assumed'] = all(row is not None and row['status']=='ok_assumed' for row in raw_r1['results']) and raw_r1['errors']==[None]*5
            checks['coarse_original_r4_raw_all_not_available'] = all(row is not None and row['status']=='not_available' and isinstance(row['estimated_latency_ns'],int) for row in raw_r4['results']) and raw_r4['errors']==[None]*5
            runs = coarse['client_runs'] = {}
            for name, method in [('full',oracle.estimate),('cost',oracle.estimate_cost)]:
                r1 = method(r1_control);e1 = list(oracle.last_errors)
                r4 = method(original_r4);e4 = list(oracle.last_errors)
                runs[name]={'R1_results':r1,'R1_errors':e1,'original_R4_results':r4,'original_R4_errors':e4}
                checks['coarse_'+name+'_r1_accepted'] = all(x is not None and x['status']=='ok_assumed' for x in r1) and e1==[None]*5
                checks['coarse_'+name+'_original_r4_rejected'] = r4==[None]*5 and all(isinstance(error,str) and 'not_available' in error for error in e4)
            coarse['raw_cost_R1'] = raw(r1_control,True)
            coarse['raw_cost_original_R4'] = raw(original_r4,True)
            checks['coarse_cost_r1_equals_full_latency'] = all(a['estimated_latency_ns']==b['estimated_latency_ns'] for a,b in zip(raw_r1['results'],coarse['raw_cost_R1']['results']))
            checks['coarse_cost_original_r4_raw_not_available'] = all(x is not None and x['status']=='not_available' for x in coarse['raw_cost_original_R4']['results'])
            rejects=[]
            for spec in original_r4:
                try:oracle.estimate_one(spec)
                except RuntimeError as error:rejects.append({'op_kind':spec['op_kind'],'rejected':'not_available' in str(error),'error':str(error)})
                else:rejects.append({'op_kind':spec['op_kind'],'rejected':False})
            coarse['estimate_one_original_R4']=rejects
            checks['coarse_estimate_one_rejects_original_r4'] = all(row['rejected'] for row in rejects)
        checks['original_r4_wire_unchanged_after_all_queries'] = original_wire_sha==WIRE_SHA(original_r4)
        checks['relevant_frozen_sources_unchanged'] = all(SHA(stage/rel)==sha for rel,sha in source_identity.items())
        checks['both_fresh_wslru_libraries_unchanged'] = all(SHA(Path(path))==sha for path,sha in library_sha.items())
        report['status'] = 'passed' if all(checks.values()) else 'failed'
    except Exception as error:
        report['status']='failed'
        report['exception']=repr(error)
        report['traceback']=traceback.format_exc()
    report['finished_unix']=time.time()
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':checks,'result':str(output/'result.json')}))
    return int(report['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
