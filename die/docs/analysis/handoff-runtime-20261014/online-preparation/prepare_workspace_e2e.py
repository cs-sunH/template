from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'die/src/Execution.cc').is_file())
REPO = ROOT / 'astra-sim-wscllm'
WORK = ROOT / 'work/wscllm-workspace-e2e'
GEN = WORK / 'generated'
RUNTIME = GEN / 'runtime_config'
QUEUE = WORK / 'request_queue.csv'
TRACE_CONFIG = WORK / 'trace_config.csv'
HARDWARE = WORK / 'hardware_coarse_2rank.json'
SYSTEM_TEMPLATE = WORK / 'system_coarse_template.json'
COARSE_DESCRIPTOR = REPO / 'extern/die_model/coarse/configs/example_hardware.json'
CABI = Path(os.environ.get('DIE_COARSE_CABI_PATH', str(ROOT / 'work/build-die/lib/libcoarse_die_cabi.so')))

sys.path.insert(0, str(REPO / 'sh_test_mesh/workload/llama2_7b_inference'))
sys.path.insert(0, str(REPO / 'sh_test_mesh'))
os.environ['DIE_COARSE_CABI_PATH'] = str(CABI)

import config_resolver  # noqa: E402
import generate_wsc_llm_trace as trace  # noqa: E402
import plan_materializer as materializer  # noqa: E402

if not CABI.is_file():
    raise SystemExit(f'missing coarse C ABI: {CABI}')

WORK.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)

with QUEUE.open('w', newline='', encoding='utf-8') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow((
        'session_id', 'turn_index', 'request_id', 'prefill_length',
        'decode_length', 'session_arrival_time_ns',
        'inter_request_interval_ns', 'description'))
    writer.writerow((
        'workspace_e2e_s0', '0', 'workspace_e2e_s0_r0', '64', '1',
        '0', '', 'synthetic positive-workspace online fixture'))

coarse = json.loads(COARSE_DESCRIPTOR.read_text(encoding='utf-8'))
summary = config_resolver.derive_coarse_die_hardware(coarse)
capacity = int(summary['derived']['hardware_summary']['hbm_capacity_bytes'])

hardware = json.loads((REPO / 'sh_test_mesh/hardware/face_case5_config_c.json')
                      .read_text(encoding='utf-8'))
hardware['slug'] = 'workspace_e2e_coarse_2rank'
hardware['label'] = 'Workspace admission tiny coarse Die E2E'
hardware['mesh'] = {
    'rows': 1,
    'columns': 2,
    'topology-by-network-dimension': ['Line', 'Line'],
}
hardware['local-hbm']['capacity-profiles'] = {
    'workspace-e2e-coarse': {
        'bytes': capacity,
        'label': 'C ABI derived per-rank coarse Die capacity',
        'note': 'Exact capacity returned by the official coarse Die C ABI.',
    }
}
hardware.pop('die', None)
hardware['coarse-die'] = coarse
hardware['notes'] = [
    'Two-rank topology for a minimal online P/D admission fixture.',
    'Per-rank HBM capacity is copied from the official coarse Die C ABI summary.',
    'This fixture uses the production scheduler and graph builder.',
]
HARDWARE.write_text(json.dumps(hardware, indent=2) + '\n', encoding='utf-8')

system = json.loads((REPO / 'sh_test_mesh/system/llama2_7b_die_template.json')
                    .read_text(encoding='utf-8'))
system['die-model-profile'] = 'coarse_die_v2'
SYSTEM_TEMPLATE.write_text(json.dumps(system, indent=2) + '\n', encoding='utf-8')

with TRACE_CONFIG.open('w', newline='', encoding='utf-8') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow((
        'kind', 'key', 'value', 'group_name', 'pg_name', 'ranks',
        'phase_role', 'description'))
    values = (
        ('layers', '1'),
        ('hidden_size', '128'),
        ('ffn_size', '256'),
        ('num_heads', '4'),
        ('vocab_size', '512'),
        ('bytes_per_elem', '2'),
        ('model_name', 'workspace_e2e_tiny_model'),
        ('mlp_variant', 'swiglu'),
        ('request_queue_csv', str(QUEUE)),
        ('request_queue_session_limit', '0'),
        ('trace_granularity', 'request_aggregated'),
        ('prefill_chunk_size', '512'),
        ('kv_cache_policy', 'session_lru_recompute'),
        ('kv_reserve_context_tokens', '0'),
        ('position_encoding', 'none'),
        ('hardware_config', str(HARDWARE)),
        ('local_hbm_capacity_profile', 'workspace-e2e-coarse'),
        ('system_template', str(SYSTEM_TEMPLATE)),
        ('remote_operand_loads', 'false'),
    )
    for key, value in values:
        writer.writerow(('config', key, value, '', '', '', '', 'workspace E2E fixture'))
    writer.writerow(('inference_group', '', '', 'workspace_e2e_prefill', '1',
                     '0', 'prefill', 'single-rank P instance'))
    writer.writerow(('inference_group', '', '', 'workspace_e2e_decode', '2',
                     '1', 'decode', 'single-rank D instance'))

# Run the official config loader and plan materializer implementations while
# redirecting their generated outputs to work/. No production source is edited.
trace.SH_TEST_DIR = GEN
materializer.GENERATED_ROOT = GEN
materializer.load_wsc_llm_trace_config = lambda: trace.load_wsc_llm_trace_config(
    TRACE_CONFIG)
config = trace.load_wsc_llm_trace_config(TRACE_CONFIG)
if config.npus_count != 2 or tuple(group.ranks for group in config.inference_groups) != ((0,), (1,)):
    raise SystemExit('tiny config did not resolve to the intended rank topology')
materializer.main()

plan_dirs = sorted(GEN.glob('llama2_7b_wsc_llm_inference_54npus_plan_*'))
if len(plan_dirs) != 1:
    raise SystemExit(f'expected one official-format plan dir, found {plan_dirs}')
plan_dir = plan_dirs[0]
manifest = json.loads((plan_dir / 'manifest.json').read_text(encoding='utf-8'))
metrics_manifest = json.loads((plan_dir / 'metrics_manifest.json').read_text(encoding='utf-8'))
system_config = json.loads(config.system_config.read_text(encoding='utf-8'))
print(json.dumps({
    'queue': str(QUEUE),
    'trace_config': str(TRACE_CONFIG),
    'plan_dir': str(plan_dir),
    'runtime_config_dir': str(config.system_config.parent),
    'npus_count': config.npus_count,
    'inference_groups': [
        {'phase_role': group.phase_role, 'ranks': list(group.ranks)}
        for group in config.inference_groups],
    'coarse_capacity_bytes_per_rank': capacity,
    'system_die_model_profile': system_config.get('die-model-profile'),
    'manifest_schema_fields': sorted(manifest),
    'metrics_manifest_schema_version': metrics_manifest.get('schema_version'),
    'metrics_manifest_fields': sorted(metrics_manifest),
}, sort_keys=True, indent=2))
