#!/usr/bin/env python3
"""Build and validate a fixed finite LRU workspace-envelope research fixture.

All outputs stay under this script's work/ directory. Production LRU sources
are imported read-only; P shapes come from its real generator primitive and D
subsets from its real scheduler and GraphBatchBuilder methods.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/lru-manifest-envelope-20261014"
INPUTS = WORK / "inputs"
RESULTS = WORK / "results"
LOGS = WORK / "logs"
GENERATED = WORK / "generated"
REPO = ROOT / "astra-sim-wscllm-LRU"
WORKLOAD = REPO / "sh_test_mesh/workload/llama2_7b_inference"
ONLINE = WORKLOAD / "online"
OFFICIAL_PROTO_ROOT = REPO / "work/chakra-python-root"
CABI = ROOT / "die/work/lru-workspace-envelope-20261014/build/lib/libcoarse_die_cabi.so"
BASE_FIXTURE = ROOT / "work/lru-online-workspace-20261014/e2e-large-workspace-probe"
BASE_HARDWARE = BASE_FIXTURE / "hardware_coarse_8rank.json"
BASE_SYSTEM = BASE_FIXTURE / "system_coarse_template.json"
BASE_TRACE = BASE_FIXTURE / "trace_config.csv"

for path in (INPUTS, RESULTS, LOGS, GENERATED):
    path.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False, sort_keys=True) + "\n")


def prepare_fixture() -> tuple[Path, Path, Path, Path]:
    """Copy the official tiny 8-rank coarse fixture and add a legal 2-session queue."""
    for path in (GENERATED, RESULTS, LOGS):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    (GENERATED / "config_root").mkdir(parents=True)

    hardware = INPUTS / "hardware_coarse_8rank.json"
    system = INPUTS / "system_coarse_template.json"
    trace = INPUTS / "trace_config.csv"
    queue = INPUTS / "manifest_queue.csv"
    shutil.copyfile(BASE_HARDWARE, hardware)
    shutil.copyfile(BASE_SYSTEM, system)

    # The existing fixture is a real 8-rank 4x2 P/D layout, MHA4, hidden=512,
    # FFN=1024, bf16, one layer. Only its queue path is redirected here.
    with BASE_TRACE.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0].keys())
    for row in rows:
        if row["kind"] == "config" and row["key"] == "request_queue_csv":
            row["value"] = str(queue.resolve())
        elif row["kind"] == "config" and row["key"] == "hardware_config":
            row["value"] = str(hardware.resolve())
        elif row["kind"] == "config" and row["key"] == "system_template":
            row["value"] = str(system.resolve())
        elif row["kind"] == "config" and row["key"] == "output_dir":
            row["value"] = str((GENERATED / "unused_output_dir").resolve())
    with trace.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Two independent sessions, each with consecutive turns. Their turn-1
    # requests share the same session-turn index and are co-batched on D.
    # Different lengths make the full-run max D=5; each current turn has prior
    # history, and P lengths cross the configured 512-token chunk boundary.
    queue_rows = [
        ("session_manifest_a", 0, "manifest_a_turn0", 64, 2, 0, "", "generated fixture seed turn"),
        ("session_manifest_a", 1, "manifest_a_turn1", 700, 5, "", 1000, "generated fixture history turn"),
        ("session_manifest_b", 0, "manifest_b_turn0", 96, 3, 100, "", "generated fixture seed turn"),
        ("session_manifest_b", 1, "manifest_b_turn1", 900, 3, "", 2000, "generated fixture history turn"),
    ]
    with queue.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow((
            "session_id", "turn_index", "request_id", "prefill_length",
            "decode_length", "session_arrival_time_ns",
            "inter_request_interval_ns", "description"))
        writer.writerows(queue_rows)
    return hardware, system, trace, queue


def configure_imports() -> tuple[Any, Any, dict[str, Any]]:
    """Import official generated protobuf first, then the actual LRU modules."""
    sys.path.insert(0, str(OFFICIAL_PROTO_ROOT))
    proto_module = importlib.import_module(
        "extern.graph_frontend.chakra.schema.protobuf.et_def_pb2")
    proto_path = Path(proto_module.__file__).resolve()
    expected_proto_path = (
        OFFICIAL_PROTO_ROOT
        / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py"
    ).resolve()
    if proto_path != expected_proto_path:
        raise RuntimeError(
            f"Chakra schema did not import from the official work root: {proto_path}")

    sys.path.insert(0, str(REPO / "sh_test_mesh"))
    sys.path.insert(0, str(WORKLOAD))
    sys.path.insert(0, str(ONLINE))

    import google.protobuf  # noqa: E402
    import generate_trace  # noqa: E402
    import generate_wsc_llm_trace as trace_module  # noqa: E402
    import plan_materializer  # noqa: E402
    from online.die_model_adapter import (  # noqa: E402
        OP_KINDS,
        DieModelContext,
        build_die_spec,
    )
    from online.die_workspace_envelope import (  # noqa: E402
        DieWorkspaceEnvelopeClient,
        WorkspaceEnvelopeError,
    )
    from online.graph_batch_builder import (  # noqa: E402
        GraphBatchBuilder,
        _prefix_of,
    )
    from online.wsc_llm_online_scheduler import (  # noqa: E402
        _OnlineRequestRuntime,
        WscLlmOnlineScheduler,
    )

    if getattr(generate_trace, "PROJECT_ROOT", None) != REPO:
        raise RuntimeError(
            f"generate_trace selected an unexpected project root: {generate_trace.PROJECT_ROOT}")
    imports = {
        "chakra_module": proto_module.__name__,
        "chakra_python_file": str(proto_path),
        "protobuf_version": str(google.protobuf.__version__),
        "generate_trace_file": str(Path(generate_trace.__file__).resolve()),
        "trace_config_loader_file": str(Path(trace_module.__file__).resolve()),
        "plan_materializer_file": str(Path(plan_materializer.__file__).resolve()),
        "die_model_adapter_file": str(Path(importlib.import_module("online.die_model_adapter").__file__).resolve()),
        "graph_batch_builder_file": str(Path(importlib.import_module("online.graph_batch_builder").__file__).resolve()),
        "scheduler_file": str(Path(importlib.import_module("online.wsc_llm_online_scheduler").__file__).resolve()),
    }
    modules = {
        "trace": trace_module,
        "materializer": plan_materializer,
        "DieModelContext": DieModelContext,
        "build_die_spec": build_die_spec,
        "OP_KINDS": OP_KINDS,
        "DieWorkspaceEnvelopeClient": DieWorkspaceEnvelopeClient,
        "WorkspaceEnvelopeError": WorkspaceEnvelopeError,
        "GraphBatchBuilder": GraphBatchBuilder,
        "prefix_of": _prefix_of,
        "OnlineRuntime": _OnlineRequestRuntime,
        "Scheduler": WscLlmOnlineScheduler,
        "generate_trace": generate_trace,
    }
    return trace_module, plan_materializer, {"imports": imports, "modules": modules}


def main() -> int:
    if not CABI.is_file():
        raise RuntimeError(f"missing real workspace-envelope C ABI: {CABI}")
    if not OFFICIAL_PROTO_ROOT.is_dir():
        raise RuntimeError(f"missing official generated Chakra Python root: {OFFICIAL_PROTO_ROOT}")

    hardware_path, system_template_path, trace_path, queue_path = prepare_fixture()
    trace_module, materializer, imported = configure_imports()
    modules = imported["modules"]
    os.environ["DIE_COARSE_CABI_PATH"] = str(CABI.resolve())
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Redirect runtime_config outputs to this scratch directory, while keeping
    # all source inputs and production module code unchanged.
    scratch_sh_test = (GENERATED / "config_root").resolve()
    trace_module.SH_TEST_DIR = scratch_sh_test

    config = trace_module.load_wsc_llm_trace_config(trace_path)
    manifest_rows = materializer._derive_manifest_requests(config, {})
    system_runtime = json.loads(config.system_config.read_text(encoding="utf-8"))
    resolved_hardware_path = Path(system_runtime["die-model-config"]).resolve()
    if not resolved_hardware_path.is_file():
        raise RuntimeError(f"resolved hardware config missing: {resolved_hardware_path}")
    manifest = {
        "requests": manifest_rows,
        "compute-cost-model": "die",
        "die-model-config": str(resolved_hardware_path),
        "manifest_source": "generated-fixed-finite-envelope-fixture",
        "repo_variant": "astra-sim-wscllm-LRU",
    }
    write_json(INPUTS / "generated_manifest.json", manifest)
    write_json(INPUTS / "resolved_system.json", system_runtime)

    by_id = {row["request_id"]: row for row in manifest_rows}
    current = [by_id["manifest_a_turn1"], by_id["manifest_b_turn1"]]
    if len(current) != 2 or {row["turn_index"] for row in current} != {1}:
        raise AssertionError("current co-batch must contain two same-turn requests")
    if not all(row["history_tokens_before"] > 0 for row in current):
        raise AssertionError("both co-batched current requests must carry history")
    if not all(row["decode_length"] > 1 for row in current):
        raise AssertionError("the co-batch fixture must exercise D>1")

    from generate_wsc_llm_trace import _emit_prefill_stage  # noqa: E402
    from online.wsc_llm_online_scheduler import _OnlineInstanceState  # noqa: E402
    from wsc_llm_scheduler import DECODE_ROLE  # noqa: E402

    context = modules["DieModelContext"].from_service(manifest, config)
    if context is None:
        raise RuntimeError("official DieModelContext.from_service did not select DIE")
    if context.dtype != "bf16" or context.model_dims["hidden"] != 512 or context.model_dims["ffn"] != 1024:
        raise AssertionError("unexpected model/dtype for the fixed official tiny fixture")

    p_group_index = next(i for i, group in enumerate(config.inference_groups)
                         if group.phase_role == "prefill")
    d_group_index = next(i for i, group in enumerate(config.inference_groups)
                         if group.phase_role == "decode")
    p_group = config.inference_groups[p_group_index]
    d_group = config.inference_groups[d_group_index]
    tp = len(p_group.ranks)
    if tp != 4 or len(d_group.ranks) != 4:
        raise AssertionError("expected actual 2x2 P/D TP=4 groups")

    # Generate every request's P body with the exact shared helper called by
    # GraphBatchBuilder._emit_prelim. This is shape/protobuf emission only; the
    # full P lifecycle (KV transfers and gates) is intentionally out of scope.
    p_graph = modules["GraphBatchBuilder"](config)
    p_graph.set_die_context(context)
    p_graph.begin_batch()
    p_spans_by_request: dict[str, list[tuple[int, int]]] = {}
    for record in manifest_rows:
        plan = dict(record)
        plan["prefill_instance_index"] = p_group_index
        plan["decode_instance_index"] = d_group_index
        p_graph._set_context(plan, "prefill", 0)
        marker = p_graph._mark()
        prefix = modules["prefix_of"](plan)
        _emit_prefill_stage(
            config=config,
            builders=p_graph.builders,
            group=p_group,
            prefix=prefix,
            stage="current_prefill",
            tokens=int(record["prefill_length"]),
            initial_context_tokens=int(record["history_tokens_before"]),
            die_ctx=context,
        )
        p_graph._collect(marker)
        rank_specs: dict[int, dict[str, Any]] = {}
        for node in p_graph.batch["nodes"]:
            compute = node.get("compute", {})
            spec = compute.get("die_op_spec") if isinstance(compute, dict) else None
            if spec and node["rank"] in p_group.ranks:
                rank_specs.setdefault(int(node["rank"]), spec)
        if set(rank_specs) != set(p_group.ranks):
            raise RuntimeError(f"actual P generator emitted incomplete rank specs for {record['request_id']}")
        exemplar = next(iter(rank_specs.values()))
        p_spans_by_request[record["request_id"]] = [
            (int(span["q_tokens"]), int(span["kv_tokens"]))
            for span in exemplar["spans"]
        ]
        if int(exemplar["weight_passes"]) != len(exemplar["spans"]):
            raise AssertionError("P weight_passes must equal actual chunk count")

    p_nodes = list(p_graph.batch["nodes"])
    p_emitted_specs = []
    for node in p_nodes:
        compute = node.get("compute", {})
        spec = compute.get("die_op_spec") if isinstance(compute, dict) else None
        if spec:
            p_emitted_specs.append({"node_name": node["name"], "rank": int(node["rank"]), "spec": spec})
    if not p_emitted_specs:
        raise RuntimeError("actual P GraphBatchBuilder helper emitted no die_op_spec")

    # The conservative fixed E is the per-(phase, op_kind, TP rank) maximum
    # over every manifest request's full-chunk P task and the compact union of
    # every request's complete remaining D run. No workspace arithmetic here.
    op_kinds = sorted(modules["OP_KINDS"])
    if "local_region" not in op_kinds or len(op_kinds) != 15:
        raise AssertionError(f"adapter op-kind vocabulary changed: {op_kinds}")
    legal_ops = [op for op in op_kinds if op != "local_region"]
    if len(legal_ops) != 14:
        raise AssertionError("expected fourteen legal op kinds plus local_region rejection")
    e_rows: list[dict[str, Any]] = []
    expected_rejections: list[dict[str, Any]] = []
    all_queries: list[dict[str, Any]] = []
    envelope_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    workspace_client = modules["DieWorkspaceEnvelopeClient"](
        resolved_hardware_path, CABI)

    try:
        hardware_fingerprint = workspace_client.hardware_fingerprint
        for request_id, pass_spans in p_spans_by_request.items():
            request = by_id[request_id]
            namespace = f"manifest_{request_id}_prefill"
            for relative_rank, physical_rank in enumerate(p_group.ranks):
                context.begin_emission(
                    phase="prefill", tensor_parallel=tp,
                    tensor_parallel_rank=relative_rank,
                    span_namespace=namespace, pass_spans=pass_spans)
                for op_kind in legal_ops:
                    fusion = "auto" if op_kind in ("attention_region", "ffn_region") else "none"
                    spec = context.build_spec(
                        op_kind, layer_count=config.layers,
                        weight_passes=len(pass_spans), fusion_policy=fusion)
                    envelope = workspace_client.query(spec)
                    key = ("prefill", op_kind, relative_rank)
                    row = {
                        "cohort": "E-prefill-full-chunks",
                        "request_id": request_id,
                        "phase": "prefill",
                        "op_kind": op_kind,
                        "physical_rank": int(physical_rank),
                        "tensor_parallel_rank": relative_rank,
                        "task_spec": spec,
                        "normalized_task": envelope.normalized_task,
                        "normalized_task_fingerprint": envelope.normalized_task_fingerprint,
                        "hardware_fingerprint": envelope.hardware_fingerprint,
                        "candidate_count": envelope.candidate_count,
                        "max_hbm_workspace_peak_bytes": envelope.max_hbm_workspace_peak_bytes,
                    }
                    all_queries.append(row)
                    previous = envelope_by_key.get(key)
                    if previous is None or envelope.max_hbm_workspace_peak_bytes > previous["max_hbm_workspace_peak_bytes"]:
                        envelope_by_key[key] = {
                            **row,
                            "max_hbm_workspace_peak_bytes": envelope.max_hbm_workspace_peak_bytes,
                        }
        print(f"E prefill complete: {len(all_queries)} real C ABI queries", flush=True)

        # All manifest requests are compacted into their complete remaining D
        # runs. Weight passes are the max manifest D length (same full-train
        # rule as scheduler._plan_train), not the selected subset's 1 pass.
        d_spans = []
        expanded_d_passes = []
        for record in manifest_rows:
            count = int(record["decode_length"])
            context_tokens = int(record["prefill_context_tokens"])
            if count <= 0:
                raise AssertionError("manifest D lengths must be positive")
            d_spans.append({
                "request_id_prefix": str(record["request_id"]),
                "first_index": 0,
                "run": {
                    "q_tokens": 1,
                    "kv_first": context_tokens + 1,
                    "kv_stride": 1,
                    "count": count,
                },
            })
            expanded_d_passes.extend((1, context_tokens + step)
                                     for step in range(1, count + 1))
        full_manifest_d_passes = max(int(r["decode_length"]) for r in manifest_rows)
        for relative_rank, physical_rank in enumerate(d_group.ranks):
            context.begin_emission(
                phase="decode", tensor_parallel=len(d_group.ranks),
                tensor_parallel_rank=relative_rank,
                span_namespace="manifest-wide-full-decode",
                pass_spans=expanded_d_passes)
            for op_kind in legal_ops:
                fusion = "auto" if op_kind in ("attention_region", "ffn_region") else "none"
                spec = modules["build_die_spec"](
                    op_kind, context, d_spans,
                    layer_count=config.layers,
                    weight_passes=full_manifest_d_passes,
                    fusion_policy=fusion,
                )
                envelope = workspace_client.query(spec)
                key = ("decode", op_kind, relative_rank)
                row = {
                    "cohort": "E-decode-manifest-run-union",
                    "request_ids": [str(r["request_id"]) for r in manifest_rows],
                    "phase": "decode",
                    "op_kind": op_kind,
                    "physical_rank": int(physical_rank),
                    "tensor_parallel_rank": relative_rank,
                    "weight_passes": full_manifest_d_passes,
                    "task_spec": spec,
                    "normalized_task": envelope.normalized_task,
                    "normalized_task_fingerprint": envelope.normalized_task_fingerprint,
                    "hardware_fingerprint": envelope.hardware_fingerprint,
                    "candidate_count": envelope.candidate_count,
                    "max_hbm_workspace_peak_bytes": envelope.max_hbm_workspace_peak_bytes,
                }
                all_queries.append(row)
                envelope_by_key[("decode", op_kind, relative_rank)] = {
                    **row,
                    "max_hbm_workspace_peak_bytes": envelope.max_hbm_workspace_peak_bytes,
                }

        # local_region is part of the legacy Python enum but must remain
        # fail-closed because v1 supplies no typed mathematical graph.
        for phase, relative_rank, physical_rank, spans, weight_passes in (
            ("prefill", 0, int(p_group.ranks[0]), p_spans_by_request[manifest_rows[-1]["request_id"]], len(p_spans_by_request[manifest_rows[-1]["request_id"]])),
            ("decode", 0, int(d_group.ranks[0]), expanded_d_passes, full_manifest_d_passes),
        ):
            context.begin_emission(
                phase=phase,
                tensor_parallel=tp,
                tensor_parallel_rank=relative_rank,
                span_namespace="expected-local-region-rejection",
                pass_spans=spans,
            )
            spec = context.build_spec(
                "local_region", layer_count=config.layers,
                weight_passes=weight_passes, fusion_policy="none")
            try:
                workspace_client.query(spec)
            except modules["WorkspaceEnvelopeError"] as error:
                message = str(error)
                if "local_region" not in message:
                    raise
                expected_rejections.append({
                    "phase": phase,
                    "physical_rank": physical_rank,
                    "op_kind": "local_region",
                    "expected": "reject unsupported legacy local_region",
                    "observed_error": message,
                })
            else:
                raise AssertionError("local_region was unexpectedly accepted")

        # Validate finite inclusion monotonicity at the envelope-set level and
        # one explicit count=1 <= count=maxD C ABI probe for every D op/rank.
        monotonicity_checks = []
        for relative_rank, physical_rank in enumerate(d_group.ranks):
            for op_kind in legal_ops:
                one_token_spans = [
                    {"request_id_prefix": str(r["request_id"]),
                     "first_index": 0,
                     "run": {"q_tokens": 1,
                             "kv_first": int(r["prefill_context_tokens"]) + 1,
                             "kv_stride": 1,
                             "count": 1}}
                    for r in manifest_rows
                ]
                context.begin_emission(
                    phase="decode", tensor_parallel=len(d_group.ranks),
                    tensor_parallel_rank=relative_rank,
                    span_namespace="manifest-wide-one-decode-token",
                    pass_spans=[(1, int(r["prefill_context_tokens"]) + 1)
                                for r in manifest_rows])
                fusion = "auto" if op_kind in ("attention_region", "ffn_region") else "none"
                one_spec = modules["build_die_spec"](
                    op_kind, context, one_token_spans,
                    layer_count=config.layers, weight_passes=1,
                    fusion_policy=fusion)
                one_env = workspace_client.query(one_spec)
                full_env = envelope_by_key[("decode", op_kind, relative_rank)]
                full_bytes = int(full_env["max_hbm_workspace_peak_bytes"])
                ok = one_env.max_hbm_workspace_peak_bytes <= full_bytes
                monotonicity_checks.append({
                    "condition": "same fixed manifest D run union, count=1 per request <= complete count per request",
                    "op_kind": op_kind,
                    "tensor_parallel_rank": relative_rank,
                    "physical_rank": int(physical_rank),
                    "one_token_bytes": one_env.max_hbm_workspace_peak_bytes,
                    "full_run_bytes": full_bytes,
                    "pass": ok,
                })
                if not ok:
                    raise AssertionError(f"observed D count monotonicity failed for {op_kind}/rank{relative_rank}")

        # Query the exact P node specs produced above; they must be covered by
        # the matching finite E row for that real op/rank.
        actual_checks: list[dict[str, Any]] = []
        for item in p_emitted_specs:
            spec = item["spec"]
            envelope = workspace_client.query(spec)
            key = ("prefill", str(spec["op_kind"]), int(spec["tp_slice"]["tensor_parallel_rank"]))
            bound = envelope_by_key[key]["max_hbm_workspace_peak_bytes"]
            passed = envelope.max_hbm_workspace_peak_bytes <= bound
            actual_checks.append({
                "source": "GraphBatchBuilder._emit_prelim shared P body helper",
                "node_name": item["node_name"],
                "physical_rank": item["rank"],
                "phase": envelope.phase,
                "op_kind": envelope.op_kind,
                "normalized_task_fingerprint": envelope.normalized_task_fingerprint,
                "candidate_count": envelope.candidate_count,
                "actual_bytes": envelope.max_hbm_workspace_peak_bytes,
                "E_bytes": bound,
                "pass": passed,
            })
            if not passed:
                raise AssertionError(f"actual P task exceeded E: {item['node_name']}")

        # Make two real runtime ledgers for the history-bearing turn-1 requests,
        # then drive the production _plan_train and GraphBatchBuilder methods.
        def make_scheduler(train_max_iter: int, split: bool):
            os.environ["SH_TRAIN_MAX_ITER"] = str(train_max_iter)
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "1" if split else "0"
            graph = modules["GraphBatchBuilder"](config)
            scheduler = modules["Scheduler"](
                manifest=manifest, config=config, graph=graph,
                mode="strategy", workspace_admission="off")
            scheduler._begin_graph_batch()
            state = scheduler.instances[d_group_index]
            runtimes = [modules["OnlineRuntime"](dict(row)) for row in current]
            for runtime in runtimes:
                runtime.prefill_instance_index = p_group_index
                runtime.decode_instance_index = d_group_index
                state.active_decode.append(runtime)
                state.active_decode_lookup.add(runtime)
                scheduler.runtime_by_request_id[runtime.request_id] = runtime
            return scheduler, state, runtimes

        def train_dict(plan: dict, state: Any, exit_members: list[Any], first_token: Any = None) -> dict:
            result = {
                "train_id": plan["train_id"],
                "instance_index": state.index,
                "joiners": [],
                "pass_spans": list(plan["pass_spans"]),
                "iterations": int(plan["iterations"]),
                "sentinel": bool(plan["sentinel"]),
                "exit_members": [{"request_id": r.request_id, "session_id": r.session_id}
                                 for r in exit_members],
            }
            if first_token is not None:
                result["first_token"] = first_token
            return result

        def audit_graph_specs(graph: Any, source: str) -> list[dict[str, Any]]:
            checked = []
            for node in graph.batch["nodes"]:
                compute = node.get("compute", {})
                spec = compute.get("die_op_spec") if isinstance(compute, dict) else None
                if not spec:
                    continue
                envelope = workspace_client.query(spec)
                tp_rank = int(spec["tp_slice"]["tensor_parallel_rank"])
                key = ("decode", str(spec["op_kind"]), tp_rank)
                bound = envelope_by_key[key]["max_hbm_workspace_peak_bytes"]
                passed = envelope.max_hbm_workspace_peak_bytes <= bound
                checked.append({
                    "source": source,
                    "node_name": node["name"],
                    "physical_rank": int(node["rank"]),
                    "phase": envelope.phase,
                    "op_kind": envelope.op_kind,
                    "normalized_task_fingerprint": envelope.normalized_task_fingerprint,
                    "candidate_count": envelope.candidate_count,
                    "actual_bytes": envelope.max_hbm_workspace_peak_bytes,
                    "E_bytes": bound,
                    "pass": passed,
                })
                if not passed:
                    raise AssertionError(f"actual D subset exceeded E: {source}:{node['name']}")
            return checked

        train_scenarios = []
        # Uncapped full run and capped train are separate production scheduler
        # instances so no mutable runtime/train sequence state is shared.
        for name, cap in (("uncapped", 0), ("capped", 2)):
            scheduler, state, runtimes = make_scheduler(cap, False)
            plan = scheduler._plan_train(state)
            if plan is None:
                raise AssertionError(f"scheduler returned no train for {name}")
            graph_plan = train_dict(plan, state, plan["exit_members"])
            scheduler.graph.emit_iteration_train(graph_plan)
            checked = audit_graph_specs(scheduler.graph, f"scheduler._plan_train/{name}/emit_iteration_train")
            if not checked:
                raise AssertionError(f"no actual D specs emitted for {name}")
            if name == "uncapped" and int(plan["iterations"]) != max(r.decode_length for r in runtimes):
                raise AssertionError("uncapped train did not reach full remaining D max")
            if name == "capped" and int(plan["iterations"]) != 2:
                raise AssertionError("capped train did not honor SH_TRAIN_MAX_ITER=2")
            train_scenarios.append({
                "name": name,
                "SH_TRAIN_MAX_ITER": cap,
                "SH_FIRST_TOKEN_SPLIT": "0",
                "train_plan": {
                    "train_id": plan["train_id"],
                    "iterations": plan["iterations"],
                    "members": [{"request_id": r.request_id, "participating_iterations": n}
                                for r, n in plan["members"]],
                    "pass_spans": plan["pass_spans"],
                    "sentinel": plan["sentinel"],
                    "exit_set": sorted(plan["exit_set"]),
                },
                "actual_task_checks": checked,
                "positive_workspace_bytes": max(row["actual_bytes"] for row in checked),
            })

        # Actual WP9 split emitter on both an uncapped and a capped _plan_train
        # result. The two subsets (first and remainder) are compiled separately.
        for name, cap in (("split_uncapped", 0), ("split_capped", 2)):
            scheduler, state, runtimes = make_scheduler(cap, True)
            plan = scheduler._plan_train(state)
            first_token = scheduler._first_token_plan(plan, runtimes)
            if not first_token or not first_token.get("split"):
                raise AssertionError(f"production first-token planner did not split {name}")
            graph_plan = train_dict(plan, state, plan["exit_members"], first_token)
            scheduler.graph.emit_train_first_step(graph_plan)
            first_checks = audit_graph_specs(scheduler.graph, f"{name}/emit_train_first_step")
            scheduler.graph.emit_train_remainder(graph_plan)
            remainder_checks = audit_graph_specs(scheduler.graph, f"{name}/emit_train_remainder")
            if not first_checks or not remainder_checks:
                raise AssertionError(f"split emitter omitted a compute subset for {name}")
            train_scenarios.append({
                "name": name,
                "SH_TRAIN_MAX_ITER": cap,
                "SH_FIRST_TOKEN_SPLIT": "1",
                "train_plan": {
                    "train_id": plan["train_id"],
                    "iterations": plan["iterations"],
                    "members": [{"request_id": r.request_id, "participating_iterations": n}
                                for r, n in plan["members"]],
                    "pass_spans": plan["pass_spans"],
                    "first_token_plan": first_token,
                    "sentinel": plan["sentinel"],
                    "exit_set": sorted(plan["exit_set"]),
                },
                "actual_task_checks": first_checks + remainder_checks,
                "first_step_task_checks": first_checks,
                "remainder_task_checks": remainder_checks,
                "positive_workspace_bytes": max(row["actual_bytes"] for row in first_checks + remainder_checks),
            })

        if not actual_checks or max(row["actual_bytes"] for row in actual_checks) <= 0:
            raise AssertionError("actual P co-batch shape query had no positive workspace")
        if not train_scenarios or max(s["positive_workspace_bytes"] for s in train_scenarios) <= 0:
            raise AssertionError("two-request history-bearing D co-batch has no positive workspace")

        for key, value in sorted(envelope_by_key.items()):
            phase, op_kind, rank = key
            e_rows.append({
                "phase": phase,
                "op_kind": op_kind,
                "tensor_parallel_rank": rank,
                "physical_rank": value["physical_rank"],
                "E_hbm_workspace_peak_bytes": value["max_hbm_workspace_peak_bytes"],
                "candidate_count_at_max_task": value["candidate_count"],
                "max_task_fingerprint": value["normalized_task_fingerprint"],
                "max_task_request_ids": value.get("request_ids", [value.get("request_id")]),
            })
        if len(e_rows) != 2 * len(legal_ops) * 4:
            raise AssertionError(f"finite E table incomplete: {len(e_rows)} rows")

        write_jsonl(RESULTS / "envelope_cabi_queries.jsonl", all_queries)
        write_jsonl(RESULTS / "actual_p_checks.jsonl", actual_checks)
        write_jsonl(RESULTS / "expected_rejections.jsonl", expected_rejections)
        write_jsonl(RESULTS / "finite_E.jsonl", e_rows)
        write_json(RESULTS / "actual_train_subset_checks.json", {
            "scope": "compile-only research validation; does not activate manager admission or set workspace_required",
            "model": {
                "name": config.model_name,
                "dtype": context.dtype,
                "layers": config.layers,
                "hidden": config.hidden_size,
                "ffn": config.ffn_size,
                "heads": config.num_heads,
                "head_dim": context.model_dims["head_dim"],
                "mlp_variant": config.mlp_variant,
                "position_encoding": config.position_encoding,
            },
            "current_cohort": current,
            "train_scenarios": train_scenarios,
            "all_actual_p_specs_checked": len(actual_checks),
            "all_actual_train_specs_checked": sum(len(s["actual_task_checks"]) for s in train_scenarios),
        })
        write_json(RESULTS / "finite_monotonicity.json", {
            "scope": "finite observed identities only; no universal monotonicity theorem",
            "structural_rule": "E is the max of a fixed, explicitly enumerated finite set; adding another set member cannot reduce the set maximum by definition",
            "observed_d_count_checks": monotonicity_checks,
            "all_pass": all(item["pass"] for item in monotonicity_checks),
            "limitations": [
                "No claim covers requests, contexts, chunk sizes, layer/pass counts, TP layouts, dtypes, hardware bytes, execution profiles, or op kinds outside this fixture.",
                "The lower-count check only compares count=1 to this fixture's full per-request D run on the same ranks; it does not establish monotonicity for arbitrary span shapes.",
                "P E covers each manifest request's actual full-chunk body shape; it does not include full lifecycle KV transfer/gate workspaces, which carry no DIE compute specs in this adapter.",
                "D E is a finite union across all manifest requests, including earlier completed turns; this is a conservative shape union, not a measured single runtime co-batch.",
            ],
        })
    finally:
        workspace_client.close()

    # Re-read outputs for completion checks and capture immutable identities.
    source_paths = [
        WORKLOAD / "generate_trace.py",
        WORKLOAD / "generate_wsc_llm_trace.py",
        WORKLOAD / "plan_materializer.py",
        REPO / "sh_test_mesh/config_resolver.py",
        ONLINE / "die_model_adapter.py",
        ONLINE / "die_workspace_envelope.py",
        ONLINE / "graph_batch_builder.py",
        ONLINE / "online_scheduler_base.py",
        ONLINE / "wsc_llm_online_scheduler.py",
        OFFICIAL_PROTO_ROOT / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py",
        Path(__file__).resolve(),
        hardware_path,
        system_template_path,
        trace_path,
        queue_path,
        INPUTS / "generated_manifest.json",
        resolved_hardware_path,
        config.system_config,
        CABI,
    ]
    source_hashes = []
    for path in source_paths:
        source_hashes.append({"path": str(path.resolve()), "sha256": sha256_path(path), "bytes": path.stat().st_size})
    git_head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    git_branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True)
    abi_hash = sha256_path(CABI)
    identity = {
        "repository": str(ROOT),
        "branch": git_branch,
        "head": git_head,
        "dirty_status_at_capture": git_status.splitlines(),
        "fixture_classification": "generated legal fixed finite fixture, not measured online evidence",
        "provenance": {
            **imported["imports"],
            "source_file_sha256": source_hashes,
            "cabi_library": str(CABI.resolve()),
            "cabi_sha256": abi_hash,
            "hardware_file_sha256": sha256_path(resolved_hardware_path),
            "hardware_fingerprint": hardware_fingerprint,
            "die_context_config_hash": context.config_hash,
            "manifest_sha256": sha256_path(INPUTS / "generated_manifest.json"),
            "queue_sha256": sha256_path(queue_path),
            "trace_config_sha256": sha256_path(trace_path),
            "hardware_fixture_sha256": sha256_path(hardware_path),
            "system_template_sha256": sha256_path(system_template_path),
            "runtime_system_sha256": sha256_path(config.system_config),
            "source_control_identity_is_commit_only": False,
        },
        "fixture_topology": {
            "npus_count": config.npus_count,
            "mesh_rows": config.hardware.mesh_rows,
            "mesh_cols": config.hardware.mesh_cols,
            "p_group_index": p_group_index,
            "p_ranks": list(p_group.ranks),
            "d_group_index": d_group_index,
            "d_ranks": list(d_group.ranks),
            "tp_degree": tp,
        },
        "finite_scope": {
            "requests": manifest_rows,
            "current_cohort_request_ids": [r["request_id"] for r in current],
            "num_legal_op_kinds": len(legal_ops),
            "local_region": "explicitly rejected by C ABI",
            "envelope_key_count": len(e_rows),
            "all_real_cabi_queries": len(all_queries),
            "positive_current_d_co_batch_workspace_bytes": max(s["positive_workspace_bytes"] for s in train_scenarios),
            "workspace_admission_activated": False,
            "workspace_required_set": False,
        },
    }
    write_json(RESULTS / "source_hardware_identity.json", identity)
    write_json(RESULTS / "run_summary.json", {
        "status": "PASS",
        "fixture_classification": "generated fixture, not measured online",
        "request_count": len(manifest_rows),
        "current_same_turn_history_requests": [r["request_id"] for r in current],
        "d_gt_1": True,
        "p_prefill_tasks_per_request": {rid: len(spans) for rid, spans in p_spans_by_request.items()},
        "p_actual_specs_passed": len(actual_checks),
        "actual_train_scenarios": [s["name"] for s in train_scenarios],
        "actual_train_specs_passed": sum(len(s["actual_task_checks"]) for s in train_scenarios),
        "E_keys": len(e_rows),
        "finite_E_query_count": len(all_queries),
        "local_region_expected_rejections": len(expected_rejections),
        "finite_monotonicity_checks": len(monotonicity_checks),
        "all_finite_monotonicity_checks_pass": all(item["pass"] for item in monotonicity_checks),
        "CABI_sha256": abi_hash,
        "hardware_fingerprint": hardware_fingerprint,
    })
    print(json.dumps({"status": "PASS", "work": str(WORK), **json.loads((RESULTS / "run_summary.json").read_text())}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
