#!/usr/bin/env python3
"""Extract and validate the real FIFO-file bridge traffic from strace logs."""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parent / "run")
BRIDGE = RUN / "bridge"
WIRE = RUN / "wire_audit"
WIRE.mkdir(parents=True, exist_ok=True)


WRITE = re.compile(
    r"write\(\d+<([^>]*)>, (\"(?:[^\"\\]|\\.)*\"), (\d+)")


def extract_temp_writes(trace_path: Path, prefix: str) -> dict[int, bytes]:
    fragments: dict[int, list[bytes]] = defaultdict(list)
    lengths: dict[int, list[int]] = defaultdict(list)
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WRITE.search(line)
        if not match:
            continue
        path_text, quoted, declared = match.groups()
        match_name = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.json\.tmp",
                                  Path(path_text).name)
        if not match_name:
            continue
        sequence = int(match_name.group(1))
        value = ast.literal_eval(quoted).encode("utf-8")
        fragments[sequence].append(value)
        lengths[sequence].append(int(declared))

    output: dict[int, bytes] = {}
    for sequence, chunks in fragments.items():
        payload = b"".join(chunks)
        if len(payload) != sum(lengths[sequence]):
            raise RuntimeError(
                f"strace byte count mismatch for {prefix}_{sequence}: "
                f"captured {len(payload)}, declared {sum(lengths[sequence])}")
        output[sequence] = payload
        name = {"request": "request", "response": "response",
                "commit_ack": "ack"}[prefix]
        (WIRE / f"{name}_{sequence}.json").write_bytes(payload)
    return output


def parse_json(payload: bytes, label: str):
    try:
        return json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"invalid captured {label}: {exc}") from exc


def uint(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


errors: list[str] = []
cpp_requests = extract_temp_writes(RUN / "cpp.strace", "request")
responses = extract_temp_writes(RUN / "python.strace", "response")
acks = extract_temp_writes(RUN / "cpp.strace", "commit_ack")

request_journal: list[dict] = []
journal_path = BRIDGE / "request_journal.jsonl"
if journal_path.exists():
    for line_number, line in enumerate(
            journal_path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            request_journal.append(parse_json(
                line.encode("utf-8"), f"request_journal line {line_number}"))

request_records: dict[int, dict] = {}
for sequence, payload in cpp_requests.items():
    request_records[sequence] = parse_json(payload, f"request_{sequence}")
for sequence, record in enumerate(request_journal):
    if sequence in request_records and request_records[sequence] != record:
        errors.append(f"request {sequence}: C++ wire bytes disagree with Python journal")
    request_records.setdefault(sequence, record)

response_records = {seq: parse_json(data, f"response_{seq}")
                    for seq, data in responses.items()}
ack_records = {seq: parse_json(data, f"ack_{seq}") for seq, data in acks.items()}

queue_path = RUN.parent / "workspace_e2e_request_queue_recompute.csv"
queue_provenance_path = Path(str(queue_path) + ".provenance.json")
queue_rows = max(0, sum(1 for line in queue_path.read_text(
    encoding="utf-8").splitlines() if line.strip()) - 1)
queue_provenance = json.loads(queue_provenance_path.read_text(encoding="utf-8"))

requests_out = []
for sequence in sorted(request_records):
    request = request_records[sequence]
    workspace = request.get("workspace_state")
    ranks = workspace.get("ranks") if isinstance(workspace, dict) else None
    expected_ranks = list(range(len(ranks))) if isinstance(ranks, list) else []
    if request.get("schema_version") != 2:
        errors.append(f"request {sequence}: schema_version is not 2")
    if request.get("delivery_sequence") != sequence:
        errors.append(f"request {sequence}: delivery_sequence mismatch")
    if not isinstance(workspace, dict) or workspace.get("protocol_version") != 1:
        errors.append(f"request {sequence}: missing workspace protocol v1")
    if not isinstance(ranks, list) or [item.get("rank") for item in ranks] != expected_ranks:
        errors.append(f"request {sequence}: workspace ranks are not complete ascending ranks")
    for item in ranks or []:
        if set(item) != {"rank", "workspace_revision", "active_workspace_bytes"}:
            errors.append(f"request {sequence}: workspace rank fields differ from schema")
        if not all(uint(item.get(key)) for key in (
                "rank", "workspace_revision", "active_workspace_bytes")):
            errors.append(f"request {sequence}: workspace rank field is not uint64-like")

    arrivals = request.get("arrivals", [])
    completed_groups = request.get("completed_groups", [])
    requests_out.append({
        "delivery_sequence": sequence,
        "schema_version": request.get("schema_version"),
        "reasons": request.get("reasons"),
        "tick": request.get("tick"),
        "arrivals": [{key: arrival.get(key) for key in (
            "queue_index", "request_id", "session_id", "turn_index",
            "prefill_length", "decode_length", "arrival_world_ns")}
            for arrival in arrivals],
        "completed_groups": [{key: group.get(key) for key in (
            "request_id", "stage", "generation", "node_count")}
            for group in completed_groups],
        "workspace_state": workspace,
    })

responses_out = []
for sequence in sorted(response_records):
    response = response_records[sequence]
    request = request_records.get(sequence, {})
    request_state = request.get("workspace_state", {}).get("ranks", [])
    state_by_rank = {entry.get("rank"): entry for entry in request_state}
    snapshot = response.get("workspace_capacity_snapshot")
    rank_rows = snapshot.get("ranks") if isinstance(snapshot, dict) else None
    if response.get("schema_version") != 2:
        errors.append(f"response {sequence}: schema_version is not 2")
    if response.get("source_delivery_sequence") != sequence or response.get("batch_id") != sequence:
        errors.append(f"response {sequence}: sequence/batch binding mismatch")
    if not isinstance(snapshot, dict) or snapshot.get("protocol_version") != 1:
        errors.append(f"response {sequence}: missing workspace capacity protocol v1")
    if not isinstance(rank_rows, list) or [item.get("rank") for item in rank_rows] != list(range(len(state_by_rank))):
        errors.append(f"response {sequence}: capacity snapshot does not cover ascending ranks")
    workspace_fit = []
    for item in rank_rows or []:
        if set(item) != {"rank", "workspace_revision", "capacity_bytes",
                         "model_weight_bytes", "resident_kv_bytes",
                         "reserved_request_bytes", "prospective_workspace_bytes"}:
            errors.append(f"response {sequence}: capacity rank fields differ from schema")
        numeric = ("rank", "workspace_revision", "capacity_bytes",
                   "model_weight_bytes", "resident_kv_bytes",
                   "reserved_request_bytes", "prospective_workspace_bytes")
        if not all(uint(item.get(key)) for key in numeric):
            errors.append(f"response {sequence}: capacity field is not uint64-like")
            continue
        request_rank = state_by_rank.get(item["rank"])
        if request_rank is None or request_rank.get("workspace_revision") != item["workspace_revision"]:
            errors.append(f"response {sequence}: rank {item['rank']} revision does not echo request")
        base = (item["model_weight_bytes"] + item["resident_kv_bytes"] +
                item["reserved_request_bytes"])
        active = request_rank.get("active_workspace_bytes", 0) if request_rank else 0
        future = item.get("prospective_workspace_bytes", 0)
        fits = base <= item["capacity_bytes"] and max(active, future) <= item["capacity_bytes"] - base
        if not fits:
            errors.append(f"response {sequence}: rank {item['rank']} base+workspace exceeds capacity")
        workspace_fit.append({
            "rank": item["rank"],
            "workspace_revision": item["workspace_revision"],
            "base_used_bytes": base,
            "active_workspace_bytes": active,
            "prospective_workspace_bytes": future,
            "capacity_bytes": item["capacity_bytes"],
            "fits": fits,
        })
    nodes = response.get("nodes", [])
    compute_nodes = [node for node in nodes
                     if node.get("type") == 4
                     and node.get("compute", {}).get("op_role") != "control_marker"
                     and "die_op_spec" in node.get("compute", {})]
    fused = sorted({node["compute"]["die_op_spec"].get("op_kind")
                    for node in compute_nodes
                    if node["compute"]["die_op_spec"].get("fusion_policy") == "auto"
                    and node["compute"]["die_op_spec"].get("op_kind") in {
                        "attention_region", "ffn_region"}})
    first_token = [node.get("name") for node in nodes
                   if "first_token" in node.get("name", "")]
    responses_out.append({
        "delivery_sequence": sequence,
        "schema_version": response.get("schema_version"),
        "batch_id": response.get("batch_id"),
        "source_delivery_sequence": response.get("source_delivery_sequence"),
        "node_count": len(nodes),
        "typed_comp_count": len(compute_nodes),
        "fused_regions": fused,
        "first_token_nodes": first_token,
        "workspace_capacity_snapshot": snapshot,
        "workspace_fit": workspace_fit,
    })

acks_out = []
for sequence in sorted(ack_records):
    ack = ack_records[sequence]
    if ack.get("schema_version") != 2 or ack.get("delivery_sequence") != sequence or ack.get("batch_id") != sequence or ack.get("success") is not True:
        errors.append(f"ack {sequence}: schema2 sequence binding or success=true failed")
    acks_out.append(ack)

cpp_log = (RUN / "cpp.log").read_text(encoding="utf-8", errors="replace")
python_log = (RUN / "python.log").read_text(encoding="utf-8", errors="replace")
exit_codes = {}
exit_path = RUN / "exit_codes.txt"
if exit_path.exists():
    for line in exit_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            exit_codes[key] = int(value)

report = {
    "run_dir": str(RUN),
    "exit_codes": exit_codes,
    "queue_rows": queue_rows,
    "queue_provenance": queue_provenance,
    "provenance_gate_matched": "provenance sidecar: matched" in cpp_log,
    "requests": requests_out,
    "responses": responses_out,
    "acks": acks_out,
    "actual_request_complete_observed": any(
        "REQUEST_COMPLETE" in record.get("reasons", []) for record in request_records.values()),
    "cpp_failure_excerpt": [line for line in cpp_log.splitlines()
                             if "[Error]" in line or "aborted" in line.lower()],
    "python_failure_excerpt": [line for line in python_log.splitlines()
                                if "fatal:" in line or "Traceback" in line],
    "protocol_errors": errors,
    "protocol_schema_valid": not errors,
}
(WIRE / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
