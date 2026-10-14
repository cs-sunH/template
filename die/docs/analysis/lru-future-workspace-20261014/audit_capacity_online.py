"""Audit the actual four-request Online capacity/graph capture; no timing model."""
import hashlib
import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path

WORK = Path(__file__).resolve().parent
RUN = Path(os.environ.get("SH_RUN_DIR", str(WORK / "run")))
WIRE = RUN / "wire_audit"
load = lambda p: json.loads(p.read_text())
audit = load(WIRE / "audit.json")
plans = load(WIRE / "compiled_plans.json")
certificate_bytes = (RUN / "bridge/workspace_envelope_certificate.json").read_bytes()
certificate = json.loads(certificate_bytes)
floors = {r["rank"]: r["prospective_workspace_bytes"] for r in certificate["ranks"]}
assert audit["exit_codes"] == {"cpp_exit": 0, "python_exit": 0}
assert audit["protocol_schema_valid"] and not audit["protocol_errors"]
assert audit["provenance_gate_matched"] and audit["actual_request_complete_observed"]
assert audit["queue_rows"] == 4
nodes, children = {}, defaultdict(set)
snapshots = {}
active_samples = []
for i in range(plans["captured_response_count"]):
    req, resp, ack = (load(WIRE / f"{kind}_{i}.json")
                      for kind in ("request", "response", "ack"))
    assert ack["success"] and ack["delivery_sequence"] == i
    request_state = {r["rank"]: r for r in req["workspace_state"]["ranks"]}
    rows = resp["workspace_capacity_snapshot"]["ranks"]
    assert [r["rank"] for r in rows] == sorted(floors)
    for row in rows:
        rank = row["rank"]
        assert row["prospective_workspace_bytes"] == floors[rank]
        assert row["workspace_revision"] == request_state[rank]["workspace_revision"]
        base = sum(row[k] for k in ("model_weight_bytes", "resident_kv_bytes",
                                   "reserved_request_bytes"))
        active = request_state[rank]["active_workspace_bytes"]
        assert base + max(active, floors[rank]) <= row["capacity_bytes"]
        snapshots[rank] = (row, active)
        active_samples.append(active)
    for n in resp["nodes"]:
        key = (n["rank"], n["id"])
        assert key not in nodes
        nodes[key] = n
    for e in resp["parent_edges"]:
        children[e["rank"], e["from"]].add((e["rank"], e["to"]))
for item in plans["items"]:
    assert item["hbm_workspace_peak_bytes"] <= floors[item["rank"]]
assert plans["compiled_plan_count"] == plans["captured_typed_comp_spec_count"]
assert plans["positive_workspace_plan_count"] > 0
assert all(active == 0 for _, active in snapshots.values())
assert sum(row["workspace_revision"] for row, _ in snapshots.values()) == 2 * plans["compiled_plan_count"]

# Include actual point-to-point match edges in the dependency DAG audit.
send, recv = {}, {}
for key, node in nodes.items():
    if node["type"] not in (5, 6):
        continue
    c = node["comm"]
    match = (c["src"], c["dst"], c["tag"], c["bytes"])
    table = send if node["type"] == 5 else recv
    assert match not in table
    table[match] = key
assert send.keys() == recv.keys()
for match in send:
    children[send[match]].add(recv[match])
indegree = {key: 0 for key in nodes}
for parent, successors in children.items():
    assert parent in nodes
    for successor in successors:
        assert successor in nodes
        indegree[successor] += 1
ready = deque(k for k, degree in indegree.items() if degree == 0)
visited = 0
while ready:
    key = ready.popleft()
    visited += 1
    for successor in children[key]:
        indegree[successor] -= 1
        if indegree[successor] == 0:
            ready.append(successor)
assert visited == len(nodes)
terminal = re.search(r"terminal_total=(\d+) terminal_success=(\d+) terminal_skipped=(\d+) terminal_other=(\d+)", (RUN / "cpp.log").read_text())
assert terminal and tuple(map(int, terminal.groups())) == (len(nodes), len(nodes), 0, 0)
source_acks = [key for key, n in nodes.items()
               if n["type"] == 6 and "noc_migrate" in n["name"]
               and "_ack_from_rank" in n["name"]]
assert len(source_acks) == 8
assert all(children[key] for key in source_acks)
assert all(any(nodes[c]["type"] == 4 and "die_op_spec" in
               nodes[c].get("compute", {}) for c in children[key])
           for key in source_acks)

result = {
    "status": "passed", "classification": "generated_fixture_not_hardware_measurement",
    "cpp_exit": 0, "python_exit": 0, "requests": 4, "batches_and_successful_acks": plans["captured_response_count"],
    "typed_comp_plans_covered": plans["compiled_plan_count"], "positive_workspace_plans": plans["positive_workspace_plan_count"],
    "successful_terminals": len(nodes), "capacity_rows_fit": len(floors) * plans["captured_response_count"],
    "future_workspace_bytes_by_rank": floors,
    "actual_plan_workspace_max_by_rank": {
        rank: max(p["hbm_workspace_peak_bytes"] for p in plans["items"]
                  if p["rank"] == rank) for rank in floors},
    "final_active_workspace_bytes": 0,
    "balanced_lease_pairs_inferred_from_revisions": plans["compiled_plan_count"],
    "positive_live_workspace_sample_count": sum(v > 0 for v in active_samples),
    "migration_source_acks_with_capacity_consumer_dependencies": 8,
    "parent_and_matched_p2p_dag": "passed",
    "remote_store_restore_online_coverage": "not_observed",
    "certificate_canonical_sha256": hashlib.sha256(certificate_bytes.rstrip(b"\n")).hexdigest(),
    "library_and_source_identity": "../../source-identity",
}
(WIRE / "capacity_online_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
