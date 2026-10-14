"""Capture observed exact envelope values and selected Plan comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ONLINE = (
    ROOT
    / "astra-sim-wscllm-LRU"
    / "sh_test_mesh"
    / "workload"
    / "llama2_7b_inference"
    / "online"
)
sys.path.insert(0, str(ONLINE))

from die_workspace_envelope import DieWorkspaceEnvelopeClient  # noqa: E402
from test_die_workspace_envelope import (  # noqa: E402
    _HARDWARE_CONFIG,
    attention_task,
    ffn_task,
    selected_plan_workspace,
)


LIBRARY = Path(os.environ["DIE_COARSE_CABI_PATH"]).resolve()
results = []
with DieWorkspaceEnvelopeClient(_HARDWARE_CONFIG) as client:
    for task in (attention_task(), ffn_task()):
        envelope = client.query(task)
        selected = selected_plan_workspace(client, task)
        results.append(
            {
                "op_kind": envelope.op_kind,
                "phase": envelope.phase,
                "candidate_count": envelope.candidate_count,
                "max_hbm_workspace_peak_bytes": envelope.max_hbm_workspace_peak_bytes,
                "selected_plan_hbm_workspace_peak_bytes": selected,
                "envelope_exceeds_selected": (
                    envelope.max_hbm_workspace_peak_bytes > selected
                ),
                "hardware_fingerprint": envelope.hardware_fingerprint,
                "normalized_task_fingerprint": envelope.normalized_task_fingerprint,
            }
        )

record = {
    "hardware_config": str(_HARDWARE_CONFIG),
    "hardware_config_sha256": hashlib.sha256(
        _HARDWARE_CONFIG.read_bytes()
    ).hexdigest(),
    "library_path": str(LIBRARY),
    "library_sha256": hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
    "results": results,
}
output = Path(__file__).with_name("real-cabi-observed.json")
output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True, indent=2))
