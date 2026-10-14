"""Focused real-C-ABI tests for die_workspace_envelope.py."""

from __future__ import annotations

import ctypes
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

_ONLINE_DIR = str(Path(__file__).resolve().parent)
if _ONLINE_DIR not in sys.path:
    sys.path.insert(0, _ONLINE_DIR)

from die_workspace_envelope import (
    DieWorkspaceEnvelopeClient,
    WorkspaceEnvelopeError,
    _read_owned_json,
    coarse_cabi_candidates,
    discover_coarse_cabi_library,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_HARDWARE_CONFIG = _REPOSITORY_ROOT / "die" / "configs" / "example_hardware.json"
_REAL_LIBRARY = Path(
    os.environ.get(
        "DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY",
        os.environ.get(
            "DIE_COARSE_CABI_PATH",
            str(
                _REPOSITORY_ROOT
                / "die"
                / "work"
                / "lru-workspace-envelope-20261014"
                / "build"
                / "lib"
                / "libcoarse_die_cabi.so"
            ),
        ),
    )
)
_OLD_LIBRARY = Path(
    os.environ.get(
        "DIE_WORKSPACE_ENVELOPE_OLD_LIBRARY",
        _REPOSITORY_ROOT
        / "work"
        / "five-os-20261014"
        / "source-identity"
        / "lru-pre-v2-binaries"
        / "libcoarse_die_cabi.so",
    )
)


def attention_task(q_tokens: int = 64, kv_tokens: int = 64) -> dict:
    return {
        "schema": "coarse_die_task_v2",
        "op_kind": "attention_region",
        "phase": "prefill",
        "tensor_shapes": {
            "m": q_tokens,
            "q_heads": 1,
            "kv_heads": 1,
            "head_dim": 32,
        },
        "spans": [
            {
                "q_tokens": q_tokens,
                "kv_tokens": kv_tokens,
                "q_start_pos": kv_tokens - q_tokens if kv_tokens >= q_tokens else 0,
                "kv_start_pos": 0,
            }
        ],
        "fusion_policy": "auto",
    }


def ffn_task() -> dict:
    return {
        "schema": "coarse_die_task_v2",
        "op_kind": "ffn_region",
        "phase": "prefill",
        "tensor_shapes": {
            "m": 64,
            "hidden": 128,
            "ffn": 64,
            "mlp_variant": "swiglu",
        },
        "fusion_policy": "auto",
    }


def selected_plan_workspace(client: DieWorkspaceEnvelopeClient, task: dict) -> int:
    """Compare the envelope with the real C ABI's ordinarily selected Plan."""
    library = client._library
    library.die_coarse_compile_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    library.die_coarse_compile_batch_v2.restype = ctypes.c_void_p
    library.die_coarse_release_plan_v2.argtypes = [ctypes.c_uint64]
    library.die_coarse_release_plan_v2.restype = ctypes.c_int
    request = json.dumps(
        {"specs": [task]}, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    pointer = library.die_coarse_compile_batch_v2(client.model_handle, request)
    if not pointer:
        detail = library.die_coarse_last_error_v2()
        raise AssertionError(
            "compile_batch_v2 returned null: "
            + (detail.decode("utf-8", errors="replace") if detail else "no last_error")
        )
    batch = _read_owned_json(library, pointer, "compile_batch_v2 in test")
    row = batch["results"][0]
    if not isinstance(row, dict):
        raise AssertionError(f"compile_batch_v2 did not return a Plan row: {row!r}")
    plan_handle = row["plan_handle"]
    try:
        value = row["summary"]["hbm_workspace_peak_bytes"]
        if type(value) is not int:
            raise AssertionError("selected Plan workspace must be an exact JSON integer")
        return value
    finally:
        status = library.die_coarse_release_plan_v2(plan_handle)
        if status != 0:
            detail = library.die_coarse_last_error_v2()
            raise AssertionError(
                "release_plan_v2 failed: "
                + (detail.decode("utf-8", errors="replace") if detail else str(status))
            )


class CabiEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _HARDWARE_CONFIG.is_file():
            raise RuntimeError(f"missing test hardware config: {_HARDWARE_CONFIG}")
        if not _REAL_LIBRARY.is_file():
            raise unittest.SkipTest(
                "real workspace-envelope C ABI library is unavailable; set "
                f"DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY or DIE_COARSE_CABI_PATH: {_REAL_LIBRARY}"
            )

    def setUp(self) -> None:
        with mock.patch.dict(
            os.environ, {"DIE_COARSE_CABI_PATH": str(_REAL_LIBRARY)}
        ):
            self.client = DieWorkspaceEnvelopeClient(_HARDWARE_CONFIG)

    def tearDown(self) -> None:
        self.client.close()

    def test_attention_and_ffn_cover_all_real_mapping_candidates(self) -> None:
        for task in (attention_task(), ffn_task()):
            with self.subTest(op_kind=task["op_kind"]):
                envelope = self.client.query(task)
                selected = selected_plan_workspace(self.client, task)
                self.assertGreater(envelope.candidate_count, 1)
                self.assertGreater(envelope.max_hbm_workspace_peak_bytes, selected)

    def test_dimension_and_span_aliases_normalize_to_same_identity(self) -> None:
        canonical = attention_task()
        aliases = attention_task()
        aliases["tensor_shapes"] = {
            "M": 64,
            "num_q_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 32,
        }
        aliases["spans"] = [
            {"q_len": 64, "kv_len": 64, "query_start": 0, "kv_start": 0}
        ]
        left = self.client.query(canonical)
        right = self.client.query(aliases)
        self.assertEqual(left.normalized_task, right.normalized_task)
        self.assertEqual(
            left.normalized_task_fingerprint,
            right.normalized_task_fingerprint,
        )
        self.assertEqual(left.candidate_count, right.candidate_count)
        self.assertEqual(
            left.max_hbm_workspace_peak_bytes,
            right.max_hbm_workspace_peak_bytes,
        )

    def test_legacy_typed_die_op_spec_returns_canonical_task(self) -> None:
        task = {
            "die_op_spec": {
                "schema_version": 1,
                "op_kind": "attention_region",
                "phase": "prefill",
                "dtype": "bf16",
                "model_dims": {
                    "hidden": 32,
                    "ffn": 64,
                    "vocab": 128,
                    "num_q_heads": 1,
                    "num_kv_heads": 1,
                    "head_dim": 32,
                    "mlp_variant": "swiglu",
                    "norm_type": "rmsnorm",
                    "position_encoding": "rope",
                },
                "tp_slice": {
                    "tensor_parallel": 1,
                    "tensor_parallel_rank": 0,
                    "q_heads_local": 1,
                    "kv_heads_local": 1,
                    "ffn_local": 64,
                    "vocab_local": 128,
                },
                "spans": [
                    {
                        "request_id": "r0",
                        "q_tokens": 64,
                        "kv_tokens": 64,
                        "q_start_pos": 0,
                        "kv_start_pos": 0,
                    }
                ],
                "layer_count": 1,
                "weight_passes": 1,
                "kv_access": {"identity": "local"},
                "fusion_policy": "auto",
            }
        }
        envelope = self.client.query(task)
        self.assertEqual(envelope.op_kind, "attention_region")
        self.assertEqual(envelope.normalized_task["schema"], "coarse_die_task_v2")
        self.assertEqual(envelope.normalized_task["kv_identity"], "local")
        self.assertEqual(envelope.normalized_task["spans"][0]["q_tokens"], 64)

    def test_u64_above_binary64_exact_range_stays_exact(self) -> None:
        huge_kv = 1 << 53
        envelope = self.client.query(attention_task(1, huge_kv))
        self.assertIs(type(envelope.max_hbm_workspace_peak_bytes), int)
        self.assertEqual(
            envelope.max_hbm_workspace_peak_bytes,
            (1 << 56) + 128,
        )
        self.assertEqual(
            envelope.normalized_task["spans"][0]["kv_tokens"], huge_kv
        )

    def test_invalid_spec_error_is_preserved_and_success_clears_it(self) -> None:
        invalid = attention_task()
        invalid["op_kind"] = "unsupported_for_workspace_test"
        with self.assertRaises(WorkspaceEnvelopeError) as caught:
            self.client.query(invalid)
        self.assertIn("die_coarse_workspace_envelope_v2 failed", str(caught.exception))
        self.assertIn("unsupported operator", str(caught.exception))
        self.assertIn("status_code", str(caught.exception))

        valid = self.client.query(attention_task())
        self.assertEqual(valid.op_kind, "attention_region")
        self.assertEqual(self.client._library.die_coarse_last_error_v2(), b"")

    def test_u64_inputs_reject_bool_and_float(self) -> None:
        for invalid_value in (True, 64.0):
            task = attention_task()
            task["spans"][0]["kv_tokens"] = invalid_value
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(
                    WorkspaceEnvelopeError, "exact uint64 integer"
                ):
                    self.client.query(task)

    def test_response_u64_and_identity_checks_reject_tampering(self) -> None:
        result = self.client.query(attention_task())
        raw = asdict(result)

        mistyped = dict(raw)
        mistyped["candidate_count"] = True
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "exact uint64 integer"):
            self.client._validate_envelope(mistyped)

        mistyped = dict(raw)
        mistyped["max_hbm_workspace_peak_bytes"] = float(
            raw["max_hbm_workspace_peak_bytes"]
        )
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "exact uint64 integer"):
            self.client._validate_envelope(mistyped)

        mismatched = dict(raw)
        mismatched["hardware_fingerprint"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "hardware_fingerprint"):
            self.client._validate_envelope(mismatched)

        mismatched = dict(raw)
        mismatched["normalized_task_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "normalized_task_fingerprint"):
            self.client._validate_envelope(mismatched)

    def test_context_manager_closes_single_model_and_close_is_idempotent(self) -> None:
        scoped = DieWorkspaceEnvelopeClient(_HARDWARE_CONFIG, _REAL_LIBRARY)
        with scoped as opened:
            self.assertIs(opened, scoped)
            self.assertEqual(opened.query(attention_task()).schema_version, 2)
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "client is closed"):
            scoped.query(attention_task())
        scoped.close()


class DiscoveryAndOwnershipTests(unittest.TestCase):
    def test_environment_path_precedes_local_lib_and_bin_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_lib, local_bin = coarse_cabi_candidates(root)
            local_lib.parent.mkdir(parents=True)
            local_bin.parent.mkdir(parents=True)
            local_lib.touch()
            local_bin.touch()
            env_lib = root / "env" / "libcoarse_die_cabi.so"
            env_lib.parent.mkdir()
            env_lib.touch()

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DIE_COARSE_CABI_PATH", None)
                self.assertEqual(discover_coarse_cabi_library(repo_root=root), local_lib.resolve())
                os.environ["DIE_COARSE_CABI_PATH"] = str(env_lib)
                self.assertEqual(
                    discover_coarse_cabi_library(repo_root=root), env_lib.resolve()
                )

    def test_explicit_archived_old_library_fails_closed_on_missing_symbol(self) -> None:
        if not _OLD_LIBRARY.is_file():
            self.skipTest(f"archived pre-envelope library unavailable: {_OLD_LIBRARY}")
        for explicit_path in (True, False):
            with self.subTest(explicit_path=explicit_path):
                with mock.patch.dict(
                    os.environ, {"DIE_COARSE_CABI_PATH": str(_OLD_LIBRARY)}
                ):
                    with self.assertRaisesRegex(
                        WorkspaceEnvelopeError, "die_coarse_workspace_envelope_v2"
                    ) as caught:
                        if explicit_path:
                            DieWorkspaceEnvelopeClient(_HARDWARE_CONFIG, _OLD_LIBRARY)
                        else:
                            DieWorkspaceEnvelopeClient(_HARDWARE_CONFIG)
                self.assertIn(str(_OLD_LIBRARY), str(caught.exception))

    def test_owned_buffer_is_freed_when_decoding_raises(self) -> None:
        class FreeRecorder:
            def __init__(self) -> None:
                self.freed = []

            def die_coarse_free_result_v2(self, pointer: ctypes.c_void_p) -> None:
                self.freed.append(pointer.value)

        backing = ctypes.create_string_buffer(b"not-json")
        pointer = ctypes.c_void_p(ctypes.addressof(backing))
        fake_library = FreeRecorder()
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "invalid UTF-8 JSON"):
            _read_owned_json(fake_library, pointer, "owned test result")
        self.assertEqual(fake_library.freed, [pointer.value])


if __name__ == "__main__":
    unittest.main(verbosity=2)

