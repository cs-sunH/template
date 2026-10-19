#!/usr/bin/env python3
"""test_die_model_adapter.py -- die 发射语境/region 编译/spec 构造回归
（joint 推广第一期；移植自 face 参考实现并按本仓发射面适配）。

License: MIT (see LICENSE in the repository root).

覆盖：
  - derive_span_records：本仓 span 构造点（prefill chunk / 等长重放
    形态 / decode 迭代）的绝对位置还原恒等式与 fail-closed（原
    "D10 静默重建"构造点已随 2026-10-15 recompute 退役；span 纯函数
    的 (X, X) 等长恒等式与 fail-closed 用例保留）；
  - CATEGORY_TO_OP_KIND 全表（joint 命名：swiglu 族 rmsnorm/gate_up/
    swiglu、final_rmsnorm）；
  - compile_regions 融合条件矩阵（完整链/缺链/外部分支消费者/链内边界/
    链尾边界放行/锚点名/显式 watch 目标/异质 identity/全序保持）；
  - build_die_spec schema 形状（layer_count=活跃层区间、weight_passes、
    kv_identity 枚举、spans 覆盖/子集）；
  - spec 经 ctypes 过库严格解析（库可用性门；契约 §2 的真正校验者）；
  - DieModelContext.from_service 的 manifest 门全矩阵。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

_ONLINE_DIR = Path(__file__).resolve().parent
_WORKLOAD_DIR = _ONLINE_DIR.parent
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from die_model_adapter import (  # noqa: E402
    CATEGORY_TO_OP_KIND,
    DieModelContext,
    derive_span_records,
    expand_span_records,
    expanded_span_count,
    slice_span_records,
    compile_regions,
)


def _library_available() -> bool:
    if os.environ.get("DIE_MODEL_CABI_PATH") or os.environ.get(
            "DIE_MODEL_CTXT_SKIP_LIB"):
        pass
    sys.path.insert(0, str(_WORKLOAD_DIR.parent.parent))
    try:
        from config_resolver import discover_cabi_library
        discover_cabi_library()
    except Exception:  # noqa: BLE001 -- 缺库 = 跳过过库用例
        return False
    return True


def _spec_record(category: str, *, boundary=False, external=False,
                 identity=None, name=None) -> dict:
    return {
        "name": name or f"train_all_layers_{category}",
        "category": category,
        "op_kind": CATEGORY_TO_OP_KIND[category],
        "num_ops": 10,
        "tensor_size": 20,
        "remote_read": 5,
        "layer_count": 32,
        "weight_passes": 3,
        "external_consumers": external,
        "boundary_after": boundary,
        "identity": dict(identity or {"tp": 6, "rank": 0, "layer": 32}),
    }


class SpanRecordTests(unittest.TestCase):
    def test_prefill_chunk_span_identity(self) -> None:
        # (chunk_tokens, history + completed + chunk_tokens)：q 起点 =
        # history + completed，kv 自 0 覆盖到当前。
        records = derive_span_records([(512, 1000 + 512)])
        self.assertEqual(records[0]["q_tokens"], 512)
        self.assertEqual(records[0]["kv_tokens"], 1512)
        self.assertEqual(records[0]["q_start_pos"], 1000)
        self.assertEqual(records[0]["kv_start_pos"], 0)

    def test_d10_rebuild_span_identity(self) -> None:
        # (X, X)：等长整体重放形态 [0, X)（span 纯函数性质——原 D10
        # 静默重建构造点已随 2026-10-15 recompute 退役，恒等式保留）。
        records = derive_span_records([(2048, 2048)])
        self.assertEqual(records[0]["q_start_pos"], 0)
        self.assertEqual(records[0]["kv_start_pos"], 0)

    def test_decode_iteration_span_identity(self) -> None:
        # (1, context + consumed + step)：q = 当前末 token（= kv-1）。
        records = derive_span_records([(1, 1000 + 7 + 3)])
        self.assertEqual(records[0]["q_start_pos"], 1009)
        self.assertEqual(records[0]["kv_start_pos"], 0)
        self.assertEqual(records[0]["request_id"], "span#s0")

    def test_decode_run_compression_and_expansion_identity(self) -> None:
        # v5 紧凑 run：等 q、常数步长的连续 span 压成单条 run 记录；
        # 展开与逐 span 派生逐项相等（无损）。
        flat = [(1, 1010 + step) for step in range(1, 206)]
        records = derive_span_records(flat, namespace="train_a")
        self.assertEqual(len(records), 1)
        run = records[0]["run"]
        self.assertEqual(run["q_tokens"], 1)
        self.assertEqual(run["kv_first"], 1011)
        self.assertEqual(run["kv_stride"], 1)
        self.assertEqual(run["count"], 205)
        self.assertEqual(records[0]["request_id_prefix"], "train_a#s")
        self.assertEqual(records[0]["first_index"], 0)
        expanded = expand_span_records(records)
        legacy = [
            {"request_id": "train_a#s{}".format(index),
             "q_tokens": tokens, "kv_tokens": kv,
             "q_start_pos": kv - tokens, "kv_start_pos": 0}
            for index, (tokens, kv) in enumerate(flat)]
        self.assertEqual(expanded, legacy)
        self.assertEqual(expanded_span_count(records), 205)

    def test_run_breaks_at_stride_or_q_change(self) -> None:
        # chunk(512) 头 + 成员 decode 步进 run + 等长重放形态 span：序列
        # 在 q 或步长变化处断开，分段计数与平铺一致。
        flat = [(512, 512), (512, 1024), (2048, 2048), (1, 3000), (1, 3001)]
        records = derive_span_records(flat)
        kinds = [
            "run" if "run" in record else "span" for record in records]
        self.assertEqual(kinds, ["run", "span", "run"])
        self.assertEqual(records[0]["run"]["count"], 2)
        self.assertEqual(records[0]["run"]["kv_stride"], 512)
        self.assertEqual(records[2]["run"]["count"], 2)
        # 展开 == 旧逐 span 派生（run 断点不改变展开结果）。
        expanded = expand_span_records(records)
        legacy = [
            {"request_id": "span#s{}".format(index),
             "q_tokens": tokens, "kv_tokens": kv,
             "q_start_pos": kv - tokens, "kv_start_pos": 0}
            for index, (tokens, kv) in enumerate(flat)]
        self.assertEqual(expanded, legacy)

    def test_slice_span_records_splits_runs(self) -> None:
        flat = [(1, 10 + step) for step in range(1, 11)]
        records = derive_span_records(flat)
        # 保留展开序 2..5（run 中段）→ 一条子 run，kv_first 对齐。
        subset = slice_span_records(records, range(2, 6))
        self.assertEqual(len(subset), 1)
        self.assertEqual(subset[0]["first_index"], 2)
        self.assertEqual(subset[0]["run"]["kv_first"], 13)
        self.assertEqual(subset[0]["run"]["count"], 4)
        # 切片展开 == 全展开的同索引过滤。
        expanded = expand_span_records(records)
        expected = [expanded[i] for i in range(2, 6)]
        self.assertEqual(expand_span_records(subset), expected)

    def test_kv_below_tokens_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "kv_length 2 < tokens 4"):
            derive_span_records([(4, 2)])

    def test_nonpositive_span_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            derive_span_records([(0, 10)])

    def test_empty_spans_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one pass span"):
            derive_span_records([])


class CategoryMapTests(unittest.TestCase):
    def test_joint_category_names_covered(self) -> None:
        expected = {
            "attention_input_rmsnorm": "norm",
            "attention_qkv_projection": "qkv_rope",
            "attention_qk_matmul": "qk",
            "attention_scale_mask": "scale_mask",
            "attention_softmax": "softmax",
            "attention_av_matmul": "av",
            "attention_output_projection": "o_proj",
            "attention_residual_add": "residual",
            "mlp_input_rmsnorm": "norm",
            "mlp_gate_up_projection": "mlp_up",
            "mlp_swiglu": "mlp_activation",
            "mlp_down_projection": "mlp_down",
            "mlp_residual_add": "residual",
            "final_rmsnorm": "norm",
            "logits_projection": "logits",
        }
        for category, op_kind in expected.items():
            self.assertEqual(CATEGORY_TO_OP_KIND[category], op_kind)

    def test_unknown_category_fails_closed(self) -> None:
        ctx = _make_ctx()
        with self.assertRaisesRegex(ValueError, "no op_kind mapping"):
            ctx.op_kind_of("mystery_op")


class CompileRegionsTests(unittest.TestCase):
    def test_full_chains_fuse_in_order(self) -> None:
        categories = [
            "attention_input_rmsnorm", "attention_qkv_projection",
            "attention_qk_matmul", "attention_scale_mask",
            "attention_softmax", "attention_av_matmul",
            "attention_output_projection", "attention_residual_add",
            "mlp_input_rmsnorm", "mlp_gate_up_projection", "mlp_swiglu",
            "mlp_down_projection", "mlp_residual_add",
        ]
        groups = compile_regions(
            [_spec_record(c) for c in categories])
        kinds = [g["op_kind"] for g in groups]
        self.assertEqual(
            kinds,
            ["norm", "qkv_rope", "attention_region", "o_proj", "residual",
             "norm", "ffn_region", "residual"])
        fused = [g for g in groups if g["fused"]]
        self.assertEqual(len(fused), 2)
        attention = fused[0]
        self.assertEqual(
            attention["member_categories"],
            ["attention_qk_matmul", "attention_scale_mask",
             "attention_softmax", "attention_av_matmul"])
        self.assertEqual(attention["num_ops"], 40)
        self.assertEqual(attention["tensor_size"], 80)
        # remote_read = 权重远载字节（generate_trace 真值路径，保留
        # 红线 §1.2 第 2 条）；调度器侧供给点已随远读退役删除，该键在
        # echo/loopback 形态下可缺席（见 test_records_without_remote_read）。
        self.assertEqual(attention["remote_read"], 20)
        self.assertTrue(attention["name"].endswith("attention_region"))
        ffn = fused[1]
        self.assertEqual(
            ffn["member_categories"],
            ["mlp_gate_up_projection", "mlp_swiglu", "mlp_down_projection"])
        self.assertTrue(ffn["name"].endswith("ffn_region"))

    def test_records_without_remote_read_key_fuse_and_sum_zero(self) -> None:
        # 2026-10-14 远读退役 D-2 口径：sh30 供给点删除后，echo/loopback
        # 路径的 spec 记录不含 remote_read 键——compile_regions 容错读
        # （node.get("remote_read", 0)），融合不受影响、合计为 0。
        categories = [
            "attention_qk_matmul", "attention_scale_mask",
            "attention_softmax", "attention_av_matmul",
        ]
        records = [_spec_record(c) for c in categories]
        for record in records:
            del record["remote_read"]
        groups = compile_regions(records)
        fused = [g for g in groups if g["fused"]]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["remote_read"], 0)
        # 非融合组同款容错（缺席键 → 0）。
        singles = compile_regions(records[:1])
        self.assertEqual(singles[0]["remote_read"], 0)

    def test_boundary_inside_chain_blocks_fusion(self) -> None:
        records = [
            _spec_record("attention_qk_matmul"),
            _spec_record("attention_scale_mask", boundary=True),
            _spec_record("attention_softmax"),
            _spec_record("attention_av_matmul"),
        ]
        groups = compile_regions(records)
        self.assertTrue(all(not g["fused"] for g in groups))
        self.assertEqual(len(groups), 4)

    def test_boundary_after_chain_tail_is_allowed(self) -> None:
        records = [
            _spec_record("attention_qk_matmul"),
            _spec_record("attention_scale_mask"),
            _spec_record("attention_softmax"),
            _spec_record("attention_av_matmul", boundary=True),
        ]
        groups = compile_regions(records)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["fused"])

    def test_external_consumer_blocks_fusion(self) -> None:
        records = [
            _spec_record("attention_qk_matmul"),
            _spec_record("attention_scale_mask", external=True),
            _spec_record("attention_softmax"),
            _spec_record("attention_av_matmul"),
        ]
        groups = compile_regions(records)
        self.assertTrue(all(not g["fused"] for g in groups))

    def test_heterogeneous_identity_blocks_fusion(self) -> None:
        records = [
            _spec_record("mlp_gate_up_projection",
                         identity={"tp": 6, "rank": 0}),
            _spec_record("mlp_swiglu", identity={"tp": 6, "rank": 0}),
            _spec_record("mlp_down_projection",
                         identity={"tp": 6, "rank": 1}),
        ]
        groups = compile_regions(records)
        self.assertTrue(all(not g["fused"] for g in groups))

    def test_marker_anchor_names_never_fuse(self) -> None:
        records = [
            _spec_record("attention_qk_matmul", name="x_first_token_qk"),
            _spec_record("attention_scale_mask", name="x_scale"),
            _spec_record("attention_softmax", name="x_softmax"),
            _spec_record("attention_av_matmul", name="x_av"),
        ]
        groups = compile_regions(records)
        self.assertTrue(all(not g["fused"] for g in groups))

    def test_explicit_watch_targets_block_fusion(self) -> None:
        records = [
            _spec_record("attention_qk_matmul", name="watched_qk"),
            _spec_record("attention_scale_mask"),
            _spec_record("attention_softmax"),
            _spec_record("attention_av_matmul"),
        ]
        groups = compile_regions(records, watch_targets=["watched_qk"])
        self.assertTrue(all(not g["fused"] for g in groups))

    def test_missing_link_keeps_separate_groups(self) -> None:
        records = [
            _spec_record("attention_qk_matmul"),
            _spec_record("attention_softmax"),
            _spec_record("attention_av_matmul"),
        ]
        groups = compile_regions(records)
        self.assertEqual([g["op_kind"] for g in groups],
                         ["qk", "softmax", "av"])


class BuildSpecTests(unittest.TestCase):
    def test_spec_schema_shape(self) -> None:
        ctx = _make_ctx()
        ctx.begin_emission(
            phase="decode", tensor_parallel=6, tensor_parallel_rank=2,
            span_namespace="train1", pass_spans=[(512, 1024), (1, 33)])
        spec = ctx.build_spec("norm", layer_count=8, weight_passes=3)
        self.assertEqual(spec["schema_version"], 1)
        self.assertEqual(spec["op_kind"], "norm")
        self.assertEqual(spec["phase"], "decode")
        self.assertEqual(spec["model_dims"]["num_q_heads"], 32)
        self.assertEqual(spec["model_dims"]["num_kv_heads"], 32)
        self.assertEqual(spec["model_dims"]["head_dim"], 128)
        self.assertEqual(spec["tp_slice"]["tensor_parallel"], 6)
        self.assertEqual(spec["tp_slice"]["tensor_parallel_rank"], 2)
        # 32 heads / TP 6 -> rank 2 分片 5（shard_extent 同源）。
        self.assertEqual(spec["tp_slice"]["q_heads_local"], 5)
        self.assertEqual(len(spec["spans"]), 2)
        self.assertEqual(spec["layer_count"], 8)
        self.assertEqual(spec["weight_passes"], 3)
        self.assertEqual(spec["kv_access"], {"identity": "local"})
        self.assertEqual(spec["provenance"], "transformer_pass_aggregated")
        self.assertEqual(len(spec["config_hash"]), 64)

    def test_kv_identity_staged_annotation(self) -> None:
        # 2026-10-14 远读退役：kv_access.identity 词表缩为 {"local"}
        # （"staged"/"remote_ready" 死值删除，KV_IDENTITIES 单值）——
        # 旧 "staged" 注解用例改为钉退役后的 fail-closed 面。
        ctx = _make_ctx()
        ctx.begin_emission(
            phase="decode", tensor_parallel=1, tensor_parallel_rank=0,
            span_namespace="t", pass_spans=[(1, 8)])
        with self.assertRaisesRegex(ValueError, "kv_access.identity"):
            ctx.build_spec(
                "qk", layer_count=1, weight_passes=1, kv_identity="staged")

    def test_bad_kv_identity_fails(self) -> None:
        ctx = _make_ctx()
        ctx.begin_emission(
            phase="prefill", tensor_parallel=1, tensor_parallel_rank=0,
            span_namespace="t", pass_spans=[(1, 8)])
        with self.assertRaisesRegex(ValueError, "kv_access.identity"):
            ctx.build_spec(
                "qk", layer_count=1, weight_passes=1, kv_identity="remote")

    def test_span_subset_override(self) -> None:
        ctx = _make_ctx()
        ctx.begin_emission(
            phase="prefill", tensor_parallel=1, tensor_parallel_rank=0,
            span_namespace="t", pass_spans=[(1, 8), (2048, 2048), (1, 9)])
        subset = [ctx.span_records[0], ctx.span_records[2]]
        spec = ctx.build_spec(
            "logits", layer_count=1, weight_passes=1, spans=subset)
        self.assertEqual(len(spec["spans"]), 2)
        self.assertNotIn(2048, [s["q_tokens"] for s in spec["spans"]])

    def test_layer_count_must_be_positive(self) -> None:
        ctx = _make_ctx()
        ctx.begin_emission(
            phase="prefill", tensor_parallel=1, tensor_parallel_rank=0,
            span_namespace="t", pass_spans=[(1, 8)])
        with self.assertRaisesRegex(ValueError, "layer_count"):
            ctx.build_spec("norm", layer_count=0, weight_passes=1)


@unittest.skipUnless(
    _library_available(),
    "die cost library (libdie_model_cabi.so) not discoverable")
class SpecThroughCostLibraryTests(unittest.TestCase):
    """spec 经 ctypes 过库严格解析（die_estimate_batch；库不可达跳过）。"""

    def test_spec_shapes_parse_in_the_library(self) -> None:
        import ctypes
        from config_resolver import discover_cabi_library
        ctx = _make_ctx()
        ctx.begin_emission(
            phase="prefill", tensor_parallel=6, tensor_parallel_rank=2,
            span_namespace="t", pass_spans=[(512, 1024), (1, 33)])
        specs = []
        for op_kind, layers, passes in (
                ("norm", 8, 3), ("qkv_rope", 8, 3), ("attention_region", 8, 3),
                ("ffn_region", 8, 3), ("o_proj", 8, 3), ("logits", 1, 3)):
            specs.append(ctx.build_spec(
                op_kind, layer_count=layers, weight_passes=passes,
                kv_identity="local"))
        library = ctypes.CDLL(str(discover_cabi_library()))
        library.die_estimate_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        library.die_estimate_batch.restype = ctypes.c_void_p
        library.die_free_result.argtypes = [ctypes.c_char_p]
        library.die_free_result.restype = None
        library.die_last_error.restype = ctypes.c_char_p
        # 无硬件 JSON 也能先严格解析：die_create_model 需要合法 die 配置；
        # 用 resolver 的派生入口取一份 resolved 配置。阶段2（2026-10-18）
        # 起 checked-in 出厂默认为 coarse-die 节，legacy v1 库解析改用
        # 显式 die 节夹具（原 checked-in die 参数）。
        from config_resolver import derive_die_hardware
        die_node = {
            "sram-mb-per-core": 0.75,
            "compute-area-mm2": 800,
            "hbm-per-side": 4,
            "hbm-sides": 1,
            "noc-width-bits": 512,
            "mesh-nx": 10,
            "mesh-ny": 10,
        }
        summary = derive_die_hardware(die_node)
        library.die_create_model.argtypes = [ctypes.c_char_p]
        library.die_create_model.restype = ctypes.c_void_p
        library.die_destroy_model.argtypes = [ctypes.c_void_p]
        library.die_destroy_model.restype = None
        model = library.die_create_model(
            json.dumps(summary, sort_keys=True).encode("utf-8"))
        self.assertTrue(bool(model))
        try:
            payload = json.dumps({"specs": specs}).encode("utf-8")
            ptr = library.die_estimate_batch(model, payload)
            self.assertTrue(bool(ptr), msg=library.die_last_error())
            text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
            library.die_free_result(ctypes.cast(ptr, ctypes.c_char_p))
            result = json.loads(text)
            errors = [r.get("error") for r in result.get("results", [])]
            self.assertEqual(
                [e for e in errors if e is not None], [],
                msg=f"library rejected emitted specs: {errors}")
        finally:
            library.die_destroy_model(model)


class FromServiceManifestGateTests(unittest.TestCase):
    class _Config:
        bytes_per_elem = 2
        mlp_variant = "swiglu"
        hidden_size = 4096
        num_heads = 32
        ffn_size = 11008
        vocab_size = 32000
        position_encoding = "rope"
        system_config = None

    def test_missing_backend_key_returns_none(self) -> None:
        self.assertIsNone(DieModelContext.from_service(
            {"requests": []}, self._Config()))

    def test_roofline_and_replay_backends_return_none(self) -> None:
        for backend in ("roofline", "replay"):
            self.assertIsNone(DieModelContext.from_service(
                {"compute-cost-model": backend}, self._Config()))

    def test_unknown_backend_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            DieModelContext.from_service(
                {"compute-cost-model": "quantum"}, self._Config())

    def test_die_backend_requires_config_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "die-model-config"):
            DieModelContext.from_service(
                {"compute-cost-model": "die"}, self._Config())

    def test_die_backend_requires_absolute_existing_path(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved_die_model.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absolute"):
                DieModelContext.from_service(
                    {"compute-cost-model": "die",
                     "die-model-config": path.name}, self._Config())
            with self.assertRaisesRegex(ValueError, "does not exist"):
                DieModelContext.from_service(
                    {"compute-cost-model": "die",
                     "die-model-config": str(Path(tmp) / "missing.json")},
                    self._Config())


def _make_ctx() -> DieModelContext:
    return DieModelContext(
        profile="mesh_die_v1",
        die_config_path=Path("/nonexistent/resolved_die_model.json"),
        config_hash="ab" * 32,
        dtype="bf16",
        model_dims={
            "hidden": 4096, "ffn": 11008, "vocab": 32000,
            "mlp_variant": "swiglu", "norm_type": "rmsnorm",
            "position_encoding": "rope", "num_q_heads": 32,
            "num_kv_heads": 32, "head_dim": 128,
        },
        num_q_heads=32,
        num_kv_heads=32,
        ffn_size=11008,
        vocab_size=32000,
    )


if __name__ == "__main__":
    unittest.main()
