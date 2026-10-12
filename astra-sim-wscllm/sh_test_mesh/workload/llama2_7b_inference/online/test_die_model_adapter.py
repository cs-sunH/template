#!/usr/bin/env python3
"""test_die_model_adapter.py -- die 后端适配层单测(阶段 C,2026-10-03)。

覆盖三块:
  1. derive_span_records:face (tokens, kv) 平铺表 -> 绝对位置还原
     (kv_tokens == q_start_pos + q_tokens、kv_start_pos == 0 恒等式;
     违反即 fail-closed);
  2. 13 类旧 COMP 类别 -> op_kind 映射(契约 §2"语义"小节全表);
  3. compile_regions 融合条件矩阵(§7.2:同质/外部分支消费者/链内边界/
     观察锚点逐项阻断;满足时融合且工作量守恒、命名不含锚点子串);
  4. build_die_spec 契约 §2 schema 形状 + DieModelContext.from_service
     的 manifest 门(die/roofline/replay/未知值/缺键)。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_die_model_adapter.py   （或 pytest 同路径）
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.die_model_adapter import (  # noqa: E402
    CATEGORY_TO_OP_KIND,
    DieModelContext,
    build_die_spec,
    compile_regions,
    derive_span_records,
)


def _records(categories, *, ops=1, tensor=2, remote=0, **overrides):
    """按类别序构造 compile_regions 输入记录(缺省无阻断)。"""
    nodes = []
    for category in categories:
        node = {
            "name": "train_all_layers_{}".format(category),
            "category": category,
            "op_kind": CATEGORY_TO_OP_KIND[category],
            "num_ops": ops,
            "tensor_size": tensor,
            "remote_read": remote,
            "layer_count": 4,
            "weight_passes": 2,
            "external_consumers": False,
            "boundary_after": False,
            "identity": {"tp": 6, "rank": 0, "layer": 4, "span_set": "s"},
        }
        node.update(overrides.get(category, {}))
        nodes.append(node)
    return nodes


class SpanPositionTests(unittest.TestCase):
    def test_wscllm_span_identity_restores_absolute_positions(self):
        """三类 face span 的恒等式还原:prefill chunk(含 recompute/当前
        段)与 decode 迭代。"""
        spans = [(128, 128), (128, 256), (44, 300), (1, 101), (1, 102)]
        records = derive_span_records(spans, namespace="batch_train_i1_9")
        expected_starts = [0, 128, 256, 100, 101]
        for record, q_start in zip(records, expected_starts):
            self.assertEqual(record["q_start_pos"], q_start)
            self.assertEqual(record["kv_start_pos"], 0)
            self.assertEqual(record["q_tokens"] + record["q_start_pos"],
                             record["kv_tokens"])
        self.assertEqual(
            [record["request_id"] for record in records],
            ["batch_train_i1_9#s{}".format(index) for index in range(5)])

    def test_span_identity_violation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invariant"):
            derive_span_records([(5, 3)])
        with self.assertRaisesRegex(ValueError, "positive"):
            derive_span_records([(0, 1)])
        with self.assertRaisesRegex(ValueError, "at least one"):
            derive_span_records([])


class CategoryMappingTests(unittest.TestCase):
    def test_full_category_table(self):
        expected = {
            "attention_input_rmsnorm": "norm",
            "attention_input_layernorm": "norm",
            "mlp_input_rmsnorm": "norm",
            "mlp_input_layernorm": "norm",
            "final_rmsnorm": "norm",
            "final_layernorm": "norm",
            "attention_qkv_projection": "qkv_rope",
            "attention_qk_matmul": "qk",
            "attention_scale_mask": "scale_mask",
            "attention_softmax": "softmax",
            "attention_av_matmul": "av",
            "attention_output_projection": "o_proj",
            "attention_residual_add": "residual",
            "mlp_gate_up_projection": "mlp_up",
            "mlp_up_projection": "mlp_up",
            "mlp_swiglu": "mlp_activation",
            "mlp_gelu": "mlp_activation",
            "mlp_down_projection": "mlp_down",
            "mlp_residual_add": "residual",
            "logits_projection": "logits",
        }
        self.assertEqual(CATEGORY_TO_OP_KIND, expected)

    def test_unknown_category_rejected_via_context(self):
        context = _make_context()
        with self.assertRaisesRegex(ValueError, "unknown aggregate COMP"):
            context.op_kind_of("attention_rope_rotation")


class CompileRegionTests(unittest.TestCase):
    ATTENTION = ("attention_qk_matmul", "attention_scale_mask",
                 "attention_softmax", "attention_av_matmul")
    FFN = ("mlp_gate_up_projection", "mlp_swiglu", "mlp_down_projection")

    def _chain(self, categories):
        return [CATEGORY_TO_OP_KIND[category] for category in categories]

    def test_happy_path_fuses_both_regions_and_conserves_work(self):
        nodes = _records(self.ATTENTION + self.FFN, ops=3, tensor=5, remote=7)
        groups = compile_regions(nodes)
        self.assertEqual(
            [group["op_kind"] for group in groups],
            ["attention_region", "ffn_region"])
        for group in groups:
            self.assertTrue(group["fused"])
            self.assertEqual(group["num_ops"], 3 * len(group["members"]))
            self.assertEqual(group["tensor_size"], 5 * len(group["members"]))
            self.assertEqual(group["remote_read"], 7 * len(group["members"]))
            self.assertEqual(group["layer_count"], 4)
            self.assertEqual(group["weight_passes"], 2)
        self.assertEqual(
            [group["name"] for group in groups],
            ["train_all_layers_attention_region",
             "train_all_layers_ffn_region"])

    def test_partial_chain_stays_separate(self):
        # softmax 缺席:qk/scale_mask/av 不构成完整链,全部分离执行。
        nodes = _records(("attention_qk_matmul", "attention_scale_mask",
                          "attention_av_matmul"))
        groups = compile_regions(nodes)
        self.assertEqual(
            [group["op_kind"] for group in groups], ["qk", "scale_mask", "av"])
        self.assertTrue(all(not group["fused"] for group in groups))

    def test_external_consumer_blocks_fusion(self):
        nodes = _records(
            self.ATTENTION,
            **{"attention_softmax": {"external_consumers": True}})
        groups = compile_regions(nodes)
        self.assertEqual(
            [group["op_kind"] for group in groups],
            ["qk", "scale_mask", "softmax", "av"])

    def test_boundary_inside_chain_blocks_fusion(self):
        # 链中间有通信/KV gate/提交边界(scale_mask 之后):不得融合。
        nodes = _records(
            self.ATTENTION,
            **{"attention_scale_mask": {"boundary_after": True}})
        groups = compile_regions(nodes)
        self.assertEqual(
            [group["op_kind"] for group in groups],
            ["qk", "scale_mask", "softmax", "av"])

    def test_boundary_after_chain_tail_is_allowed(self):
        # mlp_down 之后的 All-Reduce 是链尾边界:ffn 融合不受影响。
        nodes = _records(
            self.FFN, **{"mlp_down_projection": {"boundary_after": True}})
        groups = compile_regions(nodes)
        self.assertEqual([group["op_kind"] for group in groups], ["ffn_region"])

    def test_watch_anchor_name_blocks_fusion(self):
        nodes = _records(self.ATTENTION)
        nodes[1]["name"] = "train_all_layers_attention_softmax_first_token"
        groups = compile_regions(nodes)
        self.assertEqual(
            [group["op_kind"] for group in groups],
            ["qk", "scale_mask", "softmax", "av"])

    def test_explicit_watch_target_blocks_fusion(self):
        nodes = _records(self.FFN)
        groups = compile_regions(nodes, watch_targets=(
            nodes[0]["name"],))
        self.assertEqual(
            [group["op_kind"] for group in groups],
            ["mlp_up", "mlp_activation", "mlp_down"])

    def test_heterogeneous_identity_blocks_fusion(self):
        nodes = _records(
            self.ATTENTION,
            **{"attention_softmax": {"identity": {
                "tp": 6, "rank": 1, "layer": 4, "span_set": "s"}}})
        groups = compile_regions(nodes)
        self.assertEqual(
            [group["op_kind"] for group in groups],
            ["qk", "scale_mask", "softmax", "av"])

    def test_fused_name_colliding_with_watch_anchor_fails_closed(self):
        # _fused_name 是防御层(compile_regions 的成员名锚点检查先行,
        # 正常不可达):融合名撞锚点子串必须 fail-closed,直接钉住。
        from online.die_model_adapter import _fused_name
        node = {"name": "train_drain_attention_qk_matmul",
                "category": "attention_qk_matmul"}
        with self.assertRaisesRegex(ValueError, "watch anchor substring"):
            _fused_name(node, "attention_region")

    def test_full_emission_order_preserved(self):
        """完整 13 类序列:融合仅吸收两条链,其余原序分离。"""
        categories = (
            "attention_input_rmsnorm", "attention_qkv_projection",
            "attention_qk_matmul", "attention_scale_mask",
            "attention_softmax", "attention_av_matmul",
            "attention_output_projection", "attention_residual_add",
            "mlp_input_rmsnorm", "mlp_gate_up_projection", "mlp_swiglu",
            "mlp_down_projection", "mlp_residual_add",
        )
        groups = compile_regions(_records(categories))
        self.assertEqual(
            [group["op_kind"] for group in groups],
            ["norm", "qkv_rope", "attention_region", "o_proj", "residual",
             "norm", "ffn_region", "residual"])


def _make_context(**overrides):
    """构造 DieModelContext(from_service 等价的直造形态,供 spec 测试)。"""
    context = DieModelContext(
        profile="mesh_die_v1",
        die_config_path=Path("/nonexistent/resolved_die_model.json"),
        config_hash="0" * 64,
        dtype="bf16",
        model_dims={
            "hidden": 64, "ffn": 128, "vocab": 256, "mlp_variant": "gelu",
            "norm_type": "layernorm", "position_encoding": "rope",
            "num_q_heads": 8, "num_kv_heads": 8, "head_dim": 8,
        },
        num_q_heads=8,
        num_kv_heads=8,
        ffn_size=128,
        vocab_size=256,
    )
    context.begin_emission(
        phase="decode", tensor_parallel=4, tensor_parallel_rank=1,
        span_namespace="batch_train_i1_1", pass_spans=[(1, 41), (1, 42)])
    for key, value in overrides.items():
        setattr(context, key, value)
    return context


class BuildSpecTests(unittest.TestCase):
    def setUp(self):
        # 真实可解析的 resolved_die_model.json(config_hash/路径语义用例)。
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.resolved = Path(self._temp.name) / "resolved_die_model.json"
        self.resolved.write_text(json.dumps({
            "schema_version": 1, "profile": "mesh_die_v1",
            "die": {"sram-mb-per-core": 0.75, "core-type": "large",
                    "compute-area-mm2": 800, "hbm-per-side": 4,
                    "hbm-sides": 1, "noc-width-bits": 512},
            "derived": {}}))

    def _make_context(self):
        context = _make_context()
        context.die_config_path = self.resolved
        return context

    def test_spec_schema_shape(self):
        spec = build_die_spec(
            "attention_region", self._make_context(),
            [{"request_id": "r", "q_tokens": 1, "kv_tokens": 41,
              "q_start_pos": 40, "kv_start_pos": 0}],
            layer_count=32, weight_passes=2, fusion_policy="auto")
        self.assertEqual(
            set(spec),
            {"schema_version", "op_kind", "phase", "dtype", "model_dims",
             "tp_slice", "spans", "layer_count", "weight_passes",
             "kv_access", "fusion_policy", "provenance", "config_hash"})
        self.assertEqual(spec["schema_version"], 1)
        self.assertEqual(spec["phase"], "decode")
        self.assertEqual(spec["dtype"], "bf16")
        self.assertEqual(spec["kv_access"], {"identity": "local"})
        self.assertEqual(spec["fusion_policy"], "auto")
        self.assertEqual(spec["provenance"], "transformer_pass_aggregated")
        self.assertEqual(spec["model_dims"]["position_encoding"], "rope")
        self.assertEqual(spec["spans"][0]["q_start_pos"], 40)
        # 非整除 TP 用 shard_extent(8 头 / TP4 / rank1 -> 各 2)。
        self.assertEqual(
            spec["tp_slice"],
            {"tensor_parallel": 4, "tensor_parallel_rank": 1,
             "q_heads_local": 2, "kv_heads_local": 2, "ffn_local": 32,
             "vocab_local": 64})

    def test_spec_validation(self):
        context = self._make_context()
        with self.assertRaisesRegex(ValueError, "op_kind"):
            build_die_spec("gemm", context, context.span_records,
                           layer_count=1, weight_passes=1)
        with self.assertRaisesRegex(ValueError, "fusion_policy"):
            build_die_spec("norm", context, context.span_records,
                           layer_count=1, weight_passes=1,
                           fusion_policy="always")
        with self.assertRaisesRegex(ValueError, "layer_count"):
            build_die_spec("norm", context, context.span_records,
                           layer_count=0, weight_passes=1)
        with self.assertRaisesRegex(ValueError, "weight_passes"):
            build_die_spec("norm", context, context.span_records,
                           layer_count=1, weight_passes=0)

    def test_spec_shape_matches_contract_schema(self):
        """契约 §2 schema 形状钉子(wscllm 无 oracle 客户端,形状校验
        由 C++ parse_die_op_spec 承担,由 parsed_graph_batch_test 的 N12
        die 用例覆盖;此处钉 build_die_spec 的输出键集)。"""
        spans = [{"request_id": "t#s0", "q_tokens": 8, "kv_tokens": 128,
                  "q_start_pos": 120, "kv_start_pos": 0}]
        ctx = _make_context()
        spec = build_die_spec("norm", ctx, spans, layer_count=3,
                              weight_passes=2, fusion_policy="none")
        self.assertEqual(
            set(spec),
            {"schema_version", "op_kind", "phase", "dtype", "model_dims",
             "tp_slice", "spans", "layer_count", "weight_passes",
             "kv_access", "fusion_policy", "provenance", "config_hash"})


class FromServiceTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        resolved = Path(self._temp.name) / "resolved_die_model.json"
        resolved.write_text(json.dumps(
            {"schema_version": 1, "profile": "mesh_die_v1",
             "die": {}, "derived": {}}))
        self.resolved = resolved
        system = Path(self._temp.name) / "system.json"
        system.write_text(json.dumps(
            {"compute-cost-model": "die",
             "die-model-profile": "mesh_die_v1"}))
        self.config = SimpleNamespace(
            layers=2, hidden_size=64, ffn_size=128, vocab_size=256,
            num_heads=8, bytes_per_elem=2, mlp_variant="swiglu",
            position_encoding="rope", system_config=system)

    def _manifest(self, **overrides):
        manifest = {"requests": [], "compute-cost-model": "die",
                    "die-model-config": str(self.resolved)}
        manifest.update(overrides)
        return manifest

    def test_die_backend_constructs_context(self):
        context = DieModelContext.from_service(
            self._manifest(), self.config)
        self.assertIsNotNone(context)
        self.assertEqual(context.backend, "die")
        self.assertEqual(context.profile, "mesh_die_v1")
        self.assertEqual(context.dtype, "bf16")
        # MHA 解释(§4.2.6)+ swiglu -> rmsnorm + rope。
        self.assertEqual(context.model_dims["num_q_heads"],
                         context.model_dims["num_kv_heads"])
        self.assertEqual(context.model_dims["head_dim"], 8)
        self.assertEqual(context.model_dims["norm_type"], "rmsnorm")
        self.assertEqual(context.model_dims["position_encoding"], "rope")
        self.assertEqual(len(context.config_hash), 64)

    def test_legacy_manifests_return_none_or_fail(self):
        # 无键(旧 manifest)与显式 roofline/replay -> None(legacy)。
        for manifest in ({"requests": []},
                         {"requests": [],
                          "compute-cost-model": "roofline"},
                         {"requests": [], "compute-cost-model": "replay"}):
            self.assertIsNone(
                DieModelContext.from_service(manifest, self.config))
        # 未知后端 -> fail-closed。
        with self.assertRaisesRegex(ValueError, "unsupported"):
            DieModelContext.from_service(
                self._manifest(**{"compute-cost-model": "quantum"}),
                self.config)
        # die 后端缺 die-model-config -> fail-closed(契约 §7 裁定 2)。
        manifest = self._manifest()
        del manifest["die-model-config"]
        with self.assertRaisesRegex(ValueError, "die-model-config"):
            DieModelContext.from_service(manifest, self.config)
        # 相对路径 -> fail-closed(必须绝对路径)。
        with self.assertRaisesRegex(ValueError, "absolute"):
            DieModelContext.from_service(
                self._manifest(**{"die-model-config": "resolved_die_model.json"}),
                self.config)

    def test_missing_die_config_file_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            DieModelContext.from_service(
                self._manifest(**{
                    "die-model-config": str(
                        Path(self._temp.name) / "missing.json")}),
                self.config)


if __name__ == "__main__":
    unittest.main()
