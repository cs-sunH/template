#!/usr/bin/env python3
"""test_weight_passes.py -- A0 静态夹具:transformer_pass_aggregated 的
weight_passes 权重口径单测(拼 batch 改造,2026-08-22;不跑仿真;
照母本 sh_1.0 定型版移植,本仓 generate_trace.py 增量参数化同构)。

验证(设计文档 §3.3 的字节守恒不变量):
  1. 默认 weight_passes(= len(spans))与拆分前历史 batch=1 串行口径
     逐字节一致 —— 用独立推导的参考公式(first principles,与旧实现
     逐项对应)对拍每个算子类的 num_ops/tensor_size/remote_read;
  2. B=2 与 B=1 的同迭代权重字节必须相等(权重 ×weight_passes,与批
     成员数无关 —— 权重×B 是拼 batch 最致命口径错误,陷阱 1);
  3. 激活/KV/AllReduce 字节按成员-迭代逐 span 恒等(与 weight_passes
     无关);
  4. weight_passes 越界(> span 数 / 非正)fail-closed。

remote_read 口径(2026-10-14 远读退役 D-2 裁定):本文件的 remote_read
断言锚定**权重远载字节**(generate_trace 真值供给,保留红线 §1.2 第
2 条;trace_config 现值 remote_operand_loads=false ⇒ 缺省恒 0)——
该统计键与已退役的 remote-read 执行流无关,调度器侧供给点已删。
"""

import os
import sys

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_trace import (  # noqa: E402
    matmul_ops,
    shard_extent,
    tensor_bytes,
    transformer_pass_aggregated,
)


class _RecordingBuilder:
    """TraceBuilder API 子集(comp/all_reduce)的只记录替身。"""

    def __init__(self):
        self.comp_nodes = []
        self.all_reduces = []

    def comp(self, name, num_ops, tensor_size, remote_read_size=0,
             *, die_spec=None, op_role=None):
        self.comp_nodes.append({
            "name": name, "num_ops": num_ops,
            "tensor_size": tensor_size,
            "remote_read": remote_read_size,
            "die_spec": die_spec,
            "op_role": op_role,
        })

    def all_reduce(self, name, comm_size, pg_name):
        self.all_reduces.append({"name": name, "bytes": comm_size})


KWARGS = dict(
    layers=32,
    hidden_size=4096,
    ffn_size=11008,
    tensor_parallel=6,
    pg_name="tp6",
    vocab_size=32000,
    bytes_per_elem=2,
    num_heads=32,
    tensor_parallel_rank=0,
    mlp_variant="swiglu",
)


def _run(spans, **overrides):
    builder = _RecordingBuilder()
    transformer_pass_aggregated(builder, phase="test", pass_spans=spans,
                                **{**KWARGS, **overrides})
    # die 分支不 return spans 数（legacy 相同）；_run 只消费节点记录。
    totals = {}
    for node in builder.comp_nodes:
        category = node["name"].rsplit("_", 1)[-1]
        # name 形如 "test_all_layers_attention_qkv_projection":取
        # "all_layers_"/"all_passes_" 后缀段做类别键。
        name = node["name"]
        if "_all_layers_" in name:
            category = name.split("_all_layers_", 1)[1]
        elif "_all_passes_" in name:
            category = "pass_" + name.split("_all_passes_", 1)[1]
        totals.setdefault(category, [0, 0, 0])
        totals[category][0] += node["num_ops"]
        totals[category][1] += node["tensor_size"]
        totals[category][2] += node["remote_read"]
    ar_bytes = sum(node["bytes"] for node in builder.all_reduces)
    return totals, ar_bytes


def _reference_weight_bytes():
    """每算子类的权重常量(与旧实现逐 span 计入的常量逐项对应)。"""
    attention_hidden_per_rank = (
        shard_extent(32, 6, 0) * (4096 // 32))
    ffn_per_rank = shard_extent(11008, 6, 0)
    vocab_per_rank = shard_extent(32000, 6, 0)
    return {
        "attention_input_rmsnorm": tensor_bytes(4096, 2),
        "attention_qkv_projection":
            4096 * (3 * attention_hidden_per_rank) * 2,
        "attention_output_projection":
            attention_hidden_per_rank * 4096 * 2,
        "mlp_input_rmsnorm": tensor_bytes(4096, 2),
        "mlp_gate_up_projection": 2 * 4096 * ffn_per_rank * 2,
        "mlp_down_projection": ffn_per_rank * 4096 * 2,
        "pass_final_rmsnorm": tensor_bytes(4096, 2),
        "pass_logits_projection": 4096 * vocab_per_rank * 2,
    }


def _reference_activation_parts(spans):
    """逐 span 激活/KV 分量参考(不含权重常量;与旧实现的非常量部分
    逐项对应)。返回 {category: [ops, tensor, remote]}。"""
    attention_hidden_per_rank = (
        shard_extent(32, 6, 0) * (4096 // 32))
    ffn_per_rank = shard_extent(11008, 6, 0)
    vocab_per_rank = shard_extent(32000, 6, 0)
    heads_per_rank = shard_extent(32, 6, 0)
    totals = {}
    for tokens, kv in spans:
        act = tensor_bytes(tokens * 4096, 2)
        shard = tensor_bytes(tokens * attention_hidden_per_rank, 2)
        ffn_shard = tensor_bytes(tokens * ffn_per_rank, 2)
        score = tensor_bytes(tokens * kv * heads_per_rank, 2)
        cache = tensor_bytes(kv * attention_hidden_per_rank, 2)
        logits_out = tensor_bytes(tokens * vocab_per_rank, 2)
        per_span = {
            "attention_input_rmsnorm": (
                5 * tokens * 4096, 3 * act, act),
            "attention_qkv_projection": (
                matmul_ops(tokens, 3 * attention_hidden_per_rank, 4096),
                act + 3 * shard, act),
            "attention_qk_matmul": (
                matmul_ops(tokens, kv, attention_hidden_per_rank),
                shard + cache + score, shard + cache),
            "attention_scale_mask": (2 * tokens * kv * heads_per_rank,
                                     2 * score, score),
            "attention_softmax": (5 * tokens * kv * heads_per_rank,
                                  3 * score, score),
            "attention_av_matmul": (
                matmul_ops(tokens, attention_hidden_per_rank, kv),
                score + cache + shard, score + cache),
            "attention_output_projection": (
                matmul_ops(tokens, 4096, attention_hidden_per_rank),
                shard + act, shard),
            "attention_residual_add": (tokens * 4096, 3 * act, 2 * act),
            "mlp_input_rmsnorm": (5 * tokens * 4096, 3 * act, act),
            "mlp_gate_up_projection": (
                matmul_ops(tokens, 2 * ffn_per_rank, 4096),
                act + 2 * ffn_shard, act),
            "mlp_swiglu": (6 * tokens * ffn_per_rank,
                           3 * ffn_shard, 2 * ffn_shard),
            "mlp_down_projection": (
                matmul_ops(tokens, 4096, ffn_per_rank),
                ffn_shard + act, ffn_shard),
            "mlp_residual_add": (tokens * 4096, 3 * act, 2 * act),
            "pass_final_rmsnorm": (5 * tokens * 4096, 3 * act, act),
            "pass_logits_projection": (
                matmul_ops(tokens, vocab_per_rank, 4096),
                act + logits_out, act),
        }
        for category, parts in per_span.items():
            totals.setdefault(category, [0, 0, 0])
            for index in range(3):
                totals[category][index] += parts[index]
    return totals


def test_default_matches_legacy_reference_byte_exact():
    """默认 weight_passes 与拆分前旧口径逐字节一致(独立参考公式)。"""
    spans = [(1, 33), (1, 34), (512, 900), (7, 120)]
    totals, _ = _run(spans)
    reference = _reference_activation_parts(spans)
    weights = _reference_weight_bytes()
    for category, ref in reference.items():
        weight = weights.get(category, 0)
        expected = [ref[0], ref[1] + weight * len(spans),
                    ref[2] + weight * len(spans)]
        # aggregate_comp 对层类乘 layers=32;all_passes 类不乘。
        if not category.startswith("pass_"):
            expected = [expected[0] * 32, expected[1] * 32,
                        expected[2] * 32]
        actual = totals[category]
        assert actual == expected, (
            f"{category}: got {actual}, expected {expected}")
    # 显式 weight_passes=len(spans) 与缺省逐字节一致。
    explicit, ar_explicit = _run(spans, weight_passes=len(spans))
    assert explicit == totals and ar_explicit is not None


def test_b2_b1_same_iteration_weight_bytes_equal():
    """陷阱 1 防护:B=2 与 B=1 同迭代权重字节必须相等。

    同一上下文 c=100:B=1 = spans[(1,100)] weight_passes=1;
    B=2 = spans[(1,100),(1,100)] weight_passes=1。
    差值必须恰好等于逐 span 激活/KV 分量(权重零增量)。
    """
    b1, _ = _run([(1, 100)], weight_passes=1)
    b2, _ = _run([(1, 100), (1, 100)], weight_passes=1)
    weights = _reference_weight_bytes()
    one_span = _reference_activation_parts([(1, 100)])
    for category in b1:
        weight = weights.get(category, 0)
        # 每 rank 权重常量被 layers 乘(层类);B=1 总量 = act + 1×weight
        layers_factor = 1 if category.startswith("pass_") else 32
        expected_b1 = [
            one_span[category][0] * layers_factor,
            (one_span[category][1] + weight) * layers_factor,
            (one_span[category][2] + weight) * layers_factor,
        ]
        assert b1[category] == expected_b1, (
            f"B=1 {category}: {b1[category]} != {expected_b1}")
        expected_b2 = [
            2 * expected_b1[0],
            expected_b1[1] + one_span[category][1] * layers_factor,
            expected_b1[2] + one_span[category][2] * layers_factor,
        ]
        assert b2[category] == expected_b2, (
            f"B=2 {category}: {b2[category]} != {expected_b2}")


def test_activation_kv_ar_independent_of_weight_passes():
    """激活/KV/AllReduce/num_ops 分量与 weight_passes 无关;只有权重
    分量随其缩放(差值 = 权重常量 × Δpasses)。"""
    spans = [(1, 50), (1, 51), (512, 600)]
    w1, ar1 = _run(spans, weight_passes=1)
    w2, ar2 = _run(spans, weight_passes=2)
    assert ar1 == ar2, "AllReduce bytes must not depend on weight_passes"
    weights = _reference_weight_bytes()
    for category in w1:
        delta_tensor = w2[category][1] - w1[category][1]
        delta_remote = w2[category][2] - w1[category][2]
        delta_ops = w2[category][0] - w1[category][0]
        assert delta_ops == 0, f"num_ops changed for {category}"
        weight = weights.get(category, 0)
        layers_factor = 1 if category.startswith("pass_") else 32
        assert delta_tensor == weight * layers_factor, category
        assert delta_remote == weight * layers_factor, category


def test_weight_passes_validation_fail_closed():
    for bad in (0, -1, 4):
        try:
            _run([(1, 10), (1, 11)], weight_passes=bad)
        except ValueError:
            continue
        raise AssertionError(
            f"weight_passes={bad} must fail closed (spans=2)")


# ---------------------------------------------------------------------------
# die 后端（第一期）：同 span 的 legacy/die 发射守恒 + spec 口径。
# ---------------------------------------------------------------------------


def _make_die_context():
    from die_model_adapter import DieModelContext
    return DieModelContext(
        profile="mesh_die_v1",
        die_config_path="/nonexistent/resolved_die_model.json",
        config_hash="ef" * 32,
        dtype="bf16",
        model_dims={
            "hidden": 4096, "ffn": 11008, "vocab": 32000,
            "mlp_variant": "swiglu", "norm_type": "rmsnorm",
            "position_encoding": "rope", "num_q_heads": 32,
            "num_kv_heads": 32, "head_dim": 128,
        },
        num_q_heads=32, num_kv_heads=32, ffn_size=11008, vocab_size=32000)


def _run_die(spans, **overrides):
    from generate_trace import transformer_pass_aggregated as tpa
    builder = _RecordingBuilder()
    die_ctx = _make_die_context()
    # 发射语境的 phase 是列车 stage 语义（prefill/decode）；测试用 decode
    # （节点命名 phase 形参仍为 "test"，互不影响）。
    die_ctx.begin_emission(
        phase="decode", tensor_parallel=KWARGS["tensor_parallel"],
        tensor_parallel_rank=KWARGS["tensor_parallel_rank"],
        span_namespace="test", pass_spans=spans)
    tpa(builder, phase="test", pass_spans=spans,
        die_ctx=die_ctx, **{**KWARGS, **overrides})
    return builder


def _spec_layers_in_category(builder, suffix):
    for node in builder.comp_nodes:
        if node["name"].endswith(suffix) and node["die_spec"] is not None:
            return node["die_spec"]["layer_count"], node["die_spec"][
                "weight_passes"], len(node["die_spec"]["spans"])
    raise AssertionError(f"no die node ends with {suffix!r}")


def test_die_emission_conserves_totals_vs_legacy():
    """同 spans/weight_passes 的 die 与 legacy 发射：逐类别 num_ops/
    tensor_size/remote_read 总量与 AllReduce 字节逐字节相等（融合组 =
    成分之和）。"""
    spans = [(1, 33), (1, 34), (512, 900), (7, 120)]
    legacy_totals, legacy_ar = _run(spans, weight_passes=2)
    die_builder = _run_die(spans, weight_passes=2)
    die_totals = {}
    for node in die_builder.comp_nodes:
        name = node["name"]
        if "_all_layers_" in name:
            category = name.split("_all_layers_", 1)[1]
            # 融合组名（attention_region/ffn_region）不对应 legacy 类别键
            # ——按成分分解：融合组的总量在守恒对拍中按"层类合计"核对。
            category = "LAYER_AGG"
        elif "_all_passes_" in name:
            category = "pass_" + name.split("_all_passes_", 1)[1]
        else:
            category = "LAYER_AGG"
        die_totals.setdefault(category, [0, 0, 0])
        die_totals[category][0] += node["num_ops"]
        die_totals[category][1] += node["tensor_size"]
        die_totals[category][2] += node["remote_read"]
    die_ar = sum(node["bytes"] for node in die_builder.all_reduces)

    legacy_layer = [0, 0, 0]
    for category, totals in legacy_totals.items():
        if category.startswith("pass_"):
            assert die_totals[category] == totals, category
        else:
            for i in range(3):
                legacy_layer[i] += totals[i]
    assert die_totals["LAYER_AGG"] == legacy_layer, (
        f"layer totals diverged: {die_totals['LAYER_AGG']} != {legacy_layer}")
    assert die_ar == legacy_ar


def test_die_spec_carries_weight_passes_and_full_layer_count():
    """spec 的 layer_count=全模层数（全区间发射）、weight_passes 恒 1、
    spans 绝对位置与平铺表一致。

    2026-10-19 wp 语义修复（M1）：coarse 合同下 spec 的 weight_passes
    恒为 1（repeat 计数 = layer_instances × weight_passes，spans 已逐
    迭代枚举全部工作，整段模板只执行一遍）；上游传入的列车迭代数只
    服务旧引擎的字节簿记（generate_trace.py 口径），不再进入 spec。"""
    spans = [(1, 33), (512, 900), (1, 901)]
    builder = _run_die(spans, weight_passes=3)
    layers, passes, span_count = _spec_layers_in_category(
        builder, "_mlp_residual_add")
    assert layers == 32
    assert passes == 1
    assert span_count == 3
    tail_layers, tail_passes, tail_spans = _spec_layers_in_category(
        builder, "_all_passes_logits_projection")
    assert tail_layers == 1
    assert tail_passes == 1
    assert tail_spans == 3
    # 绝对位置恒等式（q_start = kv - tokens, kv_start = 0）。
    # v5 紧凑 run：spec.spans 为混合 [span|run]，先规范展开再逐项断言。
    from die_model_adapter import expand_span_records
    for node in builder.comp_nodes:
        if node["die_spec"] is None:
            continue
        expanded = expand_span_records(node["die_spec"]["spans"])
        assert len(expanded) == len(spans)
        for index, span in enumerate(expanded):
            tokens, kv = spans[index]
            assert span["q_tokens"] == tokens
            assert span["kv_tokens"] == kv
            assert span["q_start_pos"] == kv - tokens
            assert span["kv_start_pos"] == 0


def test_die_b2_b1_weight_bytes_equal():
    """陷阱 1 防护在 die 分支同样成立：B=2 与 B=1 同迭代（同
    weight_passes=1）的 tensor 总量差 = 第二成员的逐 span 激活/KV 分量
    （权重零增量；参考公式与 A0 静态夹具同源）。"""
    b1 = _run_die([(1, 100)], weight_passes=1)
    b2 = _run_die([(1, 100), (1, 100)], weight_passes=1)

    def tensor_sum(builder):
        return sum(node["tensor_size"] for node in builder.comp_nodes)

    member_activation = sum(
        parts[1]
        for parts in _reference_activation_parts([(1, 100)]).values())
    # 层类参考按单 span 计；发射侧对层类乘 layers、tail 不乘——激活
    # 参考须按同口径折叠（层类 ×32，pass_ 类 ×1）。
    folded = 0
    for category, parts in _reference_activation_parts([(1, 100)]).items():
        factor = 1 if category.startswith("pass_") else 32
        folded += parts[1] * factor
    assert tensor_sum(b2) - tensor_sum(b1) == folded, (
        f"B=2 - B=1 = {tensor_sum(b2) - tensor_sum(b1)}, "
        f"member activation = {folded} (weights must not double)")
