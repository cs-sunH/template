#!/usr/bin/env python3
"""test_weight_passes.py -- A0 静态夹具:transformer_pass_aggregated 的
weight_passes 权重口径单测(拼 batch 改造,2026-08-22;不跑仿真)。

验证(设计文档 §3.3 的字节守恒不变量):
  1. 默认 weight_passes(= len(spans))与拆分前历史 batch=1 串行口径
     逐字节一致 —— 用独立推导的参考公式(first principles,与旧实现
     逐项对应)对拍每个算子类的 num_ops/tensor_size/remote_read;
  2. B=2 与 B=1 的同迭代权重字节必须相等(权重 ×weight_passes,与批
     成员数无关 —— 权重×B 是拼 batch 最致命口径错误,陷阱 1);
  3. 激活/KV/AllReduce 字节按成员-迭代逐 span 恒等(与 weight_passes
     无关);
  4. weight_passes 越界(> span 数 / 非正)fail-closed。
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

    def comp(self, name, num_ops, tensor_size, remote_read_size=0):
        self.comp_nodes.append({
            "name": name, "num_ops": num_ops,
            "tensor_size": tensor_size,
            "remote_read": remote_read_size,
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
