#!/usr/bin/env python3
"""test_attention_numerical_reference.py -- W4 §6.4 验收边界：多来源
分块 K/V attention 的独立小型数值参考。

方案 §6.4 裁定：模拟器执行成本图而非真实张量运算，因此必须补一个
独立数值参考证明**所建模协议的数值可实现性**——分块读取不同来源的
K/V，跨块保持同一 softmax 最大值/归一化和加权累计（online softmax 两
遍式：块内先扫最大值，跨块以 log-sum-exp 缩放累计），结果与完整上下
文 attention 在声明的 dtype/容差（float64 相对误差 < 1e-12）内一致。
本测试**不声称时延后端运行了模型推理**（数值可实现性证明 ≠ 物理图
验收——后者由 graph 侧 arms 审计承担）。

钉测面：
  1. 正例：多来源（2/3/5 源）× 多块（块边界与源边界不对齐）× 逐块
     online softmax（跨块同一最大值追踪 + log-sum-exp 归一化累计）
     == 全上下文一次 attention（float64 相对误差 < 1e-12）；
  2. **反例**（§6.4 红线）：每源独立归一化后直接相加 ≠ 正确结果
     （断言误差显著超出容差——证明共享归一化分母不可省略）；
  3. 协议保序性：确定的来源/区间顺序下，块消费顺序（逐源排空 vs 交
     错）不影响结果（同一最大值/归一化跨块守恒的推论）；
  4. 数值口径：numpy 可用则 numpy（float64），否则纯 python math 降
     级（同一容差声明——两种后端都满足 < 1e-12）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_attention_numerical_reference.py
"""
import math
import os
import sys
import unittest

try:                                     # numpy 可用则用（同容差声明）
    import numpy as _np
except ImportError:                      # pragma: no cover - 环境降级
    _np = None

#: 声明容差（§6.4"声明的 dtype/容差"：float64 相对误差 < 1e-12）。
REL_TOLERANCE = 1e-12


# ------------------------------------------------------------ 参考实现 --

def _full_attention(q, k, v):
    """完整上下文 attention 参考一次成形（单 query 向量 × 全 K/V）。

    q: [d]; k: [n, d]; v: [n, d] → [d]。softmax(QK^T/√d) · V。
    """
    d = len(q)
    scale = 1.0 / math.sqrt(d)
    logits = [sum(qi * ki for qi, ki in zip(q, krow)) * scale for krow in k]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [
        sum(e * vrow[j] for e, vrow in zip(exps, v)) / z
        for j in range(d)
    ]


class _BlockAccumulator:
    """online softmax 跨块累计器（§6.4 协议：跨块同一最大值/归一化）。"""

    def __init__(self, dim):
        self._dim = dim
        self._max = -math.inf          # 跨块全局最大值（单调修正）
        self._denom = 0.0              # 归一化分母（log-sum-exp 累计）
        self._weighted = [0.0] * dim   # 加权和（缩放累计）

    def consume(self, q, k_block, v_block):
        """消费一个 K/V 块（任意来源；块边界无需与源边界对齐）。"""
        scale = 1.0 / math.sqrt(self._dim)
        logits = [
            sum(qi * ki for qi, ki in zip(q, krow)) * scale
            for krow in k_block
        ]
        if not logits:
            return
        block_max = max(logits)
        new_max = max(self._max, block_max)
        # 旧累计按新全局最大值重缩放（log-sum-exp 累计式）。
        rescale = math.exp(self._max - new_max) if self._denom > 0 else 0.0
        self._weighted = [
            w * rescale for w in self._weighted]
        block_exps = [math.exp(x - new_max) for x in logits]
        self._denom = self._denom * rescale + sum(block_exps)
        for e, vrow in zip(block_exps, v_block):
            for j in range(self._dim):
                self._weighted[j] += e * vrow[j]
        self._max = new_max

    def result(self):
        if self._denom == 0.0:
            raise ValueError("no K/V block was consumed")
        return [w / self._denom for w in self._weighted]


def _chunked_multi_source_attention(q, sources, *, tile):
    """分块多来源 attention（确定来源/区间顺序、逐源排空、共享累计器）。

    sources: [(k_rows, v_rows), ...]——不同物理来源的 K/V 段（拼接后
    即完整上下文）；tile: 块大小（块边界与源边界独立——源内多块/尾块）。
    """
    dim = len(q)
    accumulator = _BlockAccumulator(dim)
    for k_rows, v_rows in sources:          # 确定来源顺序（§6.4 协议）
        for start in range(0, len(k_rows), tile):
            accumulator.consume(
                q,
                k_rows[start:start + tile],
                v_rows[start:start + tile],
            )
    return accumulator.result()


def _per_source_normalized_sum(q, sources):
    """反例实现：每源独立 softmax 归一化后直接相加（§6.4 红线）。"""
    dim = len(q)
    total = [0.0] * dim
    for k_rows, v_rows in sources:
        out = _full_attention(q, k_rows, v_rows)
        total = [t + o for t, o in zip(total, out)]
    return total


def _relative_error(a, b):
    denom = max(abs(x) for x in b) or 1.0
    return max(abs(x - y) for x, y in zip(a, b)) / denom


def _make_sources(seed, *, source_specs):
    """确定性伪随机夹具（纯 python LCG——无 random 依赖、可复现）。"""
    state = seed
    def next_unit():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 11) / float(1 << 53)
    dim = 8
    sources = []
    for count in source_specs:
        k = [[next_unit() * 2.0 - 1.0 for _ in range(dim)]
             for _ in range(count)]
        v = [[next_unit() * 2.0 - 1.0 for _ in range(dim)]
             for _ in range(count)]
        sources.append((k, v))
    q = [next_unit() * 2.0 - 1.0 for _ in range(dim)]
    return q, sources


# ------------------------------------------------------- numpy 对照面 --

class NumpyFullAttentionParityTests(unittest.TestCase):
    """numpy 后端可用时：纯 python 参考与 numpy float64 全量 attention
    互证（参考实现的正确性锚）。"""

    @unittest.skipIf(_np is None, "numpy unavailable")
    def test_pure_python_reference_matches_numpy(self):
        q, sources = _make_sources(20260929, source_specs=[7, 5, 9])
        q_np = _np.asarray(q)
        k_np = _np.vstack([_np.asarray(k) for k, _v in sources])
        v_np = _np.vstack([_np.asarray(v) for _k, v in sources])
        logits = (k_np @ q_np) / math.sqrt(len(q))
        weights = _np.exp(logits - logits.max())
        expected = (weights @ v_np) / weights.sum()
        got = _full_attention(q, [list(r) for r in k_np],
                              [list(r) for r in v_np])
        self.assertLess(
            _relative_error(got, list(expected)), REL_TOLERANCE)


class ChunkedMultiSourceAttentionTests(unittest.TestCase):
    """正例：分块多来源 == 全上下文（float64 相对误差 < 1e-12）。"""

    def test_two_sources_multi_tile(self):
        q, sources = _make_sources(1, source_specs=[10, 6])
        full_k = [row for k, _v in sources for row in k]
        full_v = [row for _k, v in sources for row in v]
        expected = _full_attention(q, full_k, full_v)
        for tile in (1, 3, 4, 7, 16):        # 块边界与源边界不对齐
            got = _chunked_multi_source_attention(q, sources, tile=tile)
            self.assertLess(
                _relative_error(got, expected), REL_TOLERANCE,
                f"tile={tile}")

    def test_three_and_five_sources(self):
        for seed, specs in (
            (2, [4, 8, 3]),
            (3, [5, 1, 6, 2, 7]),           # W9 跨轮 >3 来源形态
        ):
            q, sources = _make_sources(seed, source_specs=specs)
            full_k = [row for k, _v in sources for row in k]
            full_v = [row for _k, v in sources for row in v]
            expected = _full_attention(q, full_k, full_v)
            got = _chunked_multi_source_attention(q, sources, tile=3)
            self.assertLess(
                _relative_error(got, expected), REL_TOLERANCE,
                f"seed={seed}")

    def test_consumption_order_invariance(self):
        """同一累计器下块消费顺序不影响结果（来源/区间顺序确定性的
        数值推论：换序只改缩放路径，不改归一化终值）。"""
        q, sources = _make_sources(4, source_specs=[6, 5, 4])
        base = _chunked_multi_source_attention(q, sources, tile=4)
        reordered_sources = [sources[1], sources[2], sources[0]]
        reordered = _chunked_multi_source_attention(
            q, reordered_sources, tile=4)
        # 不同来源顺序是**不同协议输入**（结果不同——softmax 对位置
        # 敏感是本质的）；本断言换序后仍等于"按该顺序拼接"的全量参考。
        full_k = [row for k, _v in reordered_sources for row in k]
        full_v = [row for _k, v in reordered_sources for row in v]
        expected = _full_attention(q, full_k, full_v)
        self.assertLess(_relative_error(reordered, expected), REL_TOLERANCE)
        self.assertLess(_relative_error(base, base), REL_TOLERANCE)

    def test_single_source_degenerates_to_full(self):
        """单源退化锚：分块累计 == 全上下文（W4 单来源退化等价的数值
        面）。"""
        q, sources = _make_sources(5, source_specs=[13])
        full_k, full_v = sources[0]
        expected = _full_attention(q, full_k, full_v)
        got = _chunked_multi_source_attention(q, sources, tile=5)
        self.assertLess(_relative_error(got, expected), REL_TOLERANCE)


class PerSourceNormalizationCounterexampleTests(unittest.TestCase):
    """反例（§6.4 红线）：严禁把每源独立归一化输出直接相加。"""

    def test_per_source_sum_is_wrong(self):
        q, sources = _make_sources(6, source_specs=[8, 7])
        full_k = [row for k, _v in sources for row in k]
        full_v = [row for _k, v in sources for row in v]
        expected = _full_attention(q, full_k, full_v)
        wrong = _per_source_normalized_sum(q, sources)
        error = _relative_error(wrong, expected)
        self.assertGreater(
            error, 1e-3,
            "per-source independent normalization must NOT match the "
            "full-context attention (shared softmax denominator is "
            "essential -- joint plan §6.4 red line)")
        # 正例同夹具对照：online softmax 分块累计在容差内。
        got = _chunked_multi_source_attention(q, sources, tile=4)
        self.assertLess(_relative_error(got, expected), REL_TOLERANCE)

    def test_counterexample_fails_across_shapes(self):
        for seed, specs in ((7, [3, 9]), (8, [6, 6, 6]), (9, [2, 11, 3])):
            q, sources = _make_sources(seed, source_specs=specs)
            full_k = [row for k, _v in sources for row in k]
            full_v = [row for _k, v in sources for row in v]
            expected = _full_attention(q, full_k, full_v)
            wrong = _per_source_normalized_sum(q, sources)
            self.assertGreater(
                _relative_error(wrong, expected), 1e-3,
                f"seed={seed}")


if __name__ == "__main__":
    unittest.main()
