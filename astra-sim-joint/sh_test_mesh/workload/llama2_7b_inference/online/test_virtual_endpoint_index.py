"""Virtual-endpoint membership index 的合同测试（joint 改造 v10）。

is_virtual_endpoint 的线性全量扫描已由 _VirtualEndpointLookup（按 first
排序副本 + last 的 max 线段树，bisect 截断 + 剪枝下钻）直接替换。本测试
驱动真实 OnlineTraceBuilder 走全部注册分支（追加 / 三个尾合并分支 /
交错重叠等差数列 / 乱序防御路径），把 is_virtual_endpoint 与旧线性扫描
的逐字参考实现在全 id 域 + 随机 id 上逐项对齐，并在生产量级（数万
progression）复核机制。只验证合同与规模机制，不做镜像实现堆叠。
"""

import random

from graph_batch_builder import OnlineTraceBuilder


def reference_is_virtual_endpoint(ranges, node_id):
    """v10 之前的线性扫描参考（逐字），仅用于等价对照。"""
    if node_id is None:
        return False
    node_id = int(node_id)
    for first, count, stride in ranges:
        delta = node_id - first
        if delta >= 0 and delta % stride == 0 and delta // stride < count:
            return True
    return False


def checked_register(builder, first, count, stride):
    """走真实 reserve/mark 入口：先占位再注册，保持 id 语义。"""
    builder.next_id = max(builder.next_id, first + (count - 1) * stride + 1)
    builder.mark_virtual_endpoint_progression(first, count, stride)


def assert_equivalent(builder, extra_ids=()):
    ranges = list(builder.virtual_endpoint_ranges)
    for node_id in list(range(0, builder.next_id + 3)) + list(extra_ids):
        assert builder.is_virtual_endpoint(node_id) == \
            reference_is_virtual_endpoint(ranges, node_id), (
                f"membership diverged at id={node_id} with "
                f"ranges={ranges}")


def test_singleton_and_adjacent_merges():
    builder = OnlineTraceBuilder(0, remote_operand_loads=False)
    checked_register(builder, 10, 1, 1)
    checked_register(builder, 13, 1, 1)   # 分支1：两 singleton 合并为 stride
    assert builder.virtual_endpoint_ranges == [(10, 2, 3)]
    assert builder.is_virtual_endpoint(10)
    assert builder.is_virtual_endpoint(13)
    assert not builder.is_virtual_endpoint(11)
    assert not builder.is_virtual_endpoint(12)
    assert_equivalent(builder)


def test_same_stride_extension_merge():
    builder = OnlineTraceBuilder(0, remote_operand_loads=False)
    checked_register(builder, 0, 4, 2)     # {0,2,4,6}
    checked_register(builder, 8, 3, 2)     # 分支3：expected=8, 同 stride
    assert builder.virtual_endpoint_ranges == [(0, 7, 2)]
    assert_equivalent(builder)


def test_expected_singleton_merge_branch():
    builder = OnlineTraceBuilder(0, remote_operand_loads=False)
    checked_register(builder, 5, 3, 1)     # {5,6,7}, expected=8
    checked_register(builder, 8, 1, 4)     # 分支2条件外：stride 不同于尾项
    checked_register(builder, 9, 1, 1)     # 分支1：与 singleton (8,1,4) 合并
    assert builder.virtual_endpoint_ranges == [(5, 3, 1), (8, 2, 1)]
    assert_equivalent(builder)


def test_overlapping_interleaved_progressions():
    builder = OnlineTraceBuilder(0, remote_operand_loads=False)
    checked_register(builder, 0, 50, 2)    # 偶数
    checked_register(builder, 1, 50, 2)    # 奇数（区间 [0,98]x[1,99] 重叠）
    checked_register(builder, 90, 5, 1)    # 尾部跨奇偶
    assert_equivalent(builder)
    assert builder.is_virtual_endpoint(97)
    assert not builder.is_virtual_endpoint(99 + 1)


def test_out_of_order_registration_defensive_path():
    builder = OnlineTraceBuilder(0, remote_operand_loads=False)
    checked_register(builder, 40, 2, 1)
    checked_register(builder, 10, 3, 3)    # 乱序：触发防御性重建路径
    checked_register(builder, 25, 1, 1)
    assert_equivalent(builder)


def test_none_and_boundary_ids():
    builder = OnlineTraceBuilder(0, remote_operand_loads=False)
    checked_register(builder, 7, 1, 1)
    assert not builder.is_virtual_endpoint(None)
    assert builder.is_virtual_endpoint(7)
    assert not builder.is_virtual_endpoint(6)
    assert not builder.is_virtual_endpoint(8)
    assert not builder.is_virtual_endpoint(-1)
    assert_equivalent(builder)


def test_production_scale_mechanism():
    """生产量级（数万 progression，对齐实测 ~5-8K 并持续增长）机制复核。"""
    rng = random.Random(20261013)
    builder = OnlineTraceBuilder(0, remote_operand_loads=False)
    next_first = 0
    for _ in range(20000):
        count = rng.randint(1, 4)
        stride = rng.randint(1, 3)
        checked_register(builder, next_first, count, stride)
        next_first += count * stride + rng.randint(0, 2)
    ranges = list(builder.virtual_endpoint_ranges)
    assert len(ranges) > 8000  # 合并后仍达生产量级
    probes = ([rng.randrange(0, next_first) for _ in range(4000)]
              + [next_first - 1, next_first, next_first + 1, -3])
    for node_id in probes:
        assert builder.is_virtual_endpoint(node_id) == \
            reference_is_virtual_endpoint(ranges, node_id), (
                f"scale divergence at id={node_id}")
