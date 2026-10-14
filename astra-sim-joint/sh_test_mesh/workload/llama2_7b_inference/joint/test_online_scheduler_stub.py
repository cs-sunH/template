#!/usr/bin/env python3
"""test_online_scheduler_stub.py -- joint/ 内共享的 Sh30OnlineScheduler
轻量替身夹具（pytest 采集但零用例，仅供同目录与其余测试 import）。

来源（2026-10-14 远读退役 §7.3.1 O3 夹具搬迁）：原 ``online/
test_remote_credit_stream.py`` 的 ``EXEC_INSTANCE``/``HOME_INSTANCE``/
``_scheduler`` 三夹具——该文件随远读 credit 流机制整删，存活消费点
（online/test_joint_o_batch.py 的 O3 merge_tail_gated 纯度门用例）改从
本文件取用。搬迁时按三动作化同步适配：

* 原 ``_scheduler(tile_bytes=...)`` 形参删除——其唯一作用是注入
  ``JOINT_REMOTE_READ_TILE_BYTES``，该 env 已随远读退役进入
  ``_RETIRED_ENV_KEYS``（设值即 fail-closed），替身改用缺省机制配置；
* 构图器配置自包含（不再依赖 online/ 测试文件的 ``_make_config``），
  2 实例缺省形态与原 ``_make_config()`` 逐字段一致。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest joint/test_online_scheduler_stub.py -q
"""
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import FaceHardware, FaceModel  # noqa: E402
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import LinkFlowRegistry  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
)

EXEC_INSTANCE = 0   # ranks (0, 1)
HOME_INSTANCE = 1   # ranks (2, 3)


def _graph_config_for(instance_count):
    """N 实例构图器配置（2 实例缺省 = 原 online 测试 _make_config 形态：
    tp_prefill/tp_decode 两分组；3 实例及以上随实例数扩展）。"""
    if instance_count == 2:
        groups = [
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
            SimpleNamespace(ranks=(2, 3), pg_name="tp_decode"),
        ]
    else:
        groups = [
            SimpleNamespace(ranks=(2 * i, 2 * i + 1),
                            pg_name="tp_instance%d" % i)
            for i in range(instance_count)]
    return SimpleNamespace(
        npus_count=2 * instance_count,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=groups,
        layers=2,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
    )


def _scheduler(layers=2, instance_count=2):
    s = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    s.p_chunk = 512
    # N5：_emit_train 的 merge_tail_gated 纯度标记读 _pending_merge_
    # alarms（__init__ 恒设——替身补设，F6）。
    s._pending_merge_alarms = {}
    s.instances = [
        _OnlineInstanceState(index=i) for i in range(instance_count)]
    s.runtime_by_request_id = {}
    s._train_max_iter = 0
    s._train_instance_index = {}
    s._ready_frontier = set()
    ranks_by_instance = {
        i: (2 * i, 2 * i + 1) for i in range(instance_count)}
    s.topology = SimpleNamespace(
        instance=lambda i: SimpleNamespace(
            size=2, ranks=ranks_by_instance[i]),
        instances=[SimpleNamespace(size=2)] * instance_count)
    s.hardware = FaceHardware(
        mesh_rows=2 * instance_count, mesh_cols=2,
        local_hbm_capacity_bytes=10**9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    s.model = FaceModel(
        layers=layers, hidden_size=64, ffn_size=128, num_heads=8,
        vocab_size=256, bytes_per_elem=2, mlp_variant="gelu")
    s._prefill_task_cache = {}
    s._decode_task_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError）。
    s._decode_task_load_cache = {}
    s._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    s._quota_tracker = None  # off 档 __init__ 初值（F6 销账）
    s.graph = GraphBatchBuilder(_graph_config_for(instance_count))
    s.graph.begin_batch()
    s._batch = {"watches": []}
    s.train_ledger_rows = []
    s.train_ledger_sink = None
    s.joint_config = parse_joint_config({})
    s._joint_flows = LinkFlowRegistry()
    s._pool_ports = SimpleNamespace(
        register=lambda *args, **kwargs: None,
        release_owner=lambda *args, **kwargs: None)
    # 对齐 __init__ 初值（F6 销账：软门已删，_register/_release_
    # transfer_flows 直达 _hbm_ports——替身漏设 = AttributeError）。
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    s._hbm_ports = HbmPortFlowRegistry()
    s._kv_ledger_epoch = 0
    return s
