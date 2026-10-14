#!/usr/bin/env python3
"""die_model_adapter.py -- die 后端的 spec 构造与生成期 region 编译
（joint 推广第一期；移植自 face 参考实现并按本仓发射面适配）。

License: MIT (see LICENSE in the repository root).

本模块是 Python 生成侧对 extern/die_model 共享成本库的唯一桥接层（设计文档
§8.1："sh_test_mesh/.../die_model_adapter.py -- 配置、形状和 region 编译；
不实现第二套公式"）。只做三件事，不实现任何派生/时间公式：

  1. DieModelContext：数据载体（backend/模型维度/TP/config_hash 等），由
     online_scheduler_base 从 manifest 的 compute-cost-model /
     die-model-config 键构造，供 graph_batch_builder 的发射路径共用；本类
     不做任何决策；
  2. build_die_spec()：按契约 §2 schema 构造 die_op_spec dict（旧 COMP
     类别 -> op_kind 映射、spans 绝对位置、kv_access.identity、
     provenance=transformer_pass_aggregated、config_hash 取 resolved 配置
     内容摘要）；
  3. compile_regions()：§7.2 纯本地链融合（QK->scale/mask->softmax->AV =>
     attention_region；gate/up->activation->down => ffn_region），条件全部
     满足才融合，否则分离执行。

spans 绝对位置管线（设计文档 §4.2.7）：joint 在线调度器把 chunk/成员×迭代
平铺成 (tokens, kv_length) 二元组（sh30_online_scheduler._plan_train；调度
器属第二期改造面，本仓不改写其 span 构造）。本仓全部 span 构造点满足同一
恒等式：kv_tokens == q_start_pos + q_tokens 且 KV 自序列位置 0 连续覆盖
（kv_start_pos == 0）：

  - prefill/重算段 chunk:  span = (chunk_tokens, history + completed + chunk_tokens)
                           -> q 起点 = 该 chunk 前已生成 token 数
                              history + completed（跨实例 recompute 的
                              working-copy 基数亦自 0 起，R13 口径）；
  - D10 静默重建 span:     span = (H_B, H_B)
                           -> H_B 前向整体重放：q 覆盖 [0, H_B)（q 起点 0，
                              kv 覆盖 [0, H_B)）；
  - decode 迭代:           span = (1, context + consumed + step)
                           -> q = 当前末 token 位置（= kv-1）；kv =
                              [会话起点, 当前]。

derive_span_records() 按该恒等式还原绝对位置（上游 session KV 账本口径，
不发明新状态）；违反即 fail-closed，绝不猜位置。spec 的 spans[].request_id
用列车命名空间合成 id（聚合图无逐 span 请求身份；C++ 库按 opaque 身份处
理，attention 逐 span Σ(q_i·k_i) 计费不受影响）。

kv_access.identity（契约 §2）：三动作（stay/copy/recompute）下列车体
消费的 KV 恒为本地 HBM 就绪数据（copy 交接/池恢复腿的门控保证体块计算
节点只在其门就绪后发射），identity 恒 "local"。该注记只进 plan cache
key 与报表（库内无成本语义），不改变计费。

watches 绑定核实（融合阻断条件的事实基础，2026-10-03）：本仓 watch 全部
经发射函数返回的 per-rank 节点 id 映射或标记节点名绑定
（drain/exit/sentinel/first_token/wakeup/merge_done 等标记 COMP 节点、
end barrier collective 节点），从不按类别名绑定列车体 13 类聚合节点；
唯一按名字绑定的 C++ 锚点是 first_token 子串（code 8），其载体也只是标记
节点名。列车体 13 类聚合节点既无 watch 也无标记子串，region 融合不改变任
何 watch 引用的节点身份；_WATCH_NAME_ANCHORS 仍作为防御性阻断条件（融合
节点名不得含任何锚点子串）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# --------------------------------------------------------------------------
# import 路径：本文件位于 workload/llama2_7b_inference/online/，共享发射
# 模块（shard_extent）在上一级。只做 import 用途。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_trace import shard_extent  # noqa: E402

# 契约 §7 裁定 2 的 manifest 新键（与 face 参考实现同名常量）。
MANIFEST_COMPUTE_COST_MODEL_KEY = "compute-cost-model"
MANIFEST_DIE_MODEL_CONFIG_KEY = "die-model-config"

# spec 固定元数据。
SPEC_SCHEMA_VERSION = 1
PROVENANCE = "transformer_pass_aggregated"
DEFAULT_DIE_PROFILE = "mesh_die_v1"
COARSE_DIE_PROFILE = "coarse_die_v2"
LEGACY_DIE_TASK_SCHEMA = "die_op_spec_v1"
COARSE_DIE_TASK_SCHEMA = "coarse_die_task_v2"

# 旧 COMP 类别 -> op_kind 映射（契约 §2"语义"小节；norm/residual 在
# attention/mlp/final 三处复用，qkv 投影恒映射 qkv_rope，是否含新 Q/K
# 旋转由 model_dims.position_encoding 区分）。joint 的类别命名与 face
# 同源（mlp_variant=swiglu 时 rmsnorm/gate_up/swiglu 名）。
CATEGORY_TO_OP_KIND = {
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
    "mlp_residual_add": "residual",
    "mlp_gate_up_projection": "mlp_up",
    "mlp_up_projection": "mlp_up",
    "mlp_swiglu": "mlp_activation",
    "mlp_gelu": "mlp_activation",
    "mlp_down_projection": "mlp_down",
    "logits_projection": "logits",
}

OP_KINDS = frozenset({
    "norm", "qkv_rope", "qk", "scale_mask", "softmax", "av", "o_proj",
    "residual", "attention_region", "mlp_up", "mlp_activation", "mlp_down",
    "ffn_region", "logits", "local_region",
})

# §7.2 融合链（op_kind 序列）。
ATTENTION_CHAIN = ("qk", "scale_mask", "softmax", "av")
FFN_CHAIN = ("mlp_up", "mlp_activation", "mlp_down")
_FUSION_CHAINS = (ATTENTION_CHAIN, FFN_CHAIN)

# 契约 §7 裁定 8 的 marker/哨兵名字锚点（C++ first_token 子串锚点 +
# 本仓标记命名：drain/exit/sentinel/join/pstart/wakeup、credit 消费
# marker（*_consumed_*）、eviction_done、merge_done）。任何含这些子串的
# 类别名不参与融合，融合节点名含这些子串即 fail-closed。
_WATCH_NAME_ANCHORS = (
    "first_token", "drain", "exit", "sentinel", "join", "pstart", "wakeup",
    "consumed", "eviction_done", "merge_done",
)

# 契约 §2：die_op_spec 存在 => runtime_ns 必须为 0（OnlineTraceBuilder 的
# compute 缺省形即 0；此处仅为发射侧自检）。
SPEC_RUNTIME_NS = 0

# kv_access.identity 合法值（契约 §2）。
KV_IDENTITIES = frozenset({"local"})


def derive_span_records(
    pass_spans: Iterable[Sequence[int]],
    *,
    namespace: str = "span",
) -> list[dict[str, Any]]:
    """把本仓的 (tokens, kv_length) span 平铺表还原为带绝对位置的 die
    span 记录（契约 §2 spans[] schema）。

    恒等式（kv_tokens == q_start_pos + q_tokens、kv_start_pos == 0）来自
    sh30_online_scheduler._plan_train 的三类 span 构造点（模块 docstring）；
    违反即 fail-closed，绝不猜位置。request_id 用 "<namespace>#s<index>"
    合成（聚合图无逐 span 请求身份；库按 opaque 身份处理）。

    v5 G 主线（紧凑 run）：相邻等 q_tokens、常数 kv 步长（步长 >= 1）的
    span 压成一条 run 记录 {"run": {q_tokens, kv_first, kv_stride,
    count}}（展开恒等 kv_i = kv_first + i*stride，逐展开项不变式在压缩时
    校验）；wire/native/计划缓存键按 run 形态消费（见 Workload.cc
    parse_die_op_spec 与 CostModel.cc plan_cache_key v2）。展开后的数值
    内容与逐 span 记录逐项相等（无损压缩）；单 span 记录形态不变。
    """
    flat = [(int(span[0]), int(span[1])) for span in pass_spans]
    records: list[dict[str, Any]] = []
    index = 0
    prefix = "{}#s".format(namespace)
    i = 0
    total = len(flat)
    while i < total:
        tokens, kv_length = flat[i]
        _validate_span_invariants(tokens, kv_length)
        run_len = 1
        if i + 1 < total:
            nxt_tokens, nxt_kv = flat[i + 1]
            stride = nxt_kv - kv_length
            if (nxt_tokens == tokens and stride >= 1
                    and kv_length >= tokens):
                run_len = 2
                j = i + 2
                while (j < total and flat[j][0] == tokens
                       and flat[j][1] - flat[j - 1][1] == stride):
                    run_len += 1
                    j += 1
        if run_len == 1:
            records.append({
                "request_id": "{}{}".format(prefix, index),
                "q_tokens": tokens,
                "kv_tokens": kv_length,
                "q_start_pos": kv_length - tokens,
                "kv_start_pos": 0,
            })
        else:
            records.append({
                "request_id_prefix": prefix,
                "first_index": index,
                "run": {
                    "q_tokens": tokens,
                    "kv_first": kv_length,
                    "kv_stride": stride,
                    "count": run_len,
                },
            })
        index += run_len
        i += run_len
    if not records:
        raise ValueError("die emission requires at least one pass span")
    return records


def _validate_span_invariants(tokens: int, kv_length: int) -> None:
    if tokens < 1 or kv_length < 1:
        raise ValueError(
            "die span must carry positive tokens and kv_length, got "
            f"{(tokens, kv_length)!r}")
    if kv_length < tokens:
        raise ValueError(
            "die span invariant violated: kv_length {} < tokens {} "
            "(joint spans satisfy kv_tokens == q_start_pos + q_tokens; "
            "see sh30_online_scheduler._plan_train)".format(
                kv_length, tokens))


def expand_span_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """把混合 [span | run] 记录逐项展开回单 span 记录（等价性验收与
    遗留消费方的规范形；run 展开 = derive_span_records 压缩的逆）。"""
    out: list[dict[str, Any]] = []
    for record in records:
        run = record.get("run")
        if run is None:
            out.append(dict(record))
            continue
        q = int(run["q_tokens"])
        kv_first = int(run["kv_first"])
        stride = int(run["kv_stride"])
        count = int(run["count"])
        prefix = str(record["request_id_prefix"])
        first = int(record["first_index"])
        for i in range(count):
            kv = kv_first + i * stride
            _validate_span_invariants(q, kv)
            out.append({
                "request_id": "{}{}".format(prefix, first + i),
                "q_tokens": q,
                "kv_tokens": kv,
                "q_start_pos": kv - q,
                "kv_start_pos": 0,
            })
    return out


def expanded_span_count(records: Iterable[Mapping[str, Any]]) -> int:
    """混合记录的展开后 span 数（与旧逐 span 记录列表长度等价）。"""
    total = 0
    for record in records:
        run = record.get("run")
        total += int(run["count"]) if run is not None else 1
    return total


def slice_span_records(
    records: Sequence[Mapping[str, Any]],
    keep: Iterable[int],
) -> list[Mapping[str, Any]]:
    """按展开序号取子集（tail 输出抑制等按索引筛选用）；run 被切割时
    分裂为前缀/中段子 run，数值恒等与展开序保持不变。"""
    keep_set = sorted({int(i) for i in keep})
    out: list[Mapping[str, Any]] = []
    base = 0
    for record in records:
        run = record.get("run")
        if run is None:
            if base in keep_set:
                out.append(record)
            base += 1
            continue
        count = int(run["count"])
        lo, hi = base, base + count  # [lo, hi)
        selected = [i - lo for i in keep_set if lo <= i < hi]
        if selected:
            out.append({
                "request_id_prefix": record["request_id_prefix"],
                "first_index": int(record["first_index"]) + selected[0],
                "run": {
                    "q_tokens": int(run["q_tokens"]),
                    "kv_first": int(run["kv_first"])
                    + int(run["kv_stride"]) * selected[0],
                    "kv_stride": int(run["kv_stride"]),
                    "count": selected[-1] - selected[0] + 1,
                },
            })
        base = hi
    return out


class DieModelContext:
    """die 后端发射语境的数据载体（joint 推广第一期）。

    构造来源：online_scheduler_base 从 manifest（compute-cost-model ==
    "die" + die-model-config 绝对路径）与 FaceTraceConfig 模型行构造。
    每次列车体发射前由 graph_batch_builder._emit_train_body 的 emit_pass
    漏斗调 begin_emission() 写入当次发射的 phase/TP/span 记录/层区间/输出
    抑制（发射序串行，无并发）；transformer_pass_aggregated 经
    build_spec()/compile_regions() 消费。本类不持有任何调度决策状态。
    """

    def __init__(
        self,
        *,
        profile: str,
        die_config_path: Path,
        config_hash: str,
        dtype: str,
        model_dims: dict[str, Any],
        num_q_heads: int,
        num_kv_heads: int,
        ffn_size: int,
        vocab_size: int,
        task_schema: str = LEGACY_DIE_TASK_SCHEMA,
        execution_profile: Optional[str] = None,
    ) -> None:
        self.backend = "die"
        self.profile = profile
        self.die_config_path = Path(die_config_path)
        self.config_hash = config_hash
        self.dtype = dtype
        if task_schema not in (LEGACY_DIE_TASK_SCHEMA, COARSE_DIE_TASK_SCHEMA):
            raise ValueError("unsupported die task schema {!r}".format(
                task_schema))
        self.task_schema = task_schema
        self.execution_profile = execution_profile
        self.model_dims = dict(model_dims)
        self._num_q_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._ffn_size = ffn_size
        self._vocab_size = vocab_size
        # 每次发射的语境（begin_emission 写入）。
        self.phase = "prefill"
        self.tensor_parallel = 1
        self.tensor_parallel_rank = 0
        self.span_namespace = ""
        self.span_records: list[dict[str, Any]] = []
        # 单列车生命周期的 span 记录/JSON 片段缓存（begin_emission 换
        # namespace 时整体换新；同一列车的全部 rank/体块发射共享）。
        self._span_cache_namespace: str = ""
        self._span_cache: dict[tuple, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------- 构造 --

    @classmethod
    def from_service(cls, manifest: dict, config: Any) -> Optional["DieModelContext"]:
        """从 manifest + FaceTraceConfig 构造；backend 非 die 时返回 None
        （legacy 回归保护：无 manifest 键的旧 plan 目录行为不变）。"""
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a dict for the die backend")
        backend = manifest.get(MANIFEST_COMPUTE_COST_MODEL_KEY)
        if backend is None or backend in ("roofline", "replay"):
            return None
        if backend != "die":
            raise ValueError(
                "unsupported compute-cost-model {!r} in manifest (expected "
                "'die', 'roofline' or 'replay')".format(backend))
        die_config = manifest.get(MANIFEST_DIE_MODEL_CONFIG_KEY)
        if not die_config:
            raise ValueError(
                "manifest key 'die-model-config' is required when "
                "compute-cost-model == 'die' (absolute path of "
                "resolved_die_model.json, contract ruling 2)")
        die_config_path = Path(die_config)
        if not die_config_path.is_absolute():
            raise ValueError(
                "manifest 'die-model-config' must be an absolute path, got "
                "{!r}".format(die_config))
        if not die_config_path.is_file():
            raise ValueError(
                "manifest 'die-model-config' file does not exist: "
                "{}".format(die_config_path))
        config_hash = hashlib.sha256(
            die_config_path.read_bytes()).hexdigest()
        try:
            hardware_config = json.loads(
                die_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "cannot parse resolved die hardware config {}: {}".format(
                    die_config_path, exc)) from exc
        if not isinstance(hardware_config, dict):
            raise ValueError("resolved die hardware config must be a JSON object")
        task_schema = (COARSE_DIE_TASK_SCHEMA
                       if hardware_config.get("schema") == "coarse_die_v2"
                       else LEGACY_DIE_TASK_SCHEMA)
        execution_profile = None
        if task_schema == COARSE_DIE_TASK_SCHEMA:
            performance = hardware_config.get("performance_profile")
            if not isinstance(performance, dict):
                raise ValueError(
                    "coarse_die_v2 hardware config requires an explicit "
                    "performance_profile.execution_profile")
            execution_profile = performance.get("execution_profile")
            if not isinstance(execution_profile, str) or not execution_profile:
                raise ValueError(
                    "coarse_die_v2 hardware config requires an explicit "
                    "performance_profile.execution_profile")

        bytes_per_elem = int(config.bytes_per_elem)
        dtype = {2: "bf16", 4: "fp32"}.get(bytes_per_elem)
        if dtype is None:
            raise ValueError(
                "die backend supports bytes_per_elem 2 (bf16) or 4 (fp32), "
                "got {!r}".format(bytes_per_elem))
        mlp_variant = str(config.mlp_variant)
        if mlp_variant not in ("swiglu", "gelu"):
            raise ValueError(
                "die backend mlp_variant must be swiglu or gelu, got "
                "{!r}".format(mlp_variant))
        hidden = int(config.hidden_size)
        num_heads = int(config.num_heads)
        if hidden % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        # §4.2.6：trace_config 只有 heads 一个头数键 -> MHA 解释
        # （num_q_heads == num_kv_heads == heads，head_dim = hidden//heads）。
        num_q_heads = num_heads
        num_kv_heads = num_heads
        head_dim = hidden // num_heads
        position_encoding = str(getattr(config, "position_encoding", None)
                                or "none")
        if position_encoding not in ("rope", "none"):
            raise ValueError(
                "position_encoding must be rope or none, got {!r}".format(
                    position_encoding))
        model_dims = {
            "hidden": hidden,
            "ffn": int(config.ffn_size),
            "vocab": int(config.vocab_size),
            "mlp_variant": mlp_variant,
            "norm_type": "rmsnorm" if mlp_variant == "swiglu" else "layernorm",
            "position_encoding": position_encoding,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        profile = DEFAULT_DIE_PROFILE
        system_config = getattr(config, "system_config", None)
        if system_config is not None:
            with open(system_config, "r", encoding="utf-8") as source:
                resolved_system = json.load(source)
            profile = str(resolved_system.get("die-model-profile",
                                             DEFAULT_DIE_PROFILE))
        if task_schema == COARSE_DIE_TASK_SCHEMA:
            if profile != COARSE_DIE_PROFILE:
                raise ValueError(
                    "coarse_die_v2 hardware requires explicit "
                    "die-model-profile='coarse_die_v2'; got {!r}".format(
                        profile))
        elif profile == COARSE_DIE_PROFILE:
            raise ValueError(
                "die-model-profile='coarse_die_v2' requires a "
                "coarse_die_v2 hardware config")
        return cls(
            profile=profile,
            die_config_path=die_config_path,
            config_hash=config_hash,
            dtype=dtype,
            model_dims=model_dims,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            ffn_size=int(config.ffn_size),
            vocab_size=int(config.vocab_size),
            task_schema=task_schema,
            execution_profile=execution_profile,
        )

    # ------------------------------------------------------------- 发射 --

    def begin_emission(
        self,
        *,
        phase: str,
        tensor_parallel: int,
        tensor_parallel_rank: int,
        span_namespace: str,
        pass_spans: Iterable[Sequence[int]],
    ) -> None:
        """写入一次列车体发射的语境（每 rank 一次；phase 取列车 stage：
        "decode"（含混拼）或 "prefill"）。

        span 记录按 (namespace, span 内容) 只派生一次、跨 rank/体块共享
        （v4 G 主线紧凑 span：同一列车的每个 rank/层段/体块发射不再逐
        span 重建记录 dict；记录对象只读共享，子集切片共享同一批 dict，
        无任何下游原地修改——序列化侧经 build_spec_json 复用缓存片段）。
        缓存随 namespace 更替整体换新（单列车生命周期，有界）。
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(
                "die emission phase must be 'prefill' or 'decode', got "
                "{!r}".format(phase))
        if tensor_parallel < 1 or not 0 <= tensor_parallel_rank < tensor_parallel:
            raise ValueError("invalid tensor-parallel slice specification")
        self.phase = phase
        self.tensor_parallel = int(tensor_parallel)
        self.tensor_parallel_rank = int(tensor_parallel_rank)
        self.span_namespace = str(span_namespace)
        spans_key = tuple((int(span[0]), int(span[1])) for span in pass_spans)
        if self._span_cache_namespace != self.span_namespace:
            self._span_cache_namespace = self.span_namespace
            self._span_cache = {}
        cached = self._span_cache.get(spans_key)
        if cached is None:
            cached = derive_span_records(spans_key, namespace=span_namespace)
            self._span_cache[spans_key] = cached
        self.span_records = cached

    def tp_slice(self) -> dict[str, int]:
        """契约 §2 tp_slice（非整除 TP 用 shard_extent，空分片合法）。"""
        return {
            "tensor_parallel": self.tensor_parallel,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "q_heads_local": shard_extent(
                self._num_q_heads, self.tensor_parallel,
                self.tensor_parallel_rank),
            "kv_heads_local": shard_extent(
                self._num_kv_heads, self.tensor_parallel,
                self.tensor_parallel_rank),
            "ffn_local": shard_extent(
                self._ffn_size, self.tensor_parallel,
                self.tensor_parallel_rank),
            "vocab_local": shard_extent(
                self._vocab_size, self.tensor_parallel,
                self.tensor_parallel_rank),
        }

    def build_spec(
        self,
        op_kind: str,
        *,
        layer_count: int,
        weight_passes: int,
        fusion_policy: str = "none",
        spans: Optional[Sequence[dict[str, Any]]] = None,
        kv_identity: str = "local",
    ) -> dict[str, Any]:
        """当前发射语境下的 legacy die_op_spec。

        spans 覆盖：缺省用 begin_emission 写入的全量 span 记录；本仓
        tail（final_norm/logits）在 D10 静默重建 span 抑制时只覆盖未被
        抑制的 span 子集（发射骨架如实传递）。kv_identity 恒 local
        （见模块 docstring）。"""
        if self.task_schema == COARSE_DIE_TASK_SCHEMA:
            raise RuntimeError(
                "coarse_die_v2 graph emission is not wired to JOINT's "
                "die_task_json/Native Runtime path yet; refusing to emit a "
                "legacy die_op_spec (unsupported_projection_state)")
        return build_die_spec(
            op_kind,
            self,
            self.span_records if spans is None else spans,
            layer_count=layer_count,
            weight_passes=weight_passes,
            fusion_policy=fusion_policy,
            kv_identity=kv_identity,
        )

    def build_task(
        self,
        op_kind: str,
        *,
        layer_count: int,
        weight_passes: int,
        fusion_policy: str = "none",
        spans: Optional[Sequence[dict[str, Any]]] = None,
        kv_identity: str = "local",
    ) -> dict[str, Any]:
        """Build the selected wire payload while keeping both schemas typed.

        The coarse branch is an additive projection of the same authored
        category/dimension/span facts. It deliberately does not normalize a
        legacy spec through the C++ compatibility path.
        """
        selected_spans = self.span_records if spans is None else spans
        if self.task_schema == COARSE_DIE_TASK_SCHEMA:
            if kv_identity != "local":
                raise ValueError(
                    "coarse_die_v2 JOINT emission only supports local KV")
            return build_coarse_die_task(
                op_kind, self, selected_spans,
                layer_count=layer_count,
                weight_passes=weight_passes,
                fusion_policy=fusion_policy,
            )
        if self.task_schema != LEGACY_DIE_TASK_SCHEMA:
            raise ValueError("unsupported die task schema {!r}".format(
                self.task_schema))
        return self.build_spec(
            op_kind,
            layer_count=layer_count,
            weight_passes=weight_passes,
            fusion_policy=fusion_policy,
            spans=selected_spans,
            kv_identity=kv_identity,
        )

    def op_kind_of(self, category: str) -> str:
        """旧 COMP 类别 -> 契约 §2 op_kind（模块级映射的唯一入口；
        生成侧 generate_trace 经本方法取映射，不复制映射表）。"""
        try:
            return CATEGORY_TO_OP_KIND[category]
        except KeyError:
            raise ValueError(
                "unknown aggregate COMP category {!r} for the die backend "
                "(no op_kind mapping in contract section 2)".format(
                    category)) from None

    def compile_regions(
        self,
        category_nodes: Sequence[dict[str, Any]],
        watch_targets: Optional[Iterable[str]] = None,
    ) -> list[dict[str, Any]]:
        """生成期 region 编译（委托模块级 compile_regions；缺省阻断 =
        marker 名字锚点子串）。"""
        return compile_regions(category_nodes, watch_targets)


def build_die_spec(
    op_kind: str,
    ctx: DieModelContext,
    spans: Sequence[dict[str, Any]],
    *,
    layer_count: int,
    weight_passes: int,
    fusion_policy: str = "none",
    kv_identity: str = "local",
) -> dict[str, Any]:
    """按契约 §2 schema 构造一个 die_op_spec dict。

    校验只覆盖 schema 形状（枚举/正整数）；数值语义校验由 C++ 库的
    parse_die_op_spec 承担（严格解析）。kv_identity 见模块 docstring
    （本仓恒 local；库内无成本语义，只进缓存键与报表）。
    """
    if op_kind not in OP_KINDS:
        raise ValueError(
            "unknown die op_kind {!r} (contract section 2 enum)".format(
                op_kind))
    if fusion_policy not in ("auto", "none"):
        raise ValueError(
            "fusion_policy must be 'auto' or 'none', got {!r}".format(
                fusion_policy))
    if kv_identity not in KV_IDENTITIES:
        raise ValueError(
            "kv_access.identity must be one of {}, got {!r}".format(
                sorted(KV_IDENTITIES), kv_identity))
    layer_count = int(layer_count)
    weight_passes = int(weight_passes)
    if layer_count < 1:
        raise ValueError("layer_count must be >= 1")
    if weight_passes < 1:
        raise ValueError("weight_passes must be >= 1")
    if not spans:
        raise ValueError("die_op_spec requires at least one span")
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "op_kind": op_kind,
        "phase": ctx.phase,
        "dtype": ctx.dtype,
        "model_dims": dict(ctx.model_dims),
        "tp_slice": ctx.tp_slice(),
        # v4 G-line: span records are shared read-only objects (derived once
        # per train by begin_emission; subset slices reference the same
        # dicts).  No downstream consumer mutates them -- the spec is only
        # serialized (online batch JSON / offline ET attribute), so the
        # historical per-category dict copies only churn memory.
        "spans": list(spans),
        "layer_count": layer_count,
        "weight_passes": weight_passes,
        "kv_access": {"identity": kv_identity},
        "fusion_policy": fusion_policy,
        "provenance": PROVENANCE,
        "config_hash": ctx.config_hash,
    }


_COARSE_DIE_OP_KINDS = frozenset({
    "norm", "qkv_rope", "qk", "scale_mask", "softmax", "av", "o_proj",
    "residual", "attention_region", "mlp_up", "mlp_activation", "mlp_down",
    "ffn_region", "logits",
})


def _coarse_query_rows(spans: Sequence[dict[str, Any]]) -> int:
    """Return the row extent represented by compact/ordinary span records.

    This only projects already-authored graph shapes into the v2 schema; it
    does not estimate work or latency.
    """
    if not spans:
        raise ValueError("coarse_die_task_v2 requires at least one span")
    rows = 0
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError("span[{}] must be an object".format(index))
        run = span.get("run")
        fields = run if isinstance(run, dict) else span
        q_tokens = fields.get("q_tokens", fields.get("q_len"))
        count = fields.get("count", span.get("count", 1))
        batch = span.get("batch", 1)
        for name, value in (("q_tokens", q_tokens), ("count", count),
                            ("batch", batch)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "span[{}].{} must be a positive integer".format(
                        index, name))
        rows += q_tokens * count * batch
    return rows


def build_coarse_die_task(
    op_kind: str,
    ctx: DieModelContext,
    spans: Sequence[dict[str, Any]],
    *,
    layer_count: int,
    weight_passes: int,
    fusion_policy: str = "none",
    storage_dtype: Optional[str | dict[str, str]] = None,
    graph_identity: Optional[dict[str, Any]] = None,
    residency: Optional[dict[str, Any]] = None,
    readiness_scope: Optional[dict[str, Any]] = None,
    causal: Optional[bool] = None,
) -> dict[str, Any]:
    """Build a typed ``coarse_die_task_v2`` from JOINT's typed emission data.

    Category mapping, absolute span positions, TP extents, and model dimensions
    remain owned by ``DieModelContext``/``build_die_spec``. This adapter only
    serializes those facts into the public coarse schema. The legacy model-wide
    dtype is passed as one explicit scalar unless a caller supplies a
    role-specific dtype object; no storage role is guessed independently.
    """
    if ctx.task_schema != COARSE_DIE_TASK_SCHEMA:
        raise ValueError(
            "build_coarse_die_task requires a coarse_die_v2 DieModelContext")
    if op_kind not in _COARSE_DIE_OP_KINDS:
        raise ValueError(
            "unsupported coarse_die_task_v2 op_kind {!r}".format(op_kind))
    if fusion_policy not in ("auto", "none"):
        raise ValueError(
            "fusion_policy must be 'auto' or 'none', got {!r}".format(
                fusion_policy))
    layer_count = int(layer_count)
    weight_passes = int(weight_passes)
    if layer_count < 1 or weight_passes < 1:
        raise ValueError("layer_count and weight_passes must be positive")
    if not isinstance(ctx.execution_profile, str) or not ctx.execution_profile:
        raise ValueError(
            "coarse_die_v2 task requires an explicit execution_profile")
    dtype = ctx.dtype if storage_dtype is None else storage_dtype
    if isinstance(dtype, dict):
        if not dtype:
            raise ValueError("storage_dtype role object must not be empty")
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in dtype.items()):
            raise ValueError("storage_dtype must map role names to strings")
        dtype = dict(dtype)
    elif not isinstance(dtype, str) or not dtype:
        raise ValueError("storage_dtype must be a non-empty string or object")

    # Reuse build_die_spec's established validation/mapping source and then
    # translate its exact typed fields; this deliberately does not send the
    # v1 object through Compiler::normalize_old_spec.
    legacy = build_die_spec(
        op_kind, ctx, spans,
        layer_count=layer_count,
        weight_passes=weight_passes,
        fusion_policy=fusion_policy,
    )
    tp_slice = ctx.tp_slice()
    dimensions = legacy["model_dims"]
    local = tp_slice
    shapes = {
        "m": _coarse_query_rows(spans),
        "hidden": int(dimensions["hidden"]),
        "ffn": int(local["ffn_local"]),
        "vocab": int(local["vocab_local"]),
        "q_heads": int(local["q_heads_local"]),
        "kv_heads": int(local["kv_heads_local"]),
        "head_dim": int(dimensions["head_dim"]),
        "mlp_variant": str(dimensions["mlp_variant"]),
        "norm_type": str(dimensions["norm_type"]),
        "position_encoding": str(dimensions["position_encoding"]),
    }
    # The new compiler rejects zero dimensions. Make the reason visible here,
    # before a later ABI call obscures which source shard was empty.
    for key in ("hidden", "ffn", "vocab", "q_heads", "kv_heads", "head_dim"):
        if shapes[key] <= 0:
            raise ValueError(
                "coarse_die_task_v2 tensor_shapes.{} must be positive; "
                "the selected TP shard is empty".format(key))

    task: dict[str, Any] = {
        "schema": COARSE_DIE_TASK_SCHEMA,
        "op_kind": op_kind,
        "phase": str(ctx.phase),
        "tensor_shapes": shapes,
        "spans": [dict(span) for span in spans],
        "tp_slice": tp_slice,
        "storage_dtype": dtype,
        "execution_profile": ctx.execution_profile,
        "layer_instances": layer_count,
        "weight_passes": weight_passes,
        "fusion_policy": fusion_policy,
    }
    for key, value in (("graph_identity", graph_identity),
                       ("residency", residency),
                       ("readiness_scope", readiness_scope)):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("{} must be an object".format(key))
            task[key] = dict(value)
    if causal is not None:
        if not isinstance(causal, bool):
            raise ValueError("causal must be a boolean")
        task["causal"] = causal
    return task


def require_legacy_projection_context(die_context: Any) -> None:
    """Reject coarse plans at the legacy JOINT projection boundary."""
    if (getattr(die_context, "task_schema", LEGACY_DIE_TASK_SCHEMA)
            == COARSE_DIE_TASK_SCHEMA
            or getattr(die_context, "profile", None) == COARSE_DIE_PROFILE):
        raise RuntimeError(
            "coarse_die_v2 JOINT prediction is fail-closed: the current "
            "provider uses legacy hbm_share_divisor/service_factor and the "
            "frozen native projection has not been validated against the "
            "coarse shared Runtime")


def _fused_name(first_member: dict[str, Any], region_suffix: str) -> str:
    """融合节点名：取首成员名去掉其类别后缀的前缀 + region 名。

    约束（§7.2 + 契约 §7 裁定 8）：名字可读、不含任何 marker 名字锚点
    子串（违反即 fail-closed，防御 C++ 名字锚点误命中）。"""
    name = str(first_member["name"])
    category = str(first_member["category"])
    index = name.rfind(category)
    prefix = name[:index] if index >= 0 else name + "_"
    fused = prefix + region_suffix
    for anchor in _WATCH_NAME_ANCHORS:
        if anchor in fused:
            raise ValueError(
                "fused region name {!r} contains watch anchor substring "
                "{!r}; rename to avoid C++ name-anchored markers".format(
                    fused, anchor))
    return fused


def _chain_fusable(
    members: Sequence[dict[str, Any]],
    watch_targets: frozenset[str],
) -> bool:
    """§7.2 融合条件逐项判定（全部满足才融合）：

      - 同 rank/span/layer/TP：identity 键（若提供）全成员相等；
      - 内部节点无外部分支消费者：任何成员 external_consumers 为真即拒；
      - 中间无通信/KV gate/提交边界/观察锚点：链内相邻成员间
        boundary_after 为真即拒（链尾之后的边界不影响）；任何成员名含
        观察锚点（或 id 出现在 watch_targets）即拒；
      - 输入 ready 相容：线性链发射下入口依赖唯一（由发射骨架保证，
        这里以相邻成员 identity 相等复核）。"""
    identity_keys = ("rank", "tp", "layer", "span_set")
    reference = members[0].get("identity") or {}
    for member in members:
        if member.get("external_consumers"):
            return False
        identity = member.get("identity") or {}
        for key in identity_keys:
            if key in identity or key in reference:
                if identity.get(key) != reference.get(key):
                    return False
        name = str(member.get("name", ""))
        for anchor in _WATCH_NAME_ANCHORS:
            if anchor in name:
                return False
        if name in watch_targets or str(member.get("id", "")) in watch_targets:
            return False
    for member in members[:-1]:
        if member.get("boundary_after"):
            return False
    return True


def compile_regions(
    category_nodes: Sequence[dict[str, Any]],
    watch_targets: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """把有序 COMP 类别记录编译成发射组（§7.2 纯本地链融合）。

    category_nodes（发射序）每项：{name, category, op_kind, num_ops,
    tensor_size, layer_count, weight_passes} + 可选阻断/
    同质键（remote_read = 权重远载字节，可选键——缺省 0）：external_consumers（外部分支消费者）、boundary_after（紧随
    其后有通信/KV gate/提交边界）、identity（{rank,tp,layer,span_set}
    任意子集，同质判定）、id（节点 id，watch 绑定用）。

    返回发射组列表（顺序保持）：{name, category, op_kind, num_ops,
    tensor_size, layer_count, weight_passes, fused, members
    （成分全名）， member_categories（成分类别）}（可选 remote_read 同样
    透传/求和——权重远载统计）。融合组的 num_ops/
    tensor_size = 各成分之和（兼容统计）；collective 位置/
    数量/参与者/bytes 与节点身份不在本函数职责内（发射骨架保持不变，
    成分类别经 member_categories 供骨架在原位发射 collective）。
    """
    protected = frozenset(str(target) for target in (watch_targets or ()))
    # 融合节点名也不得撞任何显式 watch 目标（名字/id 双口径）。
    protected_names = protected | {
        anchor for anchor in _WATCH_NAME_ANCHORS
    }
    nodes = [dict(node) for node in category_nodes]
    groups: list[dict[str, Any]] = []
    index = 0
    while index < len(nodes):
        matched = None
        for chain in _FUSION_CHAINS:
            length = len(chain)
            window = nodes[index:index + length]
            if len(window) != length:
                continue
            if tuple(node.get("op_kind") for node in window) != chain:
                continue
            matched = (window, chain)
            break
        if matched is not None:
            window, chain = matched
            region_kind = ("attention_region" if chain == ATTENTION_CHAIN
                           else "ffn_region")
            if _chain_fusable(window, protected_names):
                fused_name = _fused_name(window[0], region_kind)
                groups.append({
                    "name": fused_name,
                    "category": region_kind,
                    "op_kind": region_kind,
                    "num_ops": sum(node["num_ops"] for node in window),
                    "tensor_size": sum(node["tensor_size"] for node in window),
                    "remote_read": sum(
                        int(node.get("remote_read", 0)) for node in window),
                    "layer_count": window[0]["layer_count"],
                    "weight_passes": window[0]["weight_passes"],
                    "fused": True,
                    "members": [node["name"] for node in window],
                    "member_categories": [node["category"] for node in window],
                })
                index += len(window)
                continue
        node = nodes[index]
        groups.append({
            "name": node["name"],
            "category": node["category"],
            "op_kind": node["op_kind"],
            "num_ops": node["num_ops"],
            "tensor_size": node["tensor_size"],
            "remote_read": int(node.get("remote_read", 0)),
            "layer_count": node["layer_count"],
            "weight_passes": node["weight_passes"],
            "fused": False,
            "members": [node["name"]],
            "member_categories": [node["category"]],
        })
        index += 1
    return groups
