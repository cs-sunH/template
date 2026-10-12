#!/usr/bin/env python3
"""die_model_adapter.py -- die 后端的 spec 构造与生成期 region 编译(阶段 C)。

License: MIT (see LICENSE in the repository root).

本模块是 Python 生成侧对 extern/die_model 共享成本库的唯一桥接层(设计文档
§8.1:"sh_test_mesh/.../die_model_adapter.py -- 配置、形状和 region 编译;不
实现第二套公式")。只做三件事,不实现任何派生/时间公式:

  1. DieModelContext:数据载体(backend/模型维度/TP/config_hash 等),由
     online_scheduler_base 从 manifest 的 compute-cost-model /
     die-model-config 键构造,供 graph_batch_builder 的发射路径与 D 阶段
     die_model_oracle 共用;本类不做任何决策;
  2. build_die_spec():按契约 §2 schema 构造 die_op_spec dict(13 类旧
     COMP 类别 -> op_kind 映射、spans 绝对位置、kv_access.identity=local、
     provenance=transformer_pass_aggregated、config_hash 取 resolved
     配置内容摘要);
  3. compile_regions():§7.2 纯本地链融合(QK->scale/mask->softmax->AV =>
     attention_region;gate/up->activation->down => ffn_region),条件全部
     满足才融合,否则分离执行。

spans 绝对位置管线(设计文档 §4.2.7,wscllm-LRU 仓适配):本仓两类 COMP
体的 span 平铺表全部满足同一恒等式:kv_tokens == q_start_pos + q_tokens
且 KV 自序列位置 0 连续覆盖(kv_start_pos == 0):

  - prefill 阶段 chunk(generate_wsc_llm_trace._emit_prefill_stage):
        span = (chunk, initial_context_tokens + processed + chunk)
        -> q 起点 initial_context + processed,kv 覆盖 [0, 当前];
  - decode 列车迭代(online/wsc_llm_online_scheduler 平铺):
        span = (1, context + consumed + step) 形
        -> q = 当前末 token 位置,kv = [会话起点, 当前]。

derive_span_records() 按该恒等式还原绝对位置(上游 session KV 账本口径,
不发明新状态);spec 的 spans[].request_id 用发射命名空间合成 id(聚合图
无逐 span 请求身份,C++ 库按 opaque span 身份处理,attention 逐 span
Σ(q_i·k_i) 计费不受影响)。

kv_access.identity 恒 "local"(本仓裁定):本仓三态 KV(LOCAL/PARTIAL/
REMOTE)的远端读/恢复/迁移 DMA 全部由既有图路径计费(remote_store/
noc_migrate 端点、local_hbm_kv_restore 的 RESTORE 作业),计算节点发射
时其输入就绪门槛已保证所读 KV 在本地 HBM——设计文档 §7.4"KV 已在本地/
已准备范围内,计算核心读取它 → 新计算计划的本地内存及片内 NoC 阶段"。
远端搬运绝不被吞进本地 GEMM 的内存腿(同节红线)。

watches 绑定核实(融合阻断条件的事实基础,2026-10-03):本仓 watch 全部
经发射函数返回的 per-rank 节点 id 映射绑定(drain_members/exit_members/
sentinel_members/first_token_members/wakeup_members -> 标记 COMP 节点;
block_ends -> end barrier collective 节点;face_online_scheduler 把它们
写入 self._batch["watches"] 的 members 字段),从不按类别/名字绑定列车体
节点;唯一按名字绑定的 C++ 锚点是 first_token 子串(code 8),其载体也只是
标记节点名。列车体 13 类聚合节点既无 watch 也无标记子串,region 融合不改
变任何 watch 引用的节点身份;_WATCH_NAME_ANCHORS 仍作为防御性阻断条件
(融合节点名不得含任何锚点子串)。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享发射
# 模块(shard_extent)在上一级。只做 import 用途。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_trace import shard_extent  # noqa: E402

# 契约 §7 裁定 2 的 manifest 新键(与 die_model_oracle.py 同名常量)。
MANIFEST_COMPUTE_COST_MODEL_KEY = "compute-cost-model"
MANIFEST_DIE_MODEL_CONFIG_KEY = "die-model-config"

# spec 固定元数据。
SPEC_SCHEMA_VERSION = 1
PROVENANCE = "transformer_pass_aggregated"
DEFAULT_DIE_PROFILE = "mesh_die_v1"

# 13 类旧 COMP 类别 -> op_kind 映射(契约 §2"语义"小节;norm/residual 在
# attention/mlp/final 三处复用,qkv 投影恒映射 qkv_rope,是否含新 Q/K
# 旋转由 model_dims.position_encoding 区分)。
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

# §7.2 融合链(op_kind 序列)。
ATTENTION_CHAIN = ("qk", "scale_mask", "softmax", "av")
FFN_CHAIN = ("mlp_up", "mlp_activation", "mlp_down")
_FUSION_CHAINS = (ATTENTION_CHAIN, FFN_CHAIN)

# 契约 §7 裁定 8 的 marker/哨兵名字锚点(C++ first_token 子串锚点 +
# drain/exit/sentinel/join/pstart/wakeup 标记命名)。任何含这些子串的
# 类别名不参与融合,融合节点名含这些子串即 fail-closed。
_WATCH_NAME_ANCHORS = (
    "first_token", "drain", "exit", "sentinel", "join", "pstart", "wakeup",
)

# 契约 §2:die_op_spec 存在 => runtime_ns 必须为 0(OnlineTraceBuilder 的
# compute 缺省形即 0;此处仅为发射侧自检)。
SPEC_RUNTIME_NS = 0


def derive_span_records(
    pass_spans: Iterable[Sequence[int]],
    *,
    namespace: str = "span",
) -> list[dict[str, Any]]:
    """把 face 的 (tokens, kv_length) span 平铺表还原为带绝对位置的
    die span 记录(契约 §2 spans[] schema)。

    恒等式(kv_tokens == q_start_pos + q_tokens、kv_start_pos == 0)来自
    face_online_scheduler 的三类 span 构造点(模块 docstring);违反即
    fail-closed,绝不猜位置。request_id 用 "<namespace>#s<index>" 合成
    (聚合图无逐 span 请求身份;库按 opaque 身份处理)。
    """
    records: list[dict[str, Any]] = []
    for index, span in enumerate(pass_spans):
        tokens, kv_length = int(span[0]), int(span[1])
        if tokens < 1 or kv_length < 1:
            raise ValueError(
                "die span must carry positive tokens and kv_length, got "
                f"{(tokens, kv_length)!r}")
        if kv_length < tokens:
            raise ValueError(
                "die span invariant violated: kv_length {} < tokens {} "
                "(face spans satisfy kv_tokens == q_start_pos + q_tokens; "
                "see face_online_scheduler._plan_train/_plan_head_chunks)"
                .format(kv_length, tokens))
        records.append({
            "request_id": "{}#s{}".format(namespace, index),
            "q_tokens": tokens,
            "kv_tokens": kv_length,
            "q_start_pos": kv_length - tokens,
            "kv_start_pos": 0,
        })
    if not records:
        raise ValueError("die emission requires at least one pass span")
    return records


class DieModelContext:
    """die 后端发射语境的数据载体(阶段 C;oracle 由 D 阶段另行实现)。

    构造来源:online_scheduler_base 从 manifest(compute-cost-model ==
    "die" + die-model-config 绝对路径)与 WscLlmTraceConfig 模型行构造。
    每列车每 rank 发射前由 graph_batch_builder._emit_train_body 调
    begin_emission() 写入当次发射的 phase/TP/span 记录(发射序串行,无
    并发);transformer_pass_aggregated 经 build_spec()/compile_regions()
    消费。本类不持有任何调度决策状态。
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
    ) -> None:
        self.backend = "die"
        self.profile = profile
        self.die_config_path = Path(die_config_path)
        self.config_hash = config_hash
        self.dtype = dtype
        self.model_dims = dict(model_dims)
        self._num_q_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._ffn_size = ffn_size
        self._vocab_size = vocab_size
        # 每次发射的语境(begin_emission 写入)。
        self.phase = "prefill"
        self.tensor_parallel = 1
        self.tensor_parallel_rank = 0
        self.span_namespace = ""
        self.span_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------- 构造 --

    @classmethod
    def from_service(cls, manifest: dict, config: Any) -> Optional["DieModelContext"]:
        """从 manifest + FaceTraceConfig 构造;backend 非 die 时返回 None
        (legacy 回归保护:无 manifest 键的旧 plan 目录行为不变)。"""
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
        # §4.2.6:trace_config 只有 heads=32 一个头数键 -> MHA 解释
        # (num_q_heads == num_kv_heads == heads,head_dim = hidden//heads)。
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
            profile = str(resolved_system.get(
                "die-model-profile", DEFAULT_DIE_PROFILE))
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
        """写入一次列车体发射的语境(每 rank 一次;phase 取列车 stage:
        "decode"(含混拼)或 "prefill")。"""
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
        self.span_records = derive_span_records(
            pass_spans, namespace=span_namespace)

    def tp_slice(self) -> dict[str, int]:
        """契约 §2 tp_slice(非整除 TP 用 shard_extent,空分片合法)。"""
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
    ) -> dict[str, Any]:
        """当前发射语境下的 die_op_spec(委托模块级 build_die_spec)。"""
        return build_die_spec(
            op_kind,
            self,
            self.span_records,
            layer_count=layer_count,
            weight_passes=weight_passes,
            fusion_policy=fusion_policy,
        )

    def op_kind_of(self, category: str) -> str:
        """13 类旧 COMP 类别 -> 契约 §2 op_kind(模块级映射的唯一入口;
        生成侧 generate_trace 经本方法取映射,不复制映射表)。"""
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
        """生成期 region 编译(委托模块级 compile_regions;缺省阻断 =
        marker 名字锚点子串)。"""
        return compile_regions(category_nodes, watch_targets)


def build_die_spec(
    op_kind: str,
    ctx: DieModelContext,
    spans: Sequence[dict[str, Any]],
    *,
    layer_count: int,
    weight_passes: int,
    fusion_policy: str = "none",
) -> dict[str, Any]:
    """按契约 §2 schema 构造一个 die_op_spec dict。

    校验只覆盖 schema 形状(枚举/正整数);数值语义校验由 C++ 库的
    parse_die_op_spec 承担(严格解析,D 阶段 oracle 调用时执行)。
    kv_access.identity=local:计算节点发射时其 KV 输入已由本仓恢复/迁移
    路径就位于本地 HBM(模块 docstring 的本仓裁定);远端搬运计费属于
    既有 remote/restore 图路径,不属于本 spec。
    """
    if op_kind not in OP_KINDS:
        raise ValueError(
            "unknown die op_kind {!r} (contract section 2 enum)".format(
                op_kind))
    if fusion_policy not in ("auto", "none"):
        raise ValueError(
            "fusion_policy must be 'auto' or 'none', got {!r}".format(
                fusion_policy))
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
        "spans": [dict(span) for span in spans],
        "layer_count": layer_count,
        "weight_passes": weight_passes,
        "kv_access": {"identity": "local"},
        "fusion_policy": fusion_policy,
        "provenance": PROVENANCE,
        "config_hash": ctx.config_hash,
    }


def _fused_name(first_member: dict[str, Any], region_suffix: str) -> str:
    """融合节点名:取首成员名去掉其类别后缀的前缀 + region 名。

    约束(§7.2 + 契约 §7 裁定 8):名字可读、不含任何 marker 名字锚点
    子串(违反即 fail-closed,防御 C++ 名字锚点误命中)。"""
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
    """§7.2 融合条件逐项判定(全部满足才融合):

      - 同 rank/span/layer/TP:identity 键(若提供)全成员相等;
      - 内部节点无外部分支消费者:任何成员 external_consumers 为真即拒;
      - 中间无通信/KV gate/提交边界/观察锚点:链内相邻成员间
        boundary_after 为真即拒(链尾之后的边界不影响);任何成员名含
        观察锚点(或 id 出现在 watch_targets)即拒;
      - 输入 ready 相容:线性链发射下入口依赖唯一(由发射骨架保证,
        这里以相邻成员 identity 相等复核)。"""
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
    """把有序 COMP 类别记录编译成发射组(§7.2 纯本地链融合)。

    category_nodes(发射序)每项:{name, category, op_kind, num_ops,
    tensor_size, remote_read, layer_count, weight_passes} + 可选阻断/
    同质键:external_consumers(外部分支消费者)、boundary_after(紧随
    其后有通信/KV gate/提交边界)、identity({rank,tp,layer,span_set} 任意
    子集,同质判定)、id(节点 id,watch 绑定用)。

    返回发射组列表(顺序保持):{name, category, op_kind, num_ops,
    tensor_size, remote_read, layer_count, weight_passes, fused, members
    (成分全名), member_categories(成分类别)}。融合组的 num_ops/
    tensor_size/remote_read = 各成分之和(兼容统计);collective 位置/
    数量/参与者/bytes 与节点身份不在本函数职责内(发射骨架保持不变,
    成分类别经 member_categories 供骨架在原位发射 collective)。
    """
    protected = frozenset(str(target) for target in (watch_targets or ()))
    # 融合节点名也不得撞任何显式 watch 目标(名字/id 双口径)。
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
                    "remote_read": sum(node["remote_read"] for node in window),
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
            "remote_read": node["remote_read"],
            "layer_count": node["layer_count"],
            "weight_passes": node["weight_passes"],
            "fused": False,
            "members": [node["name"]],
            "member_categories": [node["category"]],
        })
        index += 1
    return groups
