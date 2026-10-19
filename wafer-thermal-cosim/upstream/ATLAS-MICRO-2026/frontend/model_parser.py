from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import dataclasses
import enum
import math
import sys

from transformers import AutoConfig


@dataclass
class ModelConfig:
    name: str = ""  # model config name
    model_type: str = None  # model type as tagged on Hugging Face (e.g., gpt2, opt, llama.)
    vocab_size: int = -1  # vocabulary size
    max_seq_len: int = None  # max sequence length
    num_layers: int = -1  # number of transformer layers (blocks)

    n_head: int = -1  # number of query heads
    n_kv_head: int = None  # the number of key value heads implementing Grouped Query Attention (GQA), If it is not specified, will default to n_head. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `n_kv_head=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. See https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py for details
    n_kv_group: int = None  # number of key value groups for GQA
    head_dim: int = None # each attention head's vector dim

    hidden_dim: int = -1  # hidden dimension
    ffn_intermediate_dim: int = None  # hidden dimension of FFN, default to 4 * hidden_dim
    expansion_ratio: float = None
    mlp_gated_linear_units: bool = False  # whether to use gated linear units for MLP

    # For MoE models
    is_moe: bool = False
    moe_num_experts: int = 1  # number of experts for mixture of experts model
    moe_shared_experts: int = 0
    moe_top_k: int = 1  # top k experts for mixture of experts model
    moe_intermediate_size: int = -1 # moe ffn's intermediate dim, used for qwen or deepseek

    # For deepseek model
    is_deepseek: bool = False
    q_lora_rank: int = -1 # W^DQ [hidden_dim, q_lora_rank]
    kv_lora_rank: int = -1 # W^DKV [hidden_dim, kv_lora_rank]
    qk_nope_head_dim: int = -1 # W^UQ [q_lora_rank, n_head*qk_nope_head_dim], W^UK/W^UV [kv_lora_rank, n_head*qk_nope_head_dim], W^O [n_head*qk_nope_head_dim, hidden_dim]
    qk_rope_head_dim: int = -1 # W^QR [kv_lora_rank, n_head*qk_rope_head_dim], W^KR [hidden_dim, qk_rope_head_dim]

    def __str__(self):
        return dataclasses.asdict(self).__str__()


@dataclass
class ParallelConfig:
    tp_size: int = 8
    ep_size: int = -1

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "ParallelConfig":
        try:
            obj = cls()
            obj.tp_size = int(config["tp_size"])
            obj.ep_size = int(config["ep_size"])
        except Exception as e:
            print(f"[Error] Parallel config is invalid: {e}", file=sys.stderr)
            sys.exit(-1)
        
        return obj


class OperatorType(enum.Enum):
    GEMM =  "gemm"
    ATTENTION = "attention"
    VECTOR = "vector"
    ALLREDUCE = "allreduce"
    ALL2ALL = "all2all"


@dataclass
class Operator:
    name: str
    op_type: OperatorType
    tp_size: int = -1
    ep_size: int = -1
    
    # For GEMM / all reduce / all to all
    B: int = -1 # used for MLA's matrix absorb, which conducts parallel GEMMs along different heads
    M: int = -1 # total token num (reused by all reduce and all to all)
    K: int = -1 # input hidden dim
    N: int = -1 # output hidden dim (reused by all reduce and all to all)

    # For Attention
    is_mla: bool = False
    kv_head_num: int = -1
    kv_group_num: int = -1 # q_head_num / kv_head_num
    head_dim: int = -1
    v_head_dim: int = -1 # used for deepseek MLA, K is longer than V
    input_length: List[int] = field(default_factory=list)
    context_length: List[int] = field(default_factory=list)

    # For all to all
    total_expert_num: int = -1

    def get_shape(self):
        if self.op_type == OperatorType.GEMM:
            return [self.B, self.M, self.K, self.N]
        elif self.op_type == OperatorType.ATTENTION:
            return [self.kv_head_num, self.kv_group_num, self.v_head_dim], self.input_length, self.context_length
        elif self.op_type == OperatorType.ALLREDUCE:
            return [self.M, self.N]
        elif self.op_type == OperatorType.ALL2ALL:
            return [self.M, self.N]
        
    def get_flop(self):
        if self.op_type == OperatorType.GEMM:
            return 2 * self.B * self.M * self.K * self.N
        elif self.op_type == OperatorType.ATTENTION:
            v_head_dim = self.v_head_dim if self.v_head_dim != -1 else self.head_dim
            total_flop = 0
            assert len(self.input_length) == len(self.context_length)
            for lin, lcontext in zip(self.input_length, self.context_length):
                gemm_qk_flop = 2 * self.kv_head_num * self.kv_group_num * lin * self.head_dim * lcontext
                # (1) element-wise exp (2) sum (3) division
                softmax_flop = 3 * self.kv_head_num * self.kv_group_num * lin * lcontext
                gemm_sv_flop = 2 * self.kv_head_num * self.kv_group_num * lin * lcontext * v_head_dim
                total_flop += gemm_qk_flop + softmax_flop + gemm_sv_flop
            return total_flop
        else:
            assert False, "only GEMM and ATTENTION needs to calculate FLOP count"
    
    def get_memory_capacity(self, element_size: int):
        # element size is each element's byte count
        if self.op_type == OperatorType.GEMM:
            return element_size * self.B * (self.M*self.K + self.K*self.N)
        elif self.op_type == OperatorType.ATTENTION:
            v_head_dim = self.v_head_dim if self.v_head_dim != -1 else self.head_dim
            total_memory_size = 0
            assert len(self.input_length) == len(self.context_length)
            for lin, lcontext in zip(self.input_length, self.context_length):
                input_size = element_size * self.kv_head_num * self.kv_group_num * lin * self.head_dim
                k_cache_size = element_size * self.kv_head_num * lcontext * self.head_dim
                v_cache_size = element_size * self.kv_head_num * lcontext * v_head_dim
                total_memory_size += input_size + k_cache_size + v_cache_size
            return total_memory_size
        elif self.op_type == OperatorType.ALLREDUCE:
            return element_size * self.M * self.N
        elif self.op_type == OperatorType.ALL2ALL:
            return element_size * self.M * self.N
        else:
            assert False, "Now we only calculate memory capacity for GEMM/ATTENTION/ALLREDUCE"
    
    def get_arithmetic_intensity(self, element_size: int):
        return self.get_flop() / self.get_memory_capacity(element_size)


def get_model_config_from_hf(name: str, path: str) -> ModelConfig:
    """Get model config from HuggingFace transformers library `AutoConfig`.

    Args:
        name (str): model nickname
        path (str): the model id of a pretrained model configuration hosted inside a model repo on huggingface.co

    Returns:
        ModelConfig: a dataclass for llm-analysis model config
    """

    ################# Extract config and parse basic information
    hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    
    if hasattr(hf_config, "num_hidden_layers"):
        num_layers = hf_config.num_hidden_layers
    elif hasattr(hf_config, "n_layers"):
        num_layers = hf_config.n_layers
    else:
        raise Exception(
            "hf config does not have num_hidden_layers or n_layers, check the config.json file"
        )

    ################# Attention settings
    if hasattr(hf_config, "num_attention_heads"):
        n_head = hf_config.num_attention_heads
    elif hasattr(hf_config, "n_heads"):
        n_head = hf_config.n_heads
    else:
        raise Exception(
            "hf config does not have num_attention_heads or n_heads, check the config.json file"
        )
    
    if hasattr(hf_config, "num_key_value_heads"):
        n_kv_head = hf_config.num_key_value_heads
    else:
        n_kv_head = n_head
    if name.lower() == "deepseek":
        n_kv_head = 1

    n_kv_group = math.ceil(n_head / n_kv_head)

    ################# FFN settings
    if hasattr(hf_config, "hidden_size"):
        hidden_dim = hf_config.hidden_size
    elif hasattr(hf_config, "d_model"):
        hidden_dim = hf_config.d_model
    else:
        raise Exception(
            "hf config does not have hidden_size or d_model, check the config.json file"
        )
    
    if name.lower() == "deepseek":
        head_dim = None
    else:
        if hasattr(hf_config, "head_dim"):
            head_dim = hf_config.head_dim
        else:
            head_dim = int(math.ceil(hidden_dim / n_head))
    
    if hasattr(hf_config, "ffn_intermediate_dim"):
        ffn_intermediate_dim = hf_config.ffn_intermediate_dim
    elif hasattr(hf_config, "intermediate_size"):
        ffn_intermediate_dim = hf_config.intermediate_size
    elif hasattr(hf_config, "ffn_dim"):
        ffn_intermediate_dim = hf_config.ffn_dim
    else:
        ffn_intermediate_dim = 4 * hidden_dim

    expansion_ratio = ffn_intermediate_dim / hidden_dim
    mlp_gated_linear_units = False
    if expansion_ratio == 3.5:
        mlp_gated_linear_units = True
    else:
        mlp_gated_linear_units = False
    if "qwen" in name.lower() or "deepseek" in name.lower():
        mlp_gated_linear_units = True

    ################# MoE settings
    is_moe = True
    if hasattr(hf_config, "moe_num_experts"):
        moe_num_experts = hf_config.moe_num_experts
    elif hasattr(hf_config, "num_experts"):
        moe_num_experts = hf_config.num_experts
    elif hasattr(hf_config, "num_local_experts"):
        moe_num_experts = hf_config.num_local_experts
    elif hasattr(hf_config, "n_routed_experts"):
        moe_num_experts = hf_config.n_routed_experts
    else:
        moe_num_experts = 1
        is_moe = False
    
    if hasattr(hf_config, "num_experts_per_tok"):
        moe_top_k = hf_config.num_experts_per_tok
    else:
        moe_top_k = 8
    
    if hasattr(hf_config, "n_shared_experts"):
        moe_shared_experts = hf_config.n_shared_experts
    else:
        moe_shared_experts = 0
    
    if hasattr(hf_config, "moe_intermediate_size"):
        moe_intermediate_size = hf_config.moe_intermediate_size
    else:
        moe_intermediate_size = ffn_intermediate_dim
    
    ################# DeepSeek settings
    if hasattr(hf_config, "q_lora_rank"):
        q_lora_rank = hf_config.q_lora_rank
    else:
        q_lora_rank = -1
    
    if hasattr(hf_config, "kv_lora_rank"):
        kv_lora_rank = hf_config.kv_lora_rank
    else:
        kv_lora_rank = -1
    
    if hasattr(hf_config, "qk_nope_head_dim"):
        qk_nope_head_dim = hf_config.qk_nope_head_dim
    else:
        qk_nope_head_dim = -1
    
    if hasattr(hf_config, "qk_rope_head_dim"):
        qk_rope_head_dim = hf_config.qk_rope_head_dim
    else:
        qk_rope_head_dim = -1
    
    deepseek_info = [q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim]
    if all(dim != -1 for dim in deepseek_info):
        is_deepseek = True
    else:
        is_deepseek = False

    ################# Create config object
    config = ModelConfig(
        # Basic Settings
        name=name,
        model_type=hf_config.model_type if hasattr(hf_config, "model_type") else None,
        vocab_size=hf_config.vocab_size,
        max_seq_len=hf_config.max_position_embeddings if hasattr(hf_config, "max_position_embeddings") else None,
        num_layers=num_layers,
        # Attention Settings
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_kv_group=n_kv_group,
        head_dim=head_dim,
        # FFN Settings
        hidden_dim=hidden_dim,
        ffn_intermediate_dim=ffn_intermediate_dim,
        expansion_ratio=expansion_ratio,
        mlp_gated_linear_units=mlp_gated_linear_units,
        # MoE Settings
        is_moe=is_moe,
        moe_num_experts=moe_num_experts,
        moe_shared_experts=moe_shared_experts,
        moe_top_k=moe_top_k,
        moe_intermediate_size=moe_intermediate_size,
        # DeepSeek Settings
        is_deepseek=is_deepseek,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim
    )
    return config


def get_layer_operator_list(
    model_config: ModelConfig, 
    parallel_config: ParallelConfig = ParallelConfig(tp_size=1, ep_size=1),
    skip_communication: bool = False,
    non_fused_attention: bool = False,
) -> Tuple[List[Operator], List[Operator]]:
    # NOTE: Each operator records per device's operator shape

    attention_block = []
    if not model_config.is_deepseek:
        # qkv projection. For megatron style TP, N (output dim) is split
        per_tp_shard_head_num = math.ceil(model_config.n_head / parallel_config.tp_size) + \
                                math.ceil(model_config.n_kv_head / parallel_config.tp_size) * 2
        attention_block.append(Operator(
            name="qkv_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=model_config.hidden_dim,
            # N=math.ceil((model_config.hidden_dim + 2 * model_config.hidden_dim // model_config.n_kv_group) / parallel_config.tp_size),
            N=model_config.head_dim * per_tp_shard_head_num
        ))
        if not non_fused_attention:
            # self-attention. For megatron style TP, kv head num is split
            attention_block.append(Operator(
                name="attention",
                op_type=OperatorType("attention"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                is_mla=False,
                kv_head_num=math.ceil(model_config.n_kv_head / parallel_config.tp_size),
                kv_group_num=model_config.n_kv_group,
                head_dim=model_config.head_dim,
                v_head_dim=model_config.head_dim,
                input_length=[], # Updated during inference
                context_length=[], # Updated during inference
            ))
        else:
            attention_block.append(Operator(
                name="attention_qk",
                op_type=OperatorType("gemm"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=model_config.n_kv_head, # Updated during inference
                M=model_config.n_kv_group,
                K=model_config.head_dim,
                N=1, # context length, updated during inference
            ))
            # Just a placeholder for edge-side inference simulation
            attention_block.append(Operator(
                name="softmax",
                op_type=OperatorType("vector"),
            ))
            attention_block.append(Operator(
                name="attention_sv",
                op_type=OperatorType("gemm"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=model_config.n_kv_head, # Updated during inference
                M=model_config.n_kv_group,
                K=1, # context length, updated during inference
                N=model_config.head_dim, 
            ))
        # output projection. For megatron style TP, K (input hidden dim) is split
        attention_block.append(Operator(
            name="o_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            # K=model_config.hidden_dim // parallel_config.tp_size,
            K=model_config.head_dim * math.ceil(model_config.n_head / parallel_config.tp_size),
            N=model_config.hidden_dim,
        ))
        if not skip_communication:
            # For megatron style TP, we need to conduct all reduce after output projection.
            attention_block.append(Operator(
                name="attention_all_reduce",
                op_type=OperatorType("allreduce"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                M=1, # Updated during inference
                N=model_config.hidden_dim,
            ))
    else:
        # DQ, DKV, KR: [hidden_dim, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
        # q_lora_rank in output dim is split to TP devices
        # kv_lora_rank + qk_rope_head_dim is duplicated across TP devices
        attention_block.append(Operator(
            name="dq_kr_dkv_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=model_config.hidden_dim,
            N=model_config.q_lora_rank // parallel_config.tp_size + model_config.kv_lora_rank + model_config.qk_rope_head_dim,
        ))
        # UQ, QR: [q_lora_rank, n_head * (qk_rope_head_dim, qk_nope_head_dim)]
        # These operators split input hidden dim to TP devices
        attention_block.append(Operator(
            name="uq_qr_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=model_config.q_lora_rank // parallel_config.tp_size,
            N=model_config.n_head * (model_config.qk_rope_head_dim + model_config.qk_nope_head_dim),
        ))
        if not skip_communication:
            # all reduce for UQ, UR
            # In fact, reduce scatter is enough. But for simplicity, we still adopt all reduce.
            attention_block.append(Operator(
                name="uq_qr_all_reduce",
                op_type=OperatorType("allreduce"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                M=1, # Updated during inference
                N=model_config.n_head * (model_config.qk_rope_head_dim + model_config.qk_nope_head_dim),
            ))
        # UK with transpose's shape: [n_head*qk_nope_head_dim, kv_lora_rank]
        # This operator is split along n_head
        # For each device, we need to compute n_head / tp_size GEMMs,
        # each GEMM's shape is [batch_size, qk_nope_head_dim] [qk_nope_head_dim, kv_lora_rank]
        # Finally, we get [n_head / tp_size, batch_size, kv_lora_rank]
        attention_block.append(Operator(
            name="uk_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=model_config.n_head // parallel_config.tp_size,
            M=1, # Updated during inference
            K=model_config.qk_nope_head_dim,
            N=model_config.kv_lora_rank,
        ))
        # MLA, k length is (qk_rope_head_dim + kv_lora_rank), v length is kv_lora_rank
        # Finally, we get [batch_size, n_head / tp_size, kv_lora_rank]
        if not non_fused_attention:
            attention_block.append(Operator(
                name="mla_attention",
                op_type=OperatorType("attention"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                is_mla=True,
                kv_head_num=1,
                kv_group_num=model_config.n_head // parallel_config.tp_size,
                head_dim=model_config.qk_rope_head_dim + model_config.kv_lora_rank,
                v_head_dim=model_config.kv_lora_rank,
                input_length=[], # Updated during inference
                context_length=[], # Updated during inference
            ))
        else:
            attention_block.append(Operator(
                name="mla_attention_qk",
                op_type=OperatorType("gemm"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=1, # Updated during inference
                M=model_config.n_head,
                K=model_config.qk_rope_head_dim + model_config.kv_lora_rank,
                N=1, # context length, updated during inference
            ))
            # Just a placeholder for edge-side inference simulation
            attention_block.append(Operator(
                name="softmax",
            ))
            attention_block.append(Operator(
                name="mla_attention_sv",
                op_type=OperatorType("gemm"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=1, # Updated during inference
                M=model_config.n_head,
                K=1, # context length, updated during inference
                N=model_config.kv_lora_rank, 
            ))
        # UV with transpose's shape: [n_head*qk_nope_head_dim, kv_lora_rank]
        # This weight is also split along n_head.
        # For each device, we need to compute n_head / tp_size GEMMs,
        # each GEMM's shape is [batch_size, kv_lora_rank] [kv_lora_rank, qk_nope_head_dim]
        # Finally, we get [n_head / tp_size, batch_size, qk_nope_head_dim]
        attention_block.append(Operator(
            name="uv_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=model_config.n_head // parallel_config.tp_size,
            M=1, # Updated during inference
            K=model_config.kv_lora_rank,
            N=model_config.qk_nope_head_dim,
        ))
        # output projection, this operator is split along the input hidden dim
        attention_block.append(Operator(
            name="o_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=model_config.n_head * model_config.qk_nope_head_dim // parallel_config.tp_size,
            N=model_config.hidden_dim,
        ))
        if not skip_communication:
            # all reduce
            attention_block.append(Operator(
                name="attention_all_reduce",
                op_type=OperatorType("allreduce"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                M=1, # Updated during inference
                N=model_config.hidden_dim,
            ))

    ffn_moe_block = []
    ffn_hidden_dim = model_config.ffn_intermediate_dim if not model_config.is_moe else model_config.moe_intermediate_size
    down_ffn_factor = 2 if model_config.mlp_gated_linear_units else 1

    if model_config.is_moe:
        assert model_config.moe_num_experts / parallel_config.ep_size >= 1
        if not skip_communication:
            ffn_moe_block.append(Operator(
                name="moe_alltoall_dispatch",
                op_type=OperatorType("all2all"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=1,
                M=1, # Updated during inference
                N=model_config.hidden_dim,
            ))
        # do not split expert, and no all reduce
        ffn_moe_block.append(Operator(
            name="down_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=model_config.hidden_dim,
            N=ffn_hidden_dim * down_ffn_factor
        ))
        ffn_moe_block.append(Operator(
            name="up_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=ffn_hidden_dim,
            N=model_config.hidden_dim
        ))
        if not skip_communication:
            ffn_moe_block.append(Operator(
                name="moe_alltoall_combine",
                op_type=OperatorType("all2all"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=1,
                M=1, # Updated during inference
                N=model_config.hidden_dim,
            ))
    else:
        ffn_moe_block.append(Operator(
            name="down_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=model_config.hidden_dim,
            N=ffn_hidden_dim * down_ffn_factor // parallel_config.tp_size
        ))
        ffn_moe_block.append(Operator(
            name="up_proj",
            op_type=OperatorType("gemm"),
            tp_size=parallel_config.tp_size,
            ep_size=parallel_config.ep_size,
            B=1,
            M=1, # Updated during inference
            K=ffn_hidden_dim // parallel_config.tp_size,
            N=model_config.hidden_dim
        ))
        if not skip_communication:
            ffn_moe_block.append(Operator(
                name="ffn_all_reduce",
                op_type=OperatorType("allreduce"),
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                M=1, # Updated during inference
                N=model_config.hidden_dim,
            ))

    # Since moe expert number is uncertain at runtime, we retrun the two blocks separatedly
    return attention_block, ffn_moe_block


def set_cloud_shape(
    model_config: ModelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
    context_length_list: List[int],
    parallel_config: ParallelConfig,
    core_num: int,
    block_size: int = 1,
):
    batch_size = len(context_length_list)
    for operator in attention_block:
        if operator.op_type == OperatorType.GEMM:
            operator.M = batch_size
        elif operator.op_type == OperatorType.ATTENTION:
            operator.input_length = [1 for _ in range(len(context_length_list))]
            # Homogeneous case: find minimum n giving perfect load balance
            if len(set(context_length_list)) == 1 and batch_size > 0:
                c = context_length_list[0]
                max_n = min(core_num, max(1, c // block_size))
                step = core_num // math.gcd(batch_size, core_num)
                n_val = step if step <= max_n else max_n
                operator.context_length = [math.ceil(c / n_val)] * batch_size
            else:
                per_core_context = []
                for c in context_length_list:
                    if batch_size >= core_num:
                        per_core_context.append(c)
                    else:
                        per_core = math.ceil(c / core_num)
                        if per_core >= block_size:
                            per_core_context.append(per_core)
                        else:
                            n = max(1, c // block_size)
                            per_core_context.append(math.ceil(c / n))
                operator.context_length = per_core_context
        elif operator.op_type == OperatorType.ALLREDUCE:
            operator.M = batch_size

    if model_config.is_moe:
        # assume a uniform distribution of expert routing
        per_ffn_token_num = math.ceil(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts) / model_config.moe_num_experts)
        per_device_ffn_num = math.ceil(min(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts), model_config.moe_num_experts) / parallel_config.ep_size)
    else:
        per_device_ffn_num = 1
        per_ffn_token_num = batch_size
    for operator in ffn_moe_block:
        if operator.op_type == OperatorType.GEMM:
            operator.B = per_device_ffn_num
            operator.M = per_ffn_token_num
        elif operator.op_type == OperatorType.ALLREDUCE:
            assert per_device_ffn_num == 1 and not model_config.is_moe
            operator.M = per_ffn_token_num
        elif operator.op_type == OperatorType.ALL2ALL:
            operator.M = per_ffn_token_num * per_device_ffn_num


def set_edge_shape(
    model_config: ModelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
    context_length_list: List[int],
):
    batch_size = len(context_length_list)
    for operator in attention_block:
        if operator.name == "softmax":
            continue
        if operator.op_type == OperatorType.GEMM:
            if operator.name in ("attention_qk", "mla_attention_qk"):
                assert all(ctx == context_length_list[0] for ctx in context_length_list), \
                    "All context lengths must be the same for non-fused attention in edge inference."
                operator.B *= batch_size
                operator.N = context_length_list[0]
            elif operator.name in ("attention_sv", "mla_attention_sv"):
                operator.B *= batch_size
                operator.K = context_length_list[0]
            else:
                operator.M = batch_size
        elif operator.op_type == OperatorType.ATTENTION:
            operator.input_length = [1 for _ in range(len(context_length_list))]
            operator.context_length = [c for c in context_length_list]
    
    if model_config.is_moe:
        # assume a uniform distribution of expert routing
        per_ffn_token_num = math.ceil(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts) / model_config.moe_num_experts)
        total_ffn_num = math.ceil(min(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts), model_config.moe_num_experts))
    else:
        total_ffn_num = 1
        per_ffn_token_num = batch_size
    for operator in ffn_moe_block:
        if operator.op_type == OperatorType.GEMM:
            operator.B = total_ffn_num
            operator.M = per_ffn_token_num


if __name__ == '__main__':
    parallel_config = ParallelConfig(
        tp_size=8, ep_size=8
    )

    opt_config = get_model_config_from_hf("opt_66b", "configs/models/opt_66b.json")
    opt_attention, opt_ffn = get_layer_operator_list(model_config=opt_config, parallel_config=parallel_config)

    llama_config = get_model_config_from_hf("llama3_70b", "configs/models/llama3_70b.json")
    llama_attention, llama_ffn = get_layer_operator_list(model_config=llama_config, parallel_config=parallel_config)

    qwen_config = get_model_config_from_hf("qwen3_30b_a3b", "configs/models/qwen3_30b_a3b.json")
    qwen_attention, qwen_expert = get_layer_operator_list(model_config=qwen_config, parallel_config=parallel_config)

    mixtral_config = get_model_config_from_hf("mixtral_8x22b", "configs/models/mixtral_8x22b.json")
    mixtral_attention, mixtral_expert = get_layer_operator_list(model_config=mixtral_config, parallel_config=parallel_config)

    deepseek_config = get_model_config_from_hf("deepseek_v2_236b", "configs/models/deepseek_v2_236b.json")
    deepseek_attention, deepseek_expert = get_layer_operator_list(model_config=deepseek_config, parallel_config=parallel_config)

    pass
