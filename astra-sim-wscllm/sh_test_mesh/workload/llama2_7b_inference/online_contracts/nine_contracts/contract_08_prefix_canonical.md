# 合同⑧ prefix canonical 合同(wscllm recompute 变体)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- wscllm 为 **recompute 变体**:turn-0 prefix 折入 prefill
  (turn-0 `prefill_length = prefix_len + prefill_length`;其余 turn
  `prefill_length = prefill_length`)。
- **历史 8 列 recompute CSV 不能作为 prefix canonical 的规范输入**:
  其已把 raw prefix 与新 prefill 合并(derive_request_queue.py 头注释
  "Semantics (prefix ignored)"),raw prefix 信息不可逆丢失——历史队列仅作
  旧基线(3min frozen 血统;物化规则登记于方案文档 §3 步骤 0-1)。
- 规范输入 = **8 列执行队列 + 按 request_id 键关联的 canonical sidecar**
  (物化文件名与规则见方案文档 §3 步骤 0-1;8 列执行队列列序固定为
  session_id, turn_index, request_id, prefill_length, decode_length,
  session_arrival_time_ns, inter_request_interval_ns, description):
  - sidecar 字段:
    request_id, turn_index, session_id, raw_prefix_tokens(源 prefix_len),
    raw_new_prefill_tokens(源 prefill_length),
    effective_prompt_tokens(= recompute 队列 prefill_length:turn-0 为
    prefix+prefill,其余为 prefill), prefill_length, decode_length,
    session_arrival_time_ns, inter_request_interval_ns, prefix_mode=recompute,
    digest;
  - digest = sha256(8 列队列数据行,UTF-8,逗号连接)——物化时
    一并生成并登记 provenance(方案 §3 步骤 0-1)。
- B0 校验:两者 digest 一致(8 列队列逐行重算 digest 与 sidecar 比对);
  数量/8 列转换/arrival/gap/窗口 turn/prefix 口径逐字段一致,容差 0。

## 裁决

- canonical sidecar 是 B0 的校验依据与阶段 2/3 输入等价比较的规范输入;
  sidecar 缺失或 digest 不一致 = 输入等价失败,不得静默跳过(步骤 0-1
  已显式记录"若源数据无法按 request_id 关联,记录 canonical sidecar 缺失
  并停下上报"——实测可关联,sidecar 已物化)。
- 50/80/120 档与 3min 历史输入禁止仿真(§0.3),仅作规则参考。

## 验证方法

- `python3 -c` 或验证脚本:遍历 8 列队列每行重算 sha256,与 sidecar digest
  逐行比对(全 1177 行一致)。
- 阶段 2 B0 比较器:online 输入物化 vs 离线队列,逐字段一致(容差 0)。
