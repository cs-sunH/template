# 合同⑧ prefix canonical 合同（sh_2.0 sidecar_restore 变体）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）；审查通过是阶段 1 开工硬门槛。

## 口径
- sh_2.0 为 **sidecar_restore** 变体：queue 的 `prefill_length` = 源新增
  token 数（**不折入 prefix**）；context sidecar 交付 `prefix_tokens` =
  源 `prefix_len`，`input_tokens_total = prefix_tokens + prefill_length`
  （对每一行，含 turn>0；`load_request_prefix_tokens` :740-744 等值校验）。
- **禁止双计 prefix**（不得一边折入 prefill_length 一边再填 prefix_tokens）。
- 规范输入 = 8 列执行队列 + context sidecar（5 列：
  session_id,turn_index,request_id,prefix_tokens,input_tokens_total）+
  canonical digest（raw_prefix_tokens/raw_new_prefill_tokens/
  effective_prompt_tokens/digest，prefix_mode=sidecar_restore）——三件套
  同一次物化遍历产出（traces/materialize_first_30s.py）。
- B0 校验：digest 逐行重算比对一致；prefix 作为远端常驻历史 KV 按现有
  恢复语义执行（PARTIAL 后缀恢复流水 / REMOTE 全层恢复），不重算。

## 裁决
- sidecar on/off：`trace_config.csv:13` 默认空 = off（当前 checked-in
  语义：窗口内 session 首请求无历史 KV，turn>0 全量复用前序 context）。
  开启 sidecar 会改变 turn-0 KV 历史语义，属实验配置变化，**须经用户
  裁决后在本合同登记**（尚未裁决，保持 off）。sidecar 文件无论 on/off
  都物化存档（canonical 输入一部分）。

## 验证方法
- 遍历 queue 每行重算 sha256(request_id|prefix|prefill|prefix+prefill)
  与 digest 文件逐行比对（1177 行一致）。
- 阶段 2 B0：online 输入 vs 离线队列逐字段一致，容差 0。
