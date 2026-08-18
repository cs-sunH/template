# 合同⑧ prefix canonical 合同(sh_3.0 sidecar_restore 变体)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- sh_3.0 为 **sidecar_restore 变体**(与蓝本 recompute 的关键差异):
  queue CSV 的 `prefill_length` = 源 `prefill_length`(新增 token 数,
  **不折叠 prefix**);prefix 经 context sidecar 交付。
- 规范输入三件套(物化于 `traces/`,规则见 PROVENANCE.md):
  1. 8 列执行队列(列序固定 session_id, turn_index, request_id,
     prefill_length, decode_length, session_arrival_time_ns,
     inter_request_interval_ns, description);
  2. context sidecar(5 列:session_id, turn_index, request_id,
     prefix_tokens, input_tokens_total;每请求一行,格式血统 =
     template_back/history/20_3mins/traces/*_request_context.csv);
  3. canonical sidecar(request_id 键关联:raw_prefix_tokens /
     raw_new_prefill_tokens / effective_prompt_tokens / prefix_mode /
     digest 等,见物化脚本列)。
- 核心等式:**`input_tokens_total == prefix_tokens + prefill_length`**
  (双防线:`load_request_prefix_tokens` 校验 `generate_face_trace.py:740-744`
  与 FaceRequest 校验 `face_scheduler.py:3304-3310`,均不得绕过)。
  prefix 作为远端常驻历史 KV 按现有恢复语义执行;`input_tokens_total`
  是该请求的有效输入长度。**禁止双计**(禁止把 prefix 折入 prefill_length
  的同时再填 prefix_tokens)。
- sidecar 语义:窗口首请求 previous_final_context=0 故 history=0
  (`history = min(prefix_tokens, previous_final_context)`,
  `face_scheduler.py:3486`),但 `prefill_context_tokens =
  input_tokens_total` 含 prefix(首请求 prefill 工作量含 prefix 重算),
  后续 turn 的 KV 账本含 prefix——与本仓历史 3mins_remote_prefix 基线同型。
- 每个 queue 请求必须有 sidecar 行——loader 缺行即 raise
  (`generate_face_trace.py:735-738`);未启用 sidecar 的输入 fail-closed
  (本仓合同⑧固定 sidecar_restore,RequestEnvelope turn-0 行携带
  prefix_tokens/input_tokens_total)。

## 裁决

- canonical sidecar 是 B0 的校验依据;digest = sha256(8 列队列数据行,
  UTF-8,逗号连接),物化时一并生成并登记 provenance。
- 历史派生队列(`*_trunc1M.csv`、各派生 request 队列)丢弃 prefix 或重复
  合并 prefix,禁止使用(§0.3);3min frozen 输入仅作格式血统参考。

## 验证方法

- 遍历 8 列队列每行重算 sha256 与 canonical digest 逐行比对(全 1177 行
  一致;物化脚本已内置)。
- 阶段 2 B0:queue/context/canonical 三者互校 + manifest 逐请求
  `source_prefix_tokens`/`source_input_tokens_total` 一致 + 双计校验
  全量成立。
