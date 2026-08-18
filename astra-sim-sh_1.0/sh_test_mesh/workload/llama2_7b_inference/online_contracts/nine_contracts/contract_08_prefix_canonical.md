# 合同⑧ prefix canonical 合同(sh_1.0 recompute 变体)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径
- sh_1.0 首版为 **recompute 变体**(prefix_mode=recompute):turn-0 prefill =
  prefix_len + prefill_length(prefix 折入);其余 turn prefill = prefill_length。
- 历史 8 列 recompute CSV 不能作为 prefix canonical 规范输入(raw prefix
  不可逆丢失);历史队列仅作旧基线规则参考。
- 规范输入 = 8 列执行队列 + 按 request_id 关联的 canonical sidecar
  (traces/astra_compute_20_first_30_seconds_canonical_sidecar.csv):
  request_id, turn_index, session_id, raw_prefix_tokens, raw_new_prefill_tokens,
  effective_prompt_tokens, prefill_length, decode_length,
  session_arrival_time_ns, inter_request_interval_ns, prefix_mode=recompute,
  digest(= sha256(8 列队列数据行逗号连接))。
- **本仓首版不走 sidecar_restore**(无 loader,§0.6 已核);sidecar 恢复属
  独立适配,需另行立项重新验收(总方案 §4.2/§5.5 既定裁决)。

## 裁决
- canonical sidecar 是 B0 校验依据;缺失或 digest 不一致 = 输入等价失败,
  不得静默跳过(实测可关联,sidecar 已物化)。
- 50/80/120 档与 3min 历史输入禁止仿真(§0.3),仅作规则参考。

## 验证方法
- 遍历 8 列队列每行重算 sha256 与 sidecar digest 逐行比对(全 1177 行一致;
  实测 md5 与 face 仓物化逐字节一致)。
- 阶段 2 B0 比较器逐字段一致(容差 0)。
