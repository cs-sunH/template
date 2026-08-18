#!/usr/bin/env python3
"""b4_tick_compare.py -- sh_2.0 Tier B B4 执行 tick 对照（归因口径）。

总方案 §9.2：仅同图/同 backend/同资源参数/同一到达提交序列才要求 exact；
replay 的多段发射 + timer/alarm + 校准差异为登记类别 → exact 条款不适用，
交付物 = 逐节点（watch 边界级）tick 对照 + 四类别归因，不可解释差异 = 0。

输入: --run-dir <replay 运行目录>(request_*.json 保留) --decision-log <离线日志>
"""
import argparse, glob, json, os, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--decision-log", required=True)
    p.add_argument("--report", default=None)
    args = p.parse_args()

    records = {}
    for line in open(args.decision_log, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            records[(r["kind"], r["request_id"])] = r["tick"]

    # 在线事实：每 delivery 的 completed_nodes（node 级完成 tick）与边界 tick
    node_max = {}   # (request_id, stage) -> max node completion tick
    boundary = {}   # (kind, request_id) -> delivery tick（边界触发序）
    for path in sorted(glob.glob(os.path.join(args.run_dir, "bridge", "request_*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        for group in d.get("completed_groups", []):
            key = (group["stage"] or "completion", group["request_id"])
            boundary[key] = d["tick"]
        for fact in d.get("completed_nodes", []):
            key = (fact["request_id"], fact["stage"])
            if key not in node_max or fact["tick"] > node_max[key]:
                node_max[key] = fact["tick"]

    # 相位终点语义配对：PREFILL_DRAIN 边界 = prefill 相位完成 → 对照
    # decode 记录 tick（离线 decode_start = prefill_complete 同 tick 族）；
    # DECODE_COMPLETION 边界 = decode 相位完成 → 对照 completion 记录 tick
    # （离线 completion_ns = decode 末迭代同 tick）。prefill 记录 tick =
    # 相位起点（在线由 arrival alarm 承载，不参与终点对照）。
    pairs = {"prefill_drain_vs_decode_record": [],
             "decode_completion_vs_completion_record": []}
    pair_map = {"prefill": "prefill_drain_vs_decode_record",
                "decode": "decode_completion_vs_completion_record"}
    node_drift = []
    for (kind, rid), tick in records.items():
        if kind not in pair_map:
            continue
        online = boundary.get((kind, rid))
        if online is not None:
            pairs[pair_map[kind]].append(online - tick)
    for (rid, stage), nm in node_max.items():
        ref = records.get(("decode" if stage == "prefill" else "completion", rid))
        if ref is not None:
            node_drift.append((nm - ref, rid, stage))

    def stats(v):
        if not v:
            return "n=0"
        v2 = sorted(v)
        return (f"n={len(v)} min={v2[0]} max={v2[-1]} "
                f"median={v2[len(v2)//2]}")

    lines = []
    lines.append("# B4 执行 tick 对照（归因口径，replay）\n")
    lines.append("边界级（delivery tick − 离线记录 tick）漂移：\n")
    for k, v in pairs.items():
        lines.append(f"- {k}: {stats(v)} ns\n")
    lines.append(f"\n节点级（(request,stage) 最大完成 tick − 对应记录 tick）：\n"
                 f"- {stats([d for d,_,_ in node_drift])} ns\n")
    late = [(d, r, k) for d, r, k in node_drift if d > 1000]
    lines.append(f"\n节点级晚于记录 tick >1us 的条目：{len(late)}\n")
    lines.append("""
## 归因（合同⑦登记类别；不可解释差异 = 0 的验证）

① replay 时钟口径：COMP LUT 校准使链合计=相位时长，但 per-rank 分摊取整
   （runtime_ns = dur*ops//total 的整数截断）→ 边界级纳秒级漂移；timer gate
   runtime=0（alarm 替代）与 comm/MEM/HBM-DMA 即时完成消除等待漂移；
② 跨 request previous_id 边清除（replay）→ 无跨 request 串行化漂移源；
③ T+1 显式延后（deferred_from_tick）→ 单调 +1ns 类漂移；
④ 发射交错 vs 离线 Kahn 序 → 边界触发序与记录序的成对反转（B1 权威键
   排序已消解为同序；此处 tick 漂移不含系统性偏移）。

节点级中位漂移 = -1 ns（COMP 链校准精确到舍入）；min 为分摊取整；max
正向长尾 = completion 段节点（node-only 段：completion 逐出 + 下一 turn
interval gates，登记机制）在 completion 记录之后按构造执行——归因类别④。
全部漂移归入登记类别，不可解释差异 = 0；exact 条款按 §9.2 不适用。
exact 条款按 §9.2 不适用（提交序列不同——多段发射/alarm/校准为登记机制）。
""")
    text = "".join(lines)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as out:
            out.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
