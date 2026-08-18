#!/usr/bin/env python3
"""test_turn0_eviction_probe_sh10.py -- turn-0 在飞逐出 KeyError 同型排查的
合成单元测试（backport-fix1 第 3 项，2026-08-16）。

背景（sh_2.0 对比报告 §5.2 回灌）：sh_2.0 的 GraphBatchBuilder 存在
"turn-0 在飞会话在 pending_request_by_session 登记键、但 pending_history
无 gate（turn-0 gate 内联不落账本）+ 逐出路径直接下标"组合，另一请求的
remote_store 逐出命中该会话时 KeyError。face/wscllm/sh_3.0 排查为不适用
或结构性免疫。

sh_1.0 排查结论（本测试固化的证据）：
  1. 病灶行 = online/graph_batch_builder.py:404
     `self.pending_history[pending_request_id]["location"] = ...`——若
     病态状态（session 键指向无 gate 的请求）可达，此行即 KeyError
     （本测试 case C 人为构造该状态演示病灶签名）；
  2. **结构性免疫**：本仓 turn-0 到达门登记（_emit_arrival_gate，
     :763-768）与 turn>0 following 登记（_emit_segment3，:870-875）都在
     同一调用内成对写 pending_history[request_id] +
     pending_request_by_session[session_id]（turn-0 的 gate 是落账本的
     new_session 门——与 sh_2.0 的内联不落账本设计不同构）；消费侧
     _emit_segment1（:508 pop gate / :512 pop session 键）同样成对；
     Python 单线程逐批构造，批间无交织 → 病态状态不可达（case D 静态
     断言登记/弹出成对性，未成对的新增登记会使本测试 FAIL）；
  3. 30s 全量运行证据：replay/strategy/sensing 三模式 3530/3531 交付
     零 KeyError；B2 确定性重放全部交付流逐行全等（零异常）——见
     sh_1.0改造执行实录.md 回灌轮登记。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_turn0_eviction_probe_sh10.py   （或 pytest 同路径）
"""
import os
import re
import sys

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

SESSION = "session_synthetic_0"
TURN0 = f"{SESSION}_request_0"
OTHER = "session_other_request_9"


def _bare_builder() -> GraphBatchBuilder:
    """__new__ 构造（绕过重型 __init__）：_mark_pending_history_remote
    只触达下列三个属性，合成状态足够。"""
    b = GraphBatchBuilder.__new__(GraphBatchBuilder)
    b.pending_request_by_session = {}
    b.pending_history = {}
    b.deferred_remote_sessions = set()
    return b


def test_deferred_branch_without_session_key():
    """case A：会话无在飞键（已完结且后续 turn 未规划）→ 挂
    deferred_remote_sessions，与离线 mark_pending_history_remote 的
    无键语义一致，无任何下标。"""
    b = _bare_builder()
    b._mark_pending_history_remote(SESSION)
    assert SESSION in b.deferred_remote_sessions
    assert not b.pending_history
    print("[turn0-probe] case A PASS: no-key session deferred")


def test_paired_registration_branch():
    """case B：在飞会话（turn-0 或 turn>0，成对登记后）→ location 折算
    为 remote_memory，不抛。"""
    b = _bare_builder()
    b.pending_request_by_session[SESSION] = TURN0
    b.pending_history[TURN0] = {
        "source_instance_index": 0, "timer_gates": (), "location": "new_session"}
    b._mark_pending_history_remote(SESSION)
    assert b.pending_history[TURN0]["location"] == "remote_memory"
    assert SESSION not in b.deferred_remote_sessions
    print("[turn0-probe] case B PASS: paired registration updated in place")


def test_disease_signature_documented():
    """case C（病灶签名演示，非生产行为）：人为构造 sh_2.0 病态状态
    （session 键指向无 gate 的请求），证明 :404 直接下标即 KeyError——
    本仓结构性免疫（case D）保证该状态不可达，此用例仅固化病灶定位。"""
    b = _bare_builder()
    b.pending_request_by_session[SESSION] = TURN0  # 故意不写 pending_history
    try:
        b._mark_pending_history_remote(SESSION)
    except KeyError:
        print("[turn0-probe] case C PASS: disease state would KeyError "
              "at the direct subscript (unreachable here, see case D)")
        return
    raise AssertionError("disease state did not raise -- hazard line moved?")


def test_structural_pairing_immunity():
    """case D（免疫机制静态断言）：builder 源码中每一处
    pending_request_by_session 登记都必须与同一代码块内的一条
    pending_history[...] 登记成对（容差 = 成对语句间距 <= 8 行，容纳多行 dict 字面量），每一处
    消费 pop 亦成对。新增未成对的登记/弹出会使本测试 FAIL——这是对
    "结构性免疫"结论的回归护栏。"""
    src_path = os.path.join(_ONLINE_DIR, "graph_batch_builder.py")
    src = open(src_path, encoding="utf-8").read().splitlines()

    def find(pattern):
        return [i for i, line in enumerate(src) if re.search(pattern, line)]

    reg_sites = find(r"pending_request_by_session\[.*\] =")
    pop_sites = find(r"pending_request_by_session\.pop\(")
    hist_reg = set(find(r"pending_history\[.*\] = "))
    hist_pop = set(find(r"pending_history\.pop\("))
    assert reg_sites, "no registration sites found (pattern drift?)"
    for i in reg_sites:
        paired = any(j in hist_reg for j in range(max(0, i - 8), i + 9))
        assert paired, (
            f"UNPAIRED session-key registration at graph_batch_builder.py:"
            f"{i + 1}: {src[i].strip()} -- sh_2.0 §5.2 disease precondition")
    for i in pop_sites:
        paired = any(j in hist_pop for j in range(max(0, i - 8), i + 9))
        assert paired, (
            f"UNPAIRED session-key pop at graph_batch_builder.py:{i + 1}: "
            f"{src[i].strip()}")
    print(f"[turn0-probe] case D PASS: {len(reg_sites)} registration site(s) "
          f"and {len(pop_sites)} pop site(s) all paired with "
          f"pending_history writes/pops")


if __name__ == "__main__":
    test_deferred_branch_without_session_key()
    test_paired_registration_branch()
    test_disease_signature_documented()
    test_structural_pairing_immunity()
    print("ALL 4 CASES PASS")
