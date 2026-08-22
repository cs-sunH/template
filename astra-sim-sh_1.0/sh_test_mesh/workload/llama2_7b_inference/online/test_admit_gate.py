#!/usr/bin/env python3
"""test_admit_gate.py -- 改法D（KV 账本纪元重试门）最小单元测试
（SH系列性能修复执行文档 §2.2 / §9-Q3，2026-08-22）。

被测逻辑 = sh10_online_scheduler.admit_waiting_requests 的门分支 +
_bump_kv_ledger_epoch + verify_run_end 的 stale-epoch 断言。构造方式沿用
本仓 test_turn0_eviction_probe_sh10.py 的 __new__ 合成状态脚手架（绕过
重型 __init__：门逻辑只触达 pending_admissions/_kv_ledger_epoch/
_admit_attempt_epoch/_admit_gate_verify 与 try_admit_request，合成状态
足够；try_admit_request 以实例属性 stub，不触真实 KV 账本）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_admit_gate.py   （或 pytest 同路径）
"""
import os
import re
import sys
from collections import deque

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.sh10_online_scheduler import Sh10OnlineScheduler  # noqa: E402

R1 = "session_a_request_0"
R2 = "session_b_request_0"


def _bare_scheduler(*, verify: bool = False) -> Sh10OnlineScheduler:
    """__new__ 构造（绕过重型 __init__）：门逻辑只触达下列属性。"""
    s = Sh10OnlineScheduler.__new__(Sh10OnlineScheduler)
    s.pending_admissions = deque()
    s._kv_ledger_epoch = 0
    s._admit_attempt_epoch = {}
    s._admit_gate_verify = verify
    return s


def _with_clean_run_end_state(s: Sh10OnlineScheduler) -> Sh10OnlineScheduler:
    """补齐 verify_run_end（基类 + 本仓）在空运行结束态所需的字段。"""
    s.ack_count = 0
    s.delivery_count = 0
    s.last_applied_sequence = -1
    s._delivery_reply_cache = None
    s.in_flight = {}
    s._provisional_kv_actions = {}
    s.sensing_enabled = False
    s.pending_admissions = deque()
    s._runtimes = {}
    s.instances = []
    return s


def test_gate_skips_unchanged_epoch_and_keeps_fifo():
    """失败纪元未变 → 不重试（stub 不被调用），FIFO 位置不变；
    纪元 bump 后恢复重试。"""
    s = _bare_scheduler()
    calls = []
    s.try_admit_request = lambda rid, now: calls.append(rid) or False
    s.pending_admissions.extend([R1, R2])
    s.admit_waiting_requests(1000)
    assert calls == [R1, R2], calls
    assert list(s.pending_admissions) == [R1, R2]  # FIFO 位置不变
    assert s._admit_attempt_epoch == {R1: 0, R2: 0}
    # 纪元未变：第二批跳过两条（零 stub 调用）。
    s.admit_waiting_requests(2000)
    assert calls == [R1, R2], calls
    # KV 变更一次 → 两条都恢复重试并再次失败（纪元更新为 1）。
    s._bump_kv_ledger_epoch()
    s.admit_waiting_requests(3000)
    assert calls == [R1, R2, R1, R2], calls
    assert s._admit_attempt_epoch == {R1: 1, R2: 1}
    print("[admit-gate] case 1 PASS: unchanged epoch skips, bump re-opens, "
          "FIFO preserved")


def test_success_clears_attempt_epoch_bounded():
    """成功准入即清除 _admit_attempt_epoch（有界性：dict 只含当前 blocked）。"""
    s = _bare_scheduler()
    results = {R1: False, R2: True}
    s.try_admit_request = lambda rid, now: results[rid]
    s.pending_admissions.extend([R1, R2])
    s.admit_waiting_requests(1000)
    assert list(s.pending_admissions) == [R1]
    assert s._admit_attempt_epoch == {R1: 0}  # R2 成功即清除
    s._bump_kv_ledger_epoch()
    results[R1] = True
    s.admit_waiting_requests(2000)
    assert not s.pending_admissions
    assert s._admit_attempt_epoch == {}  # 全部清除 → 有界
    print("[admit-gate] case 2 PASS: success pops attempt epoch (bounded)")


def test_shadow_mode_asserts_equivalence():
    """影子模式（_admit_gate_verify）：门判跳过的条目仍完整评估；
    若重试返回 True（等价性被破坏）→ RuntimeError。"""
    s = _bare_scheduler(verify=True)
    s.try_admit_request = lambda rid, now: False
    s.pending_admissions.append(R1)
    s.admit_waiting_requests(1000)
    assert s._admit_attempt_epoch == {R1: 0}
    # 第二批：影子重评估返回 False → 无异常、保持 blocked。
    s.admit_waiting_requests(2000)
    assert list(s.pending_admissions) == [R1]
    # 第三批：影子重评估返回 True（违例）→ fail-closed。
    s.try_admit_request = lambda rid, now: True
    try:
        s.admit_waiting_requests(3000)
    except RuntimeError as exc:
        assert "admit gate equivalence violated" in str(exc), exc
        print("[admit-gate] case 3 PASS: shadow re-evaluates and "
              "fail-closes on violation")
        return
    raise AssertionError("shadow mode did not raise on violated retry")


def test_verify_run_end_stale_epoch_guard():
    """收尾审计：_admit_attempt_epoch 非空（有 blocked 未清除却宣成
    结束）→ RuntimeError；空态通过。"""
    s = _with_clean_run_end_state(_bare_scheduler())
    s.verify_run_end()  # 空态通过
    s._admit_attempt_epoch = {R1: 7}
    try:
        s.verify_run_end()
    except RuntimeError as exc:
        assert "stale admit attempt epochs" in str(exc), exc
        print("[admit-gate] case 4 PASS: stale admit attempt epochs "
              "fail-closed at run end")
        return
    raise AssertionError("verify_run_end did not raise on stale epochs")


def test_nine_bump_sites_static_guard():
    """结构性回归护栏（同 test_turn0_eviction_probe_sh10.py case D 体例）：
    scheduler 源码必须恰好 9 处 `self._bump_kv_ledger_epoch()` 调用
    （各带 `# 改法D：KV 变更点 i/9` 注释），__init__ 读
    SH_ADMIT_GATE_VERIFY 开关——漏 bump/多 bump 都会使本测试 FAIL。"""
    src_path = os.path.join(_ONLINE_DIR, "sh10_online_scheduler.py")
    src = open(src_path, encoding="utf-8").read().splitlines()
    call_sites = [
        i for i, line in enumerate(src)
        if re.search(r"^\s*self\._bump_kv_ledger_epoch\(\)", line)
        and "def _bump_kv_ledger_epoch" not in line
    ]
    assert len(call_sites) == 9, (
        f"expected exactly 9 bump call sites, found {len(call_sites)}: "
        f"{[src[i].strip() for i in call_sites]}")
    for i in call_sites:
        assert re.search(r"# 改法D：KV 变更点 [1-9]/9", src[i]), (
            f"bump site sh10_online_scheduler.py:{i + 1} missing the "
            f"i/9 annotation: {src[i].strip()}")
    joined = "\n".join(src)
    assert 'os.environ.get("SH_ADMIT_GATE_VERIFY") == "1"' in joined, (
        "shadow-verify env switch missing from __init__")
    print(f"[admit-gate] case 5 PASS: {len(call_sites)} bump sites "
          f"annotated i/9 + shadow switch present")


if __name__ == "__main__":
    test_gate_skips_unchanged_epoch_and_keeps_fifo()
    test_success_clears_attempt_epoch_bounded()
    test_shadow_mode_asserts_equivalence()
    test_verify_run_end_stale_epoch_guard()
    test_nine_bump_sites_static_guard()
    print("ALL 5 CASES PASS")
