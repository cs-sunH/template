#!/usr/bin/env python3
"""test_arrived_twice_guard.py -- 基类 _process_arrivals 重复到达 fail-closed
回归钉子（中-2 问题①修复，2026-08-20）。

背景：sh_2.0 基类曾对 "prefill 仍在待办" 的重复到达加 6 行容忍块（continue
吞掉），偏离其余四仓的严格 fail-closed raise。该分支被证明为死代码（sh_2.0
无 replay 模式入口；基类放行后 heap 层守卫必 raise；C++ 注入模型每请求恰
注入一次），已删除，基类恢复与四仓逐字节一致。

用例 1 正是原容忍分支放行的场景（回归钉子，修复前必失败）：同一
request_id 到达两次且 prefill 仍在待办 -> ValueError("already-arrived")。
用例 2 sanity：不同请求各到达一次 -> in_flight 登记 prefill+decode 两段待办。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_arrived_twice_guard.py   （或 pytest 同路径）
"""
import os
import sys

import pytest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    OnlineSchedulerBase,
)


def _make_scheduler() -> OnlineSchedulerBase:
    scheduler = OnlineSchedulerBase(
        manifest={"requests": [{"request_id": "r1"}, {"request_id": "r2"}]},
        config=object(),
        mode="strategy",
    )
    # 单测直调 _process_arrivals 需自备批次累加器（正式链路由 _start_batch
    # 初始化，构造默认 None；_profile_scan 会下标写入）。
    scheduler._profile_batch = {"scanned_entries": 0, "full_scan_entries": 0}
    return scheduler


def test_duplicate_arrival_with_prefill_pending_fails_closed():
    """回归钉子：prefill 仍在待办的重复到达不再被容忍块 continue 吞掉，
    恢复与其余四仓一致的 fail-closed raise。"""
    scheduler = _make_scheduler()
    with pytest.raises(ValueError, match=r"already-arrived"):
        scheduler._process_arrivals(
            {"arrivals": [{"request_id": "r1"}, {"request_id": "r1"}]})
    # 第一次到达已合法登记（prefill+decode 待办），异常由第二次到达触发。
    assert scheduler.in_flight == {"r1": {STAGE_PREFILL, STAGE_DECODE}}


def test_distinct_arrivals_register_both_stages():
    """sanity：r1、r2 各到达一次 -> in_flight 登记两段待办集合。"""
    scheduler = _make_scheduler()
    scheduler._process_arrivals(
        {"arrivals": [{"request_id": "r1"}, {"request_id": "r2"}]})
    assert scheduler.in_flight == {
        "r1": {STAGE_PREFILL, STAGE_DECODE},
        "r2": {STAGE_PREFILL, STAGE_DECODE},
    }


if __name__ == "__main__":
    test_duplicate_arrival_with_prefill_pending_fails_closed()
    test_distinct_arrivals_register_both_stages()
    print("test_arrived_twice_guard: 2/2 PASS")
