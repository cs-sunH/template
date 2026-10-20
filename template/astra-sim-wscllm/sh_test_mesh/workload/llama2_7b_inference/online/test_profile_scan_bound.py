#!/usr/bin/env python3
"""test_profile_scan_bound.py -- 阶段 4 §7.3 每决策批扫描条目数上界。

E1(2026-09-28)自 verify/profile_scan_audit.py 规范形态化移植。原审计脚本
读真实运行的 profile.jsonl 写 md 报告,无自动调用面(全仓 grep 零引用;
清单 §910 W 口径登记为 python3 直调;2026-09-27 修复后 full-scan 计数
不存在,其仅存断言即"每批 scanned_entries <= total_requests")。本文件把
该验收契约钉进 runner:scanned_entries 计数点枚举当前全部候选访问入口
(基类 completed_groups/arrivals 逐条 _settle_completions/
_process_arrivals + 变体 arrival 堆弹出与 ready frontier 逐条
wsc_llm_online_scheduler.run_variant_policy;准入 pass 仅触增量维护的
ready frontier)。在此前提下单批计数与总 request 数无关——维护约定:
调度器新增遍历入口必须同步接 _profile_scan 计数,否则本测试对该入口
失效(与姊妹仓 wscllm-LRU 同口径)。

运行:pytest online/test_profile_scan_bound.py,或
      cd sh_test_mesh/workload/llama2_7b_inference && python3 -m unittest
      online.test_profile_scan_bound(本文件亦可直跑)。
"""
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.online_scheduler_base import (  # noqa: E402
    OnlineSchedulerBase,
)


class _NoScanScheduler(OnlineSchedulerBase):
    """最小变体:策略无动作(纯基类簿记路径)。"""

    def run_variant_policy(self, delta):
        pass


def _delta(seq, tick, arrivals=()):
    return {
        "schema_version": 1,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": tick,
        "deferred_from_tick": tick,
        "reasons": ["ARRIVAL"],
        "arrivals": [
            {"request_id": request_id, "queue_index": index, "ingress_seq": 0}
            for index, request_id in enumerate(arrivals)
        ],
        "completed_groups": [],
        "completed_nodes": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick},
    }


def _run_deliveries(manifest_size, deliveries=5):
    """喂 K 笔单到达交付,返回 profile_sink 逐批行。"""
    rows = []
    scheduler = _NoScanScheduler(
        manifest={"requests": [
            {"request_id": "r{}".format(index)}
            for index in range(manifest_size)
        ]},
        config=SimpleNamespace(),
        profile_sink=rows.append,
    )
    for seq in range(deliveries):
        scheduler.on_decision_batch(
            _delta(seq, 1000 + seq, arrivals=("r{}".format(seq),)))
    return rows


class ProfileScanBoundTest(unittest.TestCase):
    """单批扫描条目数与总 request 规模无关(无 O(总规模) 全量扫描)。

    覆盖披露(2026-09-30):本用例经 _NoScanScheduler 只驱动基类
    (OnlineSchedulerBase 的 _settle_completions/_process_arrivals)计数
    路径;真实变体(wsc_llm_online_scheduler 的 _profile_scan/
    _admit_pass 逐条 frontier 遍历)不在本用例的执行面内——变体侧的
    有界性由其自身的逐 delivery 计数与 run 末 profile 行共同披露,
    若要钉死变体路径需另用真实调度器喂多笔交付(未包含在此)。"""

    def test_per_batch_scanned_entries_independent_of_manifest_size(self):
        small = _run_deliveries(4, deliveries=3)
        large = _run_deliveries(400, deliveries=3)
        # 两档 manifest 规模下逐批计数逐字节相等 = 计数不随总规模缩放。
        self.assertEqual(small, large)
        for row in large:
            self.assertLessEqual(
                row["scanned_entries"], 8,
                "delivery {} scanned {} entries: per-batch scan cost must "
                "stay bounded, not scale with the manifest".format(
                    row["delivery_sequence"], row["scanned_entries"]))


if __name__ == "__main__":
    unittest.main()
