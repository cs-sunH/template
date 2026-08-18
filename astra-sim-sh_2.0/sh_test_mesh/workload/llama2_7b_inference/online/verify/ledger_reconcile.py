#!/usr/bin/env python3
"""ledger_reconcile.py -- sh_2.0 分层账本对账入口（wscllm 054118e 同型修复）。

血统与修复说明（2026-08-16 wscllm 同型缺陷排查）：
本文件原为本仓迁移时继承的 wscllm ledger_reconcile.py（修复前版本），在
本仓存在两个同型缺陷，均已实测确认，本次整体替换：
  1) 陈旧契约（wscllm 缺陷 1 同型）：原 R2/R3/R5b 从 bridge 的
     response_*.json 读 committed/watches/kv_actions，而本仓 bridge 与
     wscllm phase-7 §10.3 同款"response 消费即删"（decision_bridge.py
     ack 后 os.remove；run 结束后 bridge 目录只余 request_*.json——
     30s/3min 产物实测 response 计数恒 0），跑原版必然 R2/R3/R5b 空数据
     源 FAIL（R2 计数=2×请求全量爆表、R3 全无动作的签名）。
  2) 错误不变量（wscllm 缺陷 2 同型）：原 R2a 假设每 (request, stage) 恰
     注册一次 watch（"每批 watch∈{0,1}"的同型形式）；实际 PREFILL_DRAIN
     批可注册 2 个 watch，DECODE_COMPLETION+REQUEST_COMPLETE 批恒 0。
  另：原版 R3 依赖 wscllm 专属工件 kv_cache_events.csv / cpp.log 完成事实
  格式，本仓 KVCacheManager 无事件流 API，本就不适用（阶段 4 已为此另立
  ledger_reconcile_sh20.py）。

按 wscllm 054118e 同款修法处置：数据源迁移到归档 jsonl（ledger /
online_decision_log / graph_batch_digests / sensing_query_log + cpp.log
计数器），权威逻辑全量实现在 ledger_reconcile_sh20.py（R0-R6，含本次
加固的 R2a/R2b/R2c 批级正确不变量与 R3c tick 对平）；本文件仅为其薄
入口，兼容三种调用：
  --run-dir <run_dir>                      （本仓口径，原生直通）
  --bridge-dir <run_dir>/bridge            （wscllm 式；映射为 --run-dir）
  --online <dir>                           （README_COMMANDS.md 口径；同上映射）
原版 --manifest/--cpp-log/--report 参数已无对应物（expected 从 --expected
取，报告走 stdout），出现时被吞掉并提示。
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import ledger_reconcile_sh20 as _sh20  # noqa: E402


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--bridge-dir", dest="bridge_dir", default=None)
    pre.add_argument("--online", default=None)
    known, remaining = pre.parse_known_args()

    drop = {"--manifest", "--cpp-log", "--report"}
    cleaned = []
    skip_next = False
    for token in remaining:
        if skip_next:
            skip_next = False
            continue
        if token in drop:
            if "=" not in token:
                skip_next = True
            continue
        cleaned.append(token)

    has_run_dir = any(t == "--run-dir" or t.startswith("--run-dir=")
                      for t in cleaned)
    if not has_run_dir:
        for flag in (known.bridge_dir, known.online):
            if flag:
                path = os.path.abspath(flag.rstrip("/") or "/")
                if os.path.basename(path) == "bridge" and os.path.isdir(path):
                    path = os.path.dirname(path)
                cleaned = ["--run-dir", path] + cleaned
                break

    if not any(t == "--run-dir" or t.startswith("--run-dir=")
               for t in cleaned):
        print("[ledger_reconcile] 缺 --run-dir / --bridge-dir / --online",
              file=sys.stderr)
        return 2
    sys.argv = [sys.argv[0]] + cleaned
    return _sh20.main()


if __name__ == "__main__":
    sys.exit(main())
