#!/usr/bin/env python3
"""joint_runner.py -- astra-sim-joint 的 run 级编排入口（D14 采纳实现）。

设计依据：《详细版三机制联合仓库构造执行方案》D14（锁与运行）与
§8.1-1 的可适用子项。本实现为**包装层**：双进程拉起、GEN_MATCH、指标
后处理与归档沿用本仓已验证的 `run_online_strategy.sh` 链（避免复制
150 行拉起逻辑形成第二实现），其上叠加 D14 的全部实质要求：

1. 仓内仿真锁：`sh_test_mesh/runs/.single_simulation.lock`（flock -n，
   覆盖子进程——外层锁在本进程存续期间持有，runner 及其 children 全部
   处于锁内；跨栈并发仿真即拒绝启动）；
2. 二进制溯源：启动前对本仓 `AstraSim_Analytical_Congestion_Aware_Online`
   做 sha256 校验（存在性 + 可执行性 fail-closed）；
3. invocation 记录：`<run_dir>/invocation.json`（命令行、joint 开关、
   二进制 sha256、入口脚本 sha256、exit code、UTC 时间戳）；
4. env 清洗：启动前弹出全部陈旧策略变量（SH30_* / SH3_CAUSAL_* /
   SH_JOINT_*），仅保留/透传 D7 白名单内的运维变量（SH_TRAIN_MAX_ITER /
   SH_ADMIT_GATE_VERIFY / SH_SNAPSHOT_VERIFY / SH_ONLINE_VALIDATE /
   SH_FIRST_TOKEN_SPLIT / BRIDGE_TIMEOUT_MS）与 JOINT_* 开关族
   （2026-09-14 用户裁定：开关入口 = JOINT_* env，见 PROVENANCE.md
   偏差登记 D7-D9）。

用法：
    python3 sh_test_mesh/run_scripts/joint_runner.py <run_dir> <request_csv> \
        [--combo TJE|none|T|J|E|TJ|TE|JE] \
        [--category typed|lru] \
        [--scheduler joint|load-first|affinity-first|face_static] \
        [--layer adaptive|legacy_half|minimal_layer_groups] \
        [--remote on|off] [--remote-credit-iters auto|<正整数>] \
        [--quota off|static|aimd] \
        [--extra-env NAME=VALUE ...]

combo 与显式开关互斥（与 joint_config.parse 的 fail-closed 口径一致）。

C11（F7 耦合规则）：``--quota aimd``（或 ``--extra-env
JOINT_QUOTA_MODE=aimd``）⇒ 发射层**自动注入** ``--link-telemetry``
——注入点 = 本 runner 同时持有环境变量与 CLI 参数的公共发射路径：
env ``SH_LINK_TELEMETRY=1`` 置位（内层 run_online_strategy.sh 消费并
追加 C++ 旗标）；invocation.json 记录注入事实（``link_telemetry_
injected``）。不存在"aimd + 无遥测"的合法启动路径：内层 runner 对
``JOINT_QUOTA_MODE=aimd`` 且 ``SH_LINK_TELEMETRY`` 未置位 fail-closed
（防御性断言——仅注入逻辑损坏才触发；显式 ``SH_LINK_TELEMETRY=1``
直跑遥测臂不受影响）。

O11（2026-09-23，L6 断言的运行期双保险）：C++ 观测门还压在
``MetricCollector::enabled`` 下（main_online.cc:1165）——``aimd`` ∧
生效指标档 ``off`` ⇒ 遥测恒空、AIMD 静默空转。本 runner 在锁/二进制
检查前同构 fail-closed（生效档解析 = ``SH_METRICS_DETAIL`` env >
metrics_config.json detail_level，非法即拒）：aimd ∧ off ⇒ 打印指引 +
exit 1，无逃生口径（与 shell 断言一致），绕过 wrapper 直启不再静默
空转；invocation.json 记录判定所用档位（``metrics_detail``）。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_DIR = _REPO_ROOT / "sh_test_mesh" / "runs"
_LOCK_PATH = _RUNS_DIR / ".single_simulation.lock"
_LOCK_FD_ENV = "SH_SINGLE_SIMULATION_LOCK_FD"
_BINARY_RELPATH = (
    "build/astra_analytical/build_congestion_aware/bin/"
    "AstraSim_Analytical_Congestion_Aware_Online")
_INNER_RUNNER = _REPO_ROOT / "sh_test_mesh" / "run_scripts" / "run_online_strategy.sh"
# 指标档位权威配置（L6/O11 守卫的 json 回落源；绑定于 import 时的真实仓
# 根——F7CouplingRuleTest 的 stub _REPO_ROOT 补丁不应把档位解析指进桩仓）。
_METRICS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "sh_test_mesh" / "workload"
    / "llama2_7b_inference" / "metrics_config.json")

# D7 白名单运维变量 + JOINT_* 开关族（裁定后的策略入口）。
_PREFIXES_TO_SCRUB = ("SH30_", "SH3_CAUSAL_", "SH_JOINT_")
_ENV_ALLOWLIST = (
    "SH_TRAIN_MAX_ITER", "SH_ADMIT_GATE_VERIFY", "SH_SNAPSHOT_VERIFY",
    "SH_ONLINE_VALIDATE", "SH_FIRST_TOKEN_SPLIT", "BRIDGE_TIMEOUT_MS",
    # C11（F7 耦合）：aimd ⇒ --link-telemetry 自动注入的交接变量
    # （本 runner 置位，内层 run_online_strategy.sh 消费追加 C++ 旗标）。
    "SH_LINK_TELEMETRY",
)
_JOINT_SWITCH_VARS = (
    "JOINT_ABLATION_COMBO", "JOINT_CATEGORY_MODE", "JOINT_SCHEDULER_MODE",
    "JOINT_LAYER_POLICY", "JOINT_REMOTE_ACTIONS",
    "JOINT_REMOTE_CREDIT_ITERS", "JOINT_REMOTE_READ_PARTIAL",
    "JOINT_QUOTA_MODE",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scrub_environment() -> list[str]:
    """弹出陈旧策略变量（SH30_*/SH3_CAUSAL_*/SH_JOINT_*），返回弹出清单。"""
    scrubbed = []
    for name in sorted(os.environ):
        if name.startswith(_PREFIXES_TO_SCRUB):
            scrubbed.append(name)
            del os.environ[name]
    return scrubbed


def _resolve_metrics_detail() -> str:
    """同构 run_online_strategy.sh:66-92 的档位解析：env SH_METRICS_DETAIL
    > metrics_config.json detail_level；env/json 取值非法或 json 缺失均
    fail-closed（与内层 runner 同口径，先于一切副作用退出）。"""
    detail = os.environ.get("SH_METRICS_DETAIL")
    if detail:
        if detail not in ("off", "summary", "full"):
            raise SystemExit(
                f"[joint_runner] invalid SH_METRICS_DETAIL={detail!r} "
                "(expected off|summary|full)")
        return detail
    try:
        with open(_METRICS_CONFIG_PATH, encoding="utf-8") as source:
            config = json.load(source)
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"[joint_runner] cannot read {_METRICS_CONFIG_PATH}: {error}")
    detail = config.get("detail_level") if isinstance(config, dict) else None
    if detail not in ("off", "summary", "full"):
        raise SystemExit(
            f"[joint_runner] invalid detail_level {detail!r} in "
            f"{_METRICS_CONFIG_PATH} (expected off|summary|full)")
    return detail


def _acquire_lock():
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    inherited_raw = os.environ.get(_LOCK_FD_ENV)
    if inherited_raw is not None:
        if not inherited_raw.isascii() or not inherited_raw.isdecimal():
            raise SystemExit(
                f"[joint_runner] invalid {_LOCK_FD_ENV}="
                f"{inherited_raw!r} (expected inherited numeric FD)")
        lock_fd = int(inherited_raw)
        try:
            fd_stat = os.fstat(lock_fd)
            path_stat = _LOCK_PATH.stat()
        except OSError as error:
            raise SystemExit(
                f"[joint_runner] inherited simulation lock FD {lock_fd} "
                f"is unavailable: {error}")
        if (fd_stat.st_dev, fd_stat.st_ino) != (
                path_stat.st_dev, path_stat.st_ino):
            raise SystemExit(
                f"[joint_runner] inherited simulation lock FD {lock_fd} "
                f"does not refer to {_LOCK_PATH}")
        try:
            # Re-flocking the inherited FD is idempotent for the same open-file
            # description. If another process owns a different description,
            # fail closed; never open a second FD and self-deadlock.
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "[joint_runner] inherited simulation lock FD is not held; "
                f"another simulation owns {_LOCK_PATH}")
        os.environ[_LOCK_FD_ENV] = str(lock_fd)
        return lock_fd, False

    lock_fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise SystemExit(
            "[joint_runner] another simulation holds "
            f"{_LOCK_PATH}; refusing to start (single-simulation lock)")
    os.environ[_LOCK_FD_ENV] = str(lock_fd)
    return lock_fd, True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="astra-sim-joint run orchestrator (D14)")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("request_csv", type=Path)
    parser.add_argument("--combo", choices=(
        "none", "T", "J", "E", "TJ", "TE", "JE", "TJE"))
    parser.add_argument("--category", choices=("typed", "lru"))
    parser.add_argument("--scheduler", choices=(
        "joint", "load-first", "affinity-first", "face_static"),
        help="face_static = C17 静态距离对照臂（policy variant，非八"
             "组合；joint_config 强制 quota-off——F7 唯一耦合例外）")
    parser.add_argument("--layer", choices=(
        "adaptive", "legacy_half", "minimal_layer_groups"))
    parser.add_argument("--remote", choices=("on", "off"))
    parser.add_argument("--remote-credit-iters", dest="remote_credit_iters",
                        help="remote-read credit 块大小 K：auto 或正整数")
    parser.add_argument("--remote-read-partial", dest="remote_read_partial",
                        choices=("on", "off"),
                        help="PARTIAL 基 remote-read 适用面消融（缺省 on）")
    parser.add_argument("--quota", choices=("off", "static", "aimd"),
                        help="通用流量治理/配额开关（C11；缺省 off——F7；"
                             "aimd 自动注入 --link-telemetry）")
    parser.add_argument("--extra-env", action="append", default=[])
    args = parser.parse_args(argv)

    # env 清洗先于一切（陈旧策略变量零残留；白名单/开关族不在此列）。
    scrubbed = _scrub_environment()
    for entry in args.extra_env:
        name, _, value = entry.partition("=")
        if not name:
            raise SystemExit(f"[joint_runner] bad --extra-env entry: {entry!r}")
        os.environ[name] = value

    switch_conflicts = []
    if args.combo is not None:
        os.environ["JOINT_ABLATION_COMBO"] = args.combo
        for flag, var in (
            (args.category, "JOINT_CATEGORY_MODE"),
            (args.scheduler, "JOINT_SCHEDULER_MODE"),
            (args.layer, "JOINT_LAYER_POLICY"),
        ):
            if flag is not None:
                switch_conflicts.append(var)
    else:
        if args.category is not None:
            os.environ["JOINT_CATEGORY_MODE"] = args.category
        if args.scheduler is not None:
            os.environ["JOINT_SCHEDULER_MODE"] = args.scheduler
        if args.layer is not None:
            os.environ["JOINT_LAYER_POLICY"] = args.layer
    if args.remote is not None:
        os.environ["JOINT_REMOTE_ACTIONS"] = args.remote
    # remote-read credit 块大小 K（执行口径 = credit 交错流唯一机制，
    # 无开关；取值校验由 joint_config.parse fail-closed 兜底）。
    if args.remote_credit_iters is not None:
        os.environ["JOINT_REMOTE_CREDIT_ITERS"] = args.remote_credit_iters
    # PARTIAL 基 remote-read 适用面消融（需求①，2026-09-17；取值校验由
    # joint_config.parse fail-closed 兜底）。
    if args.remote_read_partial is not None:
        os.environ["JOINT_REMOTE_READ_PARTIAL"] = args.remote_read_partial
    # 通用流量治理/配额开关（C11；缺省 off——F7；取值校验由
    # joint_config.parse fail-closed 兜底）。
    if args.quota is not None:
        os.environ["JOINT_QUOTA_MODE"] = args.quota
    if switch_conflicts:
        raise SystemExit(
            "[joint_runner] combo preset conflicts with explicit switches: "
            f"{switch_conflicts} (mutually exclusive, same as "
            "joint_config.parse)")

    # C11（F7 耦合规则）：aimd ⇒ 自动注入 --link-telemetry。判定取最终
    # env（--quota 与 --extra-env JOINT_QUOTA_MODE=aimd 同效）。注入 =
    # 置位 SH_LINK_TELEMETRY=1（内层 runner 消费）；用户已显式置位 = 幂
    # 等。face_static 强制 quota-off 发生在 joint_config 解析侧——env
    # 仍为 aimd 时注入照做（遥测开启而配额被覆盖 = 合法组合，无行为
    # 冲突：遥测是独立观测面）。
    quota_mode = os.environ.get("JOINT_QUOTA_MODE", "off")
    link_telemetry_injected = False
    if quota_mode == "aimd" and os.environ.get(
            "SH_LINK_TELEMETRY", "0") != "1":
        os.environ["SH_LINK_TELEMETRY"] = "1"
        link_telemetry_injected = True

    # O11（L6 shell 断言的运行期双保险，2026-09-23）：C++ 观测门还压在
    # MetricCollector::enabled 下（main_online.cc:1165）——aimd ∧
    # SH_METRICS_DETAIL=off ⇒ 即便 observer=1、--link-telemetry 在位，
    # link_telemetry[] 仍恒空 ⇒ AIMD 静默空转无信号。与内层
    # run_online_strategy.sh:146-149 同构 fail-closed：无逃生口径（shell
    # 断言无逃生，runner 亦无——显式 no-signal 对照臂亦不被承认）；直启
    # 本 runner 不再依赖内层脚本兜底。档位解析同构 env > json。
    metrics_detail = _resolve_metrics_detail()
    if quota_mode == "aimd" and metrics_detail == "off":
        raise SystemExit(
            "[joint_runner] FAIL: JOINT_QUOTA_MODE=aimd requires metrics "
            "enabled (C++ observer gate is behind MetricCollector::enabled, "
            "main_online.cc:1165); SH_METRICS_DETAIL=off empties "
            "link_telemetry — use summary/full")

    binary = _REPO_ROOT / _BINARY_RELPATH
    if not binary.exists():
        raise SystemExit(
            f"[joint_runner] binary missing: {binary} (build first)")
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"[joint_runner] binary not executable: {binary}")
    binary_sha = _sha256(binary)

    args.run_dir = args.run_dir.resolve()
    request_csv = args.request_csv.resolve()
    lock_fd, owns_lock = _acquire_lock()
    try:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        invocation = {
            "argv": [str(a) for a in sys.argv],
            "joint_switches": {
                var: os.environ.get(var) for var in _JOINT_SWITCH_VARS
                if os.environ.get(var) is not None
            },
            "binary_sha256": binary_sha,
            "binary_relpath": _BINARY_RELPATH,
            "inner_runner_sha256": _sha256(_INNER_RUNNER),
            "scrubbed_env": scrubbed,
            "allowlisted_env": {
                name: os.environ.get(name) for name in _ENV_ALLOWLIST
                if os.environ.get(name) is not None
            },
            # C11（F7）：aimd ⇒ --link-telemetry 自动注入事实（manifest 侧
            # 的注入披露在 online_service 的 joint_mechanism_manifest.json
            # 侧车；此处为 runner 视角的启动事实）。
            "quota_mode_env": quota_mode,
            "link_telemetry_injected": link_telemetry_injected,
            # O11 守卫判定所用的生效指标档（env > json，同内层解析）。
            "metrics_detail": metrics_detail,
            "started_utc": started_utc,
            "single_simulation_lock_reused": not owns_lock,
        }
        try:
            result = subprocess.run(
                ["bash", str(_INNER_RUNNER),
                 str(args.run_dir), str(request_csv)],
                cwd=str(_REPO_ROOT),
                check=False,
                pass_fds=(lock_fd,),
            )
            invocation["exit_code"] = result.returncode
            return result.returncode
        finally:
            invocation["finished_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(args.run_dir / "invocation.json", "w",
                      encoding="utf-8") as sink:
                json.dump(invocation, sink, indent=1, sort_keys=True)
    finally:
        if owns_lock:
            # Closing the owner FD releases it after subprocess.run waited for
            # the inner shell. Do not LOCK_UN: inherited descendants may still
            # hold references to this same open-file description.
            os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
