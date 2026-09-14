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
        [--category typed|lru] [--scheduler joint|load-first|affinity-first] \
        [--layer adaptive|legacy_half|minimal_layer_groups] \
        [--remote on|off] [--extra-env NAME=VALUE ...]

combo 与显式开关互斥（与 joint_config.parse 的 fail-closed 口径一致）。
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
_BINARY_RELPATH = (
    "build/astra_analytical/build_congestion_aware/bin/"
    "AstraSim_Analytical_Congestion_Aware_Online")
_INNER_RUNNER = _REPO_ROOT / "sh_test_mesh" / "run_scripts" / "run_online_strategy.sh"

# D7 白名单运维变量 + JOINT_* 开关族（裁定后的策略入口）。
_PREFIXES_TO_SCRUB = ("SH30_", "SH3_CAUSAL_", "SH_JOINT_")
_ENV_ALLOWLIST = (
    "SH_TRAIN_MAX_ITER", "SH_ADMIT_GATE_VERIFY", "SH_SNAPSHOT_VERIFY",
    "SH_ONLINE_VALIDATE", "SH_FIRST_TOKEN_SPLIT", "BRIDGE_TIMEOUT_MS",
)
_JOINT_SWITCH_VARS = (
    "JOINT_ABLATION_COMBO", "JOINT_CATEGORY_MODE", "JOINT_SCHEDULER_MODE",
    "JOINT_LAYER_POLICY", "JOINT_REMOTE_ACTIONS",
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


def _acquire_lock():
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise SystemExit(
            "[joint_runner] another simulation holds "
            f"{_LOCK_PATH}; refusing to start (single-simulation lock)")
    return lock_fd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="astra-sim-joint run orchestrator (D14)")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("request_csv", type=Path)
    parser.add_argument("--combo", choices=(
        "none", "T", "J", "E", "TJ", "TE", "JE", "TJE"))
    parser.add_argument("--category", choices=("typed", "lru"))
    parser.add_argument("--scheduler", choices=(
        "joint", "load-first", "affinity-first"))
    parser.add_argument("--layer", choices=(
        "adaptive", "legacy_half", "minimal_layer_groups"))
    parser.add_argument("--remote", choices=("on", "off"))
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
    if switch_conflicts:
        raise SystemExit(
            "[joint_runner] combo preset conflicts with explicit switches: "
            f"{switch_conflicts} (mutually exclusive, same as "
            "joint_config.parse)")

    binary = _REPO_ROOT / _BINARY_RELPATH
    if not binary.exists():
        raise SystemExit(
            f"[joint_runner] binary missing: {binary} (build first)")
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"[joint_runner] binary not executable: {binary}")
    binary_sha = _sha256(binary)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_lock()
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
        "started_utc": started_utc,
    }
    try:
        result = subprocess.run(
            ["bash", str(_INNER_RUNNER),
             str(args.run_dir), str(args.request_csv.resolve())],
            cwd=str(_REPO_ROOT),
            check=False,
        )
        invocation["exit_code"] = result.returncode
        return result.returncode
    finally:
        invocation["finished_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(args.run_dir / "invocation.json", "w",
                  encoding="utf-8") as sink:
            json.dump(invocation, sink, indent=1, sort_keys=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
