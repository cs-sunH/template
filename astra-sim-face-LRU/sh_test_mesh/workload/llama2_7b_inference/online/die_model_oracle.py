#!/usr/bin/env python3
"""die_model_oracle.py -- die 成本库的 ctypes 预测 oracle(阶段 D,2026-10-03)。

License: MIT (see LICENSE in the repository root).

本模块是 face 放置评分的本地计算成本新来源(设计文档 §8.4:只替换本地
计算成本来源,保留评分组合/候选集合/约束/tie-break 与 KV 规则)。全部
派生与计时公式都在 extern/die_model 的共享库里(契约 §1/§4:Python 内
不得复写第二套公式),本文件只做三件事:

  1. 按契约 §7 裁定 1 的发现序定位 libdie_model_cabi.so(显式参数 →
     env DIE_MODEL_CABI_PATH → 仓内 build/astra_analytical/
     build_congestion_aware/{lib,bin}/ → fail-closed 报错"需要先构建
     die_model_cabi",绝不静默回退 roofline);
  2. 用 resolved_die_model.json(契约 §3.3;显式参数 → env
     DIE_MODEL_CONFIG_PATH;manifest 键 die-model-config 由调用方
     online_service.py 解析后显式传入)建模型句柄;
  3. 把批量估值与硬件摘要按 C ABI 封成 Python 调用:estimate 保留
     完整 CostResult;coarse 新 ABI 可选提供 estimate_cost,只返回
     所选 cost/hash/status。坏 spec 位置返回 None 并记录错误,批量级
     失败(入参畸形)经 last_error 取回抛 RuntimeError。

只依赖标准库(ctypes/json/os/hashlib)。
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

# 本文件位于 sh_test_mesh/workload/llama2_7b_inference/online/;仓根 =
# 向上 4 级(online -> llama2_7b_inference -> workload -> sh_test_mesh ->
# 仓库根)。仅用于 .so 与 resolved 配置的仓内发现(契约 §7 裁定 1)。
_REPO_ROOT = Path(__file__).resolve().parents[4]

CABI_LIBRARY_NAME = "libdie_model_cabi.so"
CABI_LIBRARY_ENV = "DIE_MODEL_CABI_PATH"
COARSE_CABI_LIBRARY_NAME = "libcoarse_die_cabi.so"
COARSE_CABI_LIBRARY_ENV = "DIE_COARSE_CABI_PATH"
DIE_MODEL_CONFIG_ENV = "DIE_MODEL_CONFIG_PATH"
# manifest 新键(契约 §7 裁定 2):resolved_die_model.json 绝对路径。
MANIFEST_COMPUTE_COST_MODEL_KEY = "compute-cost-model"
MANIFEST_DIE_MODEL_CONFIG_KEY = "die-model-config"

# die_estimate_batch 的 CostResult 冻结字段(阶段 A3 报告 §3;snake_case
# 20 键)。仅作返回值形状校验用,不参与任何数值计算。
COST_RESULT_FIELDS = frozenset({
    "profile", "total_seconds", "pe_seconds", "vec_seconds", "noc_seconds",
    "mem_seconds", "launch_seconds", "launch_instances", "flops", "eqops",
    "matrix_actual_bytes", "vector_actual_bytes",
    "matrix_service_units", "vector_service_units",
    "noc_link_bytes_total", "noc_edge_bytes_max", "sram_peak_per_core",
    "fused", "fallback", "total_ns_suggested",
})


def cabi_library_candidates(repo_root: Optional[Path] = None) -> tuple[Path, ...]:
    """契约 §7 裁定 1 的仓内发现路径序(测试/诊断也用它列候选)。"""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    build = root / "build" / "astra_analytical" / "build_congestion_aware"
    return (build / "lib" / CABI_LIBRARY_NAME, build / "bin" / CABI_LIBRARY_NAME)


def discover_cabi_library(lib_path: Optional[str] = None) -> Path:
    """发现序:显式参数 → env DIE_MODEL_CABI_PATH → 仓内 build 路径 →
    fail-closed(绝不静默回退 roofline)。"""
    candidates: list[Path] = []
    if lib_path:
        candidates.append(Path(lib_path))
    env_value = os.environ.get(CABI_LIBRARY_ENV)
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(cabi_library_candidates())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "die_model_cabi shared library not found; build it first "
        "(cmake -S extern/die_model -B <build-dir> && cmake --build <build-dir>), "
        "pass lib_path=..., or set {env}={name}. Tried: {tried}".format(
            env=CABI_LIBRARY_ENV,
            name=CABI_LIBRARY_NAME,
            tried=", ".join(str(path) for path in candidates),
        ))


def discover_coarse_cabi_library(lib_path: Optional[str] = None) -> Path:
    candidates: list[Path] = []
    if lib_path:
        candidates.append(Path(lib_path))
    env_value = os.environ.get(COARSE_CABI_LIBRARY_ENV)
    if env_value:
        candidates.append(Path(env_value))
    build = _REPO_ROOT / "build" / "astra_analytical" / "build_congestion_aware"
    candidates.extend((build / "lib" / COARSE_CABI_LIBRARY_NAME,
                       build / "bin" / COARSE_CABI_LIBRARY_NAME))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "coarse_die_v2 requires the coarse_die_v2 C ABI; build "
        "libcoarse_die_cabi.so or set {}. Tried: {}".format(
            COARSE_CABI_LIBRARY_ENV,
            ", ".join(str(path) for path in candidates)))


def resolve_die_model_config_path(
    *,
    explicit: Optional[str] = None,
    manifest: Optional[dict] = None,
    plan_dir: Optional[os.PathLike] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    """resolved_die_model.json 发现序(契约 §7 裁定 2;由 online_service
    调用后把结果显式传给 DieModelOracle):

      显式参数 → env DIE_MODEL_CONFIG_PATH → 仓内发现(plan 目录旁的
      generated/ 三件套同目录,与 sh_test_mesh/generated/)→ manifest
      的 die-model-config 绝对路径 → None(backend=die 时由调用方报错)。
    """
    if explicit:
        return Path(explicit)
    env_value = os.environ.get(DIE_MODEL_CONFIG_ENV)
    if env_value:
        return Path(env_value)
    candidates: list[Path] = []
    if plan_dir is not None:
        plan = Path(plan_dir)
        candidates.append(plan / "generated" / "resolved_die_model.json")
        candidates.append(plan / "resolved_die_model.json")
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    candidates.append(root / "sh_test_mesh" / "generated"
                      / "resolved_die_model.json")
    # 阶段 C resolver 的物化位置(generated/runtime_config/<slug>/,与
    # system.json/network.yml/comm_group.json 三件套同目录,契约 §3.3):
    # sorted 保证多 slug 并存时的确定性。
    candidates.extend(sorted(
        (root / "sh_test_mesh" / "generated" / "runtime_config").glob(
            "*/resolved_die_model.json")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if manifest is not None:
        value = manifest.get(MANIFEST_DIE_MODEL_CONFIG_KEY)
        if value:
            path = Path(str(value))
            if path.is_file():
                return path
            # manifest 声称的路径不存在:如实返回该路径,让构造期
            # fail-closed 报出准确文件名,而不是继续静默回退。
            return path
    return None


class DieModelOracle:
    """die_estimate_batch / die_hardware_summary 的 ctypes 客户端。

    单实例持有恰好一个模型句柄(库内含 4096 项 LRU 计划缓存);线程契约
    与 DieCApi.h 相同(每句柄单线程使用)。close() 幂等;析构兜底销毁。
    """

    def __init__(self, lib_path: Optional[str] = None,
                 config_json_path: Optional[str] = None):
        if config_json_path is None:
            env_value = os.environ.get(DIE_MODEL_CONFIG_ENV)
            if env_value:
                config_json_path = env_value
        if not config_json_path:
            raise RuntimeError(
                "resolved_die_model.json path is required: pass "
                "config_json_path=..., set {env}, or resolve it from the "
                "manifest {key} key via online_service (fail-closed; no "
                "roofline fallback)".format(
                    env=DIE_MODEL_CONFIG_ENV,
                    key=MANIFEST_DIE_MODEL_CONFIG_KEY))
        config_path = Path(config_json_path)
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                "cannot read die model config {!r}: {}".format(
                    str(config_path), exc)) from exc

        try:
            config_value = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "invalid die model config {!r}: {}".format(
                    str(config_path), exc)) from exc
        self.coarse = (isinstance(config_value, dict)
                       and config_value.get("schema") == "coarse_die_v2")
        resolved_lib = (discover_coarse_cabi_library(lib_path) if self.coarse
                        else discover_cabi_library(lib_path))

        self.lib_path = resolved_lib
        self.config_json_path = config_path
        self._lib = ctypes.CDLL(str(resolved_lib))
        self._declare_abi()
        self._closed = False
        # 逐条 estimate 错误的最近一次快照(诊断用;错误位以 None 返回)。
        self.last_errors: list[Optional[str]] = []

        if self.coarse:
            self._model = ctypes.c_uint64()
            status = self._lib.die_coarse_create_model_v2(
                config_text.encode("utf-8"), ctypes.byref(self._model))
            if status != 0 or not self._model.value:
                raise RuntimeError(
                    "die_coarse_create_model_v2 failed for config {!r}: {}"
                    .format(str(config_path), self._last_error()))
            self._model = self._model.value
        else:
            self._model = self._lib.die_create_model(
                config_text.encode("utf-8"))
            if not self._model:
                raise RuntimeError(
                    "die_create_model failed for config {!r}: {}".format(
                        str(config_path), self._last_error()))
        # 单元自检(契约:构造后 hardware_summary 必须非空)——同时把
        # 摘要缓存在实例上,后续 hardware_summary()/config_hash 零开销。
        self._hardware_summary: Optional[dict] = None
        summary = self.hardware_summary()
        if not summary or not summary.get("derived"):
            raise RuntimeError(
                "die_hardware_summary returned an empty summary for config "
                "{!r}".format(str(config_path)))

    # ------------------------------------------------------------- ABI --

    def _declare_abi(self) -> None:
        """声明 ctypes 参数表/restype,并记录可选的 cost-only 入口。

        库分配缓冲(die_hardware_summary/die_estimate_batch)的 restype
        用 c_void_p 而非 c_char_p:c_char_p 会在边界上复制成 Python
        bytes 并丢失原指针,后续 die_free_result 会释放在错误地址上
        (堆损坏)。取文用 ctypes.string_at,释放仍传原 c_void_p。
        """
        lib = self._lib
        self.supports_estimate_cost = False
        if self.coarse:
            lib.die_coarse_create_model_v2.argtypes = [
                ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
            lib.die_coarse_create_model_v2.restype = ctypes.c_int
            lib.die_coarse_hardware_summary_v2.argtypes = [ctypes.c_uint64]
            lib.die_coarse_hardware_summary_v2.restype = ctypes.c_void_p
            lib.die_coarse_estimate_batch_v2.argtypes = [
                ctypes.c_uint64, ctypes.c_char_p]
            lib.die_coarse_estimate_batch_v2.restype = ctypes.c_void_p
            if hasattr(lib, "die_coarse_estimate_cost_batch_v2"):
                lib.die_coarse_estimate_cost_batch_v2.argtypes = [
                    ctypes.c_uint64, ctypes.c_char_p]
                lib.die_coarse_estimate_cost_batch_v2.restype = ctypes.c_void_p
                self.supports_estimate_cost = True
            lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
            lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
            lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
            lib.die_coarse_free_result_v2.restype = None
            lib.die_coarse_last_error_v2.restype = ctypes.c_char_p
            lib.die_coarse_last_error_v2.argtypes = []
            return
        lib.die_create_model.restype = ctypes.c_void_p
        lib.die_create_model.argtypes = [ctypes.c_char_p]
        lib.die_hardware_summary.restype = ctypes.c_void_p
        lib.die_hardware_summary.argtypes = [ctypes.c_void_p]
        lib.die_estimate_batch.restype = ctypes.c_void_p
        lib.die_estimate_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.die_create_execution.restype = ctypes.c_void_p
        lib.die_create_execution.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.die_next_action.restype = ctypes.c_int
        lib.die_next_action.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.die_advance_time.restype = ctypes.c_int
        lib.die_advance_time.argtypes = [
            ctypes.c_void_p, ctypes.c_ulonglong]
        lib.die_complete_memory.restype = ctypes.c_int
        lib.die_complete_memory.argtypes = [
            ctypes.c_void_p, ctypes.c_ulonglong]
        lib.die_destroy_execution.restype = None
        lib.die_destroy_execution.argtypes = [ctypes.c_void_p]
        lib.die_destroy_model.restype = None
        lib.die_destroy_model.argtypes = [ctypes.c_void_p]
        lib.die_free_result.restype = None
        lib.die_free_result.argtypes = [ctypes.c_void_p]
        lib.die_last_error.restype = ctypes.c_char_p
        lib.die_last_error.argtypes = []

    def _last_error(self) -> str:
        raw = (self._lib.die_coarse_last_error_v2() if self.coarse
               else self._lib.die_last_error())
        return raw.decode("utf-8", "replace") if raw else ""

    def _require_model(self) -> int:
        if self._closed or not self._model:
            raise RuntimeError("DieModelOracle model handle is closed")
        return self._model

    # ---------------------------------------------------------- 查询口 --

    def hardware_summary(self) -> dict:
        """契约 §3.3 摘要(schema_version/profile/die/derived);缓存。"""
        if self._hardware_summary is not None:
            return self._hardware_summary
        if self.coarse:
            handle = self._lib.die_coarse_hardware_summary_v2(
                self._require_model())
        else:
            handle = self._lib.die_hardware_summary(self._require_model())
        if not handle:
            raise RuntimeError(
                "die_hardware_summary failed: {}".format(self._last_error()))
        try:
            summary = json.loads(ctypes.string_at(handle).decode("utf-8"))
        finally:
            if self.coarse:
                self._lib.die_coarse_free_result_v2(handle)
            else:
                self._lib.die_free_result(handle)
        if self.coarse and "derived" not in summary:
            summary["derived"] = summary.get("derived_hardware")
        self._hardware_summary = summary
        return summary

    @property
    def config_hash(self) -> str:
        """oracle 配置摘要(契约 §2 config_hash 语义):库返回的硬件摘要
        的规范 JSON sha256 前 16 位十六进制——同一配置文件/派生必同值。"""
        canonical = json.dumps(self.hardware_summary(), sort_keys=True,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _estimate(self, spec_dicts: Sequence[dict], *,
                  cost_only: bool) -> list[Optional[dict]]:
        model = self._require_model()
        payload = json.dumps({"specs": list(spec_dicts)})
        if cost_only:
            handle = self._lib.die_coarse_estimate_cost_batch_v2(
                model, payload.encode("utf-8"))
        elif self.coarse:
            handle = self._lib.die_coarse_estimate_batch_v2(
                model, payload.encode("utf-8"))
        else:
            handle = self._lib.die_estimate_batch(
                model, payload.encode("utf-8"))
        if not handle:
            endpoint = ("die_coarse_estimate_cost_batch_v2" if cost_only
                        else "die_estimate_batch")
            raise RuntimeError(
                "{} failed: {}".format(endpoint, self._last_error()))
        try:
            out = json.loads(ctypes.string_at(handle).decode("utf-8"))
        finally:
            if self.coarse:
                self._lib.die_coarse_free_result_v2(handle)
            else:
                self._lib.die_free_result(handle)
        results = out.get("results")
        errors = out.get("errors")
        if not isinstance(results, list) or not isinstance(errors, list) \
                or len(results) != len(spec_dicts) \
                or len(errors) != len(spec_dicts):
            raise RuntimeError(
                "{} returned a malformed envelope for {} specs".format(
                    "die_coarse_estimate_cost_batch_v2" if cost_only else
                    "die_estimate_batch", len(spec_dicts)))
        self.last_errors = [
            (error if isinstance(error, str) else
             error.get("reason") if isinstance(error, dict) else None)
            for error in errors]
        typed: list[Optional[dict]] = []
        for result in results:
            if result is None:
                typed.append(None)
            else:
                if cost_only:
                    if not isinstance(result, dict):
                        raise RuntimeError(
                            "coarse_die_v2 cost estimate returned a "
                            "non-object result")
                    latency_ns = result.get("estimated_latency_ns")
                    plan_hash = result.get("plan_hash")
                    status = result.get("status")
                    if (isinstance(latency_ns, bool) or
                            not isinstance(latency_ns, int) or latency_ns < 0 or
                            not isinstance(plan_hash, str) or
                            not isinstance(status, str)):
                        raise RuntimeError(
                            "coarse_die_v2 cost estimate is missing or has an "
                            "invalid estimated_latency_ns/plan_hash/status")
                    # The cost endpoint deliberately returns no legacy timing
                    # breakdown; FACE consumes only the selected cost and
                    # seconds remain a direct unit conversion of that cost.
                    typed.append({
                        "estimated_latency_ns": latency_ns,
                        "plan_hash": plan_hash,
                        "status": status,
                        "total_seconds": float(latency_ns) / 1e9,
                    })
                    continue
                elif self.coarse:
                    latency_ns = result.get("estimated_latency_ns")
                    if latency_ns is None:
                        raise RuntimeError(
                            "coarse_die_v2 estimate is missing "
                            "estimated_latency_ns")
                    # FACE's existing scorer consumes seconds. This is a
                    # unit conversion of the v2 oracle result, not a local
                    # performance model.
                    result["total_seconds"] = float(latency_ns) / 1e9
                elif not COST_RESULT_FIELDS.issubset(result):
                    raise RuntimeError(
                        "die_estimate_batch CostResult field set mismatch: "
                        "missing {}".format(
                            sorted(COST_RESULT_FIELDS - set(result))))
                typed.append(result)
        return typed

    def estimate(self, spec_dicts: Sequence[dict]) -> list[Optional[dict]]:
        """批量孤立估值,返回完整 CostResult;坏 spec 位置为 None 并
        把错误文本记入 last_errors。批量级失败抛 RuntimeError。"""
        return self._estimate(spec_dicts, cost_only=False)

    def estimate_cost(self, spec_dicts: Sequence[dict]) -> list[Optional[dict]]:
        """批量读取选中计划的 cost/hash/status。

        新 coarse C ABI 可用时走轻量 endpoint。旧 coarse ABI 与 legacy
        后端显式回退到完整 estimate 响应,保持现有兼容行为。检查
        supports_estimate_cost 可确认本次库是否提供轻量 endpoint。"""
        if not self.coarse or not self.supports_estimate_cost:
            return self.estimate(spec_dicts)
        return self._estimate(spec_dicts, cost_only=True)

    def estimate_one(self, spec_dict: dict) -> dict:
        """单 spec 便捷封装;错误位直接 fail-closed 抛错(调用方
        face_scheduler 的 die 路径用它查单条规格)。"""
        results = self.estimate([spec_dict])
        if results[0] is None:
            raise RuntimeError(
                "die_estimate_batch rejected spec {}/{}: {}".format(
                    spec_dict.get("op_kind", "?"),
                    spec_dict.get("phase", "?"),
                    self.last_errors[0]))
        return results[0]

    # ---------------------------------------------------------- 生命周期 --

    def close(self) -> None:
        if not self._closed and self._model:
            if self.coarse:
                self._lib.die_coarse_destroy_model_v2(self._model)
            else:
                self._lib.die_destroy_model(self._model)
        self._model = None
        self._closed = True
        # 连缓存摘要一并失效:close 后一切查询 fail-closed(测试钉子)。
        self._hardware_summary = None

    def __enter__(self) -> "DieModelOracle":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # noqa: D105 -- 析构兜底(进程退出路径)
        try:
            self.close()
        except Exception:  # noqa: BLE001 -- 析构期禁止再抛
            pass
