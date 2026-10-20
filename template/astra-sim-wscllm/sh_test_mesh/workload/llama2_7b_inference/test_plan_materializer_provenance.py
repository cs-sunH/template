#!/usr/bin/env python3
"""test_plan_materializer_provenance.py -- D3 (2026-09-28) plan_materializer
provenance/publication contract, pinned against the REAL main() with a
monkeypatched config loader and a temp generated root.

Contracts under test:
  - metrics_manifest.json carries npus_count (the C++ MetricCollector
    consistency gate "manifest npus_count != simulated systems count" only
    runs when the key is present; its absence kept the gate structurally
    dormant);
  - BOTH manifests carry the request-queue byte digest
    (request_queue_csv_sha256 / request_queue_csv_bytes) so a stale plan
    paired with a same-path re-derived CSV is detectably inconsistent
    instead of silently so;
  - stdout sidecar_rows_joined counts REAL join hits (queue rows that got a
    sidecar row), not the sidecar's total row count (the pre-fix number
    overstated passthrough coverage whenever the sidecar covers rows the
    queue does not carry);
  - publication is transactional: the final plan dir appears only once both
    manifests are complete, staging dirs are dot-prefixed (invisible to the
    GEN_MATCH "llama2_7b_wsc_llm_inference_54npus_*" glob), stale other-cfg8
    plan dirs and crash leftovers are swept, and a re-run is byte-idempotent.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SH_TEST_DIR = MODULE_DIR.parents[1]
for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plan_materializer  # noqa: E402
from generate_trace import RequestSpec  # noqa: E402

QUEUE_HEADER = (
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns,description"
)


class _Group:
    def __init__(self, ranks):
        self.ranks = list(ranks)


class _Hardware:
    label = "face_case5_unit"


class _Config:
    """Minimum config surface consumed by plan_materializer.main()."""

    def __init__(self, config_csv, queue_csv, specs):
        self.config_csv = config_csv
        self.request_queue_csv = queue_csv
        self.request_queue = tuple(specs)
        self.request_queue_indices = tuple(range(len(specs)))
        self.queue_position_by_index = {
            index: position for position, index in
            enumerate(self.request_queue_indices)}
        self.inference_groups = [_Group((0, 1, 2))]
        self.kv_cache_policy = "session_lru_recompute"
        self.npus_count = 54
        self.model_name = "llama2_7b_unit"
        self.hardware = _Hardware()
        self.hardware_capacity_profile = "unit-profile"


def _spec(request_id, session_id, turn_index, arrival=None, interval=None):
    return RequestSpec(
        session_id=session_id,
        turn_index=turn_index,
        request_id=request_id,
        prefill_length=100,
        decode_length=10,
        session_arrival_time_ns=arrival,
        inter_request_interval_ns=interval,
    )


class PlanMaterializerProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.gen_root = Path(self.tmp.name) / "generated"
        self.gen_root.mkdir()
        self._saved_gen_root = plan_materializer.GENERATED_ROOT
        plan_materializer.GENERATED_ROOT = self.gen_root
        self.addCleanup(setattr, plan_materializer,
                        "GENERATED_ROOT", self._saved_gen_root)

        config_csv = Path(self.tmp.name) / "trace_config.csv"
        config_csv.write_bytes(b"kind,key,value\nconfig,unit,1\n")

        # Queue: two sessions, three requests. The sidecar deliberately
        # covers r0 plus an rX that is NOT in the queue (the windowed-out
        # shape) -> joined=1, total=2; the pre-fix stdout printed 2.
        queue_csv = Path(self.tmp.name) / "unit_request_queue_recompute.csv"
        queue_csv.write_text(
            QUEUE_HEADER + "\n"
            "s0,0,s0_r0,100,10,1000,,d\n"
            "s0,1,s0_r1,100,10,,2000,d\n"
            "s1,0,s1_r0,100,10,5000,,d\n",
            encoding="utf-8")
        sidecar_csv = (Path(self.tmp.name) /
                       "unit_canonical_sidecar.csv")
        sidecar_csv.write_text(
            "request_id,human_time_ns,tool_time_ns\n"
            "s0_r0,50000000,\n"
            "sX_not_in_queue,1,\n",
            encoding="utf-8")

        config = _Config(config_csv, queue_csv, [
            _spec("s0_r0", "s0", 0, arrival=1000),
            _spec("s0_r1", "s0", 1, interval=2000),
            _spec("s1_r0", "s1", 0, arrival=5000),
        ])
        self._saved_loader = plan_materializer.load_wsc_llm_trace_config
        plan_materializer.load_wsc_llm_trace_config = lambda: config
        self.addCleanup(setattr, plan_materializer,
                        "load_wsc_llm_trace_config", self._saved_loader)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.rc = plan_materializer.main()
        self.stdout_record = json.loads(stdout.getvalue())
        self.output_dir = self.gen_root / (
            f"llama2_7b_wsc_llm_inference_54npus_plan_"
            f"{plan_materializer._config_digest8(config_csv)}")

    def _read(self, name):
        return json.loads((self.output_dir / name).read_text(encoding="utf-8"))

    def test_runs_clean_and_publishes_exactly_one_dir(self):
        self.assertEqual(self.rc, 0)
        self.assertTrue(self.output_dir.is_dir())
        plan_dirs = [p for p in self.gen_root.glob(
            "llama2_7b_wsc_llm_inference_54npus_plan_*")]
        self.assertEqual(plan_dirs, [self.output_dir])
        # No dot-prefixed staging/previous leftovers after a clean run.
        self.assertEqual(list(self.gen_root.glob(".*")), [])

    def test_metrics_manifest_carries_npus_count(self):
        record = self._read("metrics_manifest.json")
        self.assertEqual(record["npus_count"], 54)
        self.assertEqual(record["schema_version"], 1)

    def test_both_manifests_carry_queue_byte_digest(self):
        import hashlib
        manifest_record = self._read("manifest.json")
        metrics_record = self._read("metrics_manifest.json")
        queue_path = Path(manifest_record["request_queue_csv"])
        queue_bytes = queue_path.read_bytes()
        digest = hashlib.sha256(queue_bytes).hexdigest()
        for record in (manifest_record, metrics_record):
            self.assertEqual(record["request_queue_csv_sha256"], digest)
            self.assertEqual(record["request_queue_csv_bytes"],
                             len(queue_bytes))

    def test_sidecar_rows_joined_counts_real_hits_not_sidecar_size(self):
        # Sidecar carries 2 rows (s0_r0 + an sX that is not in the queue);
        # exactly ONE queue row actually joined. The pre-fix stdout printed
        # len(sidecar_rows) == 2 here -- overstating the passthrough.
        self.assertEqual(self.stdout_record["sidecar_rows_joined"], 1)
        self.assertEqual(self.stdout_record["sidecar_rows_total"], 2)

    def test_stale_other_cfg8_plan_dir_is_swept(self):
        stale = self.gen_root / "llama2_7b_wsc_llm_inference_54npus_plan_dead"
        stale.mkdir()
        (stale / "manifest.json").write_text("{}\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(plan_materializer.main(), 0)
        self.assertFalse(stale.exists())
        self.assertTrue(self.output_dir.is_dir())

    def test_crash_leftover_staging_dir_is_swept(self):
        leftover = self.gen_root / (
            ".llama2_7b_wsc_llm_inference_54npus_plan_dead.staging")
        leftover.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(plan_materializer.main(), 0)
        self.assertFalse(leftover.exists())
        self.assertEqual(list(self.gen_root.glob(".*")), [])

    def test_rerun_is_byte_idempotent(self):
        before = {
            name: (self.output_dir / name).read_bytes()
            for name in ("manifest.json", "metrics_manifest.json")
        }
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(plan_materializer.main(), 0)
        for name, payload in before.items():
            self.assertEqual(
                (self.output_dir / name).read_bytes(), payload, name)


class F2PyProvenanceReconciliationTests(unittest.TestCase):
    """F2_PY（2026-09-29）RV4-F3：derive 三件发布的"新队列 + 旧 sidecar +
    旧 provenance"混合代际窗口（A1 原子化的三次顺序 rename 之间中断）对账
    ——plan_materializer 在 join sidecar 前读队列旁
    ``<queue>.provenance.json``，其 ``csv_sha256`` 与当前队列字节摘要不符
    即 fail-closed。旧行为：sidecar 无摘要绑定、物化器不读 provenance，
    旧 sidecar 同名行静默绑新数据、静默省略透传字段，两 manifest 合法
    成形、SLO T_session 退化为 NA，全链零报错。

    中断矩阵归口（除本门外的各臂）：
      * rename1↔rename2 之间中断 → 新队列+旧 sidecar+旧 provenance →
        本门拦截（provenance 记录的是旧队列字节）；
      * rename2↔rename3 之间中断 → 新队列+新 sidecar+旧 provenance →
        本门同样拦截（C++ provenance 门亦拦）；
      * 完成前中断 → A1 的 .tmp 清理保持终路径不动；
      * 发布后 → GEN_MATCH 恰一闸（_publish_plan_dir）。
    """

    def _materialize_env(self, queue_text, sidecar_text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        gen_root = Path(tmp.name) / "generated"
        gen_root.mkdir()
        saved_root = plan_materializer.GENERATED_ROOT
        plan_materializer.GENERATED_ROOT = gen_root
        self.addCleanup(setattr, plan_materializer,
                        "GENERATED_ROOT", saved_root)

        config_csv = Path(tmp.name) / "trace_config.csv"
        config_csv.write_bytes(b"kind,key,value\nconfig,unit,1\n")
        queue_csv = Path(tmp.name) / "unit_request_queue_recompute.csv"
        queue_csv.write_text(queue_text, encoding="utf-8")
        sidecar_csv = Path(tmp.name) / "unit_canonical_sidecar.csv"
        sidecar_csv.write_text(sidecar_text, encoding="utf-8")

        config = _Config(config_csv, queue_csv, [
            _spec("s0_r0", "s0", 0, arrival=1000),
            _spec("s0_r1", "s0", 1, interval=2000),
        ])
        saved_loader = plan_materializer.load_wsc_llm_trace_config
        plan_materializer.load_wsc_llm_trace_config = lambda: config
        self.addCleanup(setattr, plan_materializer,
                        "load_wsc_llm_trace_config", saved_loader)
        return tmp, gen_root, queue_csv

    _QUEUE = (QUEUE_HEADER + "\n"
              "s0,0,s0_r0,100,10,1000,,d\n"
              "s0,1,s0_r1,100,10,,2000,d\n")
    _SIDECAR = "request_id,human_time_ns,tool_time_ns\ns0_r0,50000000,\n"

    def test_fresh_provenance_passes_and_manifests_match(self):
        import hashlib
        _, _, queue_csv = self._materialize_env(self._QUEUE, self._SIDECAR)
        digest = hashlib.sha256(queue_csv.read_bytes()).hexdigest()
        (queue_csv.parent /
         "unit_request_queue_recompute.csv.provenance.json").write_text(
            json.dumps({"schema": 1, "csv_sha256": digest}),
            encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(plan_materializer.main(), 0)

    def test_stale_provenance_sha_mismatch_fails_closed(self):
        # 中断矩阵的实靶：rename1（新队列）之后、rename2/3（sidecar/
        # provenance）之前中断——盘上留新队列 + 旧 sidecar + 旧 provenance
        # （provenance 的 sha 对应旧队列字节）。旧：物化静默成功。
        import hashlib
        _, _, queue_csv = self._materialize_env(self._QUEUE, self._SIDECAR)
        stale_digest = hashlib.sha256(b"stale previous generation").hexdigest()
        (queue_csv.parent /
         "unit_request_queue_recompute.csv.provenance.json").write_text(
            json.dumps({"schema": 1, "csv_sha256": stale_digest}),
            encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                plan_materializer.main()

    def test_malformed_provenance_fails_closed(self):
        # 在场但不可解析/缺 sha 的 provenance 记录同样拒绝（不得静默跳过
        # 对账）。
        _, _, queue_csv = self._materialize_env(self._QUEUE, self._SIDECAR)
        (queue_csv.parent /
         "unit_request_queue_recompute.csv.provenance.json").write_text(
            "{not json", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                plan_materializer.main()


if __name__ == "__main__":
    unittest.main()
