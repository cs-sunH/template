#!/usr/bin/env python3
"""decision_bridge.py -- Python 端 ShmDecisionBridge 协议 v1 (方案 §4 步骤 1-7).

joint v11 §8 保底：帧载体从散装 JSON 文件（tmp+rename 原子写 + 文件存在性
轮询 + unlink）替换为 POSIX-shm 单槽帧通道（bridge_shm.py；段指针见
<bridge_dir>/bridge_shm.json，C++ ensure 先铸段后启 Python）。帧负载字节与
旧 request_/response_/commit_ack_<seq>.json 文件内容逐字节相同；审计
journal（request_journal.jsonl + work_ledger gz 侧车）照旧从帧字节落盘。

桥接目录 <bridge_dir>/ 由 C++ 先建好(含 req_notify.fifo / resp_notify.fifo
与 shm 段, "先建载体再启动双方")。serve_forever 阻塞在 req_notify.fifo 的
read 上(不轮询):每收到一个通知字节,读 request/commit_ack 帧槽,按序处理
——commit_ack 与 request 分开记账(同一 seq 的 request 与 ack 帧可并存于
各自槽位;一次唤醒先处理 ack 再处理 request,与旧文件扫描唤醒同序)。

request 处理: 读 request 帧(门铃字节之后整帧已就绪,不会读到半截) ->
handler(request) -> 发布 response 帧 -> 往 resp_notify.fifo 写 1 字节
(C++ 长连接读端 poll)。handler 抛异常或协议错误 => 发布带 error 字段的
response、通知 C++、以非 0 退出(fail-closed, stderr 留痕);C++ 读到
error 即 abort。C++ 端崩溃/结束 => req_notify.fifo 写端关闭 => read 返回
EOF => serve_forever 返回 0;resp_notify 写端 BrokenPipe(C++ 常开读端关
闭,运行中途) => BridgePipeError fail-closed 退出(缺陷 B 修复 2026-08-16:
resp 通道双侧长连接,消除按交换开/关握手的内核竞态族——POLLHUP 假 EOF
误检与 EPIPE 楔死)。

本模块是决策通道,不是运行期 request 注入通道:Producer -> C++ 的
submit/close/EOF/error 走步骤 1-2 的 command queue,不经过 req_notify.fifo。

协议规则(冻结,合同②/④): request/ack 各自严格从 0 起 +1 单调,旧 seq
重复幂等忽略(槽内 seq < next_seq 即已消费重复,静默跳过——旧 contains()
过滤的同义物),未来 seq 在缺少前序时 fail-closed;背压(一轮至多一个在途
request,C++ 不回 response 不连续发);超时与崩溃检测在 C++ 侧(C++ poll
超时/EOF/EPIPE 即 abort);帧超槽容量/损坏/crc 不符 fail-closed
(bridge_shm.BridgeShmError 同现有 fail-closed 语义)。
"""

import gzip
import json
import os
import sys
import time
import traceback

from bridge_request_journal import (
    REQUEST_JOURNAL_NAME,
    WORK_LEDGER_REPLAY_SIDECAR_NAME,
)
from bridge_shm import (
    BridgeShmError,
    ShmAckRing,
    ShmFrameChannel,
    load_bridge_shm_info,
)

SCHEMA_VERSION = 3  # GraphBatch / bridge wire schema v3 (2026-10-14 远读退役：pipeline_descriptors 键删除，与 C++ kGraphBatchSchemaVersion 同步)

# v11 §8: 帧负载即旧文件字节;journal 每 256 行 flush 一次(旧路径在 flush
# 时批量删除散装 request 文件——散装文件已随载体替换消失,flush 语义保留
# 为 journal 流的批量落盘节奏)。
_PROCESSING_STATS_PARTIAL_NAME = ".bridge_processing_stats.partial.jsonl"
_JOURNAL_FLUSH_BATCH = 256
# W5(2026-09-29, journal 重放保真)：work_ledger 字段从 journal 记录剥离
# 后由本侧车承载重放源（gzip level=1——节点 churn 使逐交付 payload 达
# MB 级（W11 已知体量项），重复结构压缩比 ~10-20x；重放端 gzip 流式逐
# 行读回，语义与未压缩逐行等价）。
_WORK_LEDGER_SIDECAR_GZIP_LEVEL = 1
# v6 compact round: workledger-compact-v1 segments are binary frames; they
# go to a length-framed binary sidecar (record = [u64 LE delivery][u64 LE
# tick][u32 LE length][frame bytes], gzip(1)) instead of a JSON line.  The
# legacy JSON sidecar keeps serving non-compact (dict-path/embedded) frames,
# so old archives replay unchanged.
_COMPACT_SIDECAR_NAME = "request_journal_work_ledger.compact.bin.gz"
_COMPACT_FRAME_MAGIC = b"WLCP"

_RESPONSE_FIELDS = (
    "nodes",
    "parent_edges",
    "watches",
    "assignments",
    "kv_actions",
    # 阶段 5 §8.2:touched ranks 集合(缺省 [];online scheduler 总是显式给出)。
    "touched_ranks",
)


def _encode_response_payload(payload):
    """响应帧负载字节：与旧 _write_json_atomic 写盘字节逐字节相同。

    C3(2026-08-28):紧凑分隔符(separators=(",", ":"))——JSON 语义不变
    (C++ 侧 nlohmann 解析空白不敏感),通道字节显著缩小(大仓档响应数组
    占大头)。A1(2026-08-29): dumps 化——json.dump 走纯 Python
    iterencode 生成器,dumps 走 C one-shot 编码器,实测同 payload 快 ~6x;
    UTF-8 编码后即旧文件的完整字节序列(同一编码器参数
    separators=(",",":")/ensure_ascii 默认)。
    """
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class BridgeError(Exception):
    """协议层致命错误: 写带 error 的 response 后以非 0 退出。"""


class BridgePipeError(Exception):
    """缺陷 B 修复(2026-08-16):resp_notify 长连接写端 BrokenPipe/OSError。

    C++ 的常开读端关闭只能因为 C++ 进程死亡(正常结束也走这条路——C++ 析构
    先关 req 写端,本侧 serve 循环先看到 req EOF 正常返回,一般到不了这里;
    运行中途的 BrokenPipe 即 C++ 异常终止)。不可重试,fail-closed 退出。
    """


class _BoundedSequenceTracker:
    """O(1) duplicate filter for the bridge's strict monotonic protocol."""

    def __init__(self, label):
        self.label = label
        self.next_seq = 0

    def contains(self, seq):
        return seq < self.next_seq

    def validate_new(self, seq):
        if self.contains(seq):
            return False
        if seq != self.next_seq:
            raise BridgeError(
                "{} sequence {} arrived before required sequence {} "
                "(strict monotonic bridge protocol)".format(
                    self.label, seq, self.next_seq))
        return True

    def mark(self, seq):
        if not self.validate_new(seq):
            return False
        self.next_seq += 1
        return True


class BridgeServer:
    """协议 v0 的 Python 端;用法:

        server = BridgeServer(bridge_dir)
        server.serve_forever(handler, on_commit_ack=None)

    handler(request: dict) -> dict 返回 GraphBatch dict(schema_version/
    batch_id/source_delivery_sequence/nodes/parent_edges/watches/assignments/
    kv_actions 会被补全缺省)。on_commit_ack(ack: dict) 在收到 commit ack
    时被调用(provisional 账本 finalize 的挂接点,阶段 3/4 使用)。
    """

    def __init__(self, bridge_dir, *, canonical_request_producer=False,
                 work_ledger_bytes_source=None, timing_sink=None,
                 timing_row_source=None):
        self.bridge_dir = bridge_dir
        # The official C++ producer writes payload.dump() with neither
        # surrounding whitespace nor a trailing newline.  Only that trusted
        # path may bypass the fallback's whole-body normalization scan.
        self._canonical_request_producer = canonical_request_producer
        self._work_ledger_bytes_source = work_ledger_bytes_source
        if (self._canonical_request_producer
                and not callable(self._work_ledger_bytes_source)):
            raise ValueError(
                "canonical request producer requires the accepted "
                "work-ledger canonical-bytes source")
        self.req_notify = os.path.join(bridge_dir, "req_notify.fifo")
        self.resp_notify = os.path.join(bridge_dir, "resp_notify.fifo")
        # 协议严格单调,每条流只保留下一个序号,不保留全历史 seq 集。
        self._request_sequences = _BoundedSequenceTracker("request")
        self._ack_sequences = _BoundedSequenceTracker("commit_ack")
        # 缺陷 B 修复(2026-08-16):resp_notify 写端长连接 fd。serve_forever
        # 启动时一次打开(与 C++ open_notify 持有的常开读端配对),run 生命
        # 期持有,退出时关闭;_notify_response 只写不开。None = 尚未打开。
        self._resp_fd = None
        # 阶段 6 §9.1:桥接通道分项(方案 §9.1;C++ 侧对应物在
        # ShmDecisionBridge::Stats)。独立进程架构下"GIL 等待"的对应物 =
        # Python 阻塞等待 C++ 进程的时间(os.read(req_notify) 阻塞 +
        # resp_notify 写端打开阻塞)。channel_bytes = request/response/ack
        # 帧负载字节(Python 视角;v11 §8 起即旧 os.path.getsize 的文件
        # 字节——帧负载与旧文件内容逐字节相同);forced_flush_count =
        # response 帧发布次数(历史上与原子 JSON 写/os.fsync 1:1;B4
        # 2026-08-23 去除 fsync 后计数面保留为发布次数,与 C++ 侧
        # publish 的计数口径一致,计数等价对拍要求)。
        # 墙时分段账本(2026-10-05 埋点):_handle_request 三段
        # (读+解析 / handler 执行 / response dumps+帧发布)的累计墙时——
        # 逐 request 行字段见 _processing_stats_stream;累计值入本账本,
        # run 末由 online_service 汇总行消费。perf_counter_ns 与
        # processing_ns 的 monotonic_ns 同基数(CLOCK_MONOTONIC),分段和
        # 与 processing_ns 可比但不必相等(journal 等审计段不入分段)。
        self._timing_sink = timing_sink
        # 可选 callable:返回本 request 对应交付的调度侧分段行(由
        # online_service 的扇出 sink 暂存),桥接层在 response 写完后
        # 取走与桥三分段拼接成 timing.jsonl 行;None = 不拼接。
        self._timing_row_source = timing_row_source
        self._stats = {
            "gil_wait_ns": 0,
            "channel_bytes": 0,
            "forced_flush_count": 0,
            "handler_calls": 0,
            "read_parse_ns": 0,
            "handler_ns": 0,
            "write_ns": 0,
            "journal_encode_ns": 0,
            "journal_compress_write_ns": 0,
            "journal_write_ns": 0,
            "journal_total_ns": 0,
        }
        # 两条全程顺序流均不驻留 per-delivery 容器。request journal 复用
        # C++ 帧负载的紧凑 JSON 字节;每 256 行 flush 一次(旧路径在 flush
        # 时批量删除散装 request 文件;v11 §8 起散装文件不复存在,flush
        # 保留为 journal 流的批量落盘节奏),成功 run 只有单一 journal。
        self._request_journal = open(
            os.path.join(bridge_dir, REQUEST_JOURNAL_NAME),
            "w", encoding="utf-8", buffering=1024 * 1024)
        # W5(2026-09-29)：work_ledger 重放源侧车——canonical 生产路径
        # （官方 C++ 对端）恒开；fixture 宽容路径（外部 producer，请求
        # 可能已带原生 work_ledger 或根本无该字段）不开（journal 不剥
        # 离，无重放缺口）。gzip 文本流：重放端按行读回（与 jsonl 同
        # 语义），inner envelope 直接写 accepted canonical bytes，崩溃截断
        # 在行边界被 json.loads 拒绝（fail-closed）。
        self._work_ledger_sidecar = None
        self._compact_sidecar = None
        if self._canonical_request_producer:
            self._work_ledger_sidecar = gzip.open(
                os.path.join(bridge_dir, WORK_LEDGER_REPLAY_SIDECAR_NAME),
                "wb",
                compresslevel=_WORK_LEDGER_SIDECAR_GZIP_LEVEL)
            self._compact_sidecar = gzip.open(
                os.path.join(bridge_dir, _COMPACT_SIDECAR_NAME),
                "wb",
                compresslevel=_WORK_LEDGER_SIDECAR_GZIP_LEVEL)
        # v11 §8 shm 帧通道:serve_forever 启动时按 bridge_shm.json 附着
        # (C++ ensure 已先行铸段;构造期不开——夹具/测试可不带 C++ 对端
        # 直接调 _handle_request/_handle_ack 驱动 journal 路径)。None = 未开。
        self._req_channel = None    # C++ -> Python(request 帧;读)
        self._ack_channel = None    # C++ -> Python(commit_ack 环形;读)
        self._resp_channel = None   # Python -> C++(response 帧;写)
        # ack 环的消费水位(追加序单调 index;内存态,跨进程序列不延续
        # ——同旧协议:Python 重启即 fail-closed 世界,无恢复路径)。
        self._ack_tail = 0
        # journal 批量落盘节奏计数(旧 _pending_request_paths 的去文件化)。
        self._journaled_since_flush = 0
        self._processing_stats_path = os.path.join(
            bridge_dir, _PROCESSING_STATS_PARTIAL_NAME)
        self._processing_stats_stream = open(
            self._processing_stats_path,
            "w", encoding="utf-8", buffering=1024 * 1024)

    # ------------------------------------------------------------- helpers --

    def stats(self):
        """阶段 6 §9.1:桥接通道分项汇总(Python 视角)。每 request 服务
        时间行经 ``per_request_stats()`` 流式读取。"""
        return dict(self._stats)

    def per_request_stats(self):
        """流式读取每 request 的 ``{seq, processing_ns}`` 暂存行。"""
        if not self._processing_stats_stream.closed:
            self._processing_stats_stream.flush()
        with open(self._processing_stats_path, "r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

    def discard_per_request_stats(self):
        """online_stats 合并完成后删除内部暂存流。"""
        if not self._processing_stats_stream.closed:
            self._processing_stats_stream.close()
        try:
            os.unlink(self._processing_stats_path)
        except FileNotFoundError:
            pass

    def _flush_request_journal(self):
        if self._journaled_since_flush == 0:
            return
        self._request_journal.flush()
        self._journaled_since_flush = 0

    def _close_audit_streams(self):
        if not self._request_journal.closed:
            self._flush_request_journal()
            self._request_journal.close()
        if self._work_ledger_sidecar is not None and (
                not self._work_ledger_sidecar.closed):
            self._work_ledger_sidecar.close()
        if self._compact_sidecar is not None and (
                not self._compact_sidecar.closed):
            self._compact_sidecar.close()
        if not self._processing_stats_stream.closed:
            self._processing_stats_stream.close()

    def _new_frames(self):
        """返回 (待处理 ack 帧列表, 待处理 request 帧列表),每项 (seq, bytes)。

        ack 方向是环形通道(C++ 可连发多帧/幂等重发,旧文件协议一次唤醒
        可合并交付多个 ack):drain 取走水位之后全部帧,seq < next_seq 的
        重复静默跳过——旧 contains() 过滤的同义物;未来 seq 原样上交,由
        _handle_* 的严格单调 tracker fail-closed——旧 listdir 兜底的镜像。
        request 方向是单槽通道,1:1 背压(C++ 发布下一帧前对端必已消费
        上一帧)下至多一帧:空槽或 seq < next_seq(已消费的重复)跳过。
        槽损坏(魔数/长度/crc)抛 BridgeShmError,由 serve 循环走 _fail。
        ack 先于 request 返回,保持旧唤醒的处理次序。
        """
        acks = []
        # 模拟本唤醒内 tracker 的推进再过滤:同一唤醒中连发
        # (ack0, ack1, ack0-dup) 时,重复 ack0 必须在对 ack0/ack1 的
        # 模拟推进之后被判为已消费——与旧探测循环(自 next_seq 起探
        # 连续序号,重复文件不进入结果)逐点一致。
        sim_next = self._ack_sequences.next_seq
        for seq, payload in self._ack_channel.drain(self._ack_tail):
            self._ack_tail += 1
            if seq >= sim_next:
                acks.append((seq, payload))
                sim_next = seq + 1
        reqs = []
        frame = self._req_channel.read_frame(self._request_sequences.next_seq)
        if frame is not None:
            reqs.append(frame)
        return acks, reqs

    def _notify_response(self):
        """写 1 字节通知 C++。

        缺陷 B 修复(2026-08-16, face主动测试错误分析.md):resp_notify 写端改为
        serve_forever 启动时一次打开、run 生命周期持有的长连接(与 C++ 侧
        open_notify 持有的常开读端配对),这里只写不再开/关。旧的按交换
        open/write/close 握手存在无法从应用层消除的内核竞态族:pending 的
        O_WRONLY open 被读端唤醒后、write 落地前读端已关 → BrokenPipeError;
        C++ 侧新开读端的 poll_wait 恰逢写端 1→0 关闭转移且缓冲空 → POLLHUP +
        read()==0 的"Python 已崩"误检(F1),以及同族的握手楔死(F7 形态)。
        长连接下双端 run 期无 fd 开关转移:写端 BrokenPipeError 只能是 C++
        进程已死(fail-closed,异常向上传播由 serve 循环捕获留痕)。

        写阻塞时间计入 gil_wait_ns(等待 C++ 进程消费)。
        """
        t0 = time.monotonic_ns()
        try:
            os.write(self._resp_fd, b"\n")
        except OSError as exc:
            raise BridgePipeError(
                "resp_notify write failed: {}".format(exc)) from exc
        finally:
            self._stats["gil_wait_ns"] += time.monotonic_ns() - t0

    # ------------------------------------------------------------- handling --

    def _handle_request(self, seq, handler, raw_request):
        """处理一个 request 帧:``raw_request`` 即帧负载字节(旧
        request_<seq>.json 文件内容,含 canonical 路径下二进制拼接的
        work_ledger 成员)。"""
        t0 = time.monotonic_ns()
        frame_label = "<shm:request_{}>".format(seq)
        self._request_sequences.validate_new(seq)
        # 墙时分段(2026-10-05 埋点,perf_counter_ns):帧字节已就绪 + JSON
        # 解析。v4 N 主线(官方生产路径):原生结构扫描定位顶层
        # work_ledger 成员——outer 拼接字节交给 json.loads(小),巨大子
        # 对象原始字节直通原生接收器与 journal 侧车;Python 不再解析/
        # 重建 work_ledger DTO。fixture 宽容路径(外部 producer)保持
        # 整帧 json.loads(输入不带该形态契约)。
        _read_parse_t0 = time.perf_counter_ns()
        work_ledger_segment = None
        journal_ledger_note = None
        if self._canonical_request_producer:
            from joint.native_ledger import scan_request
            try:
                outer_bytes, work_ledger_segment = scan_request(raw_request)
                request = json.loads(outer_bytes)
            except Exception as exc:  # noqa: BLE001 -- native scan fail-closed
                raise BridgeError(
                    "request {} structural scan failed: {}".format(
                        seq, exc)) from exc
        else:
            try:
                request = json.loads(raw_request)
            except ValueError as exc:
                raise BridgeError(
                    "bad request frame {}: {}".format(frame_label, exc)) from exc
        _read_parse_ns = time.perf_counter_ns() - _read_parse_t0
        if request.get("schema_version") != SCHEMA_VERSION:
            raise BridgeError(
                "request {} schema_version {!r} != {}".format(
                    seq, request.get("schema_version"), SCHEMA_VERSION))
        self._stats["channel_bytes"] += len(raw_request)
        # 墙时分段:handler 执行(= 调度侧 on_decision_batch 全程)。
        _handler_t0 = time.perf_counter_ns()
        if work_ledger_segment is not None:
            response = handler(request, work_ledger_segment)
        else:
            response = handler(request)
        _handler_ns = time.perf_counter_ns() - _handler_t0
        self._stats["handler_calls"] += 1
        work_ledger_canonical = None
        work_ledger_note = None
        if self._canonical_request_producer:
            # v4 N 主线:官方路径的 work_ledger 以原始 segment 到达(或
            # dict 形态的历史 canonical 消费面)。在场断言按交付形态取
            # 相应证据;accepted 字节 + journal 摘要注记(snapshot_kind/
            # event_watermark)由调度侧回读源一并给出。
            has_segment = work_ledger_segment is not None
            has_dict_child = isinstance(request, dict) and isinstance(
                request.get("work_ledger"), dict)
            if not has_segment and not has_dict_child:
                raise BridgeError(
                    "canonical request {} lacks an object work_ledger"
                    .format(seq))
            try:
                sourced = self._work_ledger_bytes_source(seq)
            except Exception as exc:
                raise BridgeError(
                    "cannot obtain accepted canonical work_ledger bytes for "
                    "request {}: {}".format(seq, exc)) from exc
            if isinstance(sourced, tuple):
                work_ledger_canonical, work_ledger_note = sourced
            else:
                work_ledger_canonical = sourced
                work_ledger_note = None
            if (not isinstance(work_ledger_canonical, bytes)
                    or not (work_ledger_canonical.startswith(b"{")
                            and work_ledger_canonical.endswith(b"}"))
                    and not work_ledger_canonical.startswith(
                        _COMPACT_FRAME_MAGIC)):
                raise BridgeError(
                    "request {} has no accepted canonical "
                    "work_ledger bytes".format(seq))
        if not isinstance(response, dict):
            raise BridgeError("handler returned non-dict for seq {}".format(seq))
        if "kernel_contract" not in response:
            raise BridgeError(
                "handler response for seq {} lacks required "
                "kernel_contract".format(seq))
        if not isinstance(response["kernel_contract"], dict):
            raise BridgeError("kernel_contract must be an object")
        response.setdefault("schema_version", SCHEMA_VERSION)
        response.setdefault("batch_id", 0)
        response.setdefault("source_delivery_sequence", seq)
        for field in _RESPONSE_FIELDS:
            if field == "touched_ranks":
                # 字段缺省 = C++ 自算 touched ranks 且不校验声明值。legacy
                # fixture 不知道该字段,真实 scheduler 则总是显式给出。
                continue
            response.setdefault(field, [])
        # 墙时分段:response json.dumps + shm 帧发布(门铃字节由 serve 循环
        # 随后写;帧负载字节与旧 tmp+os.replace 原子写盘逐字节相同)。
        _write_t0 = time.perf_counter_ns()
        body = _encode_response_payload(response)
        try:
            self._resp_channel.publish(seq, body)
        except BridgeShmError as exc:
            raise BridgeError(
                "response frame {} publish failed: {}".format(seq, exc)
            ) from exc
        _write_ns = time.perf_counter_ns() - _write_t0
        self._stats["forced_flush_count"] += 1
        self._stats["channel_bytes"] += len(body)
        self._stats["read_parse_ns"] += _read_parse_ns
        self._stats["handler_ns"] += _handler_ns
        self._stats["write_ns"] += _write_ns
        # processing_ns 保持原口径:帧就绪到 response 帧发布;journal
        # 与内部统计流的审计开销不混入桥服务时间。
        processing_row = {
            "seq": seq,
            "processing_ns": time.monotonic_ns() - t0,
            "read_parse_ns": _read_parse_ns,
            "handler_ns": _handler_ns,
            "write_ns": _write_ns,
        }
        self._processing_stats_stream.write(json.dumps(
            processing_row, separators=(",", ":")) + "\n")
        # 逐行 flush(§10.3 治理):被杀/SIGTERM 时段落行不丢。
        self._processing_stats_stream.flush()
        _journal_t0 = time.perf_counter_ns()
        _journal_encode_t0 = _journal_t0
        if self._canonical_request_producer:
            # W1(2026-09-29)：journal 有界化——work_ledger 顶层字段从
            # journal 记录中剥离（替换为仅含 snapshot_kind/event_watermark
            # 的一行摘要），决不对在飞通道动手（Python 决策应用的是完整
            # request dict，账本照常入账）。原因：状态派生式账本的逐交付
            # 增量使请求字节膨胀数个量级（2s 冒烟即 ~24GB journal）。
            # W5(2026-09-29, 重放保真红线)：账本已接入正式计价
            # （LedgerWorldSource），剥离的字段改由
            # request_journal_work_ledger.jsonl.gz 侧车承载重放源——
            # probe 前缀重放经 iter_request_records(
            # attach_work_ledger=True) 逐交付还原 work_ledger，得到与
            # 原跑相同的账本状态（等价性测试 = test_journal_replay_
            # fidelity.py）。
            journal_record = request
            has_ledger_payload = (
                work_ledger_segment is not None
                or (isinstance(request, dict)
                    and "work_ledger" in request))
            if isinstance(request, dict) and "work_ledger" in request:
                journal_record = dict(request)
                ledger = journal_record.pop("work_ledger")
                if isinstance(ledger, dict):
                    journal_record["work_ledger_journal_stripped"] = {
                        "snapshot_kind": ledger.get("snapshot_kind"),
                        "event_watermark": ledger.get("event_watermark"),
                        "sidecar": WORK_LEDGER_REPLAY_SIDECAR_NAME,
                    }
            elif work_ledger_segment is not None:
                # segment 路径:outer 已不含子对象;摘要注记来自调度侧
                # 接收回执(注记缺席 = 协议装配缺失,fail-closed)。
                if not isinstance(work_ledger_note, dict):
                    raise BridgeError(
                        "canonical request {} has no work_ledger journal "
                        "note".format(seq))
                journal_record = dict(request)
                journal_record["work_ledger_journal_stripped"] = {
                    "snapshot_kind": work_ledger_note.get("snapshot_kind"),
                    "event_watermark": work_ledger_note.get("event_watermark"),
                    "sidecar": (
                        _COMPACT_SIDECAR_NAME
                        if work_ledger_canonical.startswith(
                            _COMPACT_FRAME_MAGIC)
                        else WORK_LEDGER_REPLAY_SIDECAR_NAME),
                }
            journal_line = json.dumps(
                journal_record, separators=(",", ":")) + "\n"
            _journal_encode_ns = time.perf_counter_ns() - _journal_encode_t0
            if has_ledger_payload:
                if self._work_ledger_sidecar is None:
                    raise BridgeError(
                        "canonical work_ledger replay sidecar is unavailable")
                _wrapper_encode_t0 = time.perf_counter_ns()
                prefix = json.dumps({
                    "delivery_sequence": seq,
                    "tick": request.get("tick"),
                }, separators=(",", ":")).encode("utf-8")[:-1]
                _journal_encode_ns += (
                    time.perf_counter_ns() - _wrapper_encode_t0)
            _journal_write_t0 = time.perf_counter_ns()
            if has_ledger_payload:
                _journal_compress_t0 = time.perf_counter_ns()
                if work_ledger_canonical.startswith(_COMPACT_FRAME_MAGIC):
                    record = (
                        int(seq).to_bytes(8, "little")
                        + int(request.get("tick") or 0).to_bytes(
                            8, "little")
                        + len(work_ledger_canonical).to_bytes(4, "little")
                        + work_ledger_canonical)
                    self._compact_sidecar.write(record)
                else:
                    self._work_ledger_sidecar.write(
                        prefix + b',"work_ledger":')
                    self._work_ledger_sidecar.write(work_ledger_canonical)
                    self._work_ledger_sidecar.write(b"}\n")
                _journal_compress_write_ns = (
                    time.perf_counter_ns() - _journal_compress_t0)
            else:
                _journal_compress_write_ns = 0
            self._request_journal.write(journal_line)
            _journal_write_ns = time.perf_counter_ns() - _journal_write_t0
        else:
            # Fixtures and external producers retain the original tolerant
            # normalization contract, including pretty/multiline JSON.
            raw_record = raw_request.strip().decode("utf-8")
            if "\n" in raw_record or "\r" in raw_record:
                raw_record = json.dumps(request, separators=(",", ":"))
            journal_line = raw_record + "\n"
            _journal_encode_ns = time.perf_counter_ns() - _journal_encode_t0
            _journal_write_t0 = time.perf_counter_ns()
            self._request_journal.write(journal_line)
            _journal_compress_write_ns = 0
            _journal_write_ns = time.perf_counter_ns() - _journal_write_t0
        _journal_total_ns = time.perf_counter_ns() - _journal_t0
        self._stats["journal_encode_ns"] += _journal_encode_ns
        self._stats["journal_compress_write_ns"] += _journal_compress_write_ns
        self._stats["journal_write_ns"] += _journal_write_ns
        self._stats["journal_total_ns"] += _journal_total_ns
        if self._timing_sink is not None:
            # Journal total is inclusive of row encoding and both journal
            # streams; gzip compression is included in compress/write time.
            timing_row = {
                "delivery_sequence": seq,
                "seq": seq,
                "read_parse_ns": _read_parse_ns,
                "handler_ns": _handler_ns,
                "write_ns": _write_ns,
                "journal_encode_ns": _journal_encode_ns,
                "journal_compress_write_ns": _journal_compress_write_ns,
                "journal_write_ns": _journal_write_ns,
                "journal_total_ns": _journal_total_ns,
            }
            scheduler_row = (self._timing_row_source()
                             if self._timing_row_source is not None else None)
            if scheduler_row:
                merged = dict(scheduler_row)
                merged.update(timing_row)
                timing_row = merged
            self._timing_sink(timing_row)
        self._journaled_since_flush += 1
        if self._journaled_since_flush >= _JOURNAL_FLUSH_BATCH:
            self._flush_request_journal()
        self._request_sequences.mark(seq)

    def _handle_ack(self, seq, on_commit_ack, raw_ack):
        """处理一个 commit_ack 帧:``raw_ack`` 即帧负载字节(旧
        commit_ack_<seq>.json 文件内容)。"""
        self._ack_sequences.validate_new(seq)
        try:
            ack = json.loads(raw_ack)
        except ValueError as exc:
            raise BridgeError(
                "bad ack frame <shm:commit_ack_{}>: {}".format(seq, exc)
            ) from exc
        if on_commit_ack is not None:
            on_commit_ack(ack)
        self._stats["channel_bytes"] += len(raw_ack)
        self._ack_sequences.mark(seq)
        # 阶段 7 §10.3:中间产物生命周期——ack 帧已被完整消费(on_commit_ack
        # 幂等落账完成)。旧实现的 ack 文件消费后立即删除;单槽通道下等价物
        # 是"槽由下一帧自然复用"(1:1 背压保证本帧已被读取),无需显式清理。

    def _fail(self, seq, message):
        """发布带 error 字段的 response 帧、通知 C++、非 0 退出(fail-closed)。"""
        # 缺陷 B 修复配套:F1 现场 python.log 0 字节、无可诊断性——_fail 现在
        # 先在 stderr 留痕(错误路径的证据行),再走 fail-closed 收尾。
        print("decision_bridge: fail-closed at seq {}: {}".format(
            seq, message), file=sys.stderr)
        response = {
            "schema_version": SCHEMA_VERSION,
            "batch_id": 0,
            "source_delivery_sequence": seq,
            "nodes": [],
            "parent_edges": [],
            "watches": [],
            "assignments": [],
            "kv_actions": [],
            "future_alarms": [],
            "touched_ranks": [],
            "error": message,
        }
        try:
            body = _encode_response_payload(response)
            self._resp_channel.publish(seq, body)
            self._notify_response()
        except (OSError, BridgeShmError):
            pass  # 已尽力;以非 0 退出为准
        sys.exit(1)

    # ----------------------------------------------------------- main loop --

    def open_transport(self):
        """附着 shm 帧通道(按 bridge_shm.json;世代校验 fail-closed)。

        serve_forever 在两个 FIFO 端打开后调用;不经 serve 的直接驱动方
        (journal/夹具单测)在调 _handle_request/_handle_ack 前调用一次。
        """
        info = load_bridge_shm_info(self.bridge_dir)
        self._req_channel = ShmFrameChannel.open(
            info["request_segment"], info["generation"],
            info["slot_capacity"])
        self._ack_channel = ShmAckRing.open(
            info["commit_ack_segment"], info["generation"])
        self._resp_channel = ShmFrameChannel.open(
            info["response_segment"], info["generation"],
            info["slot_capacity"])

    def close_transport(self):
        for channel in (self._req_channel, self._ack_channel,
                        self._resp_channel):
            if channel is not None:
                channel.close()
        self._req_channel = None
        self._ack_channel = None
        self._resp_channel = None

    def serve_forever(self, handler, on_commit_ack=None):
        """阻塞循环,不轮询: 每收到 req_notify.fifo 一个字节处理一批新帧。

        返回 0 = 正常结束(C++ 关闭写端,read 得 EOF)。错误路径已发 error
        response 并 sys.exit(1),不会从这里返回。

        缺陷 B 修复(2026-08-16):resp_notify 写端在此处一次打开、run 生命周期
        持有(长连接,配对 C++ open_notify 持有的常开读端)。打开顺序无死锁:
        C++ 先开 req 写端(重试至本侧 req 读端就绪)再开 resp 读端;本侧先开
        req 读端(阻塞至 C++ req 写端)再开 resp 写端(此时 C++ resp 读端已在
        或随后即到——非阻塞读端 open 恒成功,不互相等待)。
        v11 §8:shm 帧通道在两个 FIFO 端打开后附着——C++ ensure 已在启动
        Python 前铸段并写 bridge_shm.json(世代校验 fail-closed,拒绝上一次
        run 的陈旧段)。
        """
        t0 = time.monotonic_ns()
        fd = os.open(self.req_notify, os.O_RDONLY)  # 阻塞直到 C++ 打开写端
        t1 = time.monotonic_ns()
        # 阶段 6 §9.1:两个初始打开都是等待 C++ 进程,计入 gil_wait_ns。
        self._stats["gil_wait_ns"] += (t1 - t0)
        try:
            t0 = time.monotonic_ns()
            self._resp_fd = os.open(self.resp_notify, os.O_WRONLY)
            self._stats["gil_wait_ns"] += time.monotonic_ns() - t0
            self.open_transport()
            while True:
                t0 = time.monotonic_ns()
                data = os.read(fd, 1)
                # 阶段 6 §9.1:读端阻塞 = 等待 C++ 进程(独立进程架构下 GIL
                # 等待的对应物),计入 gil_wait_ns。
                self._stats["gil_wait_ns"] += time.monotonic_ns() - t0
                if not data:
                    return 0  # C++ 结束或崩溃: 写端关闭 => EOF
                try:
                    acks, reqs = self._new_frames()
                except BridgeShmError as exc:
                    # 槽损坏(魔数/长度/crc):fail-closed,seq 取当前期望值
                    # 以便 C++ 侧拿到带 error 的 response/超时中止。
                    self._fail(self._request_sequences.next_seq,
                               "bridge shm transport: {}".format(exc))
                for ack_seq, ack_raw in acks:
                    try:
                        self._handle_ack(ack_seq, on_commit_ack, ack_raw)
                    except BridgeError as exc:
                        self._fail(ack_seq, "ack {}: {}".format(ack_seq, exc))
                    except Exception:
                        # (2026-10-01 R2 审查修复) 与 request 通道对称的
                        # 兜底：on_commit_ack 的协议校验可抛 ValueError/
                        # RuntimeError 等（非 BridgeError），旧实现让异常
                        # 裸穿 serve_forever——无留痕、无 error response，
                        # C++ 侧只能等 poll 超时/EOF 才中止。此处留
                        # traceback 后走 _fail（fail-closed 退出 + EOF 触
                        # 发 C++ abort，与 BridgeError 臂同通道）。
                        traceback.print_exc()
                        self._fail(ack_seq, "ack {}: unhandled "
                                            "exception".format(ack_seq))
                for req_seq, req_raw in reqs:
                    try:
                        self._handle_request(req_seq, handler, req_raw)
                        self._notify_response()
                    except BridgePipeError as exc:
                        # C++ 长连接读端已关 = C++ 进程死亡(fail-closed,
                        # 留痕后退出;不是可重试状态)。
                        print("decision_bridge: C++ side is gone "
                              "(BrokenPipe on resp_notify write, seq {}): "
                              "{}".format(req_seq, exc), file=sys.stderr)
                        sys.exit(1)
                    except BridgeError as exc:
                        self._fail(req_seq, str(exc))
                    except Exception as exc:  # noqa: BLE001 -- handler 异常
                        import traceback
                        traceback.print_exc()
                        self._fail(req_seq, "{}: {}".format(type(exc).__name__, exc))
        finally:
            os.close(fd)
            if self._resp_fd is not None:
                os.close(self._resp_fd)
                self._resp_fd = None
            self.close_transport()
            self._close_audit_streams()
