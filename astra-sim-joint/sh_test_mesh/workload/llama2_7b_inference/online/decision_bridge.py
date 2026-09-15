#!/usr/bin/env python3
"""decision_bridge.py -- Python 端 FileDecisionBridge 协议 v0 (方案 §4 步骤 1-7).

桥接目录 <bridge_dir>/ 由 C++ 先建好(含 req_notify.fifo / resp_notify.fifo,
"先建 FIFO 再启动双方")。serve_forever 阻塞在 req_notify.fifo 的 read 上
(不轮询):每收到一个通知字节,扫描目录处理新出现的文件,按 seq 升序——
commit_ack 与 request 分开记账(同一 seq 的 request 与 ack 可并存)。

request 处理: 读 request_<seq>.json(原子发布: 临时文件+rename,不会读到
半截) -> handler(request) -> 写 response_<seq>.json(原子) -> 往
resp_notify.fifo 写 1 字节(C++ 长连接读端 poll)。handler 抛异常或协议
错误 => 写带 error 字段的 response、通知 C++、以非 0 退出(fail-closed,
stderr 留痕);C++ 读到 error 即 abort。C++ 端崩溃/结束 => req_notify.fifo
写端关闭 => read 返回 EOF => serve_forever 返回 0;resp_notify 写端
BrokenPipe(C++ 常开读端关闭,运行中途) => BridgePipeError fail-closed
退出(缺陷 B 修复 2026-08-16:resp 通道双侧长连接,消除按交换开/关握手
的内核竞态族——POLLHUP 假 EOF 误检与 EPIPE 楔死)。

本模块是决策通道,不是运行期 request 注入通道:Producer -> C++ 的
submit/close/EOF/error 走步骤 1-2 的 command queue,不经过 req_notify.fifo。

协议规则(冻结,合同②/④): request/ack 各自严格从 0 起 +1 单调,旧 seq
重复幂等忽略,未来 seq 在缺少前序时 fail-closed;背压(一轮至多一个在途
request,C++ 不回 response 不连续发);
超时与崩溃检测在 C++ 侧(C++ poll 超时/EOF/EPIPE 即 abort)。
"""

import json
import os
import sys
import time

from bridge_request_journal import REQUEST_JOURNAL_NAME

SCHEMA_VERSION = 1  # 阶段 4 §7.1:StateDelta schema v1(契约文档已删除)

_REQUEST_PREFIX = "request_"
_RESPONSE_PREFIX = "response_"
_ACK_PREFIX = "commit_ack_"
_JSON_SUFFIX = ".json"
_PROCESSING_STATS_PARTIAL_NAME = ".bridge_processing_stats.partial.jsonl"
_JOURNAL_FLUSH_BATCH = 256

_RESPONSE_FIELDS = (
    "nodes",
    "parent_edges",
    "watches",
    "assignments",
    "kv_actions",
    # 阶段 5 §8.2:touched ranks 集合(缺省 [];online scheduler 总是显式给出)。
    "touched_ranks",
)


def _read_json_atomic(path):
    """读 C++ 原子发布的 JSON 文件;文件缺失/损坏抛 OSError/ValueError。"""
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _write_json_atomic(path, payload):
    """原子发布: 写 <path>.tmp 再 os.replace,避免对端读到半截文件。

    B4(2026-08-23,用户裁决): 去除 os.fsync——本函数全部调用方(本模块
    _handle_request/_fail)均为桥响应瞬态文件,C++ 收到 notify 字节后立刻
    读走并 unlink,无持久审计消费者;tmp + os.replace 的原子性保留。
    断电极端场景 fail-closed(整轮重跑,与 campaign 实操一致)。
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as target:
        # C3(2026-08-28):紧凑分隔符(separators=(",", ":"))——JSON 语义
        # 不变(C++ 侧 nlohmann 解析空白不敏感),通道字节显著缩小
        # (大仓档响应数组占大头;省 I/O 与 C++ 读侧拷贝)。
        # A1(2026-08-29): dumps 化——json.dump 走纯 Python iterencode 生成器
        # (json.dump 传 _one_shot=False),dumps 走 C one-shot 编码器,实测同
        # payload 快 ~6x;单次 write 后 flush,字节序列与 dump 逐字节相同
        # (同一编码器参数 separators=(",",":")/ensure_ascii 默认)。
        target.write(json.dumps(payload, separators=(",", ":")))
        target.flush()
    os.replace(tmp, path)


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

    def __init__(self, bridge_dir, *, canonical_request_producer=False):
        self.bridge_dir = bridge_dir
        # The official C++ producer writes payload.dump() with neither
        # surrounding whitespace nor a trailing newline.  Only that trusted
        # path may bypass the fallback's whole-body normalization scan.
        self._canonical_request_producer = canonical_request_producer
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
        # FileDecisionBridge::Stats)。独立进程架构下"GIL 等待"的对应物 =
        # Python 阻塞等待 C++ 进程的时间(os.read(req_notify) 阻塞 +
        # resp_notify 写端打开阻塞)。channel_bytes = request/response/ack
        # JSON 文件字节(Python 视角,os.path.getsize);forced_flush_count =
        # 原子 JSON 写次数(历史上与 os.fsync 1:1;B4 2026-08-23 去除 fsync
        # 后计数面保留为原子写次数,与 C++ 侧 write_file_atomic 的计数口径
        # 一致,计数等价对拍要求)。
        self._stats = {
            "gil_wait_ns": 0,
            "channel_bytes": 0,
            "forced_flush_count": 0,
            "handler_calls": 0,
        }
        # 两条全程顺序流均不驻留 per-delivery 容器。request journal 复用
        # C++ 已写出的紧凑 JSON 字节;每 256 行 flush 后批量删除散装 request,
        # 因而成功 run 只有单一 journal,异常 run 的未 flush 尾部也有界。
        self._request_journal = open(
            os.path.join(bridge_dir, REQUEST_JOURNAL_NAME),
            "w", encoding="utf-8", buffering=1024 * 1024)
        self._pending_request_paths = []
        self._processing_stats_path = os.path.join(
            bridge_dir, _PROCESSING_STATS_PARTIAL_NAME)
        self._processing_stats_stream = open(
            self._processing_stats_path,
            "w", encoding="utf-8", buffering=1024 * 1024)

    # ------------------------------------------------------------- helpers --

    def stats(self):
        """阶段 6 §9.1:桥接通道分项汇总(Python 视角)。含每 request 服务
        时间行列表(per_request),供 online_stats.jsonl 合并。"""
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
        if not self._pending_request_paths:
            return
        self._request_journal.flush()
        pending, self._pending_request_paths = self._pending_request_paths, []
        for path in pending:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise BridgeError(
                    "cannot retire journaled request {}: {}".format(
                        path, exc)) from exc

    def _close_audit_streams(self):
        if not self._request_journal.closed:
            self._flush_request_journal()
            self._request_journal.close()
        if not self._processing_stats_stream.closed:
            self._processing_stats_stream.close()

    @staticmethod
    def _seq_of(name, prefix):
        if name.startswith(prefix) and name.endswith(_JSON_SUFFIX):
            body = name[len(prefix):-len(_JSON_SUFFIX)]
            if body.isdigit():
                return int(body)
        return None

    def _new_files(self):
        """返回 (未处理 request seq 升序, 未处理 ack seq 升序)。

        官方协议严格 +1 单调,正常路径只 stat 下一个序号。journal 批量回收
        将散装 request 尾部限制在 256 个以内;listdir 回退若发现未来序号,
        handler 前的 monotonic tracker 会 fail-closed,不会乱序交付。
        """
        reqs = []
        acks = []
        req_probe = self._request_sequences.next_seq
        while True:
            probe = os.path.join(
                self.bridge_dir,
                _REQUEST_PREFIX + str(req_probe) + _JSON_SUFFIX)
            if not os.path.exists(probe):
                break
            if not self._request_sequences.contains(req_probe):
                reqs.append(req_probe)
            req_probe += 1
        ack_probe = self._ack_sequences.next_seq
        while True:
            probe = os.path.join(
                self.bridge_dir,
                _ACK_PREFIX + str(ack_probe) + _JSON_SUFFIX)
            if not os.path.exists(probe):
                break
            if not self._ack_sequences.contains(ack_probe):
                acks.append(ack_probe)
            ack_probe += 1
        if reqs or acks:
            return sorted(reqs), sorted(acks)
        for name in os.listdir(self.bridge_dir):
            req_seq = self._seq_of(name, _REQUEST_PREFIX)
            if (req_seq is not None
                    and not self._request_sequences.contains(req_seq)):
                reqs.append(req_seq)
            ack_seq = self._seq_of(name, _ACK_PREFIX)
            if (ack_seq is not None
                    and not self._ack_sequences.contains(ack_seq)):
                acks.append(ack_seq)
        return sorted(reqs), sorted(acks)

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

    def _handle_request(self, seq, handler):
        t0 = time.monotonic_ns()
        path = os.path.join(self.bridge_dir, _REQUEST_PREFIX + str(seq) + _JSON_SUFFIX)
        self._request_sequences.validate_new(seq)
        try:
            with open(path, "r", encoding="utf-8") as source:
                raw_request = source.read()
            request = json.loads(raw_request)
        except (OSError, ValueError) as exc:
            raise BridgeError("bad request file {}: {}".format(path, exc)) from exc
        if request.get("schema_version") != SCHEMA_VERSION:
            raise BridgeError(
                "request {} schema_version {!r} != {}".format(
                    seq, request.get("schema_version"), SCHEMA_VERSION))
        self._stats["channel_bytes"] += os.path.getsize(path)
        response = handler(request)
        self._stats["handler_calls"] += 1
        if not isinstance(response, dict):
            raise BridgeError("handler returned non-dict for seq {}".format(seq))
        response.setdefault("schema_version", SCHEMA_VERSION)
        response.setdefault("batch_id", 0)
        response.setdefault("source_delivery_sequence", seq)
        for field in _RESPONSE_FIELDS:
            if field == "touched_ranks":
                # 字段缺省 = C++ 自算 touched ranks 且不校验声明值。legacy
                # fixture 不知道该字段,真实 scheduler 则总是显式给出。
                continue
            response.setdefault(field, [])
        response_path = os.path.join(
            self.bridge_dir, _RESPONSE_PREFIX + str(seq) + _JSON_SUFFIX)
        _write_json_atomic(response_path, response)
        self._stats["forced_flush_count"] += 1
        self._stats["channel_bytes"] += os.path.getsize(response_path)
        # processing_ns 保持原口径:读 request 到 response 原子发布;journal
        # 与内部统计流的审计开销不混入桥服务时间。
        processing_row = {
            "seq": seq,
            "processing_ns": time.monotonic_ns() - t0,
        }
        self._processing_stats_stream.write(json.dumps(
            processing_row, separators=(",", ":")) + "\n")
        if self._canonical_request_producer:
            # Reuse the exact C++ request bytes and append the JSONL delimiter
            # separately: no O(request-bytes) strip/newline scan and no
            # raw_record + "\\n" full-size temporary string.
            self._request_journal.write(raw_request)
            self._request_journal.write("\n")
        else:
            # Fixtures and external producers retain the original tolerant
            # normalization contract, including pretty/multiline JSON.
            raw_record = raw_request.strip()
            if "\n" in raw_record or "\r" in raw_record:
                raw_record = json.dumps(request, separators=(",", ":"))
            self._request_journal.write(raw_record + "\n")
        self._pending_request_paths.append(path)
        if len(self._pending_request_paths) >= _JOURNAL_FLUSH_BATCH:
            self._flush_request_journal()
        self._request_sequences.mark(seq)

    def _handle_ack(self, seq, on_commit_ack):
        path = os.path.join(self.bridge_dir, _ACK_PREFIX + str(seq) + _JSON_SUFFIX)
        self._ack_sequences.validate_new(seq)
        try:
            ack = _read_json_atomic(path)
        except (OSError, ValueError) as exc:
            raise BridgeError("bad ack file {}: {}".format(path, exc)) from exc
        if on_commit_ack is not None:
            on_commit_ack(ack)
        self._stats["channel_bytes"] += os.path.getsize(path)
        self._ack_sequences.mark(seq)
        # 阶段 7 §10.3:中间产物生命周期——ack 已被完整消费(on_commit_ack
        # 幂等落账完成),立即删除。ack 文件是纯中间产物(C++ 已不再引用,
        # 幂等 fixture 从 request journal 重放),删除后 bridge 仅保留有界
        # request 批次尾部、单一 journal 与在飞 response。实测 20.csv
        # 前30s 输入:ack 累计 ~266KB,
        # response 累计 ~202MB,消费后清理把它们压到有界。
        try:
            os.remove(path)
        except OSError:
            pass  # 尽力删除;失败不阻断(下一轮 _new_files 不再列出它)

    def _fail(self, seq, message):
        """写带 error 字段的 response、通知 C++、非 0 退出(fail-closed)。"""
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
            _write_json_atomic(
                os.path.join(self.bridge_dir, _RESPONSE_PREFIX + str(seq) + _JSON_SUFFIX),
                response)
            self._notify_response()
        except OSError:
            pass  # 已尽力;以非 0 退出为准
        sys.exit(1)

    # ----------------------------------------------------------- main loop --

    def serve_forever(self, handler, on_commit_ack=None):
        """阻塞循环,不轮询: 每收到 req_notify.fifo 一个字节处理一批新文件。

        返回 0 = 正常结束(C++ 关闭写端,read 得 EOF)。错误路径已写 error
        response 并 sys.exit(1),不会从这里返回。

        缺陷 B 修复(2026-08-16):resp_notify 写端在此处一次打开、run 生命周期
        持有(长连接,配对 C++ open_notify 持有的常开读端)。打开顺序无死锁:
        C++ 先开 req 写端(重试至本侧 req 读端就绪)再开 resp 读端;本侧先开
        req 读端(阻塞至 C++ req 写端)再开 resp 写端(此时 C++ resp 读端已在
        或随后即到——非阻塞读端 open 恒成功,不互相等待)。
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
            while True:
                t0 = time.monotonic_ns()
                data = os.read(fd, 1)
                # 阶段 6 §9.1:读端阻塞 = 等待 C++ 进程(独立进程架构下 GIL
                # 等待的对应物),计入 gil_wait_ns。
                self._stats["gil_wait_ns"] += time.monotonic_ns() - t0
                if not data:
                    return 0  # C++ 结束或崩溃: 写端关闭 => EOF
                reqs, acks = self._new_files()
                for ack_seq in acks:
                    try:
                        self._handle_ack(ack_seq, on_commit_ack)
                    except BridgeError as exc:
                        self._fail(ack_seq, "ack {}: {}".format(ack_seq, exc))
                for req_seq in reqs:
                    try:
                        self._handle_request(req_seq, handler)
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
            self._close_audit_streams()
