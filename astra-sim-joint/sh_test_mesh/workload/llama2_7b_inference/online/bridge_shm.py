#!/usr/bin/env python3
"""bridge_shm.py -- 决策桥 POSIX-shm 帧传输层（joint v11 §8 保底）。

C++ 侧载体（astra-sim/workload/execution_driven/ShmFrameChannel.{hh,cc}）
的 Python 镜像：三个单槽帧通道（request / commit_ack 由 C++ 发，response
由本侧发），帧字节与旧协议的 request_/response_/commit_ack_<seq>.json
文件内容逐字节相同。同步模型与 C++ 侧一致：写方整帧发布后才写 1 字节
FIFO 门铃，读方只在收到门铃字节后读帧——无需原子量；冻结的 1:1 背压保证
已发布帧不会被未消费覆盖。

字节布局（冻结，两侧一致）：

  段文件 = 64 字节头 + 单帧槽（slot_capacity 字节负载空间）
    头:   u32 magic | u32 version | u64 generation | u64 slot_capacity
          （一律小端；24..63 为零填充）
    槽:   [u32 frame_magic][u64 seq][u32 len][负载 len 字节][u32 crc32]
          crc32（IEEE/zlib 多项式）覆盖 16 字节帧头 + 负载。

世代校验（跨 run 污染防护）：C++ ensure_bridge_dir 铸新世代段并原子写
<bridge_dir>/bridge_shm.json 指针；本侧 serve 启动时读指针并校验段头
generation——陈旧段（上一次崩溃 run 残留）即 fail-closed。

幂等语义镜像散装文件协议：槽保留最后发布的帧；read_frame(expect_seq)
把 seq < expect_seq 视为已消费的重复（返回 None，静默忽略——与旧
contains() 过滤对重复 commit_ack_<seq>.json 的忽略同义）；未来 seq
（> expect）原样上交，由调用方的严格单调 tracker fail-closed（镜像旧
listdir 兜底发现未来序号文件的路径）。
"""

import json
import mmap
import os
import struct
import zlib

SEGMENT_MAGIC = 0x53484231
SEGMENT_VERSION = 1
FRAME_MAGIC = 0x46524D31
SEGMENT_HEADER_BYTES = 64
DEFAULT_SLOT_CAPACITY = 256 * 1024 * 1024
INFO_NAME = "bridge_shm.json"
_NAME_PREFIX = "astra_brg_"

# commit-ack 环形通道几何（与 C++ kBridgeShmAckSlotCount/Bytes 一致；ack
# 流无交换门控——C++ 可连发多帧，单槽会丢中间帧，故为定长多槽环）。
ACK_SLOT_COUNT = 4096
ACK_SLOT_BYTES = 8192

_FRAME_HEADER = struct.Struct("<IQI")     # magic, seq, len
_SEGMENT_HEADER = struct.Struct("<IIQQ")  # magic, version, generation, cap
_RING_HEADER = struct.Struct("<IIQQQ")    # magic, version, gen, n, bytes
_RING_HEAD = struct.Struct("<Q")          # head @32 / tail @40

# 段名角色（与 C++ mint_bridge_shm_info 的 <stem><role>_<gen hex> 一致）。
_ROLES = ("req", "ack", "resp")


class BridgeShmError(Exception):
    """shm 帧传输层致命错误（fail-closed；桥层转为 error response 退出）。"""


def _fnv1a64(text: str) -> int:
    """FNV-1a 64-bit over UTF-8 path bytes（与 C++ bridge_shm_name_stem 同常数）."""
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def bridge_shm_stem(bridge_dir: str) -> str:
    """本 bridge_dir 的段名公共前缀（含尾随下划线）."""
    return "{}{:016x}_".format(_NAME_PREFIX, _fnv1a64(bridge_dir))


def new_generation() -> int:
    """生产端世代铸造（ns 时钟与 pid 混合；与 C++ 同形）."""
    import time
    return (time.time_ns() ^ (os.getpid() << 32)) & 0xFFFFFFFFFFFFFFFF


def segment_names(stem: str, generation: int) -> dict:
    gen = "{:016x}".format(generation)
    return {
        "request_segment": "{}req_{}".format(stem, gen),
        "commit_ack_segment": "{}ack_{}".format(stem, gen),
        "response_segment": "{}resp_{}".format(stem, gen),
    }


def info_path(bridge_dir: str) -> str:
    return os.path.join(bridge_dir, INFO_NAME)


def load_bridge_shm_info(bridge_dir: str) -> dict:
    """读 C++ 写下的段指针（缺文件/损坏/形态不符 fail-closed）."""
    path = info_path(bridge_dir)
    try:
        with open(path, "r", encoding="utf-8") as source:
            info = json.load(source)
    except (OSError, ValueError) as exc:
        raise BridgeShmError(
            "cannot load {}: {}".format(path, exc)) from exc
    if (not isinstance(info, dict) or info.get("version") != 1
            or not isinstance(info.get("generation"), int)
            or not isinstance(info.get("slot_capacity"), int)
            or not all(isinstance(info.get(key), str) for key in (
                "request_segment", "commit_ack_segment",
                "response_segment"))):
        raise BridgeShmError("malformed bridge shm info in {}".format(path))
    return info


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as target:
        target.write(json.dumps(payload, separators=(",", ":")))
        target.flush()
    os.replace(tmp, path)


def clean_stale_bridge_shm(bridge_dir: str) -> None:
    """按前缀清除本 bridge_dir 的全部陈旧段（任意角色/世代）."""
    stem = bridge_shm_stem(bridge_dir)
    shm_dir = "/dev/shm"
    try:
        names = os.listdir(shm_dir)
    except OSError:
        return
    for name in names:
        if name.startswith(stem):
            try:
                os.unlink(os.path.join(shm_dir, name))
            except FileNotFoundError:
                pass


def ensure_bridge_shm(bridge_dir: str,
                      slot_capacity: int = DEFAULT_SLOT_CAPACITY) -> dict:
    """生产端铸段（C++ mint_bridge_shm_info 的 Python 对等物，测试/夹具用）。

    生产运行中 C++ 先行铸段并写指针，Python 只读（load_bridge_shm_info）。
    本函数供 Python 侧单测与无 C++ 对端的夹具充当"C++ 侧"：段缺失或
    校验不过则清陈旧段、铸新世代并原子写指针；已有有效段则原样采纳。
    返回指针 dict（不持映射——本函数返回后段由调用方按名打开）。
    """
    os.makedirs(bridge_dir, exist_ok=True)
    try:
        info = load_bridge_shm_info(bridge_dir)
        if info["slot_capacity"] == slot_capacity:
            req = ShmFrameChannel.open(
                info["request_segment"], info["generation"],
                info["slot_capacity"])
            req.close()
            ack = ShmAckRing.open(
                info["commit_ack_segment"], info["generation"])
            ack.close()
            resp = ShmFrameChannel.open(
                info["response_segment"], info["generation"],
                info["slot_capacity"])
            resp.close()
            return info
    except BridgeShmError:
        pass
    clean_stale_bridge_shm(bridge_dir)
    generation = new_generation()
    names = segment_names(bridge_shm_stem(bridge_dir), generation)
    info = dict(names)
    info["version"] = 1
    info["generation"] = generation
    info["slot_capacity"] = slot_capacity
    ShmFrameChannel.create(info["request_segment"], generation, slot_capacity)
    ShmAckRing.create(info["commit_ack_segment"], generation)
    ShmFrameChannel.create(info["response_segment"], generation, slot_capacity)
    _write_json_atomic(info_path(bridge_dir), {
        "version": 1,
        "generation": generation,
        "request_segment": info["request_segment"],
        "commit_ack_segment": info["commit_ack_segment"],
        "response_segment": info["response_segment"],
        "slot_capacity": slot_capacity,
    })
    return info


class ShmFrameChannel:
    """一个单槽 shm 帧通道（读/写同一映射）."""

    def __init__(self, name: str, generation: int, slot_capacity: int,
                 mapping, fh):
        self.name = name
        self.generation = generation
        self.slot_capacity = slot_capacity
        self._map = mapping
        self._fh = fh

    @classmethod
    def open(cls, name: str, generation: int, slot_capacity: int):
        """按名附着既有段并校验段头（世代/容量不符 fail-closed）."""
        if name.startswith("/") or "/" in name:
            raise BridgeShmError("illegal segment name {!r}".format(name))
        path = os.path.join("/dev/shm", name)
        try:
            fh = open(path, "r+b", buffering=0)
        except OSError as exc:
            raise BridgeShmError(
                "cannot open bridge shm segment {}: {}".format(
                    path, exc)) from exc
        try:
            size = os.fstat(fh.fileno()).st_size
            want = SEGMENT_HEADER_BYTES + slot_capacity
            if size != want:
                raise BridgeShmError(
                    "bridge shm segment {} has size {}, expected {}".format(
                        path, size, want))
            mapping = mmap.mmap(fh.fileno(), want)
            magic, version, seg_gen, seg_cap = _SEGMENT_HEADER.unpack_from(
                mapping, 0)
            if magic != SEGMENT_MAGIC or version != SEGMENT_VERSION:
                raise BridgeShmError(
                    "segment {} is not a bridge shm segment "
                    "(magic/version)".format(path))
            if seg_gen != generation:
                raise BridgeShmError(
                    "segment {} generation {} != expected {} "
                    "(stale segment from another run)".format(
                        path, seg_gen, generation))
            if seg_cap != slot_capacity:
                raise BridgeShmError(
                    "segment {} capacity mismatch ({} != {})".format(
                        path, seg_cap, slot_capacity))
        except Exception:
            fh.close()
            raise
        return cls(name, generation, slot_capacity, mapping, fh)

    @classmethod
    def create(cls, name: str, generation: int, slot_capacity: int):
        """创建新段（O_EXCL；存在即错）——生产端/夹具用."""
        if name.startswith("/") or "/" in name:
            raise BridgeShmError("illegal segment name {!r}".format(name))
        path = os.path.join("/dev/shm", name)
        try:
            fh = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise BridgeShmError(
                "cannot create bridge shm segment {}: {}".format(
                    path, exc)) from exc
        try:
            os.ftruncate(fh, SEGMENT_HEADER_BYTES + slot_capacity)
            mapping = mmap.mmap(fh, SEGMENT_HEADER_BYTES + slot_capacity)
            mapping[0:SEGMENT_HEADER_BYTES] = b"\x00" * SEGMENT_HEADER_BYTES
            _SEGMENT_HEADER.pack_into(
                mapping, 0, SEGMENT_MAGIC, SEGMENT_VERSION, generation,
                slot_capacity)
        except Exception:
            os.close(fh)
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return cls(name, generation, slot_capacity, mapping,
                   os.fdopen(fh, "r+b", buffering=0, closefd=True))

    # ------------------------------------------------------------- frames --

    def publish(self, seq: int, payload: bytes) -> None:
        """整帧发布；超槽容量 fail-closed（绝不静默截断）."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise BridgeShmError("payload must be bytes")
        payload = bytes(payload)
        if len(payload) > self.slot_capacity:
            raise BridgeShmError(
                "frame of {} bytes exceeds the shm slot capacity of {} "
                "(delivery payload too large; fail-closed)".format(
                    len(payload), self.slot_capacity))
        header = _FRAME_HEADER.pack(FRAME_MAGIC, seq, len(payload))
        # 门铃字节（调用方随后写 FIFO）是唯一同步点；槽内写入次序无需栅栏。
        self._map[SEGMENT_HEADER_BYTES:
                  SEGMENT_HEADER_BYTES + len(header)] = header
        if payload:
            self._map[SEGMENT_HEADER_BYTES + len(header):
                      SEGMENT_HEADER_BYTES + len(header) + len(payload)] = (
                          payload)
        crc = zlib.crc32(header + payload) & 0xFFFFFFFF
        offset = SEGMENT_HEADER_BYTES + len(header) + len(payload)
        self._map[offset:offset + 4] = struct.pack("<I", crc)

    def read_frame(self, expect_seq: int):
        """读槽内当前帧。

        返回 ``(seq, payload)``；空槽或 ``seq < expect_seq``（已消费的
        重复）返回 ``None``；槽损坏（魔数/长度/crc）抛 :class:`BridgeShmError`。
        未来 seq（> expect_seq）原样返回——由调用方的严格单调 tracker
        fail-closed（镜像旧 listdir 兜底语义）。
        """
        base = SEGMENT_HEADER_BYTES
        magic, seq, length = _FRAME_HEADER.unpack_from(self._map, base)
        if magic == 0:
            return None
        if magic != FRAME_MAGIC:
            raise BridgeShmError(
                "slot holds a corrupt frame (magic {})".format(magic))
        if seq < expect_seq:
            return None
        if length > self.slot_capacity:
            raise BridgeShmError(
                "frame length {} exceeds the slot capacity".format(length))
        span = base + _FRAME_HEADER.size + length
        stored_crc, = struct.unpack_from("<I", self._map, span)
        actual_crc = zlib.crc32(self._map[base:span]) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise BridgeShmError(
                "frame checksum mismatch (stored {}, computed {})".format(
                    stored_crc, actual_crc))
        # mmap 切片在 Python 3 直接返回 bytes（即帧负载字节的拷贝）。
        payload = self._map[base + _FRAME_HEADER.size:span]
        return seq, payload

    # ---------------------------------------------------------- lifecycle --

    def close(self) -> None:
        if self._map is not None:
            try:
                self._map.close()
            finally:
                self._map = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class ShmAckRing:
    """commit-ack 多槽环形通道（C++ ShmAckRing 的 Python 镜像）。

    ack 流无交换门控（C++ 可连发多帧，含重复 seq 的幂等重发），单槽会
    丢中间帧；本环形通道按追加序交付全部帧（含重复——调用方按 tracker
    过滤，即旧 contains() 过滤的同义物）。布局：

      64 字节头 [u32 magic][u32 version][u64 generation][u64 slot_count]
      [u64 slot_bytes][u64 head][u64 tail][16B 零填充]
      其后 slot_count 个 slot_bytes 槽，槽 i 的帧序 = head 单调整数，
      槽内帧编解码与 ShmFrameChannel 相同。
    """

    def __init__(self, name: str, generation: int, mapping, fh):
        self.name = name
        self.generation = generation
        self.slot_count = ACK_SLOT_COUNT
        self.slot_bytes = ACK_SLOT_BYTES
        self._map = mapping
        self._fh = fh

    @classmethod
    def _validate(cls, mapping, generation, path):
        magic, version, seg_gen, slot_count, slot_bytes = (
            _RING_HEADER.unpack_from(mapping, 0))
        if magic != SEGMENT_MAGIC or version != SEGMENT_VERSION:
            raise BridgeShmError(
                "segment {} is not a bridge shm segment "
                "(magic/version)".format(path))
        if seg_gen != generation:
            raise BridgeShmError(
                "segment {} generation {} != expected {} "
                "(stale segment from another run)".format(
                    path, seg_gen, generation))
        if slot_count != ACK_SLOT_COUNT or slot_bytes != ACK_SLOT_BYTES:
            raise BridgeShmError(
                "segment {} geometry mismatch ({}x{} != {}x{})".format(
                    path, slot_count, slot_bytes,
                    ACK_SLOT_COUNT, ACK_SLOT_BYTES))

    @classmethod
    def open(cls, name: str, generation: int):
        """按名附着既有段并校验段头（世代/几何不符 fail-closed）."""
        if name.startswith("/") or "/" in name:
            raise BridgeShmError("illegal segment name {!r}".format(name))
        path = os.path.join("/dev/shm", name)
        try:
            fh = open(path, "r+b", buffering=0)
        except OSError as exc:
            raise BridgeShmError(
                "cannot open bridge shm segment {}: {}".format(
                    path, exc)) from exc
        try:
            want = SEGMENT_HEADER_BYTES + ACK_SLOT_COUNT * ACK_SLOT_BYTES
            size = os.fstat(fh.fileno()).st_size
            if size != want:
                raise BridgeShmError(
                    "bridge shm segment {} has size {}, expected {}".format(
                        path, size, want))
            mapping = mmap.mmap(fh.fileno(), want)
            cls._validate(mapping, generation, path)
        except Exception:
            fh.close()
            raise
        return cls(name, generation, mapping, fh)

    @classmethod
    def create(cls, name: str, generation: int):
        """创建新段（O_EXCL；存在即错）——生产端对等/夹具用."""
        if name.startswith("/") or "/" in name:
            raise BridgeShmError("illegal segment name {!r}".format(name))
        path = os.path.join("/dev/shm", name)
        try:
            fh = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise BridgeShmError(
                "cannot create bridge shm segment {}: {}".format(
                    path, exc)) from exc
        try:
            want = SEGMENT_HEADER_BYTES + ACK_SLOT_COUNT * ACK_SLOT_BYTES
            os.ftruncate(fh, want)
            mapping = mmap.mmap(fh, want)
            mapping[0:SEGMENT_HEADER_BYTES] = b"\x00" * SEGMENT_HEADER_BYTES
            _RING_HEADER.pack_into(
                mapping, 0, SEGMENT_MAGIC, SEGMENT_VERSION, generation,
                ACK_SLOT_COUNT, ACK_SLOT_BYTES)
            _RING_HEAD.pack_into(mapping, 32, 0)   # head
            _RING_HEAD.pack_into(mapping, 40, 0)   # tail
        except Exception:
            os.close(fh)
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return cls(name, generation, mapping,
                   os.fdopen(fh, "r+b", buffering=0, closefd=True))

    # ------------------------------------------------------------- frames --

    def publish(self, seq: int, payload: bytes) -> None:
        """追加一帧；超槽容量或环满（未消费帧 >= 槽数）fail-closed."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise BridgeShmError("payload must be bytes")
        payload = bytes(payload)
        if len(payload) + _FRAME_HEADER.size + 4 > self.slot_bytes:
            raise BridgeShmError(
                "ack frame of {} bytes exceeds the ack slot size of {} "
                "(fail-closed)".format(len(payload), self.slot_bytes))
        head, = _RING_HEAD.unpack_from(self._map, 32)
        tail, = _RING_HEAD.unpack_from(self._map, 40)  # 可能偏旧=更保守
        if head - tail >= self.slot_count:
            raise BridgeShmError(
                "ack ring full: {} unconsumed ack frames (peer more than "
                "{} acks behind; fail-closed)".format(
                    head - tail, self.slot_count))
        base = SEGMENT_HEADER_BYTES + (head % self.slot_count) * self.slot_bytes
        header = _FRAME_HEADER.pack(FRAME_MAGIC, seq, len(payload))
        self._map[base:base + len(header)] = header
        if payload:
            self._map[base + len(header):base + len(header) +
                      len(payload)] = payload
        crc = zlib.crc32(header + payload) & 0xFFFFFFFF
        offset = base + len(header) + len(payload)
        self._map[offset:offset + 4] = struct.pack("<I", crc)
        _RING_HEAD.pack_into(self._map, 32, head + 1)

    def drain(self, tail: int):
        """取走追加序在 ``tail`` 之后的所有帧，返回 ``[(seq, bytes), ...]``。

        重复 seq 的帧原样包含在内（调用方按 tracker 过滤）；同时推进段内
        tail 水位（写方据此判断槽复用安全）。槽空洞/损坏 fail-closed。
        """
        frames = []
        while True:
            head, = _RING_HEAD.unpack_from(self._map, 32)
            if tail >= head:
                break
            base = SEGMENT_HEADER_BYTES + (tail % self.slot_count) \
                * self.slot_bytes
            magic, seq, length = _FRAME_HEADER.unpack_from(self._map, base)
            if magic == 0:
                raise BridgeShmError(
                    "ack ring hole below head (slot {} empty; writer/head "
                    "desync)".format(tail % self.slot_count))
            if magic != FRAME_MAGIC:
                raise BridgeShmError(
                    "ack slot holds a corrupt frame (magic {})".format(magic))
            if length + _FRAME_HEADER.size + 4 > self.slot_bytes:
                raise BridgeShmError(
                    "ack frame length {} exceeds the slot size".format(length))
            span = base + _FRAME_HEADER.size + length
            stored_crc, = struct.unpack_from("<I", self._map, span)
            actual_crc = zlib.crc32(self._map[base:span]) & 0xFFFFFFFF
            if stored_crc != actual_crc:
                raise BridgeShmError(
                    "ack frame checksum mismatch (stored {}, computed "
                    "{})".format(stored_crc, actual_crc))
            frames.append((seq, self._map[base + _FRAME_HEADER.size:span]))
            tail += 1
            _RING_HEAD.pack_into(self._map, 40, tail)
        return frames

    # ---------------------------------------------------------- lifecycle --

    def close(self) -> None:
        if self._map is not None:
            try:
                self._map.close()
            finally:
                self._map = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False
