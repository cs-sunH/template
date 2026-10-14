/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ShmFrameChannel -- POSIX-shm single-slot frame channel (joint v11 §8 保底).

The decision bridge's byte carrier (v11 fallback path): one named
/tmp-shm segment per frame direction, holding ONE frame slot. This is the
transport layer only; ShmDecisionBridge (DecisionBridge.hh) owns the
protocol (FIFO doorbells, seq handshake, fail-closed semantics).

Wire layout (frozen; mirrored byte-for-byte by
sh_test_mesh/.../online/bridge_shm.py):

  segment file  = 64-byte header + slot (slot_capacity bytes of payload room)
    header: u32 magic | u32 version | u64 generation | u64 slot_capacity
            (all little-endian; bytes 24..63 are zero padding)
    slot:   [u32 frame_magic][u64 seq][u32 len][payload len bytes][u32 crc32]
            crc32 (IEEE, zlib polynomial) over the 16 frame-header bytes
            concatenated with the payload.

Concurrency model: NO atomics. The bridge's FIFO doorbell is the
synchronization point -- the writer publishes the whole frame and only then
writes the 1-byte notification; the reader reads the frame only after the
blocking/poll-validated read of that byte returned. The frozen 1:1
backpressure (one in-flight request per epoch; the writer of a direction
never publishes frame N+1 before the peer consumed frame N) additionally
guarantees a published frame is never overwritten while unread.

Duplicate/idempotency semantics mirror the loose-file protocol: the slot
retains the last published frame, and try_read(expect_seq) treats
seq < expect_seq as an already-consumed duplicate (returns false), so a
duplicate doorbell byte is a no-op exactly like a rewritten
commit_ack_<seq>.json file was under the contains() filter.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_SHMFRAMECHANNEL_HH
#define EXECUTION_DRIVEN_SHMFRAMECHANNEL_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {

/// Segment header magic ("1BHS" little-endian) and frame-slot magic
/// ("1MRF"). A zeroed (never-published) slot reads magic == 0 == "empty".
inline constexpr uint32_t kBridgeShmSegmentMagic = 0x53484231u;
inline constexpr uint32_t kBridgeShmSegmentVersion = 1u;
inline constexpr uint32_t kBridgeShmFrameMagic = 0x46524D31u;
inline constexpr size_t kBridgeShmHeaderBytes = 64;
inline constexpr size_t kBridgeShmFrameHeaderBytes = 16;  // magic+seq+len

/// Payload room of one slot. 256 MiB frames: the largest observed delivery
/// payloads are MB-scale (workledger-compact frames, GraphBatch responses);
/// tmpfs commits pages only on first touch, so the untouched tail of a
/// 256 MiB ftruncate costs no memory. A larger frame is a producer bug and
/// fails closed with a clear error (never a silent truncation).
inline constexpr uint64_t kBridgeShmSlotCapacity = 256ull * 1024 * 1024;

/// The commit-ack direction is NOT exchange-gated: C++ may publish several
/// acks back-to-back (the frozen protocol has no ack-of-ack; the historical
/// fixture sends ack0, ack1, duplicate ack0 with no waiting). A single slot
/// would silently drop the middle frames, so the ack channel is a fixed
/// multi-slot ring instead: slot_count slots of slot_bytes each, with
/// monotonic head (writer) / tail (reader) indices in the segment header.
/// Duplicate acks are simply appended again -- the reader skips frames
/// whose seq is below its tracker (idempotent ignore, the loose-file
/// contains() filter's exact analogue). Ring overflow (peer more than
/// slot_count acks behind) fails closed; ack payloads are ~100B, so 4096
/// slots cover a ~400-frame-per-epoch-scale backlog many orders over.
inline constexpr uint64_t kBridgeShmAckSlotCount = 4096;
inline constexpr uint64_t kBridgeShmAckSlotBytes = 8192;

/// Segment-name stem for one bridge dir: "astra_brg_<fnv1a64(dir):016x>_".
/// Stale segments from a crashed previous run are unlinked by this prefix.
std::string bridge_shm_name_stem(const std::string& bridge_dir);

/// A fresh startup generation (pid + realtime nanoseconds mixed; unique per
/// ensure_bridge_dir call). Python validates it against bridge_shm.json so
/// a stale segment from another run can never be mistaken for this run's.
uint64_t new_bridge_shm_generation();

/// Best-effort shm_unlink of every segment carrying this bridge dir's stem
/// prefix (any role, any generation). Called by ensure_bridge_dir before
/// creating the new generation.
void clean_stale_bridge_shm(const std::string& bridge_dir);

/// IEEE crc32 (zlib polynomial) of a byte range -- the frame checksum.
uint32_t bridge_shm_crc32(const uint8_t* data, size_t len);

/// One single-slot shm frame channel. Move-only; the destructor unmaps.
/// All failures throw std::runtime_error with a diagnostic message; the
/// bridge layer converts them to bridge_fatal (fail-closed).
class ShmFrameChannel {
  public:
    ShmFrameChannel() = default;
    ~ShmFrameChannel();
    ShmFrameChannel(ShmFrameChannel&& other) noexcept;
    ShmFrameChannel& operator=(ShmFrameChannel&& other) noexcept;
    ShmFrameChannel(const ShmFrameChannel&) = delete;
    ShmFrameChannel& operator=(const ShmFrameChannel&) = delete;

    /// Producer side: create a fresh segment (fails if the name exists).
    static ShmFrameChannel create(const std::string& shm_name,
                                  uint64_t generation,
                                  uint64_t slot_capacity);

    /// Peer side: attach to an existing segment and validate the header
    /// (magic / version / generation / capacity). A stale generation or
    /// mismatched capacity throws -- this is the cross-run pollution guard.
    static ShmFrameChannel attach(const std::string& shm_name,
                                  uint64_t generation,
                                  uint64_t slot_capacity);

    /// Publish one frame. The payload crosses EXACTLY as given (the old
    /// protocol's file bytes). Throws when len exceeds slot_capacity.
    void publish(uint64_t seq, const void* data, uint32_t len);
    void publish(uint64_t seq, const std::string& body);

    /// Read the frame parked in the slot.
    ///  - returns false when the slot is empty or holds seq < expect_seq
    ///    (an already-consumed duplicate under the strict monotonic
    ///    protocol -- idempotent ignore);
    ///  - throws on corruption (bad slot magic / len > capacity / crc
    ///    mismatch);
    ///  - returns true with the payload for seq >= expect_seq. A FUTURE
    ///    seq (> expect) also returns true: the caller's sequence tracker
    ///    fail-closes it, mirroring the old listdir fallback discovering a
    ///    future request_<seq>.json.
    bool try_read(uint64_t expect_seq, std::string* payload,
                  uint64_t* seq_out) const;

    void close();

    bool valid() const { return base_ != nullptr; }
    const std::string& name() const { return name_; }
    uint64_t generation() const { return generation_; }
    uint64_t slot_capacity() const { return slot_capacity_; }

  private:
    static ShmFrameChannel open_internal(const std::string& shm_name,
                                         uint64_t generation,
                                         uint64_t slot_capacity, bool create);

    std::string name_;
    uint64_t generation_ = 0;
    uint64_t slot_capacity_ = 0;
    void* base_ = nullptr;   // mmap base (header at offset 0, slot at 64)
    size_t map_bytes_ = 0;
    int fd_ = -1;
};

/// Multi-slot SPSC ack ring (see kBridgeShmAckSlotCount above for why the
/// ack direction is a ring, not a single slot). Layout: 64-byte header
/// [u32 magic][u32 version][u64 generation][u64 slot_count][u64 slot_bytes]
/// [u64 head][u64 tail][16B zero pad], then slot_count slots of slot_bytes
/// each holding one frame with the same per-frame codec as ShmFrameChannel.
/// head/tail are monotonic append/consume indices (never reset); the slot
/// for index i lives at 64 + (i % slot_count) * slot_bytes.
class ShmAckRing {
  public:
    ShmAckRing() = default;
    ~ShmAckRing();
    ShmAckRing(ShmAckRing&& other) noexcept;
    ShmAckRing& operator=(ShmAckRing&& other) noexcept;
    ShmAckRing(const ShmAckRing&) = delete;
    ShmAckRing& operator=(const ShmAckRing&) = delete;

    static ShmAckRing create(const std::string& shm_name,
                             uint64_t generation, uint64_t slot_count,
                             uint64_t slot_bytes);
    static ShmAckRing attach(const std::string& shm_name,
                             uint64_t generation, uint64_t slot_count,
                             uint64_t slot_bytes);

    /// Append one ack frame. Throws when the payload exceeds slot_bytes or
    /// the ring is full (unconsumed frames >= slot_count) -- both are
    /// producer-side fail-closed conditions.
    void publish(uint64_t seq, const void* data, uint32_t len);
    void publish(uint64_t seq, const std::string& body);

    /// Drain every frame appended after *tail_io (the caller's persisted
    /// consume index), advancing *tail_io and the segment's tail watermark
    /// past each one. Returns the frames IN APPEND ORDER including
    /// duplicates (seq below the caller's tracker) -- filtering duplicates
    /// is the caller's policy, exactly like the loose-file probe path.
    /// Throws on corruption (empty slot below head, bad frame).
    std::vector<std::pair<uint64_t, std::string>> drain(uint64_t* tail_io);

    void close();
    bool valid() const { return base_ != nullptr; }
    const std::string& name() const { return name_; }
    uint64_t generation() const { return generation_; }
    uint64_t slot_count() const { return slot_count_; }
    uint64_t slot_bytes() const { return slot_bytes_; }

  private:
    static ShmAckRing open_internal(const std::string& shm_name,
                                    uint64_t generation, uint64_t slot_count,
                                    uint64_t slot_bytes, bool create);

    std::string name_;
    uint64_t generation_ = 0;
    uint64_t slot_count_ = 0;
    uint64_t slot_bytes_ = 0;
    void* base_ = nullptr;
    size_t map_bytes_ = 0;
    int fd_ = -1;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_SHMFRAMECHANNEL_HH
