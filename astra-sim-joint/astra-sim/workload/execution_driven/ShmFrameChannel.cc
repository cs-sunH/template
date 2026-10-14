/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ShmFrameChannel -- POSIX-shm single-slot frame channel (joint v11 §8 保底).
Transport implementation; see ShmFrameChannel.hh for the frozen wire layout
and the doorball synchronization argument.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/ShmFrameChannel.hh"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

void store_u32(uint8_t* dst, const uint32_t v) {
    std::memcpy(dst, &v, sizeof(v));
}

void store_u64(uint8_t* dst, const uint64_t v) {
    std::memcpy(dst, &v, sizeof(v));
}

uint32_t load_u32(const uint8_t* src) {
    uint32_t v = 0;
    std::memcpy(&v, src, sizeof(v));
    return v;
}

uint64_t load_u64(const uint8_t* src) {
    uint64_t v = 0;
    std::memcpy(&v, src, sizeof(v));
    return v;
}

[[noreturn]] void channel_fatal(const std::string& what) {
    throw std::runtime_error("bridge shm channel: " + what);
}

// CRC-32 (IEEE 802.3 / zlib polynomial 0xEDB88320), table-driven. The
// Python peer uses zlib.crc32 over the identical byte span.
struct Crc32Table {
    uint32_t v[256];
    Crc32Table() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            v[i] = c;
        }
    }
};
const Crc32Table g_crc32_table;

}  // namespace

uint32_t bridge_shm_crc32(const uint8_t* data, const size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc = g_crc32_table.v[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

std::string bridge_shm_name_stem(const std::string& bridge_dir) {
    // FNV-1a 64-bit over the directory path bytes -- identical constants
    // on the Python side (bridge_shm.py); the stem only needs to be unique
    // per bridge dir and stable across the two peers.
    uint64_t hash = 14695981039346656037ull;
    for (const char c : bridge_dir) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ull;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(hash));
    return std::string("astra_brg_") + buf + "_";
}

uint64_t new_bridge_shm_generation() {
    struct timespec ts {};
    ::clock_gettime(CLOCK_REALTIME, &ts);
    const uint64_t nanos =
        static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
        static_cast<uint64_t>(ts.tv_nsec);
    // Mix the pid so two ensures within one clock tick (or one process
    // re-ensuring after a crash) still mint distinct generations.
    return nanos ^ (static_cast<uint64_t>(::getpid()) << 32);
}

void clean_stale_bridge_shm(const std::string& bridge_dir) {
    const std::string stem = bridge_shm_name_stem(bridge_dir);
    DIR* dir = ::opendir("/dev/shm");
    if (dir == nullptr) {
        // No tmpfs shm namespace: nothing to clean.
        return;
    }
    std::vector<std::string> doomed;
    while (struct dirent* entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (name.compare(0, stem.size(), stem) == 0) {
            doomed.push_back(name);
        }
    }
    ::closedir(dir);
    for (const std::string& name : doomed) {
        const std::string path = std::string("/dev/shm/") + name;
        if (::unlink(path.c_str()) != 0 && errno != ENOENT) {
            channel_fatal("clean_stale: unlink " + path + ": " +
                          std::strerror(errno));
        }
    }
}

// ------------------------------------------------------------- lifecycle --

ShmFrameChannel::ShmFrameChannel(ShmFrameChannel&& other) noexcept
    : name_(std::move(other.name_)),
      generation_(other.generation_),
      slot_capacity_(other.slot_capacity_),
      base_(other.base_),
      map_bytes_(other.map_bytes_),
      fd_(other.fd_) {
    other.base_ = nullptr;
    other.map_bytes_ = 0;
    other.fd_ = -1;
    other.generation_ = 0;
    other.slot_capacity_ = 0;
}

ShmFrameChannel& ShmFrameChannel::operator=(ShmFrameChannel&& other) noexcept {
    if (this != &other) {
        close();
        name_ = std::move(other.name_);
        generation_ = other.generation_;
        slot_capacity_ = other.slot_capacity_;
        base_ = other.base_;
        map_bytes_ = other.map_bytes_;
        fd_ = other.fd_;
        other.base_ = nullptr;
        other.map_bytes_ = 0;
        other.fd_ = -1;
        other.generation_ = 0;
        other.slot_capacity_ = 0;
    }
    return *this;
}

ShmFrameChannel::~ShmFrameChannel() {
    close();
}

void ShmFrameChannel::close() {
    if (base_ != nullptr) {
        ::munmap(base_, map_bytes_);
        base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    map_bytes_ = 0;
}

ShmFrameChannel ShmFrameChannel::create(const std::string& shm_name,
                                        const uint64_t generation,
                                        const uint64_t slot_capacity) {
    return open_internal(shm_name, generation, slot_capacity, /*create=*/true);
}

ShmFrameChannel ShmFrameChannel::attach(const std::string& shm_name,
                                        const uint64_t generation,
                                        const uint64_t slot_capacity) {
    return open_internal(shm_name, generation, slot_capacity, /*create=*/false);
}

ShmFrameChannel ShmFrameChannel::open_internal(const std::string& shm_name,
                                               const uint64_t generation,
                                               const uint64_t slot_capacity,
                                               const bool create) {
    if (shm_name.empty() || shm_name.front() != '/' ||
        shm_name.find('/', 1) != std::string::npos) {
        channel_fatal("illegal segment name '" + shm_name + "'");
    }
    if (slot_capacity == 0) {
        channel_fatal("slot capacity must be positive");
    }
    ShmFrameChannel ch;
    ch.name_ = shm_name;
    ch.generation_ = generation;
    ch.slot_capacity_ = slot_capacity;
    const int flags = O_RDWR | (create ? (O_CREAT | O_EXCL) : 0);
    ch.fd_ = ::shm_open(shm_name.c_str(), flags, 0600);
    if (ch.fd_ < 0) {
        channel_fatal(std::string(create ? "shm_open(create) " : "shm_open ") +
                      shm_name + ": " + std::strerror(errno));
    }
    struct stat st {};
    if (::fstat(ch.fd_, &st) != 0) {
        channel_fatal("fstat " + shm_name + ": " + std::strerror(errno));
    }
    const size_t want = kBridgeShmHeaderBytes + slot_capacity;
    if (create) {
        if (::ftruncate(ch.fd_, static_cast<off_t>(want)) != 0) {
            channel_fatal("ftruncate " + shm_name + ": " +
                          std::strerror(errno));
        }
    } else if (st.st_size != static_cast<off_t>(want)) {
        channel_fatal("segment " + shm_name + " has size " +
                      std::to_string(st.st_size) + ", expected " +
                      std::to_string(want));
    }
    void* base = ::mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED,
                        ch.fd_, 0);
    if (base == MAP_FAILED) {
        channel_fatal("mmap " + shm_name + ": " + std::strerror(errno));
    }
    ch.base_ = base;
    ch.map_bytes_ = want;
    if (create) {
        std::memset(base, 0, kBridgeShmHeaderBytes);
        uint8_t* header = static_cast<uint8_t*>(base);
        store_u32(header + 0, kBridgeShmSegmentMagic);
        store_u32(header + 4, kBridgeShmSegmentVersion);
        store_u64(header + 8, generation);
        store_u64(header + 16, slot_capacity);
    } else {
        const uint8_t* header = static_cast<const uint8_t*>(base);
        if (load_u32(header + 0) != kBridgeShmSegmentMagic ||
            load_u32(header + 4) != kBridgeShmSegmentVersion) {
            channel_fatal("segment " + shm_name +
                          " is not a bridge shm segment (magic/version)");
        }
        const uint64_t seg_gen = load_u64(header + 8);
        if (seg_gen != generation) {
            channel_fatal("segment " + shm_name + " generation " +
                          std::to_string(seg_gen) + " != expected " +
                          std::to_string(generation) +
                          " (stale segment from another run)");
        }
        if (load_u64(header + 16) != slot_capacity) {
            channel_fatal("segment " + shm_name + " capacity mismatch");
        }
    }
    return ch;
}

// ---------------------------------------------------------------- frames --

void ShmFrameChannel::publish(const uint64_t seq, const void* data,
                              const uint32_t len) {
    if (base_ == nullptr) {
        channel_fatal("publish on a closed channel");
    }
    if (static_cast<uint64_t>(len) > slot_capacity_) {
        channel_fatal("frame of " + std::to_string(len) +
                      " bytes exceeds the shm slot capacity of " +
                      std::to_string(slot_capacity_) +
                      " (delivery payload too large; fail-closed)");
    }
    uint8_t* slot = static_cast<uint8_t*>(base_) + kBridgeShmHeaderBytes;
    // Frame header first, payload second, crc last. The FIFO doorbell
    // written by the bridge AFTER this returns is the reader's
    // synchronization point, so the store order inside the slot needs no
    // fences.
    uint8_t header[kBridgeShmFrameHeaderBytes];
    store_u32(header + 0, kBridgeShmFrameMagic);
    store_u64(header + 4, seq);
    store_u32(header + 12, len);
    std::memcpy(slot, header, sizeof(header));
    if (len > 0) {
        std::memcpy(slot + sizeof(header), data, len);
    }
    // header || payload is contiguous in the slot; crc over it in place.
    const size_t crc_span = kBridgeShmFrameHeaderBytes + len;
    store_u32(slot + crc_span, bridge_shm_crc32(slot, crc_span));
}

void ShmFrameChannel::publish(const uint64_t seq, const std::string& body) {
    // Guard BEFORE the uint32 cast: a >4GiB body must fail closed on the
    // capacity check, never silently truncate through the narrowing cast.
    if (base_ == nullptr) {
        channel_fatal("publish on a closed channel");
    }
    if (body.size() > slot_capacity_) {
        channel_fatal("frame of " + std::to_string(body.size()) +
                      " bytes exceeds the shm slot capacity of " +
                      std::to_string(slot_capacity_) +
                      " (delivery payload too large; fail-closed)");
    }
    publish(seq, body.data(), static_cast<uint32_t>(body.size()));
}

bool ShmFrameChannel::try_read(const uint64_t expect_seq,
                               std::string* payload,
                               uint64_t* seq_out) const {
    if (base_ == nullptr) {
        channel_fatal("try_read on a closed channel");
    }
    const uint8_t* slot =
        static_cast<const uint8_t*>(base_) + kBridgeShmHeaderBytes;
    const uint32_t magic = load_u32(slot + 0);
    if (magic == 0) {
        return false;  // never-published slot (doorbell without a frame)
    }
    if (magic != kBridgeShmFrameMagic) {
        channel_fatal("slot holds a corrupt frame (magic " +
                      std::to_string(magic) + ")");
    }
    const uint64_t seq = load_u64(slot + 4);
    if (seq < expect_seq) {
        return false;  // already-consumed duplicate; idempotent ignore
    }
    const uint32_t len = load_u32(slot + 12);
    if (static_cast<uint64_t>(len) > slot_capacity_) {
        channel_fatal("frame length " + std::to_string(len) +
                      " exceeds the slot capacity");
    }
    const size_t crc_span = kBridgeShmFrameHeaderBytes + len;
    const uint32_t crc_stored = load_u32(slot + crc_span);
    const uint32_t crc_actual =
        bridge_shm_crc32(slot, crc_span);
    if (crc_stored != crc_actual) {
        channel_fatal("frame checksum mismatch (stored " +
                      std::to_string(crc_stored) + ", computed " +
                      std::to_string(crc_actual) + ")");
    }
    payload->assign(reinterpret_cast<const char*>(slot +
                                                  kBridgeShmFrameHeaderBytes),
                    len);
    if (seq_out != nullptr) {
        *seq_out = seq;
    }
    return true;
}

// ------------------------------------------------------------ ShmAckRing --

namespace {

// Ack-ring header field offsets (little-endian u64s; bytes 48..63 zero).
constexpr size_t kRingOffSlotCount = 16;
constexpr size_t kRingOffSlotBytes = 24;
constexpr size_t kRingOffHead = 32;
constexpr size_t kRingOffTail = 40;

}  // namespace

ShmAckRing::ShmAckRing(ShmAckRing&& other) noexcept
    : name_(std::move(other.name_)),
      generation_(other.generation_),
      slot_count_(other.slot_count_),
      slot_bytes_(other.slot_bytes_),
      base_(other.base_),
      map_bytes_(other.map_bytes_),
      fd_(other.fd_) {
    other.base_ = nullptr;
    other.map_bytes_ = 0;
    other.fd_ = -1;
    other.generation_ = 0;
    other.slot_count_ = 0;
    other.slot_bytes_ = 0;
}

ShmAckRing& ShmAckRing::operator=(ShmAckRing&& other) noexcept {
    if (this != &other) {
        close();
        name_ = std::move(other.name_);
        generation_ = other.generation_;
        slot_count_ = other.slot_count_;
        slot_bytes_ = other.slot_bytes_;
        base_ = other.base_;
        map_bytes_ = other.map_bytes_;
        fd_ = other.fd_;
        other.base_ = nullptr;
        other.map_bytes_ = 0;
        other.fd_ = -1;
        other.generation_ = 0;
        other.slot_count_ = 0;
        other.slot_bytes_ = 0;
    }
    return *this;
}

ShmAckRing::~ShmAckRing() {
    close();
}

void ShmAckRing::close() {
    if (base_ != nullptr) {
        ::munmap(base_, map_bytes_);
        base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    map_bytes_ = 0;
}

ShmAckRing ShmAckRing::create(const std::string& shm_name,
                              const uint64_t generation,
                              const uint64_t slot_count,
                              const uint64_t slot_bytes) {
    return open_internal(shm_name, generation, slot_count, slot_bytes, true);
}

ShmAckRing ShmAckRing::attach(const std::string& shm_name,
                              const uint64_t generation,
                              const uint64_t slot_count,
                              const uint64_t slot_bytes) {
    return open_internal(shm_name, generation, slot_count, slot_bytes, false);
}

ShmAckRing ShmAckRing::open_internal(const std::string& shm_name,
                                     const uint64_t generation,
                                     const uint64_t slot_count,
                                     const uint64_t slot_bytes,
                                     const bool create) {
    if (shm_name.empty() || shm_name.front() != '/' ||
        shm_name.find('/', 1) != std::string::npos) {
        channel_fatal("illegal segment name '" + shm_name + "'");
    }
    if (slot_count == 0 || slot_bytes < kBridgeShmFrameHeaderBytes + 4) {
        channel_fatal("illegal ack ring geometry");
    }
    ShmAckRing ring;
    ring.name_ = shm_name;
    ring.generation_ = generation;
    ring.slot_count_ = slot_count;
    ring.slot_bytes_ = slot_bytes;
    const int flags = O_RDWR | (create ? (O_CREAT | O_EXCL) : 0);
    ring.fd_ = ::shm_open(shm_name.c_str(), flags, 0600);
    if (ring.fd_ < 0) {
        channel_fatal(std::string(create ? "shm_open(create) " : "shm_open ") +
                      shm_name + ": " + std::strerror(errno));
    }
    const size_t want = kBridgeShmHeaderBytes + slot_count * slot_bytes;
    if (create) {
        if (::ftruncate(ring.fd_, static_cast<off_t>(want)) != 0) {
            channel_fatal("ftruncate " + shm_name + ": " +
                          std::strerror(errno));
        }
    } else {
        struct stat st {};
        if (::fstat(ring.fd_, &st) != 0) {
            channel_fatal("fstat " + shm_name + ": " + std::strerror(errno));
        }
        if (st.st_size != static_cast<off_t>(want)) {
            channel_fatal("segment " + shm_name + " has size " +
                          std::to_string(st.st_size) + ", expected " +
                          std::to_string(want));
        }
    }
    void* base = ::mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED,
                        ring.fd_, 0);
    if (base == MAP_FAILED) {
        channel_fatal("mmap " + shm_name + ": " + std::strerror(errno));
    }
    ring.base_ = base;
    ring.map_bytes_ = want;
    if (create) {
        std::memset(base, 0, kBridgeShmHeaderBytes);
        uint8_t* header = static_cast<uint8_t*>(base);
        store_u32(header + 0, kBridgeShmSegmentMagic);
        store_u32(header + 4, kBridgeShmSegmentVersion);
        store_u64(header + 8, generation);
        store_u64(header + kRingOffSlotCount, slot_count);
        store_u64(header + kRingOffSlotBytes, slot_bytes);
        store_u64(header + kRingOffHead, 0);
        store_u64(header + kRingOffTail, 0);
    } else {
        const uint8_t* header = static_cast<const uint8_t*>(base);
        if (load_u32(header + 0) != kBridgeShmSegmentMagic ||
            load_u32(header + 4) != kBridgeShmSegmentVersion) {
            channel_fatal("segment " + shm_name +
                          " is not a bridge shm segment (magic/version)");
        }
        const uint64_t seg_gen = load_u64(header + 8);
        if (seg_gen != generation) {
            channel_fatal("segment " + shm_name + " generation " +
                          std::to_string(seg_gen) + " != expected " +
                          std::to_string(generation) +
                          " (stale segment from another run)");
        }
        if (load_u64(header + kRingOffSlotCount) != slot_count ||
            load_u64(header + kRingOffSlotBytes) != slot_bytes) {
            channel_fatal("segment " + shm_name + " geometry mismatch");
        }
    }
    return ring;
}

void ShmAckRing::publish(const uint64_t seq, const void* data,
                         const uint32_t len) {
    if (base_ == nullptr) {
        channel_fatal("publish on a closed ring");
    }
    if (static_cast<uint64_t>(len) + kBridgeShmFrameHeaderBytes + 4 >
        slot_bytes_) {
        channel_fatal("ack frame of " + std::to_string(len) +
                      " bytes exceeds the ack slot size of " +
                      std::to_string(slot_bytes_) + " (fail-closed)");
    }
    const uint8_t* header = static_cast<const uint8_t*>(base_);
    const uint64_t head = load_u64(header + kRingOffHead);
    const uint64_t tail = load_u64(header + kRingOffTail);  // may be stale;
    // a stale (smaller) tail only shrinks the apparent free space.
    if (head - tail >= slot_count_) {
        channel_fatal("ack ring full: " + std::to_string(head - tail) +
                      " unconsumed ack frames (peer more than " +
                      std::to_string(slot_count_) + " acks behind; " +
                      "fail-closed)");
    }
    uint8_t* slot = static_cast<uint8_t*>(base_) + kBridgeShmHeaderBytes +
                    (head % slot_count_) * slot_bytes_;
    uint8_t frame_header[kBridgeShmFrameHeaderBytes];
    store_u32(frame_header + 0, kBridgeShmFrameMagic);
    store_u64(frame_header + 4, seq);
    store_u32(frame_header + 12, len);
    std::memcpy(slot, frame_header, sizeof(frame_header));
    if (len > 0) {
        std::memcpy(slot + sizeof(frame_header), data, len);
    }
    const size_t crc_span = kBridgeShmFrameHeaderBytes + len;
    store_u32(slot + crc_span, bridge_shm_crc32(slot, crc_span));
    store_u64(static_cast<uint8_t*>(base_) + kRingOffHead, head + 1);
}

void ShmAckRing::publish(const uint64_t seq, const std::string& body) {
    if (body.size() + kBridgeShmFrameHeaderBytes + 4 > slot_bytes_) {
        channel_fatal("ack frame of " + std::to_string(body.size()) +
                      " bytes exceeds the ack slot size of " +
                      std::to_string(slot_bytes_) + " (fail-closed)");
    }
    publish(seq, body.data(), static_cast<uint32_t>(body.size()));
}

std::vector<std::pair<uint64_t, std::string>> ShmAckRing::drain(
    uint64_t* tail_io) {
    if (base_ == nullptr) {
        channel_fatal("drain on a closed ring");
    }
    std::vector<std::pair<uint64_t, std::string>> frames;
    while (true) {
        const uint8_t* header = static_cast<const uint8_t*>(base_);
        const uint64_t head = load_u64(header + kRingOffHead);
        if (*tail_io >= head) {
            break;
        }
        const uint8_t* slot =
            static_cast<const uint8_t*>(base_) + kBridgeShmHeaderBytes +
            (*tail_io % slot_count_) * slot_bytes_;
        const uint32_t magic = load_u32(slot + 0);
        if (magic == 0) {
            channel_fatal("ack ring hole below head (slot " +
                          std::to_string(*tail_io % slot_count_) +
                          " empty; writer/head desync)");
        }
        if (magic != kBridgeShmFrameMagic) {
            channel_fatal("ack slot holds a corrupt frame (magic " +
                          std::to_string(magic) + ")");
        }
        const uint64_t seq = load_u64(slot + 4);
        const uint32_t len = load_u32(slot + 12);
        if (static_cast<uint64_t>(len) + kBridgeShmFrameHeaderBytes + 4 >
            slot_bytes_) {
            channel_fatal("ack frame length " + std::to_string(len) +
                          " exceeds the slot size");
        }
        const size_t crc_span = kBridgeShmFrameHeaderBytes + len;
        if (load_u32(slot + crc_span) != bridge_shm_crc32(slot, crc_span)) {
            channel_fatal("ack frame checksum mismatch");
        }
        frames.emplace_back(
            seq, std::string(reinterpret_cast<const char*>(
                                 slot + kBridgeShmFrameHeaderBytes),
                             len));
        *tail_io += 1;
        store_u64(static_cast<uint8_t*>(base_) + kRingOffTail, *tail_io);
    }
    return frames;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
