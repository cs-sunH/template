/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef ASTRA_SIM_WORKLOAD_WORKSPACE_ADMISSION_HH
#define ASTRA_SIM_WORKLOAD_WORKSPACE_ADMISSION_HH

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AstraSim {

/// Absolute transient workspace held by one C++ rank.  Python echoes the
/// revision in its post-delivery KV capacity snapshot.
struct WorkspaceRankState {
    int rank = -1;
    uint64_t workspace_revision = 0;
    uint64_t active_workspace_bytes = 0;
};

/// Python's authoritative non-workspace HBM capacity split for one rank.
struct WorkspaceCapacityRank {
    int rank = -1;
    uint64_t workspace_revision = 0;
    uint64_t capacity_bytes = 0;
    uint64_t model_weight_bytes = 0;
    uint64_t resident_kv_bytes = 0;
    uint64_t reserved_request_bytes = 0;
    // Optional immutable bound for one COMP on the serialized compute gate.
    // This is a fit margin, never an active lease or a KV reservation.
    bool has_prospective_workspace = false;
    uint64_t prospective_workspace_bytes = 0;
};

/// Checked rank-local ledger for short-lived immutable-Plan workspace.
/// The simulation event loop is single-threaded; this type deliberately
/// requires all operations to run on that thread.
class WorkspaceAdmission final {
  public:
    struct LeaseKey {
        int rank = -1;
        uint64_t node_id = 0;
        uint64_t generation = 0;
        uint64_t lease_id = 0;
    };

    class Lease final {
      public:
        Lease() = default;
        ~Lease() noexcept { release_on_destroy(); }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept { move_from(other); }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release_on_destroy();
                move_from(other);
            }
            return *this;
        }

        /// Release exactly once.  A second explicit release is an invariant
        /// failure; destruction of an already released token is harmless.
        void release() {
            if (owner_ == nullptr) {
                throw std::runtime_error(
                    "workspace lease released more than once or is empty");
            }
            owner_->release(key_);
            owner_ = nullptr;
            released_ = true;
        }

        bool active() const { return owner_ != nullptr; }
        const LeaseKey& key() const { return key_; }

      private:
        friend class WorkspaceAdmission;
        Lease(WorkspaceAdmission* owner, LeaseKey key)
            : owner_(owner), key_(key) {}

        void move_from(Lease& other) noexcept {
            owner_ = other.owner_;
            key_ = other.key_;
            released_ = other.released_;
            other.owner_ = nullptr;
            other.released_ = true;
        }

        void release_on_destroy() noexcept {
            if (owner_ == nullptr) return;
            try {
                owner_->release(key_);
            } catch (...) {
                // Losing accounting while an execution may still be active
                // would permit overcommit.  There is no safe recovery from a
                // broken owner invariant during stack unwinding.
                std::terminate();
            }
            owner_ = nullptr;
            released_ = true;
        }

        WorkspaceAdmission* owner_ = nullptr;
        LeaseKey key_{};
        bool released_ = false;
    };

    enum class AcquireStatus { Acquired, Blocked, Impossible };
    struct AcquireResult {
        AcquireStatus status = AcquireStatus::Blocked;
        Lease lease;
    };

    explicit WorkspaceAdmission(const int rank) : rank_(rank) {
        if (rank < 0) {
            throw std::invalid_argument("workspace rank must be nonnegative");
        }
    }

    WorkspaceAdmission(const WorkspaceAdmission&) = delete;
    WorkspaceAdmission& operator=(const WorkspaceAdmission&) = delete;

    int rank() const { return rank_; }
    uint64_t workspace_revision() const { return workspace_revision_; }
    uint64_t active_workspace_bytes() const { return active_workspace_bytes_; }
    bool has_capacity_snapshot() const { return has_capacity_snapshot_; }
    std::size_t active_lease_count() const { return active_.size(); }

    /// Check token ownership and its exact live ledger record. Rank/id
    /// equality alone cannot distinguish a token from another ledger.
    bool owns_lease(const Lease& lease, const uint64_t workspace_bytes) const {
        if (lease.owner_ != this || lease.key_.rank != rank_) return false;
        const auto found = active_.find(
            Identity{lease.key_.node_id, lease.key_.generation});
        return found != active_.end() &&
               found->second.lease_id == lease.key_.lease_id &&
               found->second.workspace_bytes == workspace_bytes;
    }

    /// Return the installed Python capacity fields without mutation. The
    /// separate has_capacity_snapshot() bit is the authority indicator;
    /// workspace_revision() is the current lease-ledger revision, while the
    /// returned revision is the one echoed by the installed snapshot.
    WorkspaceCapacityRank capacity_snapshot_view() const {
        return WorkspaceCapacityRank{rank_, capacity_snapshot_revision_,
                                     capacity_bytes_, model_weight_bytes_,
                                     resident_kv_bytes_,
                                     reserved_request_bytes_,
                                     has_prospective_workspace_,
                                     prospective_workspace_bytes_};
    }

    WorkspaceRankState state() const {
        return WorkspaceRankState{rank_, workspace_revision_,
                                  active_workspace_bytes_};
    }

    void validate_request_state(const WorkspaceRankState& request) const {
        if (request.rank != rank_ ||
            request.workspace_revision != workspace_revision_ ||
            request.active_workspace_bytes != active_workspace_bytes_) {
            throw std::runtime_error(
                "workspace request view does not match rank-local ledger " +
                std::to_string(rank_));
        }
    }

    /// Compile/prepare is deliberately separate from acquisition.  The
    /// caller may try the same immutable Plan again after a later snapshot.
    AcquireResult try_acquire(const uint64_t node_id,
                              const uint64_t generation,
                              const uint64_t workspace_bytes) {
        if (!has_capacity_snapshot_) {
            return {AcquireStatus::Blocked, Lease{}};
        }
        if (has_prospective_workspace_ &&
            workspace_bytes > prospective_workspace_bytes_) {
            throw std::runtime_error(
                "coarse Plan workspace exceeds the immutable prospective "
                "workspace bound on rank " + std::to_string(rank_));
        }
        const uint64_t maximum_for_empty_kv =
            capacity_bytes_ - model_weight_bytes_;
        if (workspace_bytes > maximum_for_empty_kv) {
            return {AcquireStatus::Impossible, Lease{}};
        }
        const uint64_t base_used = checked_base_used();
        if (base_used > capacity_bytes_) {
            throw std::logic_error(
                "workspace capacity view exceeds rank capacity");
        }
        const uint64_t base_remaining = capacity_bytes_ - base_used;
        if (active_workspace_bytes_ > base_remaining ||
            workspace_bytes > base_remaining - active_workspace_bytes_) {
            return {AcquireStatus::Blocked, Lease{}};
        }

        const Identity identity{node_id, generation};
        if (active_.find(identity) != active_.end()) {
            throw std::logic_error(
                "workspace lease already exists for rank/node/generation");
        }
        if (next_lease_id_ == 0) {
            throw std::overflow_error("workspace lease id overflow");
        }
        ensure_revision_room_for_acquire();
        if (workspace_bytes >
            std::numeric_limits<uint64_t>::max() - active_workspace_bytes_) {
            throw std::overflow_error("active workspace byte total overflow");
        }

        const uint64_t lease_id = next_lease_id_;
        const uint64_t next_id =
            lease_id == std::numeric_limits<uint64_t>::max() ? 0
                                                              : lease_id + 1;
        const auto inserted = active_.emplace(
            identity, ActiveRecord{lease_id, workspace_bytes});
        if (!inserted.second) {
            throw std::logic_error("workspace identity insertion failed");
        }
        active_workspace_bytes_ += workspace_bytes;
        ++workspace_revision_;
        next_lease_id_ = next_id;
        return {AcquireStatus::Acquired,
                Lease(this, LeaseKey{rank_, node_id, generation, lease_id})};
    }

    /// Validate without mutation.  The response is bound to this exact
    /// request view and may not claim a different workspace ledger.
    void validate_capacity_snapshot(const WorkspaceCapacityRank& snapshot)
        const {
        if (snapshot.rank != rank_) {
            throw std::runtime_error("workspace snapshot rank mismatch");
        }
        if (snapshot.workspace_revision != workspace_revision_) {
            throw std::runtime_error(
                "workspace snapshot revision does not echo current rank " +
                std::to_string(rank_) + " view");
        }
        const uint64_t base = checked_sum(
            snapshot.model_weight_bytes, snapshot.resident_kv_bytes,
            "workspace snapshot base-used sum overflow");
        const uint64_t full_base = checked_sum(
            base, snapshot.reserved_request_bytes,
            "workspace snapshot base-used sum overflow");
        if (full_base > snapshot.capacity_bytes) {
            throw std::runtime_error(
                "workspace snapshot base-used bytes exceed capacity");
        }
        if (!snapshot.has_prospective_workspace &&
            snapshot.prospective_workspace_bytes != 0) {
            throw std::runtime_error("unmarked prospective workspace bound");
        }
        if (has_capacity_snapshot_ &&
            (snapshot.has_prospective_workspace != has_prospective_workspace_ ||
             snapshot.prospective_workspace_bytes != prospective_workspace_bytes_)) {
            throw std::runtime_error(
                "prospective workspace bound changed after initial snapshot");
        }
        if (snapshot.has_prospective_workspace &&
            snapshot.prospective_workspace_bytes > snapshot.capacity_bytes - full_base) {
            throw std::runtime_error(
                "workspace fit margin plus Python base-used bytes exceed rank "
                "capacity");
        }
        if (active_workspace_bytes_ > snapshot.capacity_bytes - full_base) {
            throw std::runtime_error(
                "active workspace plus Python base-used bytes exceed rank "
                "capacity");
        }
    }

    bool capacity_snapshot_changes(const WorkspaceCapacityRank& snapshot)
        const {
        validate_capacity_snapshot(snapshot);
        return !has_capacity_snapshot_ ||
               capacity_bytes_ != snapshot.capacity_bytes ||
               model_weight_bytes_ != snapshot.model_weight_bytes ||
               resident_kv_bytes_ != snapshot.resident_kv_bytes ||
               reserved_request_bytes_ != snapshot.reserved_request_bytes;
    }

    /// Caller validates all ranks before mutating any rank.  This operation
    /// itself uses only scalar assignments and cannot fail.
    void install_capacity_snapshot_unchecked(
        const WorkspaceCapacityRank& snapshot) noexcept {
        capacity_snapshot_revision_ = snapshot.workspace_revision;
        capacity_bytes_ = snapshot.capacity_bytes;
        model_weight_bytes_ = snapshot.model_weight_bytes;
        resident_kv_bytes_ = snapshot.resident_kv_bytes;
        reserved_request_bytes_ = snapshot.reserved_request_bytes;
        has_prospective_workspace_ = snapshot.has_prospective_workspace;
        prospective_workspace_bytes_ = snapshot.prospective_workspace_bytes;
        has_capacity_snapshot_ = true;
    }

  private:
    using Identity = std::pair<uint64_t, uint64_t>;  // node id, generation
    struct ActiveRecord {
        uint64_t lease_id;
        uint64_t workspace_bytes;
    };

    static uint64_t checked_sum(const uint64_t a, const uint64_t b,
                                const char* const what) {
        if (b > std::numeric_limits<uint64_t>::max() - a) {
            throw std::overflow_error(what);
        }
        return a + b;
    }

    uint64_t checked_base_used() const {
        const uint64_t base = checked_sum(
            model_weight_bytes_, resident_kv_bytes_,
            "workspace base-used sum overflow");
        return checked_sum(base, reserved_request_bytes_,
                           "workspace base-used sum overflow");
    }

    void ensure_revision_room_for_acquire() const {
        const uint64_t max = std::numeric_limits<uint64_t>::max();
        if (active_.size() > static_cast<std::size_t>(max - 2) ||
            workspace_revision_ >
                max - 2 - static_cast<uint64_t>(active_.size())) {
            throw std::overflow_error(
                "workspace revision cannot cover acquire and every active "
                "lease release");
        }
    }

    void release(const LeaseKey& key) {
        if (key.rank != rank_) {
            throw std::logic_error("workspace lease rank mismatch on release");
        }
        const auto found = active_.find(Identity{key.node_id, key.generation});
        if (found == active_.end() || found->second.lease_id != key.lease_id) {
            throw std::logic_error(
                "unknown, stale, or duplicate workspace lease release");
        }
        if (workspace_revision_ == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("workspace revision overflow on release");
        }
        const uint64_t bytes = found->second.workspace_bytes;
        if (bytes > active_workspace_bytes_) {
            throw std::logic_error("active workspace ledger underflow");
        }
        active_workspace_bytes_ -= bytes;
        active_.erase(found);
        ++workspace_revision_;
    }

    int rank_;
    uint64_t workspace_revision_ = 0;
    uint64_t active_workspace_bytes_ = 0;
    uint64_t next_lease_id_ = 1;
    bool has_capacity_snapshot_ = false;
    uint64_t capacity_snapshot_revision_ = 0;
    uint64_t capacity_bytes_ = 0;
    uint64_t model_weight_bytes_ = 0;
    uint64_t resident_kv_bytes_ = 0;
    uint64_t reserved_request_bytes_ = 0;
    bool has_prospective_workspace_ = false;
    uint64_t prospective_workspace_bytes_ = 0;
    std::map<Identity, ActiveRecord> active_;
};

}  // namespace AstraSim

#endif  // ASTRA_SIM_WORKLOAD_WORKSPACE_ADMISSION_HH
