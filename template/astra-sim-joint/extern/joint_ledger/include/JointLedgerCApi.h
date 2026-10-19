/* JointLedgerCApi.h -- C ABI of the native joint dual-ledger receiver.
 *
 * Production flow (one delivery, constant number of calls):
 *   jl_scan_request(bytes)              -- locate work_ledger member; splice
 *                                          offsets for the outer frame; no
 *                                          decoding, no state.
 *   [Python: outer gates on the spliced outer JSON, unchanged]
 *   jl_prepare(h, segment, ...)         -- typed SAX parse + full validation
 *                                          + COW staging of both ledgers.
 *   jl_commit(h, prepared)              -- atomic publish.
 *   jl_pricing_json(h)/jl_counters(h)   -- policy views.
 *
 * Error contract: every function returns JL_OK (0) or a JL_ERR_* code; a
 * human-readable message (with the offending row path / byte offset when
 * known) is available via jl_last_error.  No function ever leaves a
 * half-committed ledger: prepare failures discard all staged state, and
 * commit only publishes fully validated rows.  C++ exceptions never cross
 * this boundary.
 */
#ifndef JOINT_LEDGER_CAPI_H
#define JOINT_LEDGER_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JL_OK 0
#define JL_ERR_ARG 1        /* invalid argument (null handle/buffer) */
#define JL_ERR_PROTOCOL 2   /* validation failure (fail-closed message set) */
#define JL_ERR_STATE 3      /* API misuse (unknown prepared slot, etc.) */
#define JL_ERR_MEMORY 4     /* allocation failure */
#define JL_ERR_RANGE 5      /* caller buffer too small (retry with size) */

typedef struct JLedger JLedger;

JLedger *jl_create(void);
void jl_destroy(JLedger *ledger);

/* Structural scan of a full bridge request (compact JSON object bytes).
 * Fills the member byte range of the top-level "work_ledger" value and the
 * splice window that removes the whole member (name start through value
 * end, including one adjacent separator) so the remaining bytes form the
 * exact outer frame JSON.  Requires work_ledger to be present; any other
 * outcome is JL_ERR_PROTOCOL.  Handles string escapes, nesting and unicode
 * without decoding values. */
typedef struct {
    uint64_t member_start;   /* first byte of the splice window (remove from) */
    uint64_t member_end;     /* one past the last byte of the splice window */
    uint64_t value_start;    /* first byte of the work_ledger value */
    uint64_t value_end;      /* one past the last byte of that value */
} JLScanResult;

int jl_scan_request(const char *data, size_t length, JLScanResult *out);

/* Prepare one work_ledger segment (bytes [value_start, value_end)).  The
 * expected outer delivery sequence/tick must match the envelope header
 * (fail-closed otherwise).  On success *prepared receives a slot id. */
int jl_prepare(JLedger *ledger, const char *segment, size_t length,
               uint64_t expected_delivery, uint64_t expected_tick,
               int32_t *prepared, uint64_t *event_count,
               uint64_t *seq_end, uint64_t *result_state_version);

/* Commit a prepared delivery atomically (both ledgers, descriptor registry
 * and replay metadata form one publish unit). */
int jl_commit(JLedger *ledger, int32_t prepared);

/* Discard a prepared delivery (outer gate failed after prepare). */
int jl_discard_prepared(JLedger *ledger, int32_t prepared);

/* Last error message of this handle (or the last jl_scan_request call when
 * no handle was involved); copies at most buffer_size-1 bytes, NUL
 * terminates, returns the full message length. */
int jl_last_error(const JLedger *ledger, char *buffer, size_t buffer_size);

/* ---- query surface ---- */

typedef struct {
    uint64_t version;
    uint64_t event_watermark;
    int64_t last_tick;
    int64_t last_delivery_sequence;
    uint64_t full_load_count;
    uint64_t delta_load_count;
    uint64_t applied_event_count;
    uint64_t replayed_event_count;
    int64_t pipeline_delivery_sequence;   /* -1 before first delivery */
    uint64_t pipeline_watermark;
    uint64_t pipeline_state_version;
    uint64_t active_works;                /* active WorkLedger rows */
    int64_t parse_ns;                     /* last delivery SAX parse+validate */
    int64_t stage_ns;                     /* last delivery staging core */
    int64_t commit_ns;                    /* last delivery publish */
    int32_t last_snapshot_full;           /* 1 = last accepted delivery was full */
} JLCounters;

int jl_counters(const JLedger *ledger, JLCounters *out);

/* JSON exports.  Call with buffer=NULL (or *capacity=0) to query the
 * required size first; then call again with a buffer of that size.  The
 * returned bytes are valid UTF-8 JSON, NUL terminated when capacity
 * allows. */
int jl_pricing_json(const JLedger *ledger, char *buffer, size_t *capacity);
int jl_conservation_json(const JLedger *ledger, char *buffer,
                         size_t *capacity);
int jl_works_export_json(const JLedger *ledger, char *buffer,
                         size_t *capacity);
int jl_pipeline_snapshot_json(const JLedger *ledger, char *buffer,
                              size_t *capacity);

/* Abort helper for tests/diagnostics: the message bytes belong to the
 * library and are only valid until the next call on this handle. */
const char *jl_last_error_direct(const JLedger *ledger);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* JOINT_LEDGER_CAPI_H */
