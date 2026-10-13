#ifndef COARSE_DIE_CAPI_V2_H
#define COARSE_DIE_CAPI_V2_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef uint64_t DieHandle;
typedef struct {
    uint32_t struct_size, schema_version;
    uint64_t request_id, node_id, execution_id, generation;
} DieOwnerV2;
typedef struct {
    uint32_t struct_size, schema_version, role, enabled_resource_mask;
    uint64_t activity_id;
    DieOwnerV2 owner;
    double service_work[13];
    uint64_t hbm_payload_bytes[2][2];
    uint64_t executed_int4_ops, vec_eqops;
    double canonical_Q, owner_weight, private_time_bound_s;
} DieActivityV2;
typedef struct {
    uint32_t struct_size, schema_version, terminal_kind, status_code;
    DieOwnerV2 owner;
    uint64_t issue_ns, finish_ns, elapsed_ns;
    uint64_t hbm_served_bytes[2][2];
    uint64_t int4_ops_served, vec_eqops_served;
} DieCompletionV2;
/* The fixed V2 activity/completion records represent integral equivalent work
 * only and keep their original layout. Fractional work is represented by the
 * versioned Plan V3 payload (whole + fractional remainder) and by the JSON
 * progress/counter/snapshot APIs below; never round it into these V2 records.
 * No exported function populates a DieActivityV2 or DieCompletionV2. */
enum DieStatusV2 {
    DIE_OK=0, DIE_INVALID_INPUT=1, DIE_INVALID_HARDWARE=2,
    DIE_MISSING_TOPOLOGY=3, DIE_UNSUPPORTED=4, DIE_OVERFLOW=5,
    DIE_INCOMPATIBLE_SCHEMA=6, DIE_INCOMPATIBLE_FINGERPRINT=7,
    DIE_INVALID_TIME=8, DIE_BUSY=9, DIE_STALE_HANDLE=10,
    DIE_NUMERICAL_FAILURE=11, DIE_UNSUPPORTED_LEGACY_STATE=12,
    DIE_UNSUPPORTED_OP=13, DIE_UNSUPPORTED_DTYPE=14,
    DIE_UNSUPPORTED_HEAD_MAPPING=15, DIE_UNSUPPORTED_WORKSPACE=16,
    DIE_UNSUPPORTED_MAPPING=17, DIE_UNSUPPORTED_PROJECTION_STATE=18
};
/* Single-thread-per-model/runtime. All JSON strings are UTF-8. Returned
 * strings are owned by the library; free with die_coarse_free_result_v2.
 * Handles are monotonic opaque IDs, never addresses. Exceptions stay in C++.
 * Successful estimation is an assumed-model result, not measured accuracy. */
uint32_t die_coarse_abi_version(void);
int die_coarse_create_model_v2(const char*, DieHandle*);
const char* die_coarse_hardware_summary_v2(DieHandle);
/* {"specs":[task,...]}; results/errors are positional. compile_batch results
 * contain plan_handle, plan_hash, summary. release each returned plan handle. */
const char* die_coarse_compile_batch_v2(DieHandle,const char*);
const char* die_coarse_estimate_batch_v2(DieHandle,const char*);
int die_coarse_create_runtime_v2(DieHandle,DieHandle*);
int die_coarse_submit_plan_v2(DieHandle,DieHandle,const DieOwnerV2*,uint64_t,DieHandle*);
int die_coarse_submit_endpoint_v2(DieHandle,const char*,const DieOwnerV2*,uint64_t,DieHandle*);
int die_coarse_advance_to_v2(DieHandle,uint64_t);
const char* die_coarse_next_event_v2(DieHandle);
int die_coarse_cancel_execution_v2(DieHandle,uint64_t);
int die_coarse_update_capacity_v2(DieHandle,uint32_t,double,uint64_t);
const char* die_coarse_query_progress_v2(DieHandle);
const char* die_coarse_query_counters_v2(DieHandle);
/* Imported/forked executions receive private ABI handles. List them here. */
const char* die_coarse_list_executions_v2(DieHandle);
const char* die_coarse_export_snapshot_v2(DieHandle);
int die_coarse_import_snapshot_v2(DieHandle,const char*,DieHandle*);
int die_coarse_fork_runtime_v2(DieHandle,DieHandle*);
/* Private fork; never changes the caller's live runtime. Known overlay
 * accepts ordered endpoint/plan/cancel/capacity events only. */
const char* die_coarse_estimate_on_snapshot_v2(DieHandle,DieHandle,const DieOwnerV2*,const char*);
int die_coarse_release_plan_v2(DieHandle);
int die_coarse_destroy_execution_v2(DieHandle);
int die_coarse_destroy_runtime_v2(DieHandle);
int die_coarse_destroy_model_v2(DieHandle);
void die_coarse_free_result_v2(const char*);
const char* die_coarse_last_error_v2(void);
#ifdef __cplusplus
}
#endif
#endif
