/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DieCApi.h -- C ABI of the die-internal cost model library (stage A3).

Contract §4 / design doc §8.1: the Python side (config resolver, face
scheduler oracle, projection worlds) drives the SAME library through this
ABI; no second copy of any derivation or scheduling formula exists outside
the library.  The ABI is pure C (include from C or C++), throws nothing
across the boundary, and returns:

  - handles (void*) for the model and per-run execution objects; NULL on
    error, with the reason available via die_last_error();
  - library-allocated UTF-8 JSON buffers from die_hardware_summary() and
    die_estimate_batch(); the caller releases them with die_free_result()
    (never free()/delete());
  - 0 on success and -1 on error from the int-returning calls, except
    die_next_action(), which also returns -2 when the caller buffer is too
    small.

Error reporting: die_last_error() reads a single process-wide slot that the
latest failing call overwrote.  It is NOT thread-safe (concurrent failing
calls race on the slot) and the returned pointer is only valid until the
next failing call.  It never returns NULL.

Threading: each model/execution handle must be used from one thread at a
time; the library itself keeps no hidden global state besides the error
slot.  Compile results are cached per model handle (LRU 4096, keyed by the
frozen plan_cache_key; the plan itself is immutable shared state).
*/

#ifndef DIE_MODEL_CAPI_H_
#define DIE_MODEL_CAPI_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Creates a model handle from a hardware configuration JSON.  Accepted
shapes: the wrapped resolved_die_model.json form ({"schema_version":1,
"profile":"mesh_die_v1"|"tx8_reference_v1","die":{...six keys...}, plus an
optional ignored "derived" mirror}) or the bare die node (profile defaults
to mesh_die_v1); exactly the forms make_hardware() accepts.  The handle
owns the Hardware plus a plan cache; destroy it with die_destroy_model().
Returns NULL on any parse/validation error (see die_last_error()). */
void* die_create_model(const char* config_json);

/* The hardware summary JSON (contract §3.3: schema_version, profile, the
die node and the derived block) in a library-allocated buffer; the resolver
must take derived values from here instead of re-deriving them.  NULL if
model is not a valid handle.  Release with die_free_result(). */
const char* die_hardware_summary(void* model);

/* Batch isolated-run estimation.  specs_json is {"specs":[spec,...]} where
each spec is one die_op_spec JSON object (contract §2 schema, strictly
validated).  Returns a library-allocated buffer holding {"results":[...],
"errors":[...]} of equal length: results[i] is the serialized CostResult of
specs[i] (snake_case field names, frozen in the stage A3 report) or null,
and errors[i] is null or the failure message of spec i (a bad spec never
fails the whole batch).  NULL only if specs_json itself is malformed or
model is invalid (die_last_error()).  Release with die_free_result(). */
const char* die_estimate_batch(void* model, const char* specs_json);

/* Creates an execution handle for one spec: compiles it through the model's
plan cache and instantiates the externally driven state machine.  The spec
is the same single die_op_spec object estimate_batch accepts.  NULL on
error (die_last_error()).  Drive it with die_next_action/
die_advance_time/die_complete_memory; destroy with
die_destroy_execution(). */
void* die_create_execution(void* model, const char* spec_json);

/* Writes the next action of the execution as a NUL-terminated UTF-8 JSON
object into the CALLER buffer out (capacity cap bytes, including the NUL):

    {"kind":"timer","ns":N}
    {"kind":"memory","stage_id":S,"actual_bytes":A,"service_units":U}
    {"kind":"done","elapsed_ns":T}

Returns 0 on success, -2 if cap is too small for the serialization (out is
left unwritten) and -1 on any other error (die_last_error()).  Calling
again without advancing time or completing memory reports the same action
(idempotent); a memory action reported twice is the SAME outstanding
request, not a new one. */
int die_next_action(void* exec, char* out, size_t cap);

/* Advances the execution's internal clock by ns (delta).  0 on success, -1
on error (e.g. clock overflow). */
int die_advance_time(void* exec, unsigned long long ns);

/* Completes the outstanding memory request of stage stage_id.  The caller
must have advanced the clock to (or past) the completion moment first.  0
on success, -1 if no request is outstanding or stage_id does not match
it. */
int die_complete_memory(void* exec, unsigned long long stage_id);

/* Destroys an execution handle (no-op on NULL). */
void die_destroy_execution(void* exec);

/* Destroys a model handle and its plan cache (no-op on NULL). */
void die_destroy_model(void* model);

/* Releases a buffer returned by die_hardware_summary() or
die_estimate_batch() (no-op on NULL).  Never apply it to die_next_action's
caller buffer. */
void die_free_result(const char* buf);

/* The last error message of this process ("" when the last call
succeeded).  Never NULL; not thread-safe; valid until the next failing
call. */
const char* die_last_error(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* DIE_MODEL_CAPI_H_ */
