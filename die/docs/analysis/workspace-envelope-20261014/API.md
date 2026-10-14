# Compile-only workspace envelope API

`coarse_die::compile_workspace_envelope(const Hardware&, const nlohmann::json&)`
normalizes one task through the same compiler path as `compile()` and asks
`detail::enumerate_mappings_for_compile()` for its actual fixed mapping vector.
It returns the candidate count and the maximum exact
`hbm_workspace_peak_bytes` over that vector, along with the normalized task,
its stable fingerprint, hardware fingerprint, `op_kind`, and `phase`.

The C ABI entry point is:

```c
const char* die_coarse_workspace_envelope_v2(DieHandle model,
                                             const char* task_json);
```

The response is compact JSON with `schema_version: 2`,
`workspace_envelope_schema_version: 1`, `candidate_count`,
`max_hbm_workspace_peak_bytes`, `hardware_fingerprint`,
`normalized_task_fingerprint`, `op_kind`, `phase`, and `normalized_task`.
Byte counts are serialized as exact JSON uint64 integers. The returned string
uses the existing `die_coarse_free_result_v2()` ownership contract. On error,
the function returns null and `die_coarse_last_error_v2()` contains the
existing status/reason JSON; success clears the prior error.

This API skips `compile()`'s outer latency ranking. It does not submit work to
a caller's Runtime, allocate a C ABI Plan handle, or insert/touch a model cache
entry. Mapping enumeration can still use its existing internal latency
estimates when selecting component candidates for a materialized FFN mapping.
The result covers every mapping candidate for this one task, not a set of tasks
or a phase-wide admission envelope.

The focused `coarse_workspace_envelope_test` compares Attention and FFN
maxima against the actual mapping vectors and selected Plans, checks exact
large uint64 values and overflow/type failures, and exercises alias
normalization, unsupported legacy `local_region`, C ABI error/ownership,
handle, and cache behavior. The standalone build and all 30 CTests passed;
logs are in `../../evidence/workspace-envelope-20261014/public/`.
