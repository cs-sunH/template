# C API batch memo draft handoff

This draft was stopped at the parent's request before regression or syntax validation. `capi-batch-memo-draft.cc` is the complete Capi.cc snapshot at stop time. It includes the concurrent root changes already present in Capi.cc (normalized canonical cache key, lightweight cost query, and Plan v4 response marker) in addition to the unvalidated single-slot raw-batch memo experiment. The accompanying `capi-batch-memo-working-tree.diff` is against repository HEAD and therefore also includes those pre-existing changes; it is not a clean memo-only patch.

Draft design: one per-Model memo stores the exact raw batch request and pointers to canonical `std::map` cache entries. Cache insertion clears the memo before any map structure change. The memo contributes retained object/string/vector memory to `cache_bytes` and uses the same 4096 item / 64 MiB limits. Only all-success batches up to 256 rows and 64 KiB raw JSON are memoized. A memo hit bypasses JSON parsing and task normalization while compile_batch still creates new plan handles.

No build, syntax check, or tests were run. After the later handoff instruction, the saved snapshot was restored to `die/src/Capi.cc` and passed one C++17 syntax-only check. Tests were not edited or run; the root agent should validate and update accounting assertions during its unified build.
