# JOINT additive coarse adapter checkpoint

This checkpoint records the JOINT typed coarse-task, C++ runtime/admission and bridge-v3 seams. The focused build and tests passed; the exact commands, logs, source/binary hashes and initial build failure are in this directory.

Validation: six focused C++ binaries passed, including legacy Adapter/Lifecycle checks; the JOINT Python adapter/workspace suites passed 45/45 with no skips; all 14 coarse task-builder operation kinds returned successful results through JOINT's actual `libcoarse_die_cabi.so`. Exact v3 legacy request bytes are asserted, and the coarse snapshot/parser/committer and lease lifecycle cases pass.

The sh30 coarse task builder is still behind the frozen native-projection guard. No real JOINT-world native projection parity run was performed, so production sh30 coarse scheduling remains unintegrated. Keep the guard enabled. The next acceptance gate is real workload parity through JOINT's native pipeline and owner/endpoint lifecycle before making the builder reachable.

This checkpoint is not a production-integration completion claim. No commit or push was made by the adapter owner.
