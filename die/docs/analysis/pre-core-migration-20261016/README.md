# Preserved work before the hardware migration

This is an immutable archive of unfinished work from the original worktree.
It is **not adopted**, is not compiled by the current CMake project, and is not
validation of the current hardware revision. The original worktree is retained.
Do not apply the patch wholesale to the current Core model.

`unadopted-tracked.patch` contains the five tracked-file changes; the tar archive
contains the added long-row Norm implementation, tests and independent offline
oracle, all 92-task initial-run raw evidence, and the numerical/segmentation
experiments. `manifest.json` records every archived file and archive SHA-256.
Extract only into a scratch directory, then verify the individual hashes.
No objects, executables, libraries or caches are included.

The long-row prototype supported R=1, one row per participating Core, and
BF16/FP32 roles. The earlier 51-test suite passed before adding two regressions;
the later tiny-credit and synchronous-owner regressions are red on the unchanged
Runtime. The initial 92-task oracle suite completed 91 tasks; one actual C++
execution stalled. All 455 nonzero finish comparisons met 10%; six first-output
comparisons missed 10% (maximum 15.17%). The original busy metric failed broadly.
These are synthetic model comparisons, not measured hardware calibration. The
zero-reference metric correction was written after this run; its complete matrix
was not rerun. None of these results can be relabeled as a current-source pass.

`work/norm-review/runtime-archive-20261016/RUNTIME-README.md` inside the archive
records all numerical proposals and failures. The last tested candidate passed
seven of eight focused tests but still exceeded a Norm input capacity; it was
rejected. The final sub-epsilon candidate was only compiled; its validation and
budget measurements were not run. Production `Execution.cc` was never changed.

`work/norm-statistics-variant/` contains a fixed-size statistics prefix/tail
proposal and static counting evidence. It was not run against C++ Runtime and
was not adopted. Future work must first adapt fixtures and identities to the
current hardware contract, retain the red regressions, and establish numerical
correctness without enlarging tolerances, discarding real work, or replaying
tiles/K in production.
