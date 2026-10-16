# A+B Runtime stop-time handoff — applied, not rebuilt

The user explicitly requested immediate stop. All previously launched small
tests had already completed naturally when polled; no task remains running.
No new experiment, compiler, budget, production edit, commit or push was started
after the stop instruction. This archive only preserves text and hashes.

## Exact production state at stop

The parent reviewed and applied combined-A-B.patch to the working tree and
copied runtime_subunit_credit_test.cc into die/tests/. Production Execution.cc
SHA256 is6fcc637e13c876d240d92d8677e8f9d2e323b325aea46176848063ffe93e710e;
header SHA256 is4ae56348aab0f39c6e67ecc28fe157cf6aaffd9478ac107af4b2f1d3003f0d6a.
These are working-tree modifications; this subtask did not commit/push them.
Fresh AB compile, new CTest execution, full API/trace/51+new regression and
performance budgets are **not_run**. No AB object or library was built here.
Do not present previous baseline/A/B ELF results as AB acceptance.

combined-A-B.patch SHA256:
7646b0fdd81f17c4a2f4a285747703071d2e12f66935b57a95b667a56bc218d9.
Frozen test SHA256:
20dbbbab2d0568aba148b22e63d1e519d5925b61ad1d0e1b2fc435335c41fb9b.
The saved manifest records actual HEAD and whether current production still
matches these files, independently from Git integration status.

## Source combination and scope

Direct git apply --check in A-then-B and B-then-A order both failed because the
patches share context in the finite-buffer boundary block. Original failed
commands and stderr are retained in prepare-identity.json. The source-AB folder
is A-only after that attempt; source-BA is B-only. They are not combined files.
The **combined/** folder is the actual successful A+B source/header.

Actual three-way merges (A,baseline,B) and (B,baseline,A) returned0, had no
conflict markers and produced byte-identical source; merge-identity.json binds
the commands and all hashes. three-way-unresolved.cc is the verbatim output of
an earlier successful merge despite its preparatory filename. Its first Python
heredoc wrapper source is not_preserved; the exact git command/status is saved.
The later merge_combined.py was preserved and actually executed independently.

A's credit_boundary_tolerance helper and all original call sites remain.
B's private overload, one fixed initial-boundary redundant-row witness, <=32
owners/96variables/64extra rows, shared total256-pivot budget and original
same-or-one-ULP match remain. A ordinary boundary classification must also be
used inside the B reconstruction; never replace combined source with B's old
full file. No time epsilon, positive-event filter, progress closing threshold,
physical work, owner fairness or acceptance tolerance was changed by combining.
Neither fix certifies capacity1 Norm, arbitrary sub-epsilon representability,
R>1 readiness, target-hardware accuracy or general old-snapshot recovery.

## Actual frozen small-test runs

The one small C++ regression was compiled once and linked serially to three
explicit **existing** static libraries, not a new AB library. All commands,
headers/config/test/object/executable/library SHA256 and full stdout/stderr are
in existing-libraries-identity.json and adjacent logs. Fixture headers include
the saved original baseline Execution declaration; B adds no data members or
public API/layout, only a private overload. This is C++ coverage, not C ABI.

- Baseline fresh-session library e12c029d...: rc1,8subcases passed,4failed.
- A subunit candidate library81e3c232...: rc0,12/12subcases passed.
- B snapshot candidate library6573ce3f...: rc1,8passed,4failed.

All three libraries were byte-identical before/after. The four baseline/B
failures are filling and emptying at2^-60, each for half-second physical event
and independent snapshot inventory/Plan integrity. Units1 and2^-10 passed.
The test independently collects physical and snapshot-guard cases, so a failed
physical assertion cannot hide the entire guard regression. Within each guard
case the first failed credit assertion can stop the later capacity assertion;
candidate A actually executes both mutations in all six cases. Do not claim
the baseline2^-60 capacity edit was reached after its earlier credit failure.

The unchanged original half-second work/progress/busy/block/cancel assertions
are used, with explicit synthetic Plan identity on current-Core manual_hardware.
No tolerance was added and the archived strict Fraction failures were not changed.
Earlier A/B coupled matrices and all failed exact oracles remain in the separate
runtime-subunit/ and snapshot-import/ archives with their original scope.

## Continue only after a new instruction

On resumption, first check Git/source/remote identities and working-tree status.
If this working-tree candidate is retained, register runtime_subunit_credit_test
in the existing foreach list of die/CMakeLists.txt, verify source SHA and run a
new clean final-source build before any acceptance claim. The parent may still
archive/revert or commit these changes during stop handoff; use actual remote
and HANDOFF state. Original scripts here record historical baseline operations;
prepare_combined.py/merge_combined.py must not be blindly rerun against already
modified production. They are evidence, not an instruction to overwrite files.
No build ELF/object/archive/cache is included. No new SDK was published here.
