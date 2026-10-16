# Stop checkpoint: read-only Norm dependency review

The user requested immediate handoff during this review. No new Norm mapping, numerical patch, reproducer, build, test or performance run was started. No production source, root HANDOFF/PROJECT/RELAY or frozen evidence was changed. This note preserves findings only; it is not current-Core Norm validation.

## Exact restoration boundary

The immutable pre-Core archive is `die/docs/analysis/pre-core-migration-20261016/`. Its verified scratch extraction is `work/norm-audit/`; the existing extraction review already records 656/656 outer and 133/133 nested hashes. The earlier `norm-long-row-20261016/` archive is the first seven-epoch prototype; the later pre-Core archive contains BF16/FP32 restriction, extended physical-prefix route tests, mixed-role oracle and the synchronous red. Do not mechanically restore the earlier unrestricted INT4 helper over the later guarded variant.

Later archived `NormLongRow.hh` supports R1, one row per participating Core, all three storage roles BF16/FP32, hidden>512, fixed tile256, two HBM visits of X and one gamma/beta visit. Statistics retain16B/Core across a true complete-kernel barrier; application cannot begin before statistics Compute terminal. Dynamic credit capacities and static SRAM peak are different contracts. INT4 scale capacity/traffic is absent and remains excluded. Multi-row/Core and R>1 remain unsupported.

Core32 adaptation must remove only retired `architecture.core_type` injection from old synchronous/oracle fixtures, resolve a current explicit descriptor with all required W/H/grid fields, and preserve visibly synthetic service-rate/Vec overrides and a new synthetic fingerprint. Do not relabel old hardware/Plan/snapshot identities. The raw timing/busy reports remain historical synthetic evidence, not target measurements or current-Core acceptance. Current production still lacks the long-row builder, so compiling the old full task against an unchanged current library is not a valid Runtime-only reproduction.

## Next Runtime reproducer dependency

The exact original failing workload in the later archive is RMSNorm hidden65537,m1,4 Cores, X BF16/parameters FP32/output FP32,R1; two distinct owners arrive at0. Synthetic Vec peak204800/Core, HBM rows204800B/s, other enabled rates4e12, COMPUTE_GATE1, launch0. Source: extracted `die/tests/norm_long_row_test.cc:synchronous_long_row_credit_liveness`; old line221 injects retired type and must be adapted separately. The full mixed driver also injects that retired selector.

The immutable `strict-projection.stderr.log` contains the complete seven-epoch physical Plan and capacities plus the observed stalled state. At exact5.130029297e9ns both owners are in apply-middle epoch4: Load13 and Store15 done, Compute14 progress0.9999999999997531472231,rate0,reported credit[0,0]. Remaining progress2.468527769e-13 exceeds progress-close2e-13. With original signed curves L[254,0],C[-255,254],S[0,-255], the true remaining input is about6.29474581095e-11 and output about-6.27006053326e-11; max(1,capacity)*2e-10 snapping erases both. Capacity is[1,1], so the newly studied subunit-only repair does not cover it.

The existing single-epoch manual three-activity reduction uses the exact extracted epoch4 work/bytes/Q/PRIVATE but drops its predecessor fractional-time/service history. It actually passed the current baseline in subsequent runtime-subunit work; it is not an established reproducer of the seven-epoch failure. Do not change constants or import the old stalled snapshot with a substituted fingerprint to make it fail.

Suggested next preparation, not performed: extract only the numeric physical epoch contracts for all seven epochs from that logged Plan and construct a new manual synthetic Plan against a current resolved Core descriptor, preserving declared synthetic capacities, Q/PRIVATE, two real owners, actual predecessor work and zero launch. Give it a new canonical Plan fingerprint; do not import/rebrand the old state. Run from time0 and retain exact event snapshots, physical Vec/HBM work, original busy, finish lower bounds, per-channel credit and a fixed finite-event limit. Baseline and combined narrow Runtime candidate need separate actual-source results. If this full-history manual fixture passes, preserve that outcome and next adapt the actual long-row builder in an isolated source copy; no red is established by this note.

Sub-epsilon is a separate demonstrated historical mechanism: `boundary-diagnostic.stderr.log` shows output credit0.78024483619937, netrate-223750712.2507/s, integration3.487116659ns leaving3.7079714299e-17. The true remaining interval1.6571886599e-16ns was dropped by the old1e-15ns floor, then the public-tick remainder0.5128833408ns crossed credit to-0.114758. Removing the floor alone does not prove represented clock/progress advancement, bounded events or legality. Old exact-credit candidate also failed actual input1.0622468484693504>capacity1; last sub-epsilon candidate was compiled only. None is safe to apply wholesale.

## Fixed statistics prefix/tail proposal

Extracted `work/norm-statistics-variant/` has a byte-identified baseline/candidate/patch and a static checker. It partitions statistics into four fixed epochs: L(T); L(T)+C(-T); L(H-2T)+C(-(H-T-tail)); C(-tail). Capacity2T remains, entry/exit credits0->T->T->tail->0. Only final tail Compute writes16B statistics. Full-tile PRIVATE is the physical max per tile multiplied by the closed-form count; final tail has its own max. The existing five apply epochs and their numeric curves remain unchanged; absolute indices/IDs shift by2. Total9epochs,15 physical descriptions,16 with launch; statistics6 descriptions fit budget8, application9 fit budget16, at most3 activities.

Only static36 integer ledger cases,144 exact Fraction PRIVATE checks and112 small offline partitions passed. No candidate C++ Runtime, timing comparison, budget or adoption exists. The candidate addresses first-output/statistics partition approximation; it cannot by itself repair the unchanged apply-middle capacity1 numerical failure. Original tests hard-code7epochs and apply indices, so experimental tests must derive the new ledger indices. Check actual16B-final-write/barrier, full/tail513 edges, constant event count and all eight BF16/FP32 role combinations before any adoption claim.

## Why original busy genuinely fails

The finite affine service curve spreads positive low-rate service throughout a long overlapped interval. Original busy counts the union of time with any positive physical service, while the discrete oracle has real inter-tile idle gaps. Conserved bytes/operations and nearly identical completion do not make these intervals equal. This is a structural approximation error, not a reason to redefine busy or fit eta.

Concrete retained examples (old synthetic one-Core BF16/R1, no binding NoC):

- RMS65537 balanced solo: completion5.120078126 vs5.12653125s (0.126%); original compute busy5.094581026 vs2.5600390625s (99.0% error). HBM busy agrees to rounding.
- RMS65537 compute-bound solo: completion25.620136719 vs25.62013671875s; HBM busy25.50013671875 vs5.120078125s (398% error).
- Same compute-bound offset pair: reference HBM shared0, smooth-model shared49.848688401s. This is a real positive absolute error against a zero reference, not a meaningful finite relative-error ratio.

Initial12-task36-scenario evidence had all completion/first-output comparisons<=10% but all36 joint classifications failed; original busy44/108 missed10%. Later92-task mixed matrix completed91 tasks:455 completion comparisons<=10%, six first-output misses (RMS12.99795%,Layer15.16932%) and258/273joint failures. The zero-reference report correction was not rerun over the whole matrix. These identities/scopes remain separate from current Core and later Runtime work.

## Terminology audit deferred during source-tree recorder window

Read-only active scan found no obsolete Core type or deleted hardware-MD reference in public production/config/interface or five non-coarse legacy/consumer trees. Public core_type hits are strict rejection text/negative tests; core64_v2 is a negative identity test.460 matched lines in the five `extern/die_model/coarse/` trees belong to unchanged pinned canonical24 SDKs and must retain that historical source identity. No production hardware-search entry was found; deterministic legacy geometry decoding is not enumeration, and root design7.5 sensitivity is explicitly offline.

The active `analysis/core-resume-20261016/README.md` has stale progress statements: line3/29 syntax-only/no stage, line71 not_run, line87-89 native still running and next runner must fix JOINT entry, line101 consumers still being built. After the parent's fresh source-tree recorded build/test ends, add a current summary linking the final SDK scope and native-resumed record; mark original audit-only phase historical and update its actual execution section. Preserve original raw evidence and the explicitly historical single-core-handoff restore text. No such edits were made in this stop checkpoint.
