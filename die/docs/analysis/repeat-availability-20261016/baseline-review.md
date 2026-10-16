# R>1 region availability review (read-only production source)

Observed source HEAD: d16b5459a46df7f54d6e4afa36e2a5debd947690.
This review changes no die source, five-repository vendor, or configuration.
Only work/repeat-review contains new artifacts. The source hashes, exact compiler
and execution argv, static library hash before/after, and raw stdout/stderr are
in run_manifest.json. Probe library SHA-256 is
0a327ac92a3b9abfa839ff332a7b8a0e6f4849f7ade5dfab11c537205d8014d7.

## Actual availability defect

Run `python3 work/repeat-review/run_probe.py` from the repository root.
The original raw is probe.stdout; summary.json is a projection, not a second run.
Do not overwrite baseline artifacts when linking a modified library: use a
separate output directory or retain a baseline copy first.

Twenty-two cases were compiled/run against the canonical library. Twelve
zero-overhead repeated cases across residual, materialized ffn_region,
materialized attention_region, and qkv_rope (layers/passes 2/1, 1/2, 2/2)
report absent prediction_status, alone_estimate_status=ok_assumed,
estimate.status=ok_assumed, and consumer acceptance. All five R1 controls are
accepted. The three R>1 Norm controls and positive-launch residual R2 are
correctly marked not_available/rejected. Untyped region_ops is explicitly
unsupported by normalization; it is not a valid compound-region fixture.

Mapping.cc:3218 and 3231 return nullopt for zero applicable overhead before
examining generic R>1 barriers. Mapping.cc:3289 therefore skips availability
classification. Norm has its own independent R>1 guard at 1587. Compiler.cc
copies prediction_status into alone_estimate_status; Execution.cc:3201 does the
same for full estimate status. Prediction.hh rejects only explicit
not_available, retaining legacy behavior when diagnostics are absent.

## Minimal counterexamples

All resources have capacity one and active owners receive equal shares. Times
may be multiplied by any positive integer nanosecond scale; no rounding issue
is needed. An independent Fraction-based event calculation confirmed all four
schedules below. The earlier tool-only calculation did not save a raw file.
A new saved rerun at 2026-10-16T05:04:45Z is explicitly identified in
fraction_counterexamples_manifest.json; source is fraction_counterexamples.py,
raw stdout/stderr are fraction_counterexamples.stdout/.stderr, and the exact
runner is run_fraction_counterexamples.py. These are offline semantics examples,
not hardware calibration or a production repeat interpreter.

1. A=(wait1, X service1) twice; B arrives at t=2 with X service1.
   Correct: wait A 0..1, A service 1..2, B service 2..3, A service 3..4;
   finish A=4, B=3. Frontloading two waits instead produces wait0..2,
   A/B sharing2..4, A alone4..5; finish A=5, B=4. A completion error is 25%;
   B's own response latency changes from 1 to 2 (100%). Moving total idle
   time to the front is therefore observable under just one external arrival.

2. A=(Compute1, Store1) twice; Compute and Store have independent capacity
   one; B arrives at t=1 with Compute1. Correct finish A=4, B=2. Aggregating
   A as Compute2 followed by Store2 makes Compute shared1..3, then Store3..5;
   finish A=5, B=3. This is a region materialization-order error at zero delay,
   not an omitted numeric overhead. Total physical work is identical.

## Representation findings and exact scope of the obstruction

Current Plan has a single launch_ns. Execution has one ready_ns computed only
at submit (Execution.cc:871..881), and snapshot import requires
ready_ns-issue_ns == plan.launch_ns (2566..2568). repeat_count is audit metadata,
not a lifecycle repeat (Plan.hh and Execution.cc:3246). No extra integer
repeat_count field or more accurate R*delay arithmetic supplies the missing
region completion/next-launch order.

The required ordering is next_ready(k+1) >= final_store(k)+delay, followed by
the next region's initial load. During each delay all that owner's activities
must leave Q/alpha and no physical, PRIVATE, busy, or served work may accrue.
The body already requires separate service progress and SRAM credit. A new
clock/gate needs both the final materialization dependency and wall-clock time;
it cannot be a positive-Q activity, fake COMPUTE_GATE work, or a PRIVATE cap.

For exact nonzero waits, one resource-consuming owner alone has R distinct
positive-service intervals separated by R-1 real no-service windows. A
representation whose actual activity switches are bounded independently of R
cannot reproduce all those switches. A floor/mod expression with constant
storage does not evade the requirement that hidden switches also count. This
is a scoped obstruction to exact window reproduction, not a proof that every
allowed aggregate approximation fails: the design explicitly permits internal
readiness approximation. It does prevent claiming that a proposed aggregate
clock automatically preserves real eligibility and fairness.

An aggregate scalar availability/duty fraction is insufficient by itself.
For a continuously active observer C, two other equal owners active together
half the time give C average share (1/3+1)/2=2/3. The same individual marginal
availability fractions with one owner active at every instant give C share
1/2. Joint availability matters. Independent mixing, a scalar latency cap, or
fitting isolated R-times-body does not prove the original owner max-min rule
under competing regions or phase offsets. A zero-Q affine clock with a one-
instance fluid buffer also permits partial next-region preparation to overlap
the previous instance; the staircase materialization barrier is not supplied
by aggregate conservation alone.

No currently established general finite-curve representation justifies
removing R>1 guards. A narrowly exact merge is possible only after proving
there is no no-service gap, intermediate observable boundary, eligibility
change, or normalized-resource/PRIVATE change anywhere in the repeated source
region; then it is the same continuous single-owner stream with larger total
work. The actual multi-stage FACE R4 workloads have not met those conditions.
Do not use this restricted observation to claim their support.

## Recommended verification for a unified availability guard

- Cover layer_instances and weight_passes independently and their product;
  zero/absent overhead, positive launch, setup-only, and positive launch+setup.
- Guard actual repeated regions independently of whether applicable setup is
  zero. For a legitimate no-HBM region, setup is inapplicable but its region
  barrier still exists. The current valid public operator fixtures all issue
  physical HBM; do not invent a residency option or disable HBM to fake that
  fixture. If a no-HBM builder becomes available, test it explicitly.
- Preserve all R1 wire/hash and numeric timing exactly. Preserve old imported
  Plans and snapshots; add truthful diagnostics only to newly compiled plans.
- Retain physical epoch wire, Q/PRIVATE/credit, service-description count,
  body-only latency, byte/ops and workspace when only availability is fixed.
- Check public enumeration, compiler (including ranking candidates and
  compact GEMM materialization), estimate_alone, CABI full/cost-only and
  require_complete_prediction agree. Guard before resource acquisition,
  clock advance and completion publication in consumers remains necessary.
- Cover residual, Norm, GEMM/derived matrix ops, vector ops, forced fused and
  materialized FFN/Attention, QKV+RoPE, legacy normalized task inputs, and
  huge repeats without new internal events. Include overflow cases and
  repeated in-flight diagnostics cancel/snapshot to show the Runtime still
  permits body diagnostics while production consumers reject them.

This review is not a full test suite, speed acceptance, R>1 model solution,
five-repository release, or independent hardware calibration.
