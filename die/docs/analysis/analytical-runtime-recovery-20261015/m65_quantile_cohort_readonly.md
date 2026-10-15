# 65×65×4097 bounded-cohort check (read only)

**Latest user scope:** the existing 6.9163% solo-finish error already meets the new 10% target. These read-only quantile estimates are retained as prior analysis, not a reason to add production complexity or chase a tighter solo fit. The separately failing busy cases still require honest reporting.

For the single-Core FP32 fixture, the actual analytical finish is 0.219017413 s and the grouped discrete oracle is 0.235290865 s, an underestimate of 16.273452 ms (6.9163%). The compute ledger is conserved exactly; the discrepancy comes from M/N-tail K-load readiness gaps, not K-tail work or SRAM/LOCAL service.

As a size check only, exact output-class prefix counts produce a simple four-quantile `max(load, compute)` estimate of about 0.228588346 s (2.85% low). A three-segment grouping that keeps the full-width and tail classes more distinct estimates about 0.230545667 s (2.02% low). These are not fitted to the oracle, but neither is a production-precision certificate: the grouping loses within-segment order, tile-ready output release, and finite-buffer backpressure. It is evidence that a small fixed cohort description may improve this case, not that the mechanism is complete or meets the new tolerance.

Exact class-prefix counts are available from `GemmOSGeometry::class_output_count_in_range`; multiplying them by the existing per-class L/C/S ledger preserves total work. Exact finite-buffer readiness still needs an encoding for interleaved classes and start/drain boundaries. The four-segment version needs 12 L/C/S pieces before startup/drain; current OS service-description limit is 8. See [m65_n65_k4097_finish_diagnosis.md](m65_n65_k4097_finish_diagnosis.md) for source facts and oracle evidence.
