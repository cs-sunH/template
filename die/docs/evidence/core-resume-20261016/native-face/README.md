# Completed FACE native evidence

The copied configure/build/test records are complete and their log hashes match their original identities. Configure and build passed. The native consumer CTest ran four tests with no skips: WorkspaceAdmission, DieModelAdapter and LocalHbmDieMemory passed; DieExecutionLifecycle aborted, so overall remains failed (3/4, CTest exit8). No tests/builds were rerun by this review.

The failure occurs in Case J after two assertions preserving the real emitted FACE FFN task/provenance and real GraphBatch parser payload. The unchanged task has two layers and two weight passes (R4). The first issue_dep_free_nodes call prepares/compiles the task, then require_complete_prediction throws the exact logged prediction_not_available error before workspace admission. Mapping marks repeated-region timing incomplete; the consumer retains the required fail-closed guard.

This is a confirmed current availability limitation, not a successful lifecycle test. Case J workspace/cancellation assertions after the first issue call and later lifecycle cases including Case Q were not reached. Do not remove the guard, shrink this original fixture to R1, or claim those assertions passed. review.json records the source/hash reasoning and its limits.

Only completed FACE records and FACE CTest output are archived here. The native matrix and other repositories were still running when this review began; no changing overall summary or partial other-stage logs were copied as completed evidence. Source identities and four executable hashes are saved; no ELF was copied.
