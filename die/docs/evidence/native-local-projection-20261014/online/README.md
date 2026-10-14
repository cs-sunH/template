# JOINT Online reproduction evidence

This subdirectory records the fresh `astra-sim-joint` build, focused ABI
checks, and two one-request production Online attempts. Source and binary
hashes bind the final build after the native-domain integer-clock check was
added.

| Check | Result | Evidence |
|---|---:|---|
| Online, projection, both Die ABIs, ledger ABI, and Native3 targets | PASS | `build-after-final-clock-validation.log` |
| Native projection CTests | 3/3 PASS | `native-ctest-final.log` |
| Native slot/dependency join fixture | PASS | `native-coarse-slot-join-build.log`, `native-coarse-slot-join-ctest.log` |
| JOINT config resolver with explicit `DIE_MODEL_CABI_PATH` and `DIE_COARSE_CABI_PATH` | 13/13 PASS, no skips | `python-config-resolver-13.log` |
| JOINT adapter/workspace checks with both Die ABIs | 45/45 PASS, no skips | `python-joint-focused-45.log` |
| Die plan and native coarse projection with explicit Die and projection ABIs | 17/17 PASS, no skips | `python-die-native-17.log` |
| Legacy Online tiny request using the final Online binary | 1 request completed; C++ and Python exit 0 | `legacy/` |
| Coarse-intended Online tiny request | Blocked by the current fail-closed provider guard; C++ exit 134, Python exit 1 | `coarse-intended/` |

The legacy Online log reports `accepted=1 completed=1 active=0`, three
GraphBatch commits, 192 terminal successes out of 192 nodes, and zero
incomplete requests. The coarse-intended run reaches delivery sequence zero,
then stops at `_JointDiePlanProvider` because the current provider still uses
legacy `hbm_share_divisor` and `service_factor` and has not been validated
against the coarse shared Runtime. The run preserves that guard result and
does not claim coarse Online completion.

There is no supported non-sh30 Online entry in the current tree:
`online_service.py` exposes strategy mode and constructs
`Sh30OnlineScheduler`; admission calls `_joint_cost_model` before policy
selection. The typed `die_task` trace branch is downstream of this admission
guard.

`online-audit.json` records the run inputs, exit states, and boundary. The
matching source, input, build-product, and log hashes are in
`source-binary-identity.json`. The `inputs/` files are small evidence copies;
the preparer in the analysis subdirectory recreates them and all generated
plans/configurations under `work/joint-reproduction/`.
