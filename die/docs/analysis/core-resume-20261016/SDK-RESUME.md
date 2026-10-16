# Resume an interrupted isolated SDK verification

`run_isolated_sdk.py --resume-from OLD_RESULTS --output NEW_RESULTS` can retain completed terminal stages after matching their exact command, cwd, explicit environment, dependency/expectation fields, public source digest, snapshot manifest SHA and log SHA. Original passed/failed status and original runner SHA remain unchanged; the copied original identity, provenance and new resume-runner/support identities are recorded separately. Dependencies must themselves be retained completed passes before a dependent old result can be retained.

A successful retained configure is also checked against the live CMakeCache source directory, generator, compiler and explicit configuration flags. An interrupted build can continue its same contemporary build tree only behind that retained configure. A new configure still rejects an existing unfamiliar build tree. Missing final identity, including the original WSCLLM partial build log, means incomplete, never pass. The old result root is never overwritten.

Every newly launched phase writes atomic `started.json` before subprocess launch, then updates child PID. It contains command/cwd/explicit env, source/manifest/tool hashes and actual target files before execution. Link metadata and CTest registration resolve native executable paths; both executable and library hashes are captured before/after, with existence/missing status. Historical records that never captured target hashes say `not_recorded_by_original_runner`; a current retention artifact snapshot is separately labeled and does not pretend to be the old before/after evidence.

JOINT's typed hardware step uses the existing current `probe_coarse_hardware_consumer.py` online-oracle entry. It does not modify the frozen historical typed-resolver helper or add a production coarse authoring resolver.

After sourcing `/workspace/cloud-setup/activate.sh`, inspect the dry-run before executing. This command was used for the interrupted native matrix:

```sh
python3 -B die/docs/analysis/core-resume-20261016/run_isolated_sdk.py \
  --snapshot work/core-resume-sdk \
  --output work/core-resume-sdk-validation/native-resumed \
  --resume-from work/core-resume-sdk-validation/native \
  --phases native --jobs 2
```

Add `--execute` to run. Full frozen source bytes are checked before execution and between phases; dry-run only checks the manifest/result identities and does not claim source execution. Failed R4/availability stages remain failures, including in the final exit status. Source snapshots/production SDKs are never published or edited.

Eight small filesystem-fixture tests in `test_sdk_resume_support.py` cover failed-state retention, command/cwd/env changes, source/manifest drift, log tampering, incomplete/nonterminal records, live configure-cache mismatch and executable digest drift. They run without CMake or a simulator process:

```sh
python3 -B die/docs/analysis/core-resume-20261016/test_sdk_resume_support.py
```

Original interruption evidence is in `die/docs/evidence/core-resume-20261016/native-interrupted/`. It records completed FACE/LRU failures, WS configure success, partial WS build log and historical absence of started.json. No incomplete stage is upgraded to passed.
