# Four non-JOINT ABI/config checks

All four active non-JOINT repositories built both actual Die C ABIs in Release
and passed their existing configuration suites (20+15+18+17=70, zero skips).
The canonical23 source manifests match digest
`45b7bf3d8df9969b8155323c79eecf1a81f8d1c7499177e811b6aedf2c4c3264`.

`summary.json` records exact commands, source and binary hashes, toolchain and
per-repository test results. `logs/` preserves configure/build/test outputs,
including warnings. The build directories and binary artifacts stay in ignored
work space and are intentionally not delivered. These checks do not certify
whole workloads, the new JOINT bridge, model accuracy or host overhead.
