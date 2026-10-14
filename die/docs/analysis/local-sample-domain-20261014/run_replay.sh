#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${REPOSITORY_ROOT:-$(git -C "${HERE}" rev-parse --show-toplevel)}
BUILD_DIR=${BUILD_DIR:-${ROOT}/work/build-joint}
RUN_DIR=${RUN_DIR:-${ROOT}/work/local-sample-domain-20261014/portable-replay}
ASTRA=${ROOT}/astra-sim-joint
SAMPLES=${ROOT}/die/docs/evidence/sys-local-sample-20261014/samples
mkdir -p "${RUN_DIR}"
cmake --build "${BUILD_DIR}" -j2 --target NativeCoarseDieDomainTest \
  > "${RUN_DIR}/build.log" 2>&1
"${CXX:-c++}" -std=c++17 -O3 -DNDEBUG -ffp-contract=off \
  -I "${ASTRA}" -I "${ASTRA}/extern/helper" \
  -I "${ASTRA}/extern/die_model/coarse/include" \
  "${HERE}/replay_actual_sys_sample.cc" \
  "${BUILD_DIR}/lib/libjoint_pipeline_core.a" \
  "${BUILD_DIR}/AstraSim/extern/die_model/libdie_model.a" \
  "${BUILD_DIR}/AstraSim/extern/die_model/coarse/libcoarse_die.a" \
  -o "${RUN_DIR}/replay" > "${RUN_DIR}/replay-build.log" 2>&1
"${RUN_DIR}/replay" "${SAMPLES}/partial-workload-sample.json" \
  "${SAMPLES}/drained-workload-sample.json" \
  "${SAMPLES}/terminal-local-runtime-sample.json" \
  "${SAMPLES}/terminal-ticks.json" > "${RUN_DIR}/result.json"
python3 - "${SAMPLES}/drained-workload-sample.json" "${RUN_DIR}/result.json" <<'PY'
from decimal import Decimal
from pathlib import Path
import json, sys
source = json.loads(Path(sys.argv[1]).read_text())['local_runtime']['runtime_snapshot']
report_path = Path(sys.argv[2])
report = json.loads(report_path.read_text())
deltas = []
for change in report['final_runtime_diff']:
    value = source
    for key in change['path'].strip('/').split('/'):
        value = value[int(key)] if isinstance(value, list) else value[key]
    deltas.append({'path': change['path'], 'source_decimal': value,
                   'replay_decimal': change['value'],
                   'replay_minus_source_ns': str(Decimal(change['value']) - Decimal(value))})
report['ungraded_exact_busy_decimal_deltas'] = deltas
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['strict_runtime_wire_equal',
      'runtime_except_four_busy_accumulators_equal', 'actual_completions',
      'ungraded_exact_busy_decimal_deltas']}, indent=2))
PY
