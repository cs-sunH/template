#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/apply_patch_series.sh" "simulator/src/dram/ramulator2" "patches/ramulator2.patch"
