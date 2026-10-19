#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/apply_patch_series.sh" "simulator/src/noc/booksim2" "patches/booksim2.patch"
