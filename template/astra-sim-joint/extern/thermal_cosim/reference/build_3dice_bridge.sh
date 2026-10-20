#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ICE_ROOT="${THREEDICE_ROOT:-$REPO_DIR/upstream/3d-ice}"
SUPERLU_ROOT="$ICE_ROOT/superlu_mt-4.0.0"
OUTPUT="$REPO_DIR/build/ice_bridge"
LOG_DIR="${LOG_DIR:-$REPO_DIR/local-results/bridge-logs}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$LOG_DIR" "$(dirname -- "$OUTPUT")"

for tool in make gcc flex bison unzip; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required build tool not found: $tool" >&2
    exit 2
  fi
done
if [[ ! -f "$ICE_ROOT/Makefile" || ! -f "$ICE_ROOT/superlu_mt-4.0.0.zip" ]]; then
  echo "3D-ICE checkout/embedded SuperLU source missing at $ICE_ROOT" >&2
  exit 2
fi
if [[ ! -e /lib/x86_64-linux-gnu/libopenblas.so && ! -e /usr/lib/x86_64-linux-gnu/libopenblas.so ]]; then
  echo "3D-ICE's makefile requires the system OpenBLAS development symlink" >&2
  exit 2
fi

if [[ ! -d "$SUPERLU_ROOT" ]]; then
  unzip -q "$ICE_ROOT/superlu_mt-4.0.0.zip" -d "$ICE_ROOT"
fi

# The bundled SuperLU-MT 4.0 predates the compiler's C23 default and assumes
# that its LP64 SuiteSparse_long is `long`, matching 3D-ICE's LUIndex_t.
# Preserve pristine local copies before applying build-only compatibility edits.
if [[ ! -e "$SUPERLU_ROOT/SRC/Makefile.3d-ice-adapter.bak" ]]; then
  cp "$SUPERLU_ROOT/SRC/Makefile" "$SUPERLU_ROOT/SRC/Makefile.3d-ice-adapter.bak"
fi
if rg -q 'sreadmt\.o|dreadmt\.o' "$SUPERLU_ROOT/SRC/Makefile"; then
  sed -i 's/sreadmt\.o //g; s/dreadmt\.o //g' "$SUPERLU_ROOT/SRC/Makefile"
fi
if [[ ! -e "$SUPERLU_ROOT/SRC/colamd.h.3d-ice-adapter.bak" ]]; then
  cp "$SUPERLU_ROOT/SRC/colamd.h" "$SUPERLU_ROOT/SRC/colamd.h.3d-ice-adapter.bak"
fi
/usr/bin/python3 - "$SUPERLU_ROOT/SRC/colamd.h" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()
old = '''#if 0    /* commented out by Sherry */
#define SuiteSparse_long long
#define SuiteSparse_long_max LONG_MAX
#define SuiteSparse_long_idd "ld"
#endif
#define SuiteSparse_long long long int
#define SuiteSparse_long_max LONG_MAX
#define SuiteSparse_long_idd "lld"
'''.replace("\n+", "\n")
new = '''#if 1    /* 3D-ICE LP64 compatibility: use long to match int_t/LUIndex_t */
#define SuiteSparse_long long
#define SuiteSparse_long_max LONG_MAX
#define SuiteSparse_long_idd "ld"
#else
#define SuiteSparse_long long long int
#define SuiteSparse_long_max LONG_MAX
#define SuiteSparse_long_idd "lld"
#endif
'''.replace("\n+", "\n")
if "3D-ICE LP64 compatibility: use long to match int_t/LUIndex_t" in s:
    raise SystemExit(0)
if old not in s:
    raise SystemExit("Unexpected bundled colamd.h; stop rather than patching an unknown SuperLU source")
p.write_text(s.replace(old, new, 1))
PY

if [[ ! -f "$SUPERLU_ROOT/make.inc" ]]; then
  cp "$SUPERLU_ROOT/MAKE_INC/make.linux.openmp" "$SUPERLU_ROOT/make.inc"
fi
sed -i 's/^BLASDEF.*/BLASDEF   = -DUSE_VENDOR_BLAS/' "$SUPERLU_ROOT/make.inc"
sed -i 's@^BLASLIB .*@BLASLIB = -L/lib/x86_64-linux-gnu -lopenblas@' "$SUPERLU_ROOT/make.inc"
sed -i 's/^CDEFS.*/CDEFS        = -DAdd_/' "$SUPERLU_ROOT/make.inc"

SUPERLU_CLEAN_LOG="$LOG_DIR/superlu-clean.$RUN_TAG.log"
SUPERLU_BUILD_LOG="$LOG_DIR/superlu-double-build.$RUN_TAG.log"
ICE_CLEAN_LOG="$LOG_DIR/3dice-clean.$RUN_TAG.log"
ICE_BUILD_LOG="$LOG_DIR/3dice-build.$RUN_TAG.log"
BRIDGE_BUILD_LOG="$LOG_DIR/ice-bridge-build.$RUN_TAG.log"

make -C "$SUPERLU_ROOT/SRC" clean >"$SUPERLU_CLEAN_LOG" 2>&1
make -C "$SUPERLU_ROOT/SRC" double \
  CFLAGS='-std=gnu17 -O3 -D__OPENMP -DDLONG -DPRNTlevel=0 -DDEBUGlevel=0' \
  NOOPTS='-std=gnu17 -O0' >"$SUPERLU_BUILD_LOG" 2>&1
test -s "$SUPERLU_ROOT/lib/libsuperlu_mt_OPENMP.a"

# Clean compiled objects so no objects built with a different int ABI are reused.
make -C "$ICE_ROOT/sources" clean >"$ICE_CLEAN_LOG" 2>&1
make -C "$ICE_ROOT/lib" clean >>"$ICE_CLEAN_LOG" 2>&1
make -C "$ICE_ROOT/bin" clean >>"$ICE_CLEAN_LOG" 2>&1
make -C "$ICE_ROOT" all CFLAGS='-MD -O3 -fopenmp -std=gnu17' >"$ICE_BUILD_LOG" 2>&1
test -s "$ICE_ROOT/lib/libthreed-ice-4.0.0.a"
test -x "$ICE_ROOT/bin/3D-ICE-Emulator"

"${CC:-gcc}" -std=gnu17 -O2 -fopenmp \
  -I"$ICE_ROOT/include" -I"$SUPERLU_ROOT/SRC" \
  "$REPO_DIR/adapters/ice_bridge.c" \
  "$ICE_ROOT/lib/libthreed-ice-4.0.0.a" \
  "$SUPERLU_ROOT/lib/libsuperlu_mt_OPENMP.a" \
  -L/lib/x86_64-linux-gnu -lopenblas -lm -ldl \
  -o "$OUTPUT" >"$BRIDGE_BUILD_LOG" 2>&1
chmod +x "$OUTPUT"

printf 'Built %s\n' "$OUTPUT"
printf '3D-ICE commit: '
git -C "$ICE_ROOT" rev-parse HEAD
printf 'SuperLU double library: %s\n' "$SUPERLU_ROOT/lib/libsuperlu_mt_OPENMP.a"
printf 'Logs: %s\n' "$LOG_DIR"
