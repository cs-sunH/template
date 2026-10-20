#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
HOTSPOT_DIR="${HOTSPOT_ROOT:-$REPO_DIR/upstream/HotSpot}"
BUILD_JOBS="${BUILD_JOBS:-2}"
BUILD_DIR="$REPO_DIR/build"
OUTPUT="$BUILD_DIR/hotspot_bridge"

if ! command -v make >/dev/null 2>&1; then
  echo "make is required to build HotSpot" >&2
  exit 2
fi
if ! command -v gcc >/dev/null 2>&1; then
  echo "gcc is required by the upstream HotSpot Makefile" >&2
  exit 2
fi
if ! command -v "${CC:-cc}" >/dev/null 2>&1; then
  echo "C compiler '${CC:-cc}' not found (set CC to a compiler executable)" >&2
  exit 2
fi
if [[ ! -f "$HOTSPOT_DIR/hotspot-iface.h" ]]; then
  echo "HotSpot source not found at $HOTSPOT_DIR (set HOTSPOT_ROOT to its checkout)" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
make -C "$HOTSPOT_DIR" -j"$BUILD_JOBS" all
"${CC:-cc}" -O2 -std=c99 -I"$HOTSPOT_DIR" \
  "$REPO_DIR/adapters/hotspot_bridge.c" \
  "$HOTSPOT_DIR/libhotspot.a" -lm \
  -o "$OUTPUT"
chmod +x "$OUTPUT"
printf 'Built %s\n' "$OUTPUT"
