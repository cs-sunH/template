#!/bin/bash
# (2026-10-01) INERT 自检:打包文件已删,pip3 install 必失败——
# 提前短路给出明确提示,而不是让调用者撞 pip 报错。
if [ ! -f "${CHAKRA_DIR:?}/setup.py" ]; then
  echo "[install_chakra] inert: packaging files (setup.py etc.) were " >&2
  echo "removed from this vendored copy; the C++ build consumes the " >&2
  echo "vendored sources directly and does not need this install." >&2
  exit 1
fi

set -e

## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************
##
## NOTE (2026-09-28): this upstream helper is currently INERT in this repo --
## the chakra Python packaging files (setup.py/setup.cfg/pyproject.toml) were
## removed, so the `pip3 install "${CHAKRA_DIR}"` step below cannot succeed.
## The C++ build does not use it: the feeder is vendored at
## extern/graph_frontend/chakra and compiled by CMake directly. Kept for
## upstream reference only.


# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/.."
CHAKRA_DIR="${PROJECT_DIR:?}/extern/graph_frontend/chakra"
PARAM_DIR="${PROJECT_DIR:?}/extern/graph_frontend/param"

# start
echo "[ASTRA-sim] Installing Chakra Package..."

# install param: Chakra dependency in the upstream workflow (this copy is
# inert in this repo -- upstream reference only; see the NOTE above. The
# subsequent pip3 installs cannot succeed here because the Chakra Python
# packaging files were removed).
echo "[ASTRA-sim] Installing param/et_replay..."
echo ""
if [ ! -d "${PARAM_DIR:?}" ]; then
(
    cd "${PROJECT_DIR:?}"/extern/graph_frontend
    git clone https://github.com/facebookresearch/param.git
    cd param/et_replay
    git checkout 7b19f586dd8b267333114992833a0d7e0d601630
    pip3 install .
)
fi

# install Chakra
echo ""
echo "[ASTRA-sim] Installing Chakra..."
echo ""
pip3 install "${CHAKRA_DIR:?}"

# done
echo ""
echo "[ASTRA-sim] Chakra installation finished."
