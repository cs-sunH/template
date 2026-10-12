#!/bin/bash
set -e

## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/.."

# start
echo "[ASTRA-sim] Starting a Docker Container from astrasim/astra-sim:latest..."

# run an interactive container from the prebuilt astrasim/astra-sim image
# (image itself is built by utils/build_docker_image.sh)
docker run -i -t \
    -v "${PROJECT_DIR:?}:/app/astra-sim" \
    astrasim/astra-sim:latest \
    /bin/bash
