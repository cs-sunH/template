## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

## Use Ubuntu
FROM ubuntu:22.04
LABEL maintainer="Will Won <william.won@gatech.edu>"
LABEL maintainer="Jinsun Yoo <jinsun@gatech.edu>"


### ================== System Setups ======================
## Install System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
# clang-format-18 via the llvm.org apt source (S2-2/G-confirm 2026-09-28):
# the shared .clang-format uses InsertNewlineAtEOF (needs >=16) and
# InsertBraces (needs >=15); ubuntu:22.04's stock clang-format (14) rejects
# the config outright, breaking the documented formatting toolchain.
RUN apt -y update && apt -y install wget gnupg lsb-release ca-certificates
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
      > /etc/apt/sources.list.d/llvm.list \
    && apt -y update
RUN apt -y install \
    coreutils wget vim git \
    gcc g++ clang-format-18 \
    make cmake \
    python3.11 python3.11-venv

## Create Python venv: pinned to python3.11 (the unversioned python3 on
## ubuntu:22.04 is 3.10 and would silently diverge from the documented
## toolchain)
RUN python3.11 -m venv /opt/venv/astra-sim
ENV PATH="/opt/venv/astra-sim/bin:$PATH"
RUN pip3 install --upgrade pip

## Add astra-sim to PYTHONPATH
ENV PYTHONPATH="/app/astra-sim"
# (2026-09-28 cleanup, G-confirm S4-1) STG deps numpy/sympy/graphviz/pandas
# removed: zero importers/callers across this repository (apt graphviz
# likewise). PYTHONPATH itself is carried over from the upstream image for
# layout compatibility: no in-repo Python depends on it (all Python lives
# under sh_test_mesh/ and manages sys.path relative to its own files).
# pytest is installed below because the README's unified Python test entry
# (run_python_unit_tests.sh / python3 -m pytest) must be runnable in-container.
### ======================================================


### ====== Abseil Installation: Protobuf Dependency ======
## Download Abseil 20240722.0 (Latest LTS as of 10/31/2024)
ARG ABSL_VER=20240722.0

# Download source
WORKDIR /opt
RUN wget https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz
RUN tar -xf abseil-cpp-${ABSL_VER}.tar.gz
RUN rm abseil-cpp-${ABSL_VER}.tar.gz

## Compile Abseil
WORKDIR /opt/abseil-cpp-${ABSL_VER}/build
RUN cmake .. \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="/opt/abseil-cpp-${ABSL_VER}/install"
RUN cmake --build . --target install --config Release --parallel $(nproc)
ENV absl_DIR="/opt/abseil-cpp-${ABSL_VER}/install"
### ======================================================


### ============= Protobuf Installation ==================
## Download Protobuf 29.0 (=v5.29.0, latest stable version as of Feb/01/2025)
ARG PROTOBUF_VER=29.0

# Download source
WORKDIR /opt
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz
RUN tar -xf protobuf-${PROTOBUF_VER}.tar.gz
RUN rm protobuf-${PROTOBUF_VER}.tar.gz

## Compile Protobuf
WORKDIR /opt/protobuf-${PROTOBUF_VER}/build
RUN cmake .. \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_BUILD_TYPE=Release \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_ABSL_PROVIDER=package \
    -DCMAKE_INSTALL_PREFIX="/opt/protobuf-${PROTOBUF_VER}/install"
RUN cmake --build . --target install --config Release --parallel $(nproc)
ENV PATH="/opt/protobuf-${PROTOBUF_VER}/install/bin:$PATH"
ENV protobuf_DIR="/opt/protobuf-${PROTOBUF_VER}/install"

# Also, install Python protobuf package, plus pytest: the README's unified
# Python regression entry (run_python_unit_tests.sh, i.e. per-root
# python3 -m pytest) runs inside this venv and fails bare without it.
RUN pip3 install protobuf==5.${PROTOBUF_VER} pytest

# Set the environment variable
ENV PROTOBUF_FROM_SOURCE=True
### ======================================================


### ================== Finalize ==========================
## Move to the application directory
WORKDIR /app/astra-sim
### ======================================================
