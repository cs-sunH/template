# ATLAS Simulator (MICRO '26)

This is the artifact of ATLAS, the first full-stack simulator for
3D-DRAM-based LLM accelerators. It combines a cycle-level performance
simulator, a Hotspot-based thermal simulator, and three software frontends that
generate simulator inputs from model descriptions or operator programs 
written by TileLang-style domain-specific languages.

## Overview

1. [Installation](#1-installation)
2. [Kick-the-Tires](#2-kick-the-tires)
3. [Instructions for Artifact Evaluation](#3-instructions-for-artifact-evaluation)
4. [Usage and Framework Extension](#4-usage-and-framework-extension)
5. [Developer Documentation](#5-developer-documentation)

## 1. Installation

Use Python 3.10 on a POSIX host with Git, Bash, Make, CMake, and a C/C++
compiler. The following commands are relative to the repository root and do
not depend on a machine-specific path.

```bash
# Create environment
# We recommend using conda to manage python environment.
# If you have not been installed conda, please refer to https://www.anaconda.com/docs/getting-started/miniconda/main for more instructions.
conda create -n atlas python=3.10 -y
conda activate atlas

# Initialize and patch the submodules
git submodule update --init --recursive
bash patches/unpatch_all.sh

# Install the Python and native-build dependencies
python -m pip install \
  "cmake>=3.26.1" ninja cython scikit-build-core \
  "patchelf>=0.17.2; platform_system == 'Linux'" \
  "apache-tvm-ffi==0.1.10" z3-solver \
  loguru matplotlib numpy pandas psutil pyyaml transformers==4.51.0

# Install the Python binding and build all native simulator executables
cmake -S simulator -B simulator/build
cmake --build simulator/build --parallel
python -m pip install -e simulator -v

# Build the hotspot simulator required for ATLAS thermal simulator
sudo apt install libblas-dev libsuperlu-dev
bash pyta/build_hotspot.sh

# Install TileLang in editable, CPU-only mode
CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON" \
  python -m pip install -e frontend/tilelang --no-build-isolation --no-deps -v
```

The TileLang command disables GTest discovery without modifying the TVM
submodule. If a platform still fails while configuring GTest, change
`set(USE_GTEST AUTO)` to `set(USE_GTEST OFF)` in
`frontend/tilelang/3rdparty/tvm/cmake/config.cmake`, remove
`frontend/tilelang/build`, and rerun the TileLang install command.

## 2. Kick-the-Tires

ATLAS provides three frontend passes for generating simulator inputs. Each
script below runs one cloud and one edge workload; the frontends are described
in more detail in [Section 4](#4-usage-and-framework-extension).

```bash
python tests/inference_test_auto.py
python tests/inference_test_tilelang.py
python tests/inference_test_atlang.py
```

The scripts use multiprocessing and write operator descriptions, placement
files, simulator logs, and final results under `kick_the_tires/`. 

TileLang and ATLang redirect simulator output to `kernel_output.log` in each 
workload directory. If you want to follow a running simulation, run the following command:

```bash
tail -f kick_the_tires/<workload-directory>/kernel_output.log
```

After simulation finishes, each workload directory contains `e2e_performance.csv`. 
The `e2e_latency_s` and `e2e_energy_j` columns report full-model latency and energy;
divide either value by `num_layers` for a per-layer value.

## 3. Instructions for Artifact Evaluation

The complete CPU-only artifact evaluation covers cloud and edge DRAM DSE and
cloud and edge chip DSE. See [docs/ae_instructions.md](docs/ae_instructions.md)
for the tested machine configuration, all-in-one commands, expected runtime,
output layout, and result validation procedure.

## 4. Usage and Framework Extension

ATLAS separates hardware definition, thermal correction, software description,
and cycle-level simulation. Extending the framework to another accelerator or
application follows the same four-stage flow.

### 4.1 Define the Hardware

Start from the examples under `configs/architecture/` and provide all files
referenced by the target system:

- A chip YAML such as `configs/architecture/chip/test_chip_16ch.yaml` defines
  frequency, core count, controller, matrix/vector engines, SRAM bandwidth and
  capacity, power/area data, DRAM, and optional NoC information.
- The chip YAML's `dram.config_path` points to a Ramulator configuration under
  `configs/architecture/dram/`. A new DRAM organization must provide a matching
  Ramulator organization, timing, address mapping, and frontend configuration.
- For a multi-core NoC design, `noc.config_path` points to a BookSim
  configuration such as `configs/architecture/noc/4x4_mesh`; `topology` and
  `flit_size` in the chip YAML must agree with it.
- A cloud or edge system YAML under `configs/architecture/system/` references
  the chip YAML and adds system-level parallelism, interconnect, channel/NPU,
  and KV-cache parameters.

Paths are part of the configuration contract. Keep all referenced files in the
repository or use portable paths relative to the repository root.

### 4.2 Apply Thermal Correction

Before performance evaluation, pass the physical dimensions, layer count,
cooling mode, and logic/DRAM power derived from the hardware definition to
`pyta.evaluator.thermal_evaluator`. The function runs HotSpot and returns the
maximum temperature and selected `frequency_MHz`. Apply that frequency to the
chip YAML used for performance simulation, then point the system YAML to the
thermally corrected chip YAML.

The chip DSE scripts demonstrate this flow in `_run_cloud_thermal` and
`_run_edge_thermal`: they create a per-configuration thermal directory, call
PyTA, and record `thermal_summary.csv` and `thermal_map.{png,pdf}` before
running the performance cases.

### 4.3 Describe the Software

Choose one of the three frontends according to the required level of
automation and programmability:

- **Auto frontend:** `frontend/auto/` automatically converts the LLMs and
  operator execution strategies evaluated in the paper into placement and
  simulator task descriptions. Model architecture fields are supplied by JSON
  files under `configs/models/`. Start from `tests/inference_test_auto.py` when
  evaluating the supported model families or adapting their shapes and system
  settings.
- **TileLang frontend:** `frontend/ops/tilelang/` expresses custom operators
  with the extended TileLang programming interface. It is appropriate when the
  same source should retain access to TileLang's GPU code-generation stack.
  Start from `tests/inference_test_tilelang.py` and the cloud/edge operator
  implementations.
- **ATLang frontend:** `frontend/ops/atlang/` uses an ATLAS-owned DSL aligned
  with the simulator-facing TileLang interface. TileLang's GPU compilation
  stack is intentionally heavy; ATLang keeps the programming model while
  providing a lightweight, extensible AST-to-simulator path for users who only
  need operator programming and performance simulation. Start from
  `tests/inference_test_atlang.py`.

TileLang and ATLang share the same simulator programming concepts: a core
array, SPMD/MPMD operator regions, kernel tiling, serial loops, tensor layouts,
memory movement, matrix/vector operations, reductions, and inter-core
communication. Their common API and required simulator parameters are defined
in [docs/programming_interface.md](docs/programming_interface.md).

### 4.4 Run and Inspect the Simulation

All three frontends ultimately generate the data-placement and operator YAML
files consumed by `atlasim.Chip`. Use a dedicated output directory for each
hardware/software pair, preserve the generated task descriptions for
reproducibility, and read the final latency and energy from the returned result
or the generated performance CSV. The three inference scripts show the full
path from hardware/model loading through frontend generation to simulator
execution.

## 5. Developer Documentation

- [Artifact evaluation instructions](docs/ae_instructions.md)
- [Common TileLang/ATLang programming interface](docs/programming_interface.md)
- [ATLang implementation architecture](docs/atlang_architecture.md)
