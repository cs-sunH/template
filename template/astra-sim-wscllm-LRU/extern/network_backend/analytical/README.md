# astra-network-analytical

> 注：历史文档《1D-2D拓扑修改说明.md》已于 2026-09-26 移除——其 §七
> 验证命令指向的 `sh_test/`、`analytical/test` 与 congestion_unaware
> 路径均已不存在，不能再作为可执行验证入口；2D mesh（维度序路由/多维编址）
> 的现行实现以 `congestion_aware/topology/MultiDimTopology.cpp` 为准
> （fluid-net-C2 勘正 2026-09-27：运行时配置 `topology-by-network-dimension`
> 恒两维、经多维路径建拓扑——config_resolver 生成 [Line,Line]；此前注记
> 指向的 `basic-topology/Mesh.cpp`/`Mesh.h` 实现为 1D 链，按其读会错过
> XY 路由实现）。

## Overview
Analytical network simulator models communications over multi-dimensional topologies through analytical equations. Currently, the `congestion_aware` analytical network simulator is supported.

This simulator is developed as a part of the [ASTRA-sim](https://github.com/astra-sim/astra-sim) project, thereby the analytical network simulator can naturally be used as the network modeling backend of the ASTRA-sim simulator.

## Build Status
| main branch | macOS | Ubuntu |
|:---:|:---:|:---:|
| congestion_aware | [![build](https://github.com/astra-sim/astra-network-analytical/actions/workflows/test_congestion_aware_macos.yml/badge.svg?branch=main)](https://github.com/astra-sim/astra-network-analytical/actions/workflows/test_congestion_aware_macos.yml) | [![build](https://github.com/astra-sim/astra-network-analytical/actions/workflows/test_congestion_aware_ubuntu.yml/badge.svg?branch=main)](https://github.com/astra-sim/astra-network-analytical/actions/workflows/test_congestion_aware_ubuntu.yml) |

## Formatting
| main branch | format |
|:---:|:---:|
| format | [![format](https://github.com/astra-sim/astra-network-analytical/actions/workflows/check-clang-format.yml/badge.svg?branch=main)](https://github.com/astra-sim/astra-network-analytical/actions/workflows/check-clang-format.yml) |

## Documentation
- [Analytical Network Simulator Documentation](https://astra-sim.github.io/astra-network-analytical-docs/index.html)
- [ASTRA-sim Documentation](https://astra-sim.github.io/astra-sim-docs/index.html)

## References
- [ASTRA-sim website](https://astra-sim.github.io)
- [ASTRA-sim2.0 paper](https://arxiv.org/abs/2303.14006)
