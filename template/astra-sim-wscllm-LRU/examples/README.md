# examples/ — 上游继承示例（仅裸 C++ / 上游形态）

本目录为上游 astra-sim 继承件，面向**裸 C++ 上游调用形态**；本仓的在线全栈链路
（Python resolver → generated runtime_config → execution-driven 引擎）不消费本
目录，README（仓根）全文亦无引用。各条目在本仓约束下的可用性：

| 文件/目录 | 本仓在线链路 | 说明 |
| --- | --- | --- |
| `network/analytical/HGX-H100-validated.yml` | **不可用** | topology `Switch`：本仓 `MultiDimTopology` 构造仅接受 Mesh/Ring/FullyConnected，其余维度值打印诊断后 `std::exit(-1)`（extern/network_backend/analytical/congestion_aware/topology/MultiDimTopology.cpp，构造函数维度校验环），该示例在本仓在线二进制下必死。 |
| `remote_memory/analytical/no_memory_expansion.json` | **不可用（超售）** | `NO_MEMORY_EXPANSION` 被 Python resolver 白名单拒绝（sh_test_mesh/config_resolver.py，M-28）；C++ 后端单独仍接受——本仓全栈语义下为超售配置，仅裸 C++ 后端形态可用。 |
| `network/analytical/Ring_8npus.yml`、`Ring_16npus.yml` | 上游合法形态 | topology `Ring` 在本仓接受，但键面按上游形态，未经本仓 config_resolver 键集校验，不构成在线链路配置样例。 |
| `system/native_collectives/Ring_4chunks.json`、`HGX-H100-validated.json` | 上游合法形态 | ring 系统模板键面为上游合法形态；非本仓 generated runtime_config（14+5 键）口径。 |
| `system/custom_collectives/`（custom_collective.json + 8 个 .et） | 唯一自洽示例 | 键面与 CollectiveImplLookup 一致、引用的 .et 文件实存；`custom` 实现走 CustomAlgorithm/ETFeeder 动态加载路径。 |

## et_def gencode 再生口径注记（RC-S4-4，记录级）

`extern/graph_frontend/chakra/schema/protobuf/` 下两套 gencode 出处口径不同
（内容等价已证，勿单一口径再生）：

- `et_def.pb.cc`/`et_def.pb.h`：以 `-I <schema 目录>` 生成（descriptor 表名/
  内嵌文件名为短名 `et_def.proto`）；
- `et_def_pb2.py`：以 `-I <仓根>` 生成（内嵌全路径
  `extern/.../et_def.proto`）。

如需再生：C++ 侧保持 `-I <schema 目录>`、Python 侧保持 `-I <仓根>`，两套口径
不可互换统一，否则产生大面积噪声 diff（C++/Python descriptor 池不相交，运行时
无影响）。
