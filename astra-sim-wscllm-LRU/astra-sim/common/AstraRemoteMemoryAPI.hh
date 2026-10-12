/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_MEMORY_API_HH__
#define __ASTRA_MEMORY_API_HH__

#include <cstdint>

namespace AstraSim {

class Sys;
class WorkloadLayerHandlerData;

class AstraRemoteMemoryAPI {
  public:
    virtual ~AstraRemoteMemoryAPI() = default;
    virtual void set_sys(int id, Sys* sys) = 0;
    // 所有权契约（2026-10-01 补注）：调用方 new 出的 wlhd 所有权在本次
    // 调用即转移给后端——后端取得唯一所有权（实现侧 reset 进作业），
    // 交付时经 release() 移交回 Workload，未交付即 shutdown 则由后端
    // 删除。调用方 issue 之后不得再触碰该指针（Workload::issue_remote_mem
    // 即按此契约调用）。
    virtual void issue(uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd) = 0;
};

}  // namespace AstraSim

#endif /* __ASTRA_MEMORY_API_HH__ */
