/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MY_PACKET_HH__
#define __MY_PACKET_HH__

#include <cstdint>

namespace AstraSim {

// Plain data carrier for in-flight collective packets. It used to derive from
// Callable with an empty call(), but it is never dispatched as a callback.
class MyPacket {
  public:
    MyPacket(int preferred_src, int preferred_dest);
    MyPacket(uint64_t msg_size, int preferred_src, int preferred_dest);

    int preferred_dest;
    int preferred_src;
    uint64_t msg_size;
};

}  // namespace AstraSim

#endif /* __MY_PACKET_HH__ */
