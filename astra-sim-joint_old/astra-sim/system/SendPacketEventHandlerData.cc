/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/SendPacketEventHandlerData.hh"

using namespace AstraSim;

SendPacketEventHandlerData::SendPacketEventHandlerData() {
    tag = 0;
    callable = nullptr;
    wlhd = nullptr;
}
