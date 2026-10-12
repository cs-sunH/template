/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/DataSet.hh"

#include <utility>

#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

int DataSet::id_auto_increment = 0;

DataSet::DataSet(int total_streams)
    : DataSet(total_streams, Sys::boostedTick()) {}

DataSet::DataSet(int total_streams, Tick /*creation_tick*/) {
    this->my_id = id_auto_increment++;
    this->total_streams = total_streams;
    this->finished_streams = 0;
    this->active = true;
    this->notifier = nullptr;
}

DataSet::~DataSet() {
    delete notifier;
}

void DataSet::set_notifier(Callable* callable, EventType event) {
    // Broadcast used to set the notifier twice, leaking the first pair.
    delete notifier;
    notifier = new std::pair<Callable*, EventType>(callable, event);
}

void DataSet::retain_communicator_group(
    std::shared_ptr<CommunicatorGroup> communicator_group) {
    communicator_group_owner_ = std::move(communicator_group);
}

void DataSet::fire_completion() {
    Callable* c = notifier->first;
    EventType ev = notifier->second;
    delete notifier;
    notifier = nullptr;
    IntData* int_data = new IntData(my_id);
    c->call(ev, int_data);
    delete int_data;
}

void DataSet::notify_stream_finished(StreamStat* data) {
    // The StreamStat payload is identity-only since the write-only statistic
    // chain was removed (face-LRU audit 2026-09-29).
    (void)data;
    finished_streams++;
    if (finished_streams == total_streams) {
        if (notifier != nullptr) {
            fire_completion();
        }
    }
}

void DataSet::call(EventType event, CallData* data) {
    notify_stream_finished(((StreamStat*)data));
}
