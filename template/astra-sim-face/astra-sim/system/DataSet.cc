/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/DataSet.hh"

#include <utility>

#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/IntData.hh"

using namespace AstraSim;

int DataSet::id_auto_increment = 0;

DataSet::DataSet(int total_streams) {
    this->my_id = id_auto_increment++;
    this->total_streams = total_streams;
    this->finished_streams = 0;
    this->finished = false;
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
    if (total_streams == 0 && !finished) {
        // Zero-stream DataSet (a zero-byte non-custom collective generates
        // no stream, so notify_stream_finished can never fire -- the same
        // hang mode the custom-collective path guards against in
        // Sys::generate_collective). Nothing will ever complete it, so
        // complete it here and deliver the notification below instead of
        // leaving the workload waiting for CollectiveCommunicationFinished
        // forever.
        finished = true;
    }
    if (finished) {
        // Completion already happened before the notifier was registered:
        // fire immediately (same delivery as notify_stream_finished).
        Callable* c = notifier->first;
        EventType ev = notifier->second;
        delete notifier;
        notifier = nullptr;
        IntData* int_data = new IntData(my_id);
        c->call(ev, int_data);
        delete int_data;
    }
}

void DataSet::call(EventType event, CallData* data) {
    notify_stream_finished(((StreamStat*)data));
}

void DataSet::retain_communicator_group(
    std::shared_ptr<CommunicatorGroup> communicator_group) {
    communicator_group_owner_ = std::move(communicator_group);
}

void DataSet::notify_stream_finished(StreamStat* data) {
    // data is only a type tag for the notification; the per-stream stat
    // accumulation it used to feed was write-only end to end and was
    // removed (see StreamStat.hh).
    (void)data;
    finished_streams++;
    if (finished_streams == total_streams) {
        finished = true;
        if (notifier != nullptr) {
            Callable* c = notifier->first;
            EventType ev = notifier->second;
            delete notifier;
            notifier = nullptr;
            IntData* int_data = new IntData(my_id);
            c->call(ev, int_data);
            delete int_data;
        }
    }
}
