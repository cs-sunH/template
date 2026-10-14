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

DataSet::DataSet(int total_streams, Tick creation_tick) {
    this->my_id = id_auto_increment++;
    this->total_streams = total_streams;
    this->finished_streams = 0;
    this->finished = false;
    this->finish_tick = 0;
    this->active = true;
    this->creation_tick = creation_tick;
    this->notifier = nullptr;
}

DataSet::~DataSet() {
    delete notifier;
}

void DataSet::set_notifier(Callable* callable, EventType event) {
    // Broadcast used to set the notifier twice, leaking the first pair.
    delete notifier;
    notifier = new std::pair<Callable*, EventType>(callable, event);
    // (2026-10-01 R2 审查修复) 零相 collective（全部维度为单点/未涉）
    // 在 generate_collective 内就已 notify 完成——彼时 notifier 尚未
    // 安装，完成事件丢失、发起节点永不完成。此处对已完成数据集**就地
    // 补发**一次（正常路径 streams 后续才完成，finished 恒 false，零行
    // 为变化）。
    if (finished) {
        fire_notifier_once();
    }
}

void DataSet::retain_communicator_group(
    std::shared_ptr<CommunicatorGroup> communicator_group) {
    communicator_group_owner_ = std::move(communicator_group);
}

void DataSet::notify_stream_finished(StreamStat* data) {
    // The StreamStat payload is identity-only for the removed write-only
    // statistic chain (face-LRU audit 2026-09-29 mirrored to joint); the
    // bus-stat accounting lives on the SharedBusStat payloads themselves.
    (void)data;
    finished_streams++;
    // >= (not ==): a zero-phase collective completes the dataset with
    // total_streams == 0 via an explicit notify; in normal flow each
    // stream notifies exactly once so == and >= coincide.
    if (finished_streams >= total_streams) {
        finished = true;
        finish_tick = Sys::boostedTick();
        fire_notifier_once();
    }
}

void DataSet::fire_notifier_once() {
    // Single delivery point for the completion notification; consumes the
    // notifier (one-shot) exactly like the historic inline body did.
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

void DataSet::call(EventType event, CallData* data) {
    notify_stream_finished(((StreamStat*)data));
}
