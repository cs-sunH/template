/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __DATASET_HH__
#define __DATASET_HH__

#include <memory>

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/StreamStat.hh"

namespace AstraSim {

class CommunicatorGroup;

// Callable via the deferred-completion path: issue_coll_comm registers the
// DataSet itself as the event callback (register_event), and the fired
// event lands in call() -> notify_stream_finished(nullptr).
class DataSet : public Callable, public StreamStat {
  public:
    DataSet(int total_streams);
    ~DataSet();
    void set_notifier(Callable* layer, EventType event);
    void call(EventType event, CallData* data) override;
    /// Keep the communicator definition used to build this collective alive
    /// until every stream has completed. Metadata may replace the map entry
    /// while this DataSet is still active.
    void retain_communicator_group(
        std::shared_ptr<CommunicatorGroup> communicator_group);
    void notify_stream_finished(StreamStat* data);

    static int id_auto_increment;
    int my_id;
    int total_streams;
    int finished_streams;
    bool finished;
    bool active;
    std::pair<Callable*, EventType>* notifier;

  private:
    std::shared_ptr<CommunicatorGroup> communicator_group_owner_;
};

}  // namespace AstraSim

#endif /* __DATASET_HH__ */
