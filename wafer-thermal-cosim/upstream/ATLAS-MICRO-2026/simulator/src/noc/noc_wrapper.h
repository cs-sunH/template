#ifndef _ATLASIM_NOC_WRAPPER_H
#define _ATLASIM_NOC_WRAPPER_H

#include <memory>
// booksim include
#include "include/trafficmanager.hpp"
#include "include/booksim_config.hpp"
// atlasim include
#include "common/packet.h"


namespace atlasim {


class NoCWrapper {
private:
    // Basic settings
    BookSimConfig config;

    std::vector<uint64_t> prev_clk;
    uint64_t clk;

    // Components
    std::shared_ptr<TrafficManager> traffic_manager;

    // Reference counting for global booksim2 state management.
    // booksim2 uses static/global pools (Credit::_all, Flit::_free, gRoutingFunctionMap, etc.)
    // that are shared across all NoCWrapper instances. We must only clean them up
    // when the LAST NoCWrapper is destroyed.
    static int instance_count;

public:
    NoCWrapper(BookSimConfig _config, PCNInterfaceSet _send_queues, PCNInterfaceSet _received_queues);
    ~NoCWrapper();

    // Returns the number of active NoCWrapper instances.
    static int getInstanceCount() { return instance_count; }

    // Used for simulation
    void reset_execution_status();
    void tick();

    // Used for status checking
    std::vector<double> router_conflict_factors();
    bool traffic_drained() {
        return traffic_manager->flitsDrained();
    }

    // Used for statistics display
    void display_stats(std::ostream& os = std::cout);
};


}


#endif // _ATLASIM_NOC_WRAPPER_H