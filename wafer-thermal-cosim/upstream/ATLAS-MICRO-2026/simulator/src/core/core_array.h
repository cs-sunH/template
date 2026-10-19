#ifndef _ATLASIM_CORE_ARRAY_H
#define _ATLASIM_CORE_ARRAY_H

#include <vector>
// atlasim include
#include "common/packet.h"
#include "core/core.h"


namespace atlasim {


class CoreArray {
private:
    // Basic settings
    const ArchConfig arch_config;
    const OperatorList operator_list;
    const PlacementMapCollection placement_maps;

    std::vector<uint64_t> prev_clk;
    uint64_t clk;

    // Components
    int core_num;
    std::vector<std::shared_ptr<Core>> cores;
    PCNInterfaceSet send_queues;
    PCNInterfaceSet received_queues;
    std::shared_ptr<std::vector<bool>> pipe_open;

    // For deadlock checking
    std::vector<std::shared_ptr<Core>> last_cores_state;

public:
    CoreArray(
        // Basic configs
        const ArchConfig _arch_config, const OperatorList _operator_list, const PlacementMapCollection _placement_maps,
        // Booksim settings
        int _threshold,
        // Interface classes between Cores and NoC wrapper
        PCNInterfaceSet _send_queues, PCNInterfaceSet _received_queues, std::shared_ptr<std::vector<bool>> _pipe_open
    );

    // Used for simulation
    void reset_execution_status();
    void tick();

    // Used for status checking
    bool state_changed();
    bool all_core_finished();

    // Used for statistics display
    double get_max_dram_bw_per_core() { return cores[0]->get_max_dram_bw(); }
    double get_max_dram_bw() { return cores[0]->get_max_dram_bw() * cores.size(); }
    Stats get_e2e_stats();
    OperatorStatsCollection get_all_op_stats();
    CoreStatsCollection get_core_stats();
    std::vector<std::pair<std::string, CoreStatsCollection>> get_operator_stats_per_core();
    Performance get_performance();
    void display_stats(std::ostream& os = std::cout);
};


}


#endif // _ATLASIM_CORE_ARRAY_H
