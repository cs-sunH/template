#ifndef _ATLASIM_CHIP_H
#define _ATLASIM_CHIP_H

#include <memory>
#include <fstream>
// booksim include
#include "include/booksim_config.hpp"
// atlasim include
#include "common/arch.h"
#include "common/task.h"
#include "common/data.h"
#include "common/stats.h"
#include "core/core_array.h"
#include "noc/noc_wrapper.h"


namespace atlasim {


class Chip {
private:
    // Basic settings
    std::string arch_config_path;
    std::string operator_list_path;
    std::string placement_map_path;
    ArchConfig arch_config;
    BookSimConfig booksim_config;
    OperatorList operator_list;
    PlacementMapCollection placement_maps;

    // Interface classes between PEs and NoC wrapper
    PCNInterfaceSet send_queues;     
    PCNInterfaceSet received_queues;
    std::shared_ptr<std::vector<bool>> credit_board;
    // Components
    std::shared_ptr<NoCWrapper> noc;
    std::shared_ptr<CoreArray> core_array;

    std::vector<uint64_t> prev_clk;
    uint64_t clk;

    // Simulation records
    std::shared_ptr<std::ofstream> log_file;
    std::streambuf* cout_buf = nullptr;
    std::streambuf* cerr_buf = nullptr;
    
    // For deadlock checking
    using QueueSnapshot = std::vector<std::deque<Packet>>;
    std::pair<QueueSnapshot, QueueSnapshot> last_queues_state;

public:
    Chip(
        std::string _arch_config_path,
        std::string _operator_list_path,
        std::string _placement_map_path,
        std::string _log_file_path = ""
    );
    ~Chip();
    const ArchConfig get_arch_config() const { return arch_config; }

    // Used for simulation
    void reset_execution_status(std::string _operator_list_path = "");
    // Simulate until all PEs finish all operators
    Performance simulate();
    // Invoked by simulate
    void tick();
    
    // Used for status checking
    bool is_finished();
    bool check_deadlock();
    
    // Used for statistics display
    double get_max_dram_bw_per_core() { 
        // The unit is GB/s
        return core_array->get_max_dram_bw_per_core(); 
    }
    double get_max_compute_capacity_per_core() {
        // The unit is GFLOP/s
        return 2 * arch_config->matrix_config.mac_num * arch_config->frequency * 1e-3;
    }
    double get_comp_bw_ratio() {
        return get_max_compute_capacity_per_core() / get_max_dram_bw_per_core();
    }
    double get_max_dram_bw() { return core_array->get_max_dram_bw(); }
    void display_stats(std::ostream& os = std::cout);
};


}


#endif // _ATLASIM_CHIP_H
