// ramulator include
#include "base/config.h"
#include <cstdint>
// atlasim include
#include "dram/dram_wrapper.h"


namespace atlasim {


DRAMWrapper::DRAMWrapper(const std::string _name_prefix, const ArchConfig arch_config)
    : Component(_name_prefix+"_DRAM", ComponentType::DRAM) {
    config_path = arch_config->dram_config.config_path;
    power = arch_config->dram_config.power;
    frequency = arch_config->frequency;

    // Create ramulator objects
    std::vector<std::string> params;
    YAML::Node ramulator_config = Ramulator::Config::parse_config_file(config_path, params);
    frontend = std::shared_ptr<Ramulator::IFrontEnd>(Ramulator::Factory::create_frontend(ramulator_config));
    memory_system = std::shared_ptr<Ramulator::IMemorySystem>(Ramulator::Factory::create_memory_system(ramulator_config));
    // Get relative clock ratio
    frontend_tick = frontend->get_clock_ratio();
    mem_tick = frontend->get_clock_ratio();
    tick_mult = frontend_tick * mem_tick;
    if(tick_mult != 1) {
        std::cerr << "[WARNING] Setting tick_mult (frontend_tick * memory_system_tick) > 1 will hurt simulation efficiency." << std::endl;
    }
    // Let frontend fetch necessary information from memory system.
    // These information will be used for memory transaction generation.
    frontend->connect_memory_system(memory_system.get());
    memory_system->connect_frontend(frontend.get());
    frontend->set_memory_system_info();

    // Get maximum bandwidth of the DRAM and tx_bytes
    tx_bytes = frontend->get_tx_bytes();
    max_bw = memory_system->get_max_bw();
    row_size = memory_system->get_row_size();
    ch_offset = frontend->get_ch_offset();

    // Calculate clock ratio
    double tCK_DRAM = memory_system->get_tCK(); // ns
    double tCK_core = 1.0 / (arch_config->frequency / 1e3); // ns
    clk_ratio = tCK_DRAM / tCK_core;

    cur_read_byte = 0;
    cur_write_byte = 0;
    total_read_byte = 0;
    total_write_byte = 0;
}


double DRAMWrapper::get_cur_op_count() const {
    return cur_read_byte + cur_write_byte;
}


double DRAMWrapper::get_total_op_count() const {
    return total_read_byte + total_write_byte;
}


void DRAMWrapper::reset_op_count(bool reset_total) {
    cur_read_byte = 0;
    cur_write_byte = 0;
    if(reset_total) {
        total_read_byte = 0;
        total_write_byte = 0;
    }
}


// Each item must provide:
//   (1) base_addr (of tensor origin)
//   (2) element_size (3) is_write
//   (4) layout_rank
//   (5) shape_<dim>, stride_<dim> for each dim
//   (6) access_base_<dim>, access_extent_<dim> for each dim
bool DRAMWrapper::check_input(ComponentInput input) {
    for(size_t item_idx = 0; item_idx < input->size(); ++item_idx) {
        auto& d = input->at(item_idx);
        if(d.find("base_addr") == d.end() || d.find("element_size") == d.end() || d.find("is_write") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required scalar fields among {base_addr, element_size, is_write}."
            );
        }

        auto rank_iter = d.find("layout_rank");
        if(rank_iter == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `layout_rank`."
            );
        }
        int64_t rank = rank_iter->second;
        if(rank <= 0) {
            return fail_component_input_item(
                name,
                item_idx,
                "`layout_rank` must be positive."
            );
        }
        for(int64_t dim = 0; dim < rank; ++dim) {
            if(d.find("shape_" + std::to_string(dim)) == d.end()) {
                return fail_component_input_item(
                    name,
                    item_idx,
                    "layout rank and provided shape fields do not match: missing `shape_" +
                    std::to_string(dim) + "` for layout_rank=" + std::to_string(rank) + "."
                );
            }
            if(d.find("stride_" + std::to_string(dim)) == d.end()) {
                return fail_component_input_item(
                    name,
                    item_idx,
                    "layout rank and provided stride fields do not match: missing `stride_" +
                    std::to_string(dim) + "` for layout_rank=" + std::to_string(rank) + "."
                );
            }
        }
        for(int64_t dim = 0; dim < rank; ++dim) {
            if(d.find("access_base_" + std::to_string(dim)) == d.end()) {
                return fail_component_input_item(
                    name,
                    item_idx,
                    "access window rank does not match layout_rank: missing `access_base_" +
                    std::to_string(dim) + "` for layout_rank=" + std::to_string(rank) + "."
                );
            }
            if(d.find("access_extent_" + std::to_string(dim)) == d.end()) {
                return fail_component_input_item(
                    name,
                    item_idx,
                    "access window rank does not match layout_rank: missing `access_extent_" +
                    std::to_string(dim) + "` for layout_rank=" + std::to_string(rank) + "."
                );
            }
        }
    }
    return true;
}


uint64_t DRAMWrapper::simulate(ComponentInput input) {
    if(!check_input(input)) {
        print_input(input, name);
        exit(-1);
    }

    if(input->size() == 0)
        return 0;

    // Record previous cycle counts, and send DRAM access transactions
    uint64_t prev_mem_clk = memory_system->get_clk();
    int num_dram_reqs = frontend->send(input);

    // Simulate untill all DRAM transactions are finished
    for(uint64_t i=0; ; ++i) {
        if(((i%tick_mult) % mem_tick) == 0) {
            frontend->tick();
        }

        if(((i%tick_mult) % frontend_tick) == 0) {
            memory_system->tick();
        }

        if(frontend->is_finished() && memory_system->is_finished()) {
            break;
        }
    }

    // Fetch finsh status and update performance records
    int cur_mem_clk = memory_system->get_clk();
    cur_read_byte += frontend->get_read_bytes();
    cur_write_byte += frontend->get_write_bytes();
    total_read_byte += frontend->get_read_bytes();
    total_write_byte += frontend->get_write_bytes();
    return std::ceil((cur_mem_clk-prev_mem_clk) * clk_ratio);
}


}
