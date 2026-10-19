#ifndef _ATLASIM_DRAM_WRAPPER_H
#define _ATLASIM_DRAM_WRAPPER_H

#include <memory>
// ramulator include
#include "base/type.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"
// atlasim include
#include "common/arch.h"
#include "core/component.h"


namespace atlasim {


class DRAMWrapper: public Component {
public:
    DRAMWrapper(const std::string _name_prefix, const ArchConfig arch_config);

    virtual void reset_op_count(bool reset_total = false) override;
    virtual uint64_t simulate(ComponentInput input) override;

    Ramulator::Addr_t get_tx_size() const { return tx_bytes; }
    Ramulator::Addr_t get_ch_offset() const { return ch_offset; }
    Ramulator::Addr_t get_row_size() const { return row_size; }
    double get_max_bw() const { return max_bw; }

    virtual double get_cur_op_count() const override;
    virtual double get_total_op_count() const override;

private:
    std::string config_path;
    double clk_ratio; // core cycle count per DRAM cycle (DRAM is typically slower, so one DRAM cycle typically contains >1 core cycles)
    Ramulator::Addr_t tx_bytes;
    Ramulator::Addr_t ch_offset;
    Ramulator::Addr_t row_size; // The unit is byte
    double max_bw; // Maximum bandwidth of the DRAM, GB/s

    // Ramulator objects & records for cycle-accurate simulatiom
    std::shared_ptr<Ramulator::IFrontEnd> frontend;
    std::shared_ptr<Ramulator::IMemorySystem> memory_system;
    // Relative tick ratios
    int frontend_tick;
    int mem_tick;
    int tick_mult;

    // Runtime statistics
    double cur_read_byte, cur_write_byte;
    double total_read_byte, total_write_byte;

    virtual bool check_input(ComponentInput input) override;
};


}


#endif // _ATLASIM_DRAM_WRAPPER_H
