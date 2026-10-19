#ifndef _ATLASIM_BUFFER_H
#define _ATLASIM_BUFFER_H

#include "common/arch.h"
#include "core/component.h"
#include <cstdint>


namespace atlasim {


class Buffer: public Component {
public:
    Buffer(const std::string _name_prefix, const ArchConfig arch_config);

    virtual void reset_op_count(bool reset_total = false) override;
    virtual uint64_t simulate(ComponentInput input) override;

    virtual double get_cur_op_count() const override;
    virtual double get_total_op_count() const override;

private:
    // Buffer property
    int buffer_size;
    double read_bw, write_bw; // The unit is Byte/cycle

    // Runtime statistics
    double cur_read_byte, cur_write_byte;
    double total_read_byte, total_write_byte;

    virtual bool check_input(ComponentInput input) override;
};


}


#endif // _ATLASIM_BUFFER_H
