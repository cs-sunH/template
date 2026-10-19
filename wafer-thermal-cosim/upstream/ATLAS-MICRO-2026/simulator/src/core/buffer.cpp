#include <cmath>
// atlasim include
#include "core/buffer.h"


namespace atlasim {


Buffer::Buffer(const std::string _name_prefix, const ArchConfig arch_config)
    : Component(_name_prefix+"_Buffer", ComponentType::BUFFER) {
    buffer_size = arch_config->buffer_config.buffer_size;
    read_bw = arch_config->buffer_config.read_bw;
    write_bw = arch_config->buffer_config.write_bw;
    power = arch_config->buffer_config.power;
    frequency = arch_config->frequency;

    cur_read_byte = 0;
    cur_write_byte = 0;
    total_read_byte = 0;
    total_write_byte = 0;
}


double Buffer::get_cur_op_count() const {
    return cur_read_byte + cur_write_byte;
}


double Buffer::get_total_op_count() const {
    return total_read_byte + total_write_byte;
}


void Buffer::reset_op_count(bool reset_total) {
    cur_read_byte = 0;
    cur_write_byte = 0;
    if(reset_total) {
        total_read_byte = 0;
        total_write_byte = 0;
    }
}


bool Buffer::check_input(ComponentInput input) {
    if(input->size()>2) {
        return fail_component_input(
            name,
            "input item count must be at most 2."
        );
    }
    for(size_t item_idx = 0; item_idx < input->size(); ++item_idx) {
        auto& d = input->at(item_idx);
        if(d.size() != 1) {
            return fail_component_input_item(
                name,
                item_idx,
                "each item must contain exactly one field."
            );
        }
        if(d.find("read")==d.end() && d.find("write")==d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `read` or `write`."
            );
        }
    }
    return true;
}


uint64_t Buffer::simulate(ComponentInput input) {
    if(!check_input(input)) {
        print_input(input, name);
        exit(-1);
    }

    int rbyte = 0;
    int wbyte = 0;
    for(auto d: *input) {
        if(d.find("read") != d.end()) {
            rbyte += d.at("read");
        }
        else if(d.find("write") != d.end()) {
            wbyte += d.at("write");
        }
    }

    cur_read_byte += rbyte;
    cur_write_byte += wbyte;
    total_read_byte += rbyte;
    total_write_byte += wbyte;
    uint64_t read_cycle = uint64_t(std::ceil(rbyte / read_bw));
    uint64_t write_cycle = uint64_t(std::ceil(wbyte / write_bw));
    return std::max(read_cycle, write_cycle);
}


}
