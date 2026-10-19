#include <cmath>
// atlasim include
#include "core/matrix_unit.h"


namespace atlasim {


MatrixUnit::MatrixUnit(const std::string _name_prefix, const ArchConfig arch_config)
    : Component(_name_prefix+"_Matrix", ComponentType::MATRIX) {
    mac_num = arch_config->matrix_config.mac_num;
    power = arch_config->matrix_config.power;
    frequency = arch_config->frequency;

    cur_mac_computation_count = 0;
    total_mac_computation_count = 0;
}


double MatrixUnit::get_cur_op_count() const {
    return cur_mac_computation_count;
}


double MatrixUnit::get_total_op_count() const {
    return total_mac_computation_count;
}


void MatrixUnit::reset_op_count(bool reset_total) {
    cur_mac_computation_count = 0;
    if(reset_total)
        total_mac_computation_count = 0;
}


bool MatrixUnit::check_input(ComponentInput input) {
    for(size_t item_idx = 0; item_idx < input->size(); ++item_idx) {
        auto& d = input->at(item_idx);
        if(d.find("mac_count") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `mac_count`."
            );
        }
    }
    return true;
}


uint64_t MatrixUnit::simulate(ComponentInput input) {
    if(!check_input(input)) {
        print_input(input, name);
        exit(-1);
    }

    int cur_mac_count = 0;
    for(int i=0; i<(*input).size(); ++i)
        cur_mac_count += (*input)[i].at("mac_count");

    cur_mac_computation_count += cur_mac_count;
    total_mac_computation_count += cur_mac_count;
    return uint64_t(std::ceil((double)cur_mac_count / mac_num));
}


}
