#include <cmath>
// atlasim include
#include "core/vector_unit.h"


namespace atlasim {


VectorUnit::VectorUnit(const std::string _name_prefix, const ArchConfig arch_config)
    : Component(_name_prefix+"_Vector", ComponentType::VECTOR) {
    vec_num = arch_config->vector_config.vec_num;
    power = arch_config->vector_config.power;
    frequency = arch_config->frequency;

    cur_vec_computation_count = 0;
    total_vec_computation_count = 0;
}


double VectorUnit::get_cur_op_count() const {
    return cur_vec_computation_count;
}


double VectorUnit::get_total_op_count() const {
    return total_vec_computation_count;
}


void VectorUnit::reset_op_count(bool reset_total) {
    cur_vec_computation_count = 0;
    if(reset_total)
        total_vec_computation_count = 0;
}


bool VectorUnit::check_input(ComponentInput input) {
    for(size_t item_idx = 0; item_idx < input->size(); ++item_idx) {
        auto& d = input->at(item_idx);
        if(d.find("vec_count") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `vec_count`."
            );
        }
    }
    return true;
}


uint64_t VectorUnit::simulate(ComponentInput input) {
    if(!check_input(input)) {
        print_input(input, name);
        exit(-1);
    }

    int cur_vec_count = 0;
    for(int i=0; i<(*input).size(); ++i)
        cur_vec_count += (*input)[i].at("vec_count");

    cur_vec_computation_count += cur_vec_count;
    total_vec_computation_count += cur_vec_count;
    return uint64_t(std::ceil((double)cur_vec_count / vec_num));
}


}
