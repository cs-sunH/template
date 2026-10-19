#include "core/component.h"


namespace atlasim {


ComponentInput make_component_input(
    std::initializer_list<std::unordered_map<std::string, int64_t>> list
) {
    return std::make_shared<std::vector<std::unordered_map<std::string, int64_t>>>(list);
}


ComponentInputItem make_component_input_item(
    std::initializer_list<std::pair<const std::string, int64_t>> list
) {
    return std::unordered_map<std::string, int64_t>(list);
}


void print_input(ComponentInput input, std::string module_name, bool err) {
    if(err)
        std::cerr << "[ERROR] Module " << module_name << " has incorrect input format. ";
    else
        std::cout << "Print module " << module_name << "'s input. ";
    std::cout << "Input data contains:"<< std::endl;

    for(int i=0; i<(*input).size(); ++i) {
        std::cout << "Input data " << i+1 << ":" << std::endl;
        for(auto d: (*input)[i]) {
            std::cout << "    " << d.first << ": " << d.second << std::endl;
        }
    }
}


bool fail_component_input(const std::string& module_name, const std::string& message) {
    std::cerr << "[ERROR] " << module_name << " input: " << message << std::endl;
    return false;
}


bool fail_component_input_item(
    const std::string& module_name,
    std::size_t item_idx,
    const std::string& message
) {
    std::cerr << "[ERROR] " << module_name << " input item " << item_idx << ": " << message << std::endl;
    return false;
}


Component::Component()
    : name("")
    , type(ComponentType::NONE) {
    clk = 0;
    busy_cycles = 0;
    power = 0.0;
    frequency = 0.0;
    cur_begin_clk = 0;
}


Component::Component(const std::string _name, const ComponentType _type)
    : name(_name)
    , type(_type) {
    clk = 0;
    busy_cycles = 0;
    power = 0.0;
    frequency = 0.0;
    cur_begin_clk = 0;
}


void Component::reset_execution_status(bool reset_total) {
    cur_begin_clk = clk;
    reset_op_count(reset_total);
}


double Component::get_cur_energy() const {
    if(frequency <= 0) {
        return 0.0;
    }
    return (clk - cur_begin_clk) / (frequency * 1e6) * power;
}


double Component::get_total_energy() const {
    if(frequency <= 0) {
        return 0.0;
    }
    return clk / (frequency * 1e6) * power;
}


std::string Component::gen_stats_str() {
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
        emitter << YAML::Key << name;
        emitter << YAML::Value;
        emitter << YAML::BeginMap;
            emitter << YAML::Key << "total_cycle";
            emitter << YAML::Value << clk;
            emitter << YAML::Key << "busy_cycle";
            emitter << YAML::Value << busy_cycles;
            emitter << YAML::Key << "util";
            emitter << YAML::Value << get_util();
            emitter << YAML::Key << "energy (Joule)";
            emitter << YAML::Value << get_total_energy();
        emitter << YAML::EndMap;
    emitter << YAML::EndMap;
    emitter << YAML::Newline;

    std::string stats_str(emitter.c_str());
    return stats_str;
}


void Component::display_stats(std::ostream& os) {
    os << gen_stats_str() << std::endl;
}


}
