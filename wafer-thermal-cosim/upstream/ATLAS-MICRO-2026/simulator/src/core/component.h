#ifndef _ATLASIM_COMPONENT_H
#define _ATLASIM_COMPONENT_H

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>


namespace atlasim {


typedef std::shared_ptr<std::vector<std::unordered_map<std::string, int64_t>>> ComponentInput;
typedef std::unordered_map<std::string, int64_t> ComponentInputItem;

ComponentInput make_component_input(std::initializer_list<std::unordered_map<std::string, int64_t>> list);
ComponentInputItem make_component_input_item(std::initializer_list<std::pair<const std::string, int64_t>> list);
void print_input(ComponentInput input, std::string module_name, bool err=true);

bool fail_component_input(const std::string& module_name, const std::string& message);
bool fail_component_input_item(
    const std::string& module_name,
    std::size_t item_idx,
    const std::string& message
);


enum ComponentType {
    MATRIX,
    VECTOR,
    BUFFER,
    DRAM,
    NOC,
    NONE,
};
static const std::unordered_map<std::string, ComponentType> str2ctype = {
    {"Matrix", ComponentType::MATRIX},
    {"Vector", ComponentType::VECTOR},
    {"Buffer", ComponentType::BUFFER},
    {"DRAM", ComponentType::DRAM},
    {"NoC", ComponentType::NOC},
};
static const std::unordered_map<ComponentType, std::string> ctype2str = {
    {ComponentType::MATRIX, "Matrix"},
    {ComponentType::VECTOR, "Vector"},
    {ComponentType::BUFFER, "Buffer"},
    {ComponentType::DRAM, "DRAM"},
    {ComponentType::NOC, "NoC"},
};


class Component {
public:
    const std::string name;
    const ComponentType type;

    Component();
    Component(const std::string _name, const ComponentType _type);

    virtual void reset_execution_status(bool reset_total = false);
    virtual void reset_op_count(bool reset_total = false) { };
    virtual uint64_t simulate(ComponentInput input) = 0;

    virtual void tick(bool is_busy) {
        clk++; 
        if(is_busy)
            busy_cycles++;
    }
    double get_util() const { return busy_cycles*1.0/clk; }

    virtual double get_cur_op_count() const { return 0; }
    virtual double get_total_op_count() const { return 0; }
    virtual double get_cur_energy() const;
    virtual double get_total_energy() const;

    virtual std::string gen_stats_str();
    virtual void display_stats(std::ostream& os = std::cout);

protected:
    uint64_t clk;
    uint64_t busy_cycles;
    double power; // The unit is W
    double frequency; // The unit is MHz
    uint64_t cur_begin_clk;

private:
    virtual bool check_input(ComponentInput input) = 0;
};


typedef std::unordered_map<std::string, std::shared_ptr<Component>> ComponentCollection;


};

#endif // _ATLASIM_COMPONENT_H
