#pragma once
#include "coarse_die/Plan.hh"
#include <limits>
#include <stdexcept>

namespace coarse_die {
// Conservative retained-allocation accounting. Uses the containers' actual
// capacities; includes node/control/allocator allowances. This is deliberately
// distinct from allocator RSS and from the smaller serialized JSON byte count.
inline std::size_t retained_json_bytes(const nlohmann::json& j) {
    std::size_t total=0;
    const auto add=[&](std::size_t bytes) {
        if(bytes>std::numeric_limits<std::size_t>::max()-total)
            throw std::overflow_error("overflow: retained plan memory");
        total+=bytes;
    };
    if(j.is_object()) {
        const auto& object=j.get_ref<const nlohmann::json::object_t&>();
        add(sizeof(object)+16);
        for(const auto& member:object) {
            add(sizeof(nlohmann::json::object_t::value_type)+4*sizeof(void*)+16);
            add(member.first.capacity()+1+16);
            add(retained_json_bytes(member.second));
        }
    } else if(j.is_array()) {
        const auto& array=j.get_ref<const nlohmann::json::array_t&>();
        add(sizeof(array)+array.capacity()*sizeof(nlohmann::json)+32);
        for(const auto& member:array) add(retained_json_bytes(member));
    } else if(j.is_string()) {
        const auto& string=j.get_ref<const nlohmann::json::string_t&>();
        add(sizeof(string)+string.capacity()+1+32);
    } else if(j.is_binary()) {
        add(sizeof(nlohmann::json::binary_t)+j.get_binary().capacity()+32);
    }
    return total;
}
inline std::size_t retained_plan_bytes(const Plan& p) {
    std::size_t total=sizeof(Plan)+64;
    const auto add=[&](std::size_t bytes) {
        if(bytes>std::numeric_limits<std::size_t>::max()-total)
            throw std::overflow_error("overflow: retained plan memory");
        total+=bytes;
    };
    add(retained_json_bytes(p.task)); add(retained_json_bytes(p.summary));
    add(p.fingerprint.capacity()+p.hardware_fingerprint.capacity()+34);
    add(p.epochs.capacity()*sizeof(Epoch)+16);
    for(const auto& e:p.epochs) {
        add(e.name.capacity()+17);
        add(e.activities.capacity()*sizeof(Activity)+16);
        for(const auto& a:e.activities)
            if(!a.credit_curve.empty()) add(a.credit_curve.capacity()*sizeof(CreditPoint)+16);
    }
    return total;
}
}
