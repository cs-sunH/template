#include <cassert>
#include <cmath>
// atlasim include
#include "core/noc_interface.h"
#include "common/packet.h"


namespace atlasim {


NoCInterface::NoCInterface(
    const std::string _name_prefix, const ArchConfig arch_config, 
    int _threshold,
    int _core_id, 
    CNInterface _send_queue,
    CNInterface _received_queue,
    std::shared_ptr<std::vector<Packet>> _send_buffer,
    std::shared_ptr<std::vector<Packet>> _received_buffer,
    std::shared_ptr<std::vector<bool>> _pipe_open
    ) : Component(_name_prefix+"_NoC", ComponentType::NOC),
    has_noc(arch_config->has_noc),
    core_id(_core_id),
    threshold(_threshold),
    send_queue(_send_queue),
    received_queue(_received_queue),
    send_buffer(_send_buffer),
    received_buffer(_received_buffer),
    pipe_open(_pipe_open) {
    
    flit_size = has_noc ? arch_config->noc_config.flit_size : 0;
    power = has_noc ? arch_config->noc_config.power : 0;
    frequency = arch_config->frequency;

    total_send_flit_count = 0;
    cur_send_flit_count = 0;
}


void NoCInterface::reset_op_count(bool reset_total) {
    cur_send_flit_count = 0;
    if(reset_total) {
        total_send_flit_count = 0;
    }
}


double NoCInterface::get_cur_op_count() const {
    return cur_send_flit_count;
}


double NoCInterface::get_total_op_count() const {
    return total_send_flit_count;
}


bool NoCInterface::check_input(ComponentInput input) {
    for(size_t item_idx = 0; item_idx < input->size(); ++item_idx) {
        auto& d = input->at(item_idx);
        if(d.find("is_send") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `is_send`."
            );
        }
        if(d.find("src") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `src`."
            );
        }
        if(d.find("dst") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `dst`."
            );
        }
        if(d.find("flit_num") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `flit_num`."
            );
        }
        if(d.find("init_flit_id") == d.end()) {
            return fail_component_input_item(
                name,
                item_idx,
                "missing required field `init_flit_id`."
            );
        }
    }
    return true;
}


uint64_t NoCInterface::simulate(ComponentInput input) {
    if(!has_noc && !input->empty()) {
        std::cerr << "[ERROR] NoC input is not supported when architecture has no NoC." << std::endl;
        exit(-1);
    }

    if(!check_input(input)) {
        print_input(input, name);
        exit(-1);
    }

    for(auto d: *input) {
        int src = d["src"];
        int dst = d["dst"];
        bool is_send = d["is_send"];

        Packet p;
        p.fid = d["init_flit_id"];
        p.size = d["flit_num"];
        p.type = Packet::TransferType::_UNICAST;
        p.path = std::make_shared<MCTree>(src);
        p.path->add_segment(src, dst, nullptr, true);

        if(is_send) {
            assert(src == core_id);
            send_buffer->push_back(p);
            cur_send_flit_count += p.size;
            total_send_flit_count += p.size;
        }
        else {
            assert(dst == core_id);
            received_buffer->push_back(p);
        }
    }

    return 0;
}


bool NoCInterface::doorbell(const std::set<int>& dests) {
    bool dests_all_free = true;
    for(auto dst: dests) {
        assert(dst >= 0 && dst < pipe_open->size());
        dests_all_free &= (*pipe_open)[dst];
    }

    bool src_channel_available = send_queue->size() < threshold;

    return dests_all_free && src_channel_available;
}


void NoCInterface::tick(bool is_busy) {
    clk++; 
    if(is_busy)
        busy_cycles++;

    if(!has_noc) {
        return;
    }

    (*pipe_open)[core_id] = received_queue->size() < threshold;

    // Push packets into send queue
    int send_buffer_offset = 0;
    while(send_buffer_offset < send_buffer->size()) {
        Packet p = send_buffer->at(send_buffer_offset);
        if(doorbell(p.path->get_dest_nodes())) {
            send_queue->push_back(p);
            send_buffer->erase(send_buffer->begin() + send_buffer_offset);
        }
        else {
            send_buffer_offset++;
        }
    }

    // Remove packets from receive queue
    int recv_queue_offset = 0;
    while(recv_queue_offset < received_queue->size()) {
        Packet p = received_queue->at(recv_queue_offset);
        
        bool found = false;
        for(int i=0; i<received_buffer->size(); ++i) {
            Packet p_to_recv = received_buffer->at(i);
            if(p_to_recv.fid == p.fid && p_to_recv.path->root() == p.path->root()) {
                received_buffer->erase(received_buffer->begin() + i);
                found = true;
                break;
            }
        }

        if(found) {
            received_queue->erase(received_queue->begin() + recv_queue_offset);
        }
        else {
            recv_queue_offset++;
        }
    }
}


}
