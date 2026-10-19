#ifndef _ATLASIM_NOC_INTERFACE_H
#define _ATLASIM_NOC_INTERFACE_H

#include "core/component.h"
#include "common/arch.h"
#include "common/packet.h"


namespace atlasim {


class NoCInterface: public Component {
public:
    NoCInterface(
        // Raw configs
        const std::string _name_prefix, const ArchConfig arch_config, 
        // Extracted from booksim config file
        int _threshold,
        // Extracted from core object
        int _core_id, 
        CNInterface _send_queue,
        CNInterface _received_queue, 
        std::shared_ptr<std::vector<Packet>> _send_buffer,
        std::shared_ptr<std::vector<Packet>> _received_buffer, 
        std::shared_ptr<std::vector<bool>> _pipe_open
    );

    virtual void reset_op_count(bool reset_total = false) override;
    virtual uint64_t simulate(ComponentInput input) override;
    virtual void tick(bool is_busy) override;

    virtual double get_cur_op_count() const override;
    virtual double get_total_op_count() const override;

private:
    // NoC property
    bool has_noc;
    int flit_size; // The unit is Byte
    int core_id;
    int threshold;

    // Runtime statistics
    double total_send_flit_count;
    double cur_send_flit_count;

    CNInterface send_queue, received_queue; // Pipe between NoCInterface and NoCWrapper (invoked in tick function)
    std::shared_ptr<std::vector<Packet>> send_buffer, received_buffer; // Buffer to store packets pre-generated in simulate function
    std::shared_ptr<std::vector<bool>> pipe_open;

    // Helper functions for NoC send/recv simulation
    bool doorbell(const std::set<int>& dests);

    virtual bool check_input(ComponentInput input) override;
};

}


#endif // _ATLASIM_NOC_INTERFACE_H
