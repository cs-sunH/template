#ifndef _ATLASIM_MATRIX_UNIT_H
#define _ATLASIM_MATRIX_UNIT_H

#include "common/arch.h"
#include "core/component.h"


namespace atlasim {


class MatrixUnit: public Component {
public:
    MatrixUnit(const std::string _name_prefix, const ArchConfig arch_config);

    virtual void reset_op_count(bool reset_total = false) override;
    virtual uint64_t simulate(ComponentInput input) override;

    virtual double get_cur_op_count() const override;
    virtual double get_total_op_count() const override;

private:
    // Matrix unit property
    int mac_num;

    // Runtime statistics
    double cur_mac_computation_count;
    double total_mac_computation_count;

    virtual bool check_input(ComponentInput input) override;
};


}


#endif // _ATLASIM_MATRIX_UNIT_H
