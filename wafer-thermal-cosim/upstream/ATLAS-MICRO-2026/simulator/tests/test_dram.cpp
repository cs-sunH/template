#include <cstdint>
#include <iostream>
#include <string>
#include <filesystem>
#include <cassert>
#include <cmath>
#include <vector>

// yaml-cpp include
#include <yaml-cpp/yaml.h>
// atlasim include
#include "common/arch.h"
#include "core/component.h"
#include "dram/dram_wrapper.h"
#include "util/sample.h"


#define _MHz 1e6
#define _GHz 1e9
#define _KB std::pow(2, 10)
#define _MB std::pow(2, 20)
#define _GB std::pow(2, 30)


namespace {

std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& shape, bool row_major) {
    std::vector<int64_t> strides(shape.size(), 1);
    if(row_major) {
        for(int dim = static_cast<int>(shape.size()) - 2; dim >= 0; --dim) {
            strides[dim] = strides[dim + 1] * shape[dim + 1];
        }
    } else {
        for(size_t dim = 1; dim < shape.size(); ++dim) {
            strides[dim] = strides[dim - 1] * shape[dim - 1];
        }
    }
    return strides;
}


atlasim::ComponentInputItem make_general_dram_input_item(
    int64_t base_addr,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    bool is_write,
    const std::vector<int64_t>& access_base,
    const std::vector<int64_t>& access_extent,
    int64_t element_size
) {
    auto item = atlasim::make_component_input_item({
        {"base_addr", base_addr},
        {"element_size", element_size},
        {"is_write", is_write},
        {"layout_rank", static_cast<int64_t>(shape.size())},
    });
    for(size_t dim = 0; dim < shape.size(); ++dim) {
        item["shape_" + std::to_string(dim)] = shape[dim];
        item["stride_" + std::to_string(dim)] = strides[dim];
    }
    for(size_t dim = 0; dim < access_base.size(); ++dim) {
        item["access_base_" + std::to_string(dim)] = access_base[dim];
        item["access_extent_" + std::to_string(dim)] = access_extent[dim];
    }
    return item;
}


atlasim::ComponentInputItem make_matrix_access_item(
    int64_t base_addr,
    int64_t rows,
    int64_t cols,
    bool row_major,
    bool is_write,
    int64_t access_rows,
    int64_t access_cols,
    int64_t element_size
) {
    return make_general_dram_input_item(
        base_addr,
        {rows, cols},
        make_contiguous_strides({rows, cols}, row_major),
        is_write,
        {0, 0},
        {access_rows, access_cols},
        element_size
    );
}

}  // namespace


void test_attention_access(
    atlasim::ArchConfig arch_config,
    atlasim::DRAMWrapper& dram,
    // KV tensor shape
    int64_t slot_num,
    int64_t kv_vector_bytes,
    // Access config
    int64_t block_size,
    int64_t access_block_num,
    int64_t seed = 42
) {
    // Calculate total block number and sample access block list.
    int64_t total_block_num = slot_num / block_size;
    if(access_block_num > total_block_num) {
        std::cerr << "Access block number is greater than total block number." << std::endl;
        exit(-1);
    }
    std::vector<int64_t> access_block_list = atlasim::sample_unique<int64_t>(0, total_block_num, access_block_num, seed, true);
    
    // Generate for each block's access.
    atlasim::ComponentInput input = atlasim::make_component_input({});
    for(int64_t i=0; i<access_block_list.size(); ++i) {
        int64_t cur_block_addr = access_block_list[i] * block_size * kv_vector_bytes;
        input->push_back(make_matrix_access_item(cur_block_addr, slot_num, kv_vector_bytes, true, false, block_size, kv_vector_bytes, 1));
    }

    // Simulate
    uint64_t cycles = dram.simulate(input);
    int64_t total_bytes = access_block_num * block_size * kv_vector_bytes;

    // Calculate performance metrics.
    double data_volume_gb = total_bytes / _GB;
    double latency_s = cycles / (arch_config->frequency * _MHz);
    double bw = data_volume_gb / latency_s;
    double max_bw = dram.get_max_bw();
    std::cout << "Data volume: " << data_volume_gb << " GB" << std::endl;
    std::cout << "Cycles: " << cycles << std::endl;
    std::cout << "Latency: " << latency_s << " s" << std::endl;
    std::cout << "BW: " << bw << " GB/s" << std::endl;
    std::cout << "BW Util: " << bw / max_bw << std::endl;
}


void test_matrix_access(
    atlasim::ArchConfig arch_config,
    atlasim::DRAMWrapper& dram,
    // GEMM size (M, K) * (K, N) = (M, N)
    int64_t M, int64_t K, int64_t N,
    // Tile size (tM, tK) * (tK, tN) = (tM, tN)
    int64_t tM, int64_t tK, int64_t tN,
    // Tensor placement method
    bool input_row_major, bool output_row_major, bool weight_row_major,
    // Element size
    int64_t element_size,
    // dataflow, 0: output stationary, 1: weight stationary
    int dataflow
) {
    // Each tensor's base address is aligned with DRAM row
    int64_t input_bytes = M * K * element_size;
    int64_t output_bytes = M * N * element_size;
    int64_t weight_bytes = K * N * element_size;
    int64_t input_row_count = std::ceil(input_bytes / dram.get_row_size());
    int64_t output_row_count = std::ceil(output_bytes / dram.get_row_size());
    int64_t weight_row_count = std::ceil(weight_bytes / dram.get_row_size());
    int64_t input_base_addr = 0;
    int64_t output_base_addr = input_row_count * dram.get_row_size();
    int64_t weight_base_addr = (input_row_count + output_row_count) * dram.get_row_size();

    // Induce tile number of each dim
    if((M%tM != 0) || (K%tK != 0) || (N%tN != 0)) {
        std::cerr << "M, K, N must be divisible by tM, tK, tN." << std::endl;
        exit(-1);
    }
    int64_t M_tile_num = M / tM;
    int64_t K_tile_num = K / tK;
    int64_t N_tile_num = N / tN;
    int64_t loop_count = M_tile_num * N_tile_num * K_tile_num;

    int64_t cur_input_addr = input_base_addr;
    int64_t cur_output_addr = output_base_addr;
    int64_t cur_weight_addr = weight_base_addr;
    int64_t cur_M = 0;
    int64_t cur_N = 0;
    int64_t cur_K = 0;
    int64_t prev_M = 0;
    int64_t prev_N = 0;
    uint64_t cycles = 0;
    if(dataflow == 0) {
        for(int64_t i=0; i<loop_count+1; ++i) {
            atlasim::ComponentInput input = atlasim::make_component_input({});
            // Load input and weight tile
            if(i<loop_count) {
                // Generate DRAM access transactions
                input->push_back(make_matrix_access_item(cur_input_addr, M, K, input_row_major, false, tM, tK, element_size));
                input->push_back(make_matrix_access_item(cur_weight_addr, K, N, weight_row_major, false, tK, tN, element_size));
            }
            // Store fully accumulated output tile
            if(i>0 && i%K_tile_num == 0) {
                if(output_row_major) {
                    cur_output_addr = output_base_addr + (prev_M*N + prev_N) * element_size;
                }
                else {
                    cur_output_addr = output_base_addr + (prev_N*M + prev_M) * element_size;
                }

                input->push_back(make_matrix_access_item(cur_output_addr, M, N, output_row_major, true, tM, tN, element_size));
            }

            cycles += dram.simulate(input);

            // Update tile offset and tile base addr
            if(i<loop_count) {
                prev_M = cur_M;
                prev_N = cur_N;
                cur_K += tK;
                if(cur_K >= K) {
                    cur_K = 0;
                    cur_N += tN;
                    if(cur_N >= N) {
                        cur_N = 0;
                        cur_M += tM;
                        if(cur_M >= M) {
                            if(i != loop_count-1) {
                                std::cerr << "i " << i << " is not equal to loop count - 1 " << loop_count-1 << "." << std::endl;
                                exit(-1);
                            }
                        }
                    }
                }
                if(input_row_major) {
                    cur_input_addr = input_base_addr + (cur_M*K + cur_K) * element_size;
                }
                else {
                    cur_input_addr = input_base_addr + (cur_K*M + cur_M) * element_size;
                }
                if(weight_row_major) {
                    cur_weight_addr = weight_base_addr + (cur_K*N + cur_N) * element_size;
                }
                else {
                    cur_weight_addr = weight_base_addr + (cur_N*K + cur_K) * element_size;
                }
            }
        }
    }
    else if(dataflow == 1) {
        for(int64_t i=0; i<loop_count+1; ++i) {
            atlasim::ComponentInput input = atlasim::make_component_input({});
            // Load input and output partial sum tiles
            // Load weight tile every M_tile_num iterations
            if(i<loop_count) {
                // Generate DRAM access transactions
                input->push_back(make_matrix_access_item(cur_input_addr, M, K, input_row_major, false, tM, tK, element_size));
                input->push_back(make_matrix_access_item(cur_output_addr, M, N, output_row_major, false, tM, tN, element_size));
                if(i%M_tile_num == 0) {
                    input->push_back(make_matrix_access_item(cur_weight_addr, K, N, weight_row_major, false, tK, tN, element_size));
                }
            }
            // Store output partial sum tile
            if(i>0) {
                int64_t prev_output_addr;
                if(output_row_major) {
                    prev_output_addr = output_base_addr + (prev_M*N + prev_N) * element_size;
                }
                else {
                    prev_output_addr = output_base_addr + (prev_N*M + prev_M) * element_size;
                }

                input->push_back(make_matrix_access_item(prev_output_addr, M, N, output_row_major, true, tM, tN, element_size));
            }

            cycles += dram.simulate(input);

            // Update tile offset and tile base addr
            if(i<loop_count) {
                prev_M = cur_M;
                prev_N = cur_N;
                cur_M += tM;
                if(cur_M >= M) {
                    cur_M = 0;
                    cur_K += tK;
                    if(cur_K >= K) {
                        cur_K = 0;
                        cur_N += tN;
                        if(cur_N >= N) {
                            if(i != loop_count-1) {
                                std::cerr << "i " << i << " is not equal to loop count - 1 " << loop_count-1 << "." << std::endl;
                                exit(-1);
                            }
                        }
                    }
                }
                if(input_row_major) {
                    cur_input_addr = input_base_addr + (cur_M*K + cur_K) * element_size;
                }
                else {
                    cur_input_addr = input_base_addr + (cur_K*M + cur_M) * element_size;
                }
                if(output_row_major) {
                    cur_output_addr = output_base_addr + (cur_M*N + cur_N) * element_size;
                }
                else {
                    cur_output_addr = output_base_addr + (cur_N*M + cur_M) * element_size;
                }
                if(weight_row_major) {
                    cur_weight_addr = weight_base_addr + (cur_K*N + cur_N) * element_size;
                }
                else {
                    cur_weight_addr = weight_base_addr + (cur_N*K + cur_K) * element_size;
                }
            }
        }
    }
    else {
        std::cerr << "Invalid dataflow: " << dataflow << std::endl;
        exit(-1);
    }

    // Use actual DRAM transfer volume (after request dedup) instead of
    // logical tile data volume, so that bw_util cannot exceed 100%.
    double actual_bytes = dram.get_total_op_count();
    double data_volume_gb = actual_bytes / _GB;
    double latency_s = cycles / (arch_config->frequency * _MHz);
    double bw = data_volume_gb / latency_s;
    double max_bw = dram.get_max_bw();
    std::cout << "Data volume: " << data_volume_gb << " GB" << std::endl;
    std::cout << "Cycles: " << cycles << std::endl;
    std::cout << "Latency: " << latency_s << " s" << std::endl;
    std::cout << "BW: " << bw << " GB/s" << std::endl;
    std::cout << "BW Util: " << bw / max_bw << std::endl;
}


int main(int argc, char** argv) {
    if(argc<3) {
        std::cout << "Usage: ./test_dram [CHIP_CONFIG] [TEST_TYPE]" << std::endl;
        exit(-1);
    }

    std::string chip_config_path(argv[1]);
    if(!std::filesystem::exists(chip_config_path)) {
        std::cerr << "Chip config path " << chip_config_path << " does not exist." << std::endl;
        exit(-1);
    }
    YAML::Node chip_config = YAML::LoadFile(chip_config_path);
    atlasim::ArchConfig arch_config = std::make_shared<atlasim::ChipConfig>(chip_config);

    atlasim::DRAMWrapper dram("test", arch_config);

    std::string test_type(argv[2]);
    if(test_type == "attention") {
        if(argc<7) {
            std::cout << "Usage: ./test_dram [CHIP_CONFIG] attention [SLOT_NUM] [KV_VECTOR_BYTES] [BLOCK_SIZE] [ACCESS_BLOCK_NUM] [SEED=42]" << std::endl;
            exit(-1);
        }
        int64_t slot_num = std::stoll(argv[3]);
        int64_t kv_vector_bytes = std::stoll(argv[4]);
        int64_t block_size = std::stoll(argv[5]);
        int64_t access_block_num = std::stoll(argv[6]);
        int64_t seed = 42;
        if(argc==8) {
            int64_t seed = std::stoll(argv[7]);
        }
        test_attention_access(
            arch_config, 
            dram,
            slot_num,
            kv_vector_bytes,
            block_size,
            access_block_num,
            seed
        );
    }
    else if(test_type == "matrix") {
        if(argc<11) {
            std::cout << "Usage: ./test_dram [CHIP_CONFIG] matrix [M] [K] [N] [tM] [tK] [tN] [INPUT_ROW_MAJOR] [OUTPUT_ROW_MAJOR] [WEIGHT_ROW_MAJOR] [ELEMENT_SIZE] [DATAFLOW]" << std::endl;
            exit(-1);
        }
        int64_t M = std::stoll(argv[3]);
        int64_t K = std::stoll(argv[4]);
        int64_t N = std::stoll(argv[5]);
        int64_t tM = std::stoll(argv[6]);
        int64_t tK = std::stoll(argv[7]);
        int64_t tN = std::stoll(argv[8]);
        bool input_row_major = std::stoi(argv[9]);
        bool output_row_major = std::stoi(argv[10]);
        bool weight_row_major = std::stoi(argv[11]);
        int64_t element_size = std::stoll(argv[12]);
        int dataflow = std::stoi(argv[13]);
        test_matrix_access(
            arch_config, 
            dram, 
            M, K, N, 
            tM, tK, tN, 
            input_row_major, 
            output_row_major, 
            weight_row_major, 
            element_size, 
            dataflow
        );
    }
    else {
        std::cerr << "Invalid test type: " << test_type << std::endl;
        exit(-1);
    }
    return 0;
}
