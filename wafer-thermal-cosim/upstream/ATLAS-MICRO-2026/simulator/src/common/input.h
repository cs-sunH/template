#ifndef _ATLASIM_INPUT_H
#define _ATLASIM_INPUT_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
// yaml-cpp include
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
// atlasim include
#include "util/sample.h"


namespace atlasim {


// Each std::vector<int> element is one KV tile's information.
// Each int value is one token's index in the whole KV cache tensor.
// Each std::vector<int> element's size should be <= token_tile_size;
typedef std::vector<std::vector<int>> ReqTileInfo;


struct AttnInput {
    // Virtual destructor to make the class polymorphic (required for dynamic_cast)
    virtual ~AttnInput() = default;
    
    // Attention shape information
    int q_head_num;
    int kv_head_num;
    int q_head_num_per_kv;
    // Since QK has different head dim against V in MLA, we directly adopt different variables
    int qk_head_dim;
    int v_head_dim;
    
    // Tensor information of input/output/kv-cache 
    std::string input_tensor_name;
    std::string output_tensor_name;
    std::string kv_cache_tensor_name;
    // KV cache tensor's shape is: (kv_head_num, total_slot_num, qk_head_dim + v_head_dim)
    // (Since we use fused attn, placing each token's k&v consecutively can better utilize row buffer)
    int total_slot_num;
    // Therefore, each kv_head's addr gap is element_size * total_slot_num * 2 * head_dim
    int64_t kv_head_addr_offset;

    AttnInput() { }
    AttnInput(const YAML::Node config) {
        q_head_num = config["q_head_num"].as<int>();
        kv_head_num = config["kv_head_num"].as<int>();
        q_head_num_per_kv = config["q_head_num_per_kv"].as<int>();
        qk_head_dim = config["qk_head_dim"].as<int>();
        v_head_dim = config["v_head_dim"].as<int>();

        input_tensor_name = config["input_tensor_name"].as<std::string>();
        output_tensor_name = config["output_tensor_name"].as<std::string>();
        kv_cache_tensor_name = config["kv_cache_tensor_name"].as<std::string>();
        total_slot_num = config["total_slot_num"].as<int>();
        kv_head_addr_offset = config["kv_head_addr_offset"].as<int64_t>();
    }

    virtual void display_stats(std::ostream& os=std::cout, int indent=0) {
        os << std::setw(indent) << "" << "Attention Input:" << std::endl;

        os << std::setw(indent) << "" << "(1) Shape information:" << std::endl;
        os << std::setw(indent) << "" << "    Q Head Num: " << q_head_num << std::endl;
        os << std::setw(indent) << "" << "    KV Head Num: " << kv_head_num << std::endl;
        os << std::setw(indent) << "" << "    Q Head Num Per KV: " << q_head_num_per_kv << std::endl;
        os << std::setw(indent) << "" << "    QK Head Dim: " << qk_head_dim << std::endl;
        os << std::setw(indent) << "" << "    V Head Dim: " << v_head_dim << std::endl;

        os << std::setw(indent) << "" << "(2) Tensor information:" << std::endl;
        os << std::setw(indent) << "" << "    Input Tensor Name: " << input_tensor_name << std::endl;
        os << std::setw(indent) << "" << "    Output Tensor Name: " << output_tensor_name << std::endl;
        os << std::setw(indent) << "" << "    KV Cache Tensor Name: " << kv_cache_tensor_name << std::endl;
        os << std::setw(indent) << "" << "    Total Slot Num: " << total_slot_num << std::endl;
        os << std::setw(indent) << "" << "    KV Head Addr Offset: " << kv_head_addr_offset << std::endl;
    }
};


struct DAttnCoreInput {
    int core_id;
    // Request intra-batch indices which need to be computed on current core
    // Since each core may only have the KV blocks of request subset, we need to record request indices,
    // which are used to compute DRAM addresses.
    std::vector<int> request_indices;
    // Each request's KV tile count
    std::vector<int> request_tile_count;
    std::vector<ReqTileInfo> request_tile_info_list;

    DAttnCoreInput() { }
    DAttnCoreInput(
        const YAML::Node config,
        int pid,
        int token_tile_size, // token number for on-chip computation
        int total_slot_num,
        int block_size, // token number for KV cache management
        bool random,
        int random_seed
    ) {
        core_id = pid;
        request_indices = config["request_indices"].as<std::vector<int>>();
        request_tile_count = config["request_tile_count"].as<std::vector<int>>();
        int total_block_num = total_slot_num / block_size;

        if(random) {
            // Randomly sample all requests' all token slots into one vector
            int total_access_block_num = 0;
            for(int i=0; i<request_indices.size(); ++i) {
                total_access_block_num += int(std::ceil(request_tile_count[i]*token_tile_size*1.0 / block_size));
            }
            std::vector<int> access_block_list = atlasim::sample_unique<int>(
                0,
                total_block_num,
                total_access_block_num,
                random_seed,
                true
            );
            
            // Re-organize sampled token slots into each request's token tile record
            request_tile_info_list.clear();
            int cur_block_idx = -1;
            int cur_intra_block_idx = 0;
            for(int i=0; i<request_indices.size(); ++i) {
                int cur_request_tile_count = request_tile_count[i];
                cur_block_idx++;
                cur_intra_block_idx = 0;
                
                ReqTileInfo cur_req_tile_info;
                for(int j=0; j<cur_request_tile_count; ++j) {
                    std::vector<int> cur_tile_token_list;
                    for(int k=0; k<token_tile_size; ++k) {
                        cur_tile_token_list.push_back(access_block_list[cur_block_idx]*block_size + cur_intra_block_idx);
                        cur_intra_block_idx++;
                        if(cur_intra_block_idx==block_size) {
                            cur_block_idx++;
                            cur_intra_block_idx = 0;
                        }
                    }
                    cur_req_tile_info.push_back(cur_tile_token_list);
                }
                request_tile_info_list.push_back(cur_req_tile_info);
            }
        }
        else {
            request_tile_info_list = config["request_tile_info_list"].as<std::vector<ReqTileInfo>>();
        }
    }

    void display_stats(std::ostream& os=std::cout, int indent=0) {
        os << std::setw(indent) << "" << "Core " << core_id << " decoding request information:" << std::endl;
        os << std::setw(indent) << "" << "    Request Num: " << request_indices.size() << std::endl;
        os << std::setw(indent) << "" << "    Request Tile Info List: " << std::endl;
        for(int i=0; i<request_indices.size(); ++i) {
            os << std::setw(indent) << "" << "        Request " << i << " (Intra-batch index: " << request_indices[i] << ") ";
            os << "Token Tile Count: " << request_tile_count[i] << std::endl;
        }  
    }
};


// Decoding stage's attention input format
// This information should be provided by outside's runtime simulator
struct DAttnInput: AttnInput {
    // Token number in each on-chip compute tile
    // For decoding, we only need to consider KV cache's token tile size
    int token_tile_size;

    // Token number used for KV cache management
    int block_size;
    int total_block_num;

    // Each Core's decoding request information
    std::vector<DAttnCoreInput> core_input_list;

    DAttnInput() { }
    DAttnInput(const YAML::Node config): AttnInput(config) {
        token_tile_size = config["token_tile_size"].as<int>();
        block_size = config["block_size"].as<int>();
        total_block_num = total_slot_num / block_size;

        bool random = config["random"].as<bool>();
        int random_seed = config["random_seed"].as<int>();
        for(int cid=0; cid<config["core_input_list"].size(); ++cid) {
            YAML::Node core_input_node = config["core_input_list"][cid];
            core_input_list.push_back(DAttnCoreInput(
                core_input_node,
                cid, 
                token_tile_size,
                total_slot_num,
                block_size, 
                random,
                random_seed
            ));
        }
    }

    void display_stats(std::ostream& os=std::cout, int indent=0) override {
        os << std::setw(indent) << "" << "Decoding Attention Input:" << std::endl;

        os << std::setw(indent) << "" << "(1) Shape information:" << std::endl;
        os << std::setw(indent) << "" << "    Q Head Num: " << q_head_num << std::endl;
        os << std::setw(indent) << "" << "    KV Head Num: " << kv_head_num << std::endl;
        os << std::setw(indent) << "" << "    Q Head Num Per KV: " << q_head_num_per_kv << std::endl;
        os << std::setw(indent) << "" << "    QK Head Dim: " << qk_head_dim << std::endl;
        os << std::setw(indent) << "" << "    V Head Dim: " << v_head_dim << std::endl;

        os << std::setw(indent) << "" << "(2) Tensor information:" << std::endl;
        os << std::setw(indent) << "" << "    Input Tensor Name: " << input_tensor_name << std::endl;
        os << std::setw(indent) << "" << "    Output Tensor Name: " << output_tensor_name << std::endl;
        os << std::setw(indent) << "" << "    KV Cache Tensor Name: " << kv_cache_tensor_name << std::endl;
        os << std::setw(indent) << "" << "    Total Slot Num: " << total_slot_num << std::endl;
        os << std::setw(indent) << "" << "    KV Head Addr Offset: " << kv_head_addr_offset << std::endl;

        os << std::setw(indent) << "" << "(3) Decoding specific information:" << std::endl;
        os << std::setw(indent) << "" << "    Token Tile Size: " << token_tile_size << std::endl;
        
        os << std::setw(indent) << "" << "    Block Size: " << block_size << std::endl;
        os << std::setw(indent) << "" << "    Total Block Num: " << total_block_num << std::endl;
        
        os << std::setw(indent) << "" << "(4) Core Input List: " << std::endl;
        for(int i=0; i<core_input_list.size(); ++i) {
            core_input_list[i].display_stats(os, indent+4);
        }
    }
};


}


#endif // _ATLASIM_INPUT_H
