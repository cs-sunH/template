#include "coarse_die/Capi.h"
#include <json/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
using nlohmann::json;
static void check(bool b,const char* why) { if(!b) throw std::runtime_error(why); }
static json take(const char* p) { check(p!=nullptr,die_coarse_last_error_v2()); auto j=json::parse(p); die_coarse_free_result_v2(p); return j; }
int main() {
 try {
    check(die_coarse_abi_version()==2,"ABI version");
    std::ifstream in(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware.json");
    json cfg; in>>cfg; DieHandle m=0,r=0,ex=0,pr=0;
    check(die_coarse_create_model_v2(cfg.dump().c_str(),&m)==DIE_OK,die_coarse_last_error_v2());
    auto hw=take(die_coarse_hardware_summary_v2(m)); check(hw.at("schema_version")==2,"summary version");
    check(die_coarse_create_runtime_v2(m,&r)==DIE_OK,die_coarse_last_error_v2());
    check(die_coarse_destroy_model_v2(m)==DIE_BUSY,"live runtime prevents model destruction");
    DieOwnerV2 ow{sizeof(DieOwnerV2),2,1,2,3,1};
    auto ep=json{{"endpoint_id",1},{"transaction_id",1},{"owner_kind","comm"},{"direction","read"},
      {"physical_bytes",1000000},{"path_class","hbm_to_edge"},{"eta_hbm",0.7},
      {"route",{{"internal_route",false}}}};
    check(die_coarse_submit_endpoint_v2(r,ep.dump().c_str(),&ow,0,&ex)==DIE_OK,die_coarse_last_error_v2());
    check(die_coarse_destroy_execution_v2(ex)==DIE_BUSY,"active execution lifecycle");
    check(die_coarse_destroy_runtime_v2(r)==DIE_BUSY,"active runtime lifecycle");
    auto next=take(die_coarse_next_event_v2(r)).at("next_event_ns").get<uint64_t>();
    check(die_coarse_advance_to_v2(r,next/2)==DIE_OK,die_coarse_last_error_v2());
    auto snap=take(die_coarse_export_snapshot_v2(r)); DieHandle fork=0;
    check(die_coarse_import_snapshot_v2(m,snap.dump().c_str(),&fork)==DIE_OK,die_coarse_last_error_v2());
    auto corrupt=snap; corrupt["schema"]="mesh_die_v1"; DieHandle bad=999;
    check(die_coarse_import_snapshot_v2(m,corrupt.dump().c_str(),&bad)!=DIE_OK && bad==0,"atomic invalid snapshot");
    check(die_coarse_advance_to_v2(r,next)==DIE_OK,die_coarse_last_error_v2());
    check(die_coarse_advance_to_v2(fork,next)==DIE_OK,die_coarse_last_error_v2());
    check(take(die_coarse_query_counters_v2(r))==take(die_coarse_query_counters_v2(fork)),"private fork counter replay");
    check(die_coarse_advance_to_v2(r,next-1)==DIE_INVALID_TIME,"time reversal");
    check(die_coarse_destroy_execution_v2(ex)==DIE_OK,die_coarse_last_error_v2());
    check(die_coarse_query_progress_v2(ex)==nullptr,"stale execution");
    check(die_coarse_destroy_runtime_v2(r)==DIE_OK,die_coarse_last_error_v2());
    check(die_coarse_destroy_runtime_v2(fork)==DIE_OK,die_coarse_last_error_v2());
    check(die_coarse_destroy_model_v2(m)==DIE_OK,die_coarse_last_error_v2());
    check(die_coarse_destroy_model_v2(m)==DIE_STALE_HANDLE,"stale model");
    check(die_coarse_create_model_v2("{}",&pr)!=DIE_OK,"reject incomplete config");
    std::cout<<"{\"cabi_lifecycle\":\"passed\",\"abi\":2}\n";
    return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
