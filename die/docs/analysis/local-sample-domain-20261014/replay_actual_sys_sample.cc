#include "astra-sim/workload/execution_driven/NativeCoarseDieDomain.hh"
#include <fstream>
#include <iostream>
#include <stdexcept>
using nlohmann::json;
using AstraSim::ExecutionDriven::NativeCoarseDieDomain;
json read(const char* path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error(std::string("cannot read ")+path);
  json value; file >> value; return value;
}
int main(int argc,char** argv) {
 try {
  if(argc!=5) throw std::runtime_error("usage: replay partial drained terminal ticks");
  const json partial=read(argv[1]), drained=read(argv[2]);
  const json terminal=read(argv[3]), ticks=read(argv[4]);
  const auto& local=partial.at("local_runtime");
  auto hardware=std::make_shared<const coarse_die::Hardware>(
      coarse_die::resolve_hardware(local.at("hardware_config")));
  // External leg ids are supplied here by the actual captured COMP/endpoint
  // catalog. The Domain does not infer graph or ledger identities.
  std::map<std::uint64_t,std::string> bindings;
  for(const auto& row: partial.at("comp_bindings"))
   if(!bindings.emplace(row.at("handle").get<std::uint64_t>(),"captured-comp:"+row.at("node_id").dump()).second)
    throw std::runtime_error("duplicate COMP handle");
  for(const auto& row: partial.at("endpoint_bindings"))
   if(!bindings.emplace(row.at("handle").get<std::uint64_t>(),"captured-local-hbm:"+row.at("node_id").dump()).second)
    throw std::runtime_error("duplicate endpoint handle");
  const json source_before=local;
  auto domain=NativeCoarseDieDomain::import_local_sample("hbm:0",hardware,0,local,bindings);
  if(local!=source_before) throw std::runtime_error("sample changed during import");
  if(domain->export_runtime_snapshot()!=local.at("runtime_snapshot"))
   throw std::runtime_error("active raw Runtime differs immediately after import");
  json completions=json::array();
  std::uint64_t previous=local.at("sampled_at_ns");
  while(auto next=domain->next_event()) {
   if(*next<previous) throw std::runtime_error("event moved backward");
   domain->advance_to(*next); previous=*next;
   for(const auto& completion:domain->take_completions())
    completions.push_back({{"leg_id",completion.leg_id},{"finish_ns",completion.finish_ns},{"cancelled",completion.cancelled}});
  }
  const json final_runtime=domain->export_runtime_snapshot();
  const json expected=drained.at("local_runtime").at("runtime_snapshot");
  json physical_final=final_runtime, physical_expected=expected;
  const std::vector<json::json_pointer> busy_fields={
    json::json_pointer("/stats/compute_busy_ns"),
    json::json_pointer("/stats/hbm_shared_ns"),
    json::json_pointer("/stats/row_shared_ns/0"),
    json::json_pointer("/stats/row_shared_ns/1")};
  for(const auto& field:busy_fields) {
    physical_final[field]=nullptr; physical_expected[field]=nullptr;
  }
  const bool physical_subset_equal=physical_final==physical_expected;
  json report={{"schema","actual_sys_local_sample_continuation_v1"},
    {"actual_completions",completions},{"saved_sys_terminal_ticks",ticks},
    {"strict_runtime_wire_equal",final_runtime==expected},
    {"runtime_except_four_busy_accumulators_equal",physical_subset_equal},
    {"ungraded_busy_fields",{"/stats/compute_busy_ns","/stats/hbm_shared_ns",
      "/stats/row_shared_ns/0","/stats/row_shared_ns/1"}},
    {"final_runtime_diff",json::diff(expected,final_runtime)},
    {"final_runtime",final_runtime},
    {"scope","physical Runtime and pending local completion; not full production world"}};
  if(completions.size()!=2 || completions.at(0).at("leg_id")!="captured-comp:41" ||
     completions.at(0).at("finish_ns")!=ticks.at("comp_terminal_ticks_ns").at(0) ||
     completions.at(1).at("leg_id")!="captured-local-hbm:900" ||
     completions.at(1).at("finish_ns")!=ticks.at("local_hbm_transfer_terminal_tick_ns"))
   throw std::runtime_error("imported completion ticks differ from untouched real Sys queue");
  if(!domain->take_completions().empty()) throw std::runtime_error("completion repeated");
  const auto terminal_handle=terminal.at("completion_bindings").at(0).at("handle").get<std::uint64_t>();
  auto terminal_domain=NativeCoarseDieDomain::import_local_sample("terminal:hbm:0",hardware,0,terminal,{{terminal_handle,"captured-terminal"}});
  const auto pending=terminal_domain->take_completions();
  if(pending.size()!=1 || pending[0].finish_ns!=1 || !terminal_domain->take_completions().empty() ||
     !terminal_domain->export_runtime_snapshot().at("executions").empty())
    throw std::runtime_error("same-tick terminal pending completion differs");
  report["terminal_pending_exactly_once"]=true;
  std::cout << report.dump(2) << '\n';
  return physical_subset_equal?0:2;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
