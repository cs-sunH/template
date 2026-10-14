#include "coarse_die/Execution.hh"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace coarse_die;
using json = nlohmann::json;
int main(int argc,char**argv) {
 if(argc!=2) return 2;
 json input; std::ifstream(argv[1]) >> input;
 input["performance_profile"]["timing_topology_profile"]="edge_single_word_v2";
 auto hardware=resolve_hardware(input);
 for(std::size_t r=0;r<resource_count;++r) {
   hardware.rates[r]=r==COMPUTE_GATE?1:1000000;
   hardware.derived["resources"][r]["rate"]=hardware.rates[r];
 }
 json report={{"schema","arbiter_host_probe_v1"},{"fixture_kind","synthetic signed-credit L/C/S; no granularity certificate"},{"cases",json::array()}};
 for(int owners: {1,8,32}) for(int boundary: {0,1,2}) {
  json row={{"owners",owners},{"active_activities",3*owners},{"domains",13},{"credit_case",boundary}};
  try {
   auto plan=std::make_shared<Plan>(); plan->hardware_fingerprint=hardware.fingerprint;
   plan->task={{"op_kind","synthetic"}}; plan->summary={{"status","test_fixture"}};
   Epoch epoch; epoch.name="bounded-credit-probe";
   epoch.credit_initial={boundary==1?0.0:boundary==2?2.0:1.0,boundary==1?0.0:boundary==2?2.0:1.0}; epoch.credit_capacity={2,2};
   for(int stage=0;stage<3;++stage) {
    Activity a; a.id=stage+1; a.role=stage==0?Role::Load:stage==1?Role::Compute:Role::Store;
    for(std::size_t r=0;r<resource_count;++r)
     a.work[r]=r==COMPUTE_GATE?(stage==1?1:0):(stage==1?(r==LOCAL||r==SRAM_READ||r==SRAM_WRITE?2000:0):1000+(r%3)*100);
    if(stage!=1) { a.hbm_bytes[0][stage==2]=1000; a.hbm_bytes[1][stage==2]=1000; }
    a.canonical_q=canonical_basis(a); a.credit_delta=stage==0?std::array<double,2>{1,0}:stage==1?std::array<double,2>{-1,1}:std::array<double,2>{0,-1};
    epoch.activities.push_back(a);
   }
   plan->epochs={epoch}; plan->fingerprint=plan_to_json(*plan).at("fingerprint");
   Runtime runtime(hardware);
   for(int j=0;j<owners;++j) runtime.submit(plan,Owner{std::uint64_t(j+1),std::uint64_t(j+1),std::uint64_t(j+1),1},0);
   const auto snapshot=runtime.export_snapshot();
   std::vector<std::uint64_t> samples;
   for(int i=0;i<35;++i) {
    auto start=std::chrono::steady_clock::now();
    runtime.update_capacity(COMPUTE_GATE,i%2?.99:1.0,0);
    auto stop=std::chrono::steady_clock::now();
    if(i>=5) samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count());
   }
   row["samples_ns"]=samples; std::sort(samples.begin(),samples.end());
   row["p50_ns"]=samples[14]; row["p95_ns"]=samples[28]; row["p95_budget_ns"]=50000; row["budget_met"]=samples[28]<=50000;
   row["status"]="measured"; row["counters"]=runtime.counters();
   row["initial_snapshot_bytes"]=snapshot.dump().size();
  } catch(const std::exception&e) { row["status"]="failed"; row["error"]=e.what(); }
  report["cases"].push_back(row);
 }
 std::cout<<report.dump(2)<<"\n";
}
