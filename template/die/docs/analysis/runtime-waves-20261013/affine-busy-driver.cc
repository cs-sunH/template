#include "test_support.hh"
#include <iostream>
using namespace coarse_die;
using namespace coarse_die_test;
int main() {
  auto hw = example_hardware();
  for (size_t r = 0; r < resource_count; ++r) {
    if (hw.rates[r]>0) hw.rates[r] = 1e30;
  }
  hw.rates[COMPUTE_GATE] = 1;
  hw.rates[HBM0] = 1;
  for(size_t r=0;r<resource_count;++r)hw.derived["resources"][r]["rate"]=hw.rates[r];
  json results=json::array();
  for (std::uint64_t count : {8ULL,64ULL,4096ULL,1000000ULL}) {
    auto plan=std::make_shared<Plan>();
    plan->task=json::object();plan->summary=json::object();plan->hardware_fingerprint=hw.fingerprint;
    Epoch first;first.name="load_first";first.credit_capacity={2,0};
    Activity load;load.id=1;load.role=Role::Load;load.work[HBM0]=1;load.hbm_bytes[0][0]=1;load.canonical_q=1;load.credit_delta={1,0};first.activities={load};
    Epoch repeated;repeated.name="K_wave_affine_repeat";repeated.repeat_count=count;repeated.credit_initial={1,0};repeated.credit_capacity={2,0};
    load.id=2;load.work[HBM0]=count-1;load.hbm_bytes[0][0]=count-1;load.canonical_q=count-1;load.credit_delta={static_cast<double>(count-1),0};
    Activity compute;compute.id=3;compute.role=Role::Compute;compute.work[COMPUTE_GATE]=count*10;compute.canonical_q=1;compute.credit_delta={-static_cast<double>(count),0};
    repeated.activities={load,compute};plan->epochs={first,repeated};
    plan->fingerprint=plan_to_json(*plan).at("fingerprint").get<std::string>();
    Runtime rt(hw);const auto h=rt.submit(plan,Owner{1,1,1,1},0);std::uint64_t events=0;
    while(!rt.terminal_finish_ns(h)){rt.advance_to(rt.next_event().value());++events;}
    results.push_back({{"waves",count},{"actual_finish_s",rt.terminal_finish_ns(h).value()/1e9},{"actual_busy_s",rt.counters().at("hbm_busy_ns").get<double>()/1e9},{"exact_discrete_finish_s",1+10*count},{"exact_discrete_busy_s",count},{"event_advances",events},{"counters",rt.counters()}});
  }
  std::cout<<results.dump(2)<<'\n';
}
