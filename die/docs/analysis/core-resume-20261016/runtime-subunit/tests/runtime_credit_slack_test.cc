#include "test_support.hh"
#include <array>
#include <cmath>

using namespace coarse_die;
using namespace coarse_die_test;
namespace {
void verify() {
  std::array<double,resource_count> rates{};rates[COMPUTE_GATE]=1;
  const auto hw=manual_hardware(rates);
  for(double unit:{1.0,std::ldexp(1.0,-10),std::ldexp(1.0,-60)})for(bool filling:{false,true}) {
    Epoch e;e.name="scaled_credit_half_second";
    e.credit_capacity={{4*unit,0}};e.credit_initial={{(filling ? 3 : 1)*unit,0}};
    auto a=activity(1,Role::Compute,COMPUTE_GATE,1);
    a.private_time_s=1;a.canonical_q=1;a.vec_eqops=1000000000000ULL;
    a.credit_delta={{(filling ? 2 : -2)*unit,0}};
    a.credit_curve={{0,{{0,0}}},{1,a.credit_delta}};e.activities={a};
    Runtime source(hw);const auto handle=source.submit(manual_plan(hw,{e}),owner(1,1,1),0);
    require(source.next_event()==500000000ULL,
        "positive credit/headroom has the same physical half-second service regardless of credit-unit scale");
    source.advance_to(250000000ULL);
    const auto prefix=source.query_progress(handle);
    require(prefix.at("activities")[0].at("progress")==0.25 &&
        prefix.at("served").at("vec_eqops")==250000000000ULL,
        "sub-tolerance credit cannot erase significant physical compute work");
    auto restored=Runtime::import_snapshot(hw,source.export_snapshot());
    require(restored->export_snapshot()==source.export_snapshot(),"tiny positive slack restores with its actual allocation");
    source.advance_to(500000000ULL);restored->advance_to(500000000ULL);
    require(!source.next_event() && !source.drained() &&
        source.query_progress(handle).at("activities")[0].at("progress")==0.5,
        "real empty/full credit stops service at one half, rather than completing unpaid remaining work");
    source.advance_to(750000000ULL);restored->advance_to(750000000ULL);
    require(source.export_snapshot()==restored->export_snapshot() &&
        source.query_progress(handle).at("served").at("vec_eqops")==500000000000ULL &&
        source.counters().at("compute_busy_ns")==500000000ULL,
        "blocked remaining half has no service or busy and snapshot continuation is exact");
    source.cancel(handle,750000000ULL);
    require(source.drained() && source.query_progress(handle).at("cancelled")==true,
        "cancellation retains only the correctly paid half trillion operations");
  }
}
}
int main(){return run_test("runtime_credit_slack_test",verify);}
