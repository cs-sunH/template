#include "test_support.hh"
#include <array>
#include <cmath>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
void verify() {
  std::array<double,resource_count> rates{};
  rates[COMPUTE_GATE]=1;
  const auto hw=manual_hardware(rates);
  for(double unit:{std::ldexp(1.0,-10),std::ldexp(1.0,-60)})
      for(bool filling:{false,true}) {
    Epoch epoch;
    epoch.name="synthetic_scaled_credit_snapshot_guard";
    epoch.credit_capacity={{4*unit,0}};
    epoch.credit_initial={{(filling?3:1)*unit,0}};
    auto work=activity(1,Role::Compute,COMPUTE_GATE,1);
    work.private_time_s=1;work.canonical_q=1;work.vec_eqops=1000000000000ULL;
    work.credit_delta={{(filling?2:-2)*unit,0}};
    work.credit_curve={{0,{{0,0}}},{1,work.credit_delta}};
    epoch.activities={work};
    Runtime source(hw);
    const auto h=source.submit(manual_plan(hw,{epoch}),owner(1,1,1),0);
    source.advance_to(250000000ULL);
    require(source.query_progress(h).at("served").at("vec_eqops")==250000000000ULL,
        "quarter second pays the same work in every declared credit unit");
    const auto original=source.export_snapshot();
    auto correct=Runtime::import_snapshot(hw,original);
    require(correct->export_snapshot()==original,"valid small-unit snapshot round trips exactly");
    for(bool change_capacity:{false,true}) {
      auto changed=original;
      if(change_capacity)changed["executions"][0]["credit_capacity"][0]=8*unit;
      else changed["executions"][0]["credit"][0]=filling?4*unit:0;
      bool rejected=false;
      try{(void)Runtime::import_snapshot(hw,changed);}
      catch(const std::invalid_argument&){rejected=true;}
      require(rejected,"changing physical credit inventory or capacity is invalid at every unit scale");
      require(source.export_snapshot()==original,"a rejected import cannot mutate the source world");
    }
  }
}
}
int main(){return run_test("scaled_credit_snapshot_guard_test",verify);}
