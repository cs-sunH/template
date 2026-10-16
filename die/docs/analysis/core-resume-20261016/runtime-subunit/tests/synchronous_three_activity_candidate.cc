// Prepared only: not compiled or run. This is a new synthetic manual Plan,
// not an imported historical Norm Plan and not current-hardware calibration.
// Its three activity contracts are taken verbatim from the archived stalled
// apply-middle epoch documented in synchronous-contract.json. Starting that
// epoch from zero drops its historical fractional-time prelude; this source
// is therefore an unverified reduction, not an established red reproducer.
#include "test_support.hh"
#include <array>
#include <algorithm>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
void verify() {
  // Current Core descriptor is resolved by manual_hardware; these explicitly
  // synthetic capacities reproduce the numerical service contract only.
  std::array<double,resource_count> rates{{1,4e12,4e12,4e12,204800,204800,
                                         0,0,4e12,4e12,0,0,4e12}};
  auto hw=manual_hardware(rates);
  const auto current_identity=hw.fingerprint;
  hw.fingerprint="synthetic_manual_synchronous_apply_middle_v1:"+
      stable_hash(current_identity+":"+json(rates).dump()+":archived_epoch4_numeric_contract");

  Epoch epoch;
  epoch.name="synthetic_manual_apply_middle_reserved_tail";
  epoch.credit_capacity={{1,1}};
  epoch.credit_initial={{1,1}};
  // The initial units are declared pre-existing inventory, as in the source
  // epoch. No service from its omitted predecessor epochs is claimed here.
  Activity load;
  load.id=13;load.role=Role::Load;
  load.work={{0,0,390150,0,195075,195075,0,0,0,195075,0,0,585225}};
  load.hbm_bytes={{{195075,0},{195075,0}}};
  load.private_time_s=3.9015e-7;load.canonical_q=390150;
  load.credit_delta={{254,0}};
  load.credit_curve={{0,{{0,0}}},{1,load.credit_delta}};

  Activity compute;
  compute.id=14;compute.role=Role::Compute;
  compute.work={{1.59375,652800,522240,1175040,0,0,0,0,0,0,0,0,0}};
  compute.vec_eqops=326400;
  compute.private_time_s=1.59375;compute.canonical_q=1175040;
  compute.credit_delta={{-255,254}};
  compute.credit_curve={{0,{{0,0}}},{1,compute.credit_delta}};

  Activity store;
  store.id=15;store.role=Role::Store;
  store.work={{0,261120,0,0,130560,130560,0,0,130560,0,0,0,391680}};
  store.hbm_bytes={{{0,130560},{0,130560}}};
  store.private_time_s=2.6112e-7;store.canonical_q=261120;
  store.credit_delta={{0,-255}};
  store.credit_curve={{0,{{0,0}}},{1,store.credit_delta}};
  epoch.activities={load,compute,store};

  const auto plan=manual_plan(hw,{epoch},"synthetic_manual_synchronous_apply_middle_v1");
  Runtime runtime(hw);
  const auto first=runtime.submit(plan,owner(1,1,1),0);
  const auto second=runtime.submit(plan,owner(2,2,2),0);
  unsigned events=0;
  while(!runtime.drained()) {
    const auto next=runtime.next_event();
    require(next.has_value() && ++events<=64,
        "synchronous manual cumulative curves retain advancing finite-credit service");
    runtime.advance_to(*next);
  }
  std::uint64_t last_finish=0;
  for(auto handle:{first,second}) {
    const auto progress=runtime.query_progress(handle);
    require(progress.at("served").at("vec_eqops")==326400,
        "each synthetic owner pays the complete extracted Compute work");
    const auto finish=runtime.terminal_finish_ns(handle);
    require(finish.has_value(),"each extracted epoch actually reaches terminal");
    last_finish=std::max(last_finish,*finish);
  }
  require(last_finish>=3187500000ULL,
      "two owners cannot finish 3.1875 shared COMPUTE_GATE seconds early");
}
}
int main(){return run_test("synchronous_three_activity_candidate",verify);}
