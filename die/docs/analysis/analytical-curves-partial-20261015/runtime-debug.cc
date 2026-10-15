#include "../../die/tests/test_support.hh"
#include <array>
#include <algorithm>
using namespace coarse_die;
using namespace coarse_die_test;
namespace {
std::shared_ptr<const Plan> delayed_output(const Hardware& hw, std::uint64_t count=1) {
  Epoch e;e.name="progress_delayed_output";e.credit_capacity={{0,1}};
  auto c=activity(1,Role::Compute,COMPUTE_GATE,static_cast<double>(count));
  c.int4_ops=count;c.credit_delta={{0,2.0*count}};
  c.credit_curve={{0,{{0,0}}},{0.5,{{0,0}}},{1,{{0,2.0*count}}}};
  auto s=activity(2,Role::Store,HBM0,2.0*count);
  s.hbm_bytes[0][1]=2*count;s.canonical_q=2.0*count;s.credit_delta={{0,-2.0*count}};
  e.activities={c,s};return manual_plan(hw,{e},"delayed_output");
}
void drain(Runtime& r) {
  unsigned guard=0;
  while(!r.drained()) {
    const auto next=r.next_event();require(next.has_value(),"curve execution has a finite next event");
    r.advance_to(*next);require(++guard<30,"fixed description uses bounded event count");
  }
}
void verify() {
  std::array<double,resource_count> rates{};rates[COMPUTE_GATE]=1;rates[HBM0]=1;
  const auto hw=manual_hardware(rates);
  std::cerr << "checkpoint first\n";
  Runtime r(hw);const auto h=r.submit(delayed_output(hw),owner(1,1,1),0);
  require(r.next_event()==500000000ULL,"next event is the output work threshold");
  std::cerr << "checkpoint beforecurve\n";
  r.advance_to(400000000ULL);
  require(r.query_progress(h).at("served").at("hbm_bytes_by_row_direction")[0][1]==0,
          "store cannot serve any output before cumulative compute threshold");
  auto copy=Runtime::import_snapshot(hw,r.export_snapshot());
  std::cerr << "checkpoint atcurve\n";
  r.advance_to(500000000ULL);copy->advance_to(500000000ULL);
  require(r.export_snapshot()==copy->export_snapshot(),"curve boundary survives import exactly");
  std::cerr << "checkpoint aftercurve\n";
  r.advance_to(900000000ULL);
  auto fork=Runtime::import_snapshot(hw,r.export_snapshot());
  std::cerr << "checkpoint drain\n";
  drain(r);drain(*fork);
  require(r.export_snapshot()==fork->export_snapshot(),"source and post-boundary imported continuation agree");
  require_close(static_cast<double>(*r.terminal_finish_ns(h)),2500000000.0,1.0,
                "continuous output-full tangent gives analytical completion");
  require(r.query_progress(h).at("served").at("hbm_bytes_by_row_direction")[0][1]==2,
          "physical output byte endpoint is exact");
  std::cerr << "checkpoint slowed\n";
  Runtime slowed(hw);const auto slow=slowed.submit(delayed_output(hw),owner(2,2,2),0);
  slowed.update_capacity(COMPUTE_GATE,0.5,250000000ULL);
  require(slowed.next_event()==750000000ULL,"capacity change shifts threshold by work, not fixed time");
  slowed.advance_to(700000000ULL);
  require(slowed.query_progress(slow).at("served").at("hbm_bytes_by_row_direction")[0][1]==0,
          "slower compute delays output readiness");
  drain(slowed);
  require_close(static_cast<double>(*slowed.terminal_finish_ns(slow)),2750000000.0,1.0,
                "slowed progress threshold preserves actual IO work");
  std::cerr << "checkpoint cancelled\n";
  Runtime cancelled(hw);auto cancelled_h=cancelled.submit(delayed_output(hw),owner(3,3,3),0);
  cancelled.cancel(cancelled_h,700000000ULL);
  const auto state=cancelled.query_progress(cancelled_h);
  require(state.at("state")=="cancelled","curve cancel remains a terminal cancellation");
  require_close(state.at("served").at("hbm_bytes_by_row_direction")[0][1].get<double>(),0.2,1e-8,
                "cancellation retains actually served output only");
  std::uint64_t max_events=0;
  for(const auto n:{1ULL,1000ULL,1000000ULL}) {
  std::cerr << "checkpoint scale\n";
    Runtime scaled(hw);scaled.submit(delayed_output(hw,n),owner(4,4,4),0);drain(scaled);
    const auto events=scaled.counters().at("internal_events").get<std::uint64_t>();
    max_events=std::max(max_events,events);
    require(events<20,"repeat count does not create one event per output unit");
  }
  require(max_events>0,"event growth fixture exercised actual Runtime");
}
}
int main(){return run_test("service_curve_runtime_test",verify);}
