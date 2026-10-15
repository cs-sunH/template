#include "test_support.hh"
#include <array>
#include <algorithm>
#include <limits>
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
  s.credit_curve={{0,{{0,0}}},{1,s.credit_delta}};
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
  Runtime r(hw);const auto h=r.submit(delayed_output(hw),owner(1,1,1),0);
  require(r.next_event()==500000000ULL,"next event is the output work threshold");
  r.advance_to(400000000ULL);
  require(r.query_progress(h).at("served").at("hbm_bytes_by_row_direction")[0][1]==0,
          "store cannot serve any output before cumulative compute threshold");
  auto copy=Runtime::import_snapshot(hw,r.export_snapshot());
  r.advance_to(500000000ULL);copy->advance_to(500000000ULL);
  require(r.export_snapshot()==copy->export_snapshot(),"curve boundary survives import exactly");
  r.advance_to(900000000ULL);
  auto fork=Runtime::import_snapshot(hw,r.export_snapshot());
  drain(r);drain(*fork);
  require(r.export_snapshot()==fork->export_snapshot(),"source and post-boundary imported continuation agree");
  require_close(static_cast<double>(*r.terminal_finish_ns(h)),2500000000.0,1.0,
                "continuous output-full tangent gives analytical completion");
  require(r.query_progress(h).at("served").at("hbm_bytes_by_row_direction")[0][1]==2,
          "physical output byte endpoint is exact");
  Runtime slowed(hw);const auto slow=slowed.submit(delayed_output(hw),owner(2,2,2),0);
  slowed.update_capacity(COMPUTE_GATE,0.5,250000000ULL);
  require(slowed.next_event()==750000000ULL,"capacity change shifts threshold by work, not fixed time");
  slowed.advance_to(700000000ULL);
  require(slowed.query_progress(slow).at("served").at("hbm_bytes_by_row_direction")[0][1]==0,
          "slower compute delays output readiness");
  drain(slowed);
  require_close(static_cast<double>(*slowed.terminal_finish_ns(slow)),2750000000.0,1.0,
                "slowed progress threshold preserves actual IO work");
  Runtime cancelled(hw);auto cancelled_h=cancelled.submit(delayed_output(hw),owner(3,3,3),0);
  cancelled.cancel(cancelled_h,700000000ULL);
  const auto state=cancelled.query_progress(cancelled_h);
  require(state.at("state")=="cancelled","curve cancel remains a terminal cancellation");
  require_close(state.at("served").at("hbm_bytes_by_row_direction")[0][1].get<double>(),0.2,1e-8,
                "cancellation retains actually served output only");

  Epoch terminal_epoch;terminal_epoch.name="terminal_roundoff";
  terminal_epoch.credit_initial={{17,0}};terminal_epoch.credit_capacity={{34,1}};
  auto almost_done=activity(11,Role::Compute,COMPUTE_GATE,1);
  almost_done.credit_delta={{-17,0}};
  almost_done.credit_curve={{0,{{0,0}}},{1,almost_done.credit_delta}};
  auto remaining=activity(12,Role::Load,HBM0,1);
  terminal_epoch.activities={almost_done,remaining};
  Runtime terminal_runtime(hw);
  terminal_runtime.submit(manual_plan(hw,{terminal_epoch},"terminal_roundoff"),
                          owner(7,7,7),0);
  auto terminal_snapshot=terminal_runtime.export_snapshot();
  auto& terminal_execution=terminal_snapshot["executions"][0];
  auto& terminal_activities=terminal_execution["activities"];
  terminal_activities[0]["progress"]=1.0;
  terminal_activities[0]["curve_progress_exact"]="9.999999999999999940000e-01";
  terminal_activities[0]["rate"]=0.0;
  terminal_activities[0]["alpha"]=0.0;
  terminal_activities[1]["rate"]=1.0;
  terminal_activities[1]["alpha"]=1.0;
  terminal_execution["credit"][0]=0.0;
  auto terminal_resumed=Runtime::import_snapshot(hw,terminal_snapshot);
  const auto terminal_result=terminal_resumed->export_snapshot();
  require(terminal_result["executions"][0]["activities"][0]["done"]==true &&
          terminal_result["executions"][0]["activities"][0]["progress"]==1.0,
          "sub-binary64 terminal progress closes when whole-epoch credit is valid");

  Epoch coupled_epoch;coupled_epoch.name="terminal_roundoff_credit_guard";
  coupled_epoch.credit_capacity={{6,1}};
  auto positive=activity(21,Role::Compute,COMPUTE_GATE,1);
  positive.credit_delta={{1.0e17,0}};
  positive.credit_curve={{0,{{0,0}}},{1,positive.credit_delta}};
  auto negative=activity(22,Role::Load,HBM0,1);
  negative.credit_delta={{-1.0e17,0}};
  negative.credit_curve={{0,{{0,0}}},{1,negative.credit_delta}};
  coupled_epoch.activities={positive,negative};
  Runtime coupled_runtime(hw);
  coupled_runtime.submit(manual_plan(hw,{coupled_epoch},"terminal_roundoff_credit_guard"),
                         owner(8,8,8),0);
  auto coupled_snapshot=coupled_runtime.export_snapshot();
  auto& coupled_execution=coupled_snapshot["executions"][0];
  auto& coupled_activities=coupled_execution["activities"];
  coupled_activities[0]["progress"]=1.0;
  coupled_activities[0]["curve_progress_exact"]="9.999999999999999940000e-01";
  coupled_activities[1]["progress"]=std::nextafter(1.0,0.0);
  coupled_activities[1]["curve_progress_exact"]="9.999999999999999390000e-01";
  coupled_activities[0]["rate"]=1.0;
  coupled_activities[1]["rate"]=1.0;
  coupled_activities[0]["alpha"]=0.5;
  coupled_activities[1]["alpha"]=0.5;
  coupled_execution["credit"][0]=5.5;
  auto coupled_resumed=Runtime::import_snapshot(hw,coupled_snapshot);
  const auto coupled_result=coupled_resumed->export_snapshot();
  require(coupled_result["executions"][0]["activities"][0]["done"]==false,
          "terminal roundoff stays open when completing it alone crosses epoch credit bounds");

  Epoch legacy_epoch;legacy_epoch.name="legacy_snapshot_wire";
  legacy_epoch.activities={activity(31,Role::Compute,COMPUTE_GATE,1)};
  Runtime legacy_runtime(hw);
  legacy_runtime.submit(manual_plan(hw,{legacy_epoch},"legacy_snapshot_wire"),
                        owner(9,9,9),0);
  const auto legacy_snapshot=legacy_runtime.export_snapshot();
  require(!legacy_snapshot.contains("allocation_state"),
          "legacy Plan snapshots keep their original allocation wire");
  auto legacy_resumed=Runtime::import_snapshot(hw,legacy_snapshot);
  require(legacy_resumed->export_snapshot()==legacy_snapshot,
          "legacy v3 snapshot import retains its original byte-equivalent state");

  std::uint64_t max_events=0;
  for(const auto n:{1ULL,1000ULL,1000000ULL}) {
    Runtime scaled(hw);scaled.submit(delayed_output(hw,n),owner(4,4,4),0);
    scaled.advance_to(*scaled.next_event()+12345);
    const auto checkpoint=scaled.export_snapshot();
    auto resumed=Runtime::import_snapshot(hw,checkpoint);
    drain(scaled);drain(*resumed);
    require(scaled.export_snapshot()==resumed->export_snapshot(),
            "large-repeat exact progress survives snapshot without finite-credit drift");
    const auto events=scaled.counters().at("internal_events").get<std::uint64_t>();
    max_events=std::max(max_events,events);
    require(events<20,"repeat count does not create one event per output unit");
  }
  Runtime zero_rate(hw);zero_rate.submit(delayed_output(hw,1000000),owner(5,5,5),0);
  auto zero_rate_snapshot=zero_rate.export_snapshot();
  require(zero_rate_snapshot.at("executions").at(0).at("activities").at(1).at("rate")==0.0,
          "initial finite-credit boundary blocks the output activity");
  zero_rate_snapshot["executions"][0]["activities"][1]["rate"]=
      std::numeric_limits<double>::denorm_min();
  bool rejected_zero_rate=false;
  try { (void)Runtime::import_snapshot(hw,zero_rate_snapshot); }
  catch(const std::invalid_argument&) { rejected_zero_rate=true; }
  require(rejected_zero_rate,"snapshot cannot turn a structural zero allocation positive");

  Runtime tangent(hw);tangent.submit(delayed_output(hw,1000000),owner(6,6,6),0);
  nlohmann::json high_boundary_snapshot;
  for(unsigned guard=0;guard<20;++guard) {
    const auto checkpoint=tangent.export_snapshot();
    const double credit=checkpoint.at("executions").at(0).at("credit").at(0).get<double>();
    if(credit>=1.0-2.0e-10) { high_boundary_snapshot=checkpoint;break; }
    const auto next=tangent.next_event();
    require(next.has_value(),"large-credit tangent has a finite boundary event");
    tangent.advance_to(*next);
  }
  require(!high_boundary_snapshot.is_null(),"large-repeat fixture reaches its finite-credit boundary");
  auto& tangent_activities=high_boundary_snapshot["executions"][0]["activities"];
  const double compute_rate=tangent_activities[0]["rate"].get<double>();
  tangent_activities[0]["rate"]=std::nextafter(compute_rate,
      std::numeric_limits<double>::infinity());
  bool rejected_tangent=false;
  try { (void)Runtime::import_snapshot(hw,high_boundary_snapshot); }
  catch(const std::invalid_argument&) { rejected_tangent=true; }
  require(rejected_tangent,
          "one-ulp rate increase cannot point outside a large finite-credit tangent");
  require(max_events>0,"event growth fixture exercised actual Runtime");
}
}
int main(){return run_test("service_curve_runtime_test",verify);}
