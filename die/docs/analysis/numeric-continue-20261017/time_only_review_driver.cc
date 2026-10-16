// Independent current-Core synthetic controls for the narrow time-only patch.
#include "test_support.hh"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
using namespace coarse_die;using namespace coarse_die_test;namespace fs=std::filesystem;
namespace {
std::string dec(long double x){std::ostringstream s;s<<std::setprecision(24)<<x;return s.str();}
std::shared_ptr<const Plan> timed(const Hardware& hw,double seconds,bool curve,std::uint64_t launch,const std::string& name,std::uint64_t vec=1000000000000ULL){
 Epoch e;e.name=name;auto a=activity(1,Role::Compute,COMPUTE_GATE,seconds);a.private_time_s=seconds;a.canonical_q=1;a.vec_eqops=vec;
 if(curve)a.credit_curve={{0,{{0,0}}},{1,{{0,0}}}};e.activities={a};
 auto p=std::make_shared<Plan>();p->task={{"fixture",name}};p->summary={{"status","synthetic_time_review"}};
 p->hardware_fingerprint=hw.fingerprint;p->epochs={e};p->launch_ns=launch;p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();return p;
}
json observe(const Runtime& r,const std::vector<std::uint64_t>& handles){
 json q=json::array();for(auto h:handles)q.push_back(r.query_progress(h));auto next=r.next_event();
 return {{"snapshot",r.export_snapshot()},{"queries",q},{"counters",r.counters()},{"next_event_ns",next?json(*next):json(nullptr)}};
}
json case_ready(const Hardware& hw){
 Runtime r(hw);auto a=r.submit(timed(hw,std::nextafter(1e-9,0.),true,0,"pre_one_ns_predecessor",1),owner(1,1,1),0);
 auto b=r.submit(timed(hw,std::ldexp(1.,-85),true,1,"queued_exact_one_ns_ready"),owner(2,2,2),0);
 json result={{"case","exact_integer_ready_must_not_gain_pre_ready_service"},{"before",observe(r,{a,b})}};
 r.advance_to(1);result["at_ready"]=observe(r,{a,b});const auto q=r.query_progress(b);
 result["passed"]=!q.at("terminal").get<bool>()&&q.at("served").at("vec_eqops")==0&&q.at("activities")[0].at("progress")==0;
 return result;
}
json case_mixed_ready(const Hardware& hw){
 Runtime r(hw);
 auto first=timed(hw,std::nextafter(1e-9,0.),true,0,"curved_pre_ready_boundary",1);
 auto p=std::make_shared<Plan>(*first);
 auto next=timed(hw,1.0,true,0,"curved_service_after_boundary",1);
 p->epochs.push_back(next->epochs[0]);p->epochs.back().activities[0].id=2;
 p->fingerprint.clear();p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();
 auto a=r.submit(p,owner(1,1,1),0);
 auto b=r.submit(timed(hw,std::ldexp(1.,-85),false,1,"legacy_queued_exact_ready"),owner(2,2,2),0);
 json result={{"case","curved_predecessor_cannot_start_legacy_before_ready"},{"before",observe(r,{a,b})}};
 r.advance_to(1);result["at_ready"]=observe(r,{a,b});const auto q=r.query_progress(b);
 result["passed"]=!q.at("terminal").get<bool>()&&q.at("served").at("vec_eqops")==0&&q.at("activities")[0].at("progress")==0;
 return result;
}
json case_full(const Hardware& hw){
 const auto duration=std::ldexp(1.,-85);Runtime r(hw);auto h=r.submit(timed(hw,duration,true,0,"full_positive_subepsilon_terminal"),owner(1,1,1),0);
 json result={{"case","full_positive_terminal"},{"before",observe(r,{h})}};r.advance_to(1);result["after"]=observe(r,{h});
 const auto busy=std::stold(r.counters().at("compute_busy_ns_exact").get<std::string>());const auto expected=static_cast<long double>(duration)*1e9L;
 result["expected_busy_ns"]=dec(expected);result["actual_busy_ns"]=dec(busy);
 result["passed"]=r.terminal_finish_ns(h)==1&&r.query_progress(h).at("served").at("vec_eqops")==1000000000000ULL&&busy==expected;
 return result;
}
json case_tail(const Hardware& hw){
 const auto duration=std::nextafter(1e-9,std::numeric_limits<double>::infinity());Runtime r(hw);
 auto h=r.submit(timed(hw,duration,true,0,"above_integer_ns_terminal_tail"),owner(1,1,1),0);
 json result={{"case","positive_tail_above_one_ns"},{"before",observe(r,{h})}};
 r.advance_to(1);result["at_one"]=observe(r,{h});const bool live=!r.drained();const auto next=r.next_event();
 r.advance_to(2);result["at_two"]=observe(r,{h});
 const auto busy=std::stold(r.counters().at("compute_busy_ns_exact").get<std::string>());
 const auto initial_rate=result.at("before").at("queries").at(0).at("activities").at(0).at("rate").get<double>();
 // The exact service-time oracle uses the actual published binary64 rate.
 // Its reciprocal differs slightly from the private-time input after rate rounding.
 const auto expected=1.0L/static_cast<long double>(initial_rate)*1e9L;
 result["expected_from_published_rate"]=initial_rate;
 result["private_time_input_ns"]=dec(static_cast<long double>(duration)*1e9L);
 result["expected_busy_ns"]=dec(expected);result["actual_busy_ns"]=dec(busy);result["busy_difference_ns"]=dec(busy-expected);
 result["passed"]=live&&next==2&&r.terminal_finish_ns(h)==2&&std::abs(busy-expected)<=std::numeric_limits<long double>::epsilon()*2;
 return result;
}
json case_legacy(const Hardware& hw){
 Runtime retained(hw),clean(hw);auto old=retained.submit(timed(hw,std::ldexp(1.,-85),true,0,"retained_terminal_curve"),owner(1,1,1),0);retained.advance_to(1);clean.advance_to(1);
 const auto plan=timed(hw,2.75e-9,false,0,"legacy_after_terminal_curve");auto a=retained.submit(plan,owner(2,2,2),1),b=clean.submit(plan,owner(2,2,2),1);
 const auto first_a=retained.next_event(),first_b=clean.next_event();retained.advance_to(10);clean.advance_to(10);
 const auto qa=retained.query_progress(a),qb=clean.query_progress(b);
 return {{"case","terminal_curve_does_not_change_later_legacy"},{"retained",observe(retained,{old,a})},{"clean",observe(clean,{b})},
  {"passed",first_a==first_b&&retained.terminal_finish_ns(a)==clean.terminal_finish_ns(b)&&qa.at("served")==qb.at("served")&&qa.at("activities")==qb.at("activities")}};
}

json case_unrepresented_progress(const Hardware& hw){
 const auto unit=std::ldexp(1.,-60);Epoch e;e.name="positive_credit_event_at_unrepresentable_progress_increment";
 e.credit_capacity={{4*unit,0}};e.credit_initial={{unit,0}};
 auto a=activity(1,Role::Compute,COMPUTE_GATE,1);a.private_time_s=1;a.canonical_q=1;a.vec_eqops=1000000000000ULL;
 a.credit_delta={{-std::ldexp(1.,40),0}};
 a.credit_curve={{0,{{0,0}}},{.5,{{0,0}}},{1,a.credit_delta}};e.activities={a};
 Runtime r(hw);const auto h=r.submit(manual_plan(hw,{e},"unrepresented_progress_failclosed"),owner(1,1,1),0);
 r.advance_to(500000000);const auto before=r.export_snapshot();json result={{"case","unrepresentable_selected_progress_fails_before_tiny_mutation"},{"before",observe(r,{h})}};
 bool threw=false;std::string error;try{r.advance_to(500000001);}catch(const std::runtime_error& x){threw=true;error=x.what();}
 result["after"]=observe(r,{h});result["error"]=error;result["tiny_step_state_unchanged"]=before==r.export_snapshot();
 result["passed"]=threw&&error.find("represented physical progress")!=std::string::npos&&before==r.export_snapshot();return result;
}
json case_unrepresented_clock(const Hardware& hw){
 auto first=timed(hw,0.5e-9,true,0,"half_ns_clock_anchor",1);
 auto second=timed(hw,std::ldexp(1.,-100),true,0,"unrepresentable_fractional_clock_tail",1000000000000ULL);
 auto p=std::make_shared<Plan>(*first);p->epochs.push_back(second->epochs[0]);p->epochs.back().activities[0].id=2;p->fingerprint.clear();p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();
 Runtime r(hw);const auto h=r.submit(p,owner(1,1,1),0);json result={{"case","unrepresentable_clock_fails_before_tiny_service"}};
 bool threw=false;std::string error;try{r.advance_to(1);}catch(const std::runtime_error& x){threw=true;error=x.what();}
 result["after"]=observe(r,{h});result["error"]=error;const auto q=r.query_progress(h);
 result["passed"]=threw&&error.find("represented clock")!=std::string::npos&&q.at("activities")[1].at("progress")==0&&q.at("served").at("vec_eqops")==1;
 return result;
}
}
int main(int argc,char**argv){
 if(argc!=3)return 2;fs::path dest=argv[1];fs::create_directories(dest);
 std::array<double,resource_count> rates{};rates[COMPUTE_GATE]=1;auto hw=manual_hardware(rates);
 hw.fingerprint="synthetic_current_core_time_only_review_v1:"+stable_hash(json{{"base",hw.fingerprint},{"rates",rates}}.dump());
 json report={{"label",argv[2]},{"scope","independent current-Core synthetic time/readiness/legacy controls"},{"cases",json::array()}};
 for(auto fn:{case_ready,case_mixed_ready,case_full,case_tail,case_legacy,case_unrepresented_progress,case_unrepresented_clock})try{report["cases"].push_back(fn(hw));}catch(const std::exception& e){report["cases"].push_back({{"passed",false},{"error",e.what()}});}
 std::ofstream out(dest/"report.json");out<<report.dump(2)<<'\n';bool passed=true;for(const auto&c:report["cases"])passed=passed&&c.at("passed").get<bool>();
 std::cout<<json{{"label",argv[2]},{"passed",passed}}.dump()<<'\n';return passed?0:1;
}
