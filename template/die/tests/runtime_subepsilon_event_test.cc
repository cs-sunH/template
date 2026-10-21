// A physically payable, representable half-activity whose true boundary event
// is below the old 1e-15ns floor. All counts/state remain fixed and synthetic.
#include "test_support.hh"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
using namespace coarse_die;using namespace coarse_die_test;
namespace fs=std::filesystem;
std::string decimal(long double x){std::ostringstream s;s<<std::setprecision(24)<<x;return s.str();}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 fs::path dest=argv[1];fs::create_directories(dest);
 auto save=[&](const char*name,const json&j){std::ofstream o(dest/name);o<<j.dump(2)<<'\n';};
 const double unit=std::ldexp(1.,-60),private_s=std::ldexp(1.,-85);
 std::array<double,resource_count> rates{};rates[COMPUTE_GATE]=1;
 auto hw=manual_hardware(rates);hw.fingerprint="synthetic_current_core_positive_subepsilon_v1:"+
  stable_hash(json{{"base",hw.fingerprint},{"rates",rates},{"unit",unit},{"private_s",private_s}}.dump());
 Epoch e;e.name="representable_positive_half_activity_below_old_time_floor";
 e.credit_capacity={{4*unit,0}};e.credit_initial={{unit,0}};
 auto a=activity(1,Role::Compute,COMPUTE_GATE,private_s);a.private_time_s=private_s;
 a.canonical_q=1;a.vec_eqops=1000000000000ULL;a.credit_delta={{-2*unit,0}};
 a.credit_curve={{0,{{0,0}}},{1,a.credit_delta}};e.activities={a};
 const auto p=manual_plan(hw,{e},"synthetic_current_core_representable_positive_subepsilon_v1");
 json result={{"label",argv[2]},{"scope","synthetic positive representable subepsilon event; not target hardware"},
  {"unit",unit},{"private_s",private_s},{"physical_boundary_ns",decimal(static_cast<long double>(private_s)*1e9L/2)},
  {"plan",plan_to_json(*p)},{"checks",json::array()},{"observations",json::array()}};
 auto check=[&](bool passed,const char*name){result["checks"].push_back({{"name",name},{"passed",passed}});};
 Runtime r(hw);const auto h=r.submit(p,owner(1,1,1),0);
 auto observe=[&](const char* name){auto next=r.next_event();auto s=r.export_snapshot();save((std::string(name)+"-snapshot.json").c_str(),s);
 result["observations"].push_back({{"name",name},{"next_event_ns",next?json(*next):json(nullptr)},
  {"query",r.query_progress(h)},{"counters",r.counters()}});};
 try{
  observe("initial");
  check(r.next_event()==1,"positive true subepsilon event is visible at first public tick");
  r.advance_to(1);observe("after-one-ns");
  check(r.query_progress(h).at("activities")[0].at("progress")==0.5,"only exact physical half can execute inside one public tick");
  check(r.query_progress(h).at("served").at("vec_eqops")==500000000000ULL,"no unpaid half trillion operations are invented");
  check(!r.next_event()&&!r.drained(),"remaining work blocks at genuine empty boundary");
  check(r.counters().at("internal_events").get<unsigned>()<=4,"positive event handling has fixed finite event cost");
  const long double busy=std::stold(r.counters().at("compute_busy_ns_exact").get<std::string>());
  const long double expected=static_cast<long double>(private_s)*1e9L/2;
  result["busy_exact_decimal"]=decimal(busy);result["busy_difference_ns"]=decimal(busy-expected);
  check(busy==expected,"original busy counts exactly the physically serviced subepsilon interval");
  auto fork=Runtime::import_snapshot(hw,r.export_snapshot());check(fork->export_snapshot()==r.export_snapshot(),"actual post-boundary snapshot restores exactly");
  r.cancel(h,1);observe("cancelled");
  check(r.query_progress(h).at("served").at("vec_eqops")==500000000000ULL,"cancel retains exact physically paid half only");
 }catch(const std::exception& error){result["error"]=error.what();check(false,"Runtime or snapshot exception");observe("failure");}
 bool passed=true;for(const auto&c:result["checks"])passed=passed&&c.at("passed").get<bool>();
 result["status"]=passed?"passed":"failed";save("report.json",result);
 std::cout<<json{{"label",argv[2]},{"status",result["status"]},{"error",result.value("error","")}}.dump()<<'\n';return passed?0:1;
}
