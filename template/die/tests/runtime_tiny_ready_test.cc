#include "test_support.hh"
#include <filesystem>
using namespace coarse_die;using namespace coarse_die_test;
std::shared_ptr<const Plan> anchor(const Hardware& hw,bool queued) {
 const double seconds=queued?std::ldexp(1.,-85):std::nextafter(1e-9,0.);
 auto a=activity(queued?4:3,Role::Load,HBM0,seconds);a.private_time_s=seconds;a.canonical_q=1;
 a.credit_curve={{0,{{0,0}}},{1,{{0,0}}}};Epoch e;e.name=queued?"integer_ready":"near_integer_anchor";e.activities={a};
 auto p=std::make_shared<Plan>(*manual_plan(hw,{e},e.name));p->launch_ns=queued?1:0;p->fingerprint.clear();p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();return p;
}
int main(int argc,char**argv) {
 if(argc!=3)return 2;namespace fs=std::filesystem;fs::create_directories(argv[1]);
 std::array<double,resource_count> rates{};rates[COMPUTE_GATE]=rates[HBM0]=rates[HBM1]=1;
 auto hw=manual_hardware(rates);hw.fingerprint="synthetic_current_core_tiny_ready_participant:"+stable_hash(json(rates).dump());
 Epoch e;e.name="fast_initial_compute_then_slow_tangent";
 auto compute=activity(1,Role::Compute,COMPUTE_GATE,1e-9);compute.private_time_s=1e-9;compute.canonical_q=1;
 compute.credit_delta={{-.5,0}};compute.credit_curve={{0,{{0,0}}},{.5,{{0,0}}},{1,compute.credit_delta}};
 auto load=activity(2,Role::Load,HBM1,1);load.private_time_s=1;load.canonical_q=1;
 load.credit_delta={{1,0}};load.credit_curve={{0,{{0,0}}},{1,load.credit_delta}};e.activities={compute,load};
 Runtime r(hw);json report={{"label",argv[2]},{"scope","synthetic tiny-ready boundary with another active curved participant"}};
 try {
  auto slow=r.submit(manual_plan(hw,{e},e.name),owner(1,1,1),0);
  auto near=r.submit(anchor(hw,false),owner(2,2,2),0);
  auto queued=r.submit(anchor(hw,true),owner(3,3,3),0);report["before"]=r.export_snapshot();
  bool threw=false;std::string error;try{r.advance_to(1);}catch(const std::exception& ex){threw=true;error=ex.what();}
  report["after"]=r.export_snapshot();report["slow"]=r.query_progress(slow);report["near"]=r.query_progress(near);report["queued"]=r.query_progress(queued);
  report["threw"]=threw;report["error"]=error;report["passed"]=threw&&error.find("participating represented physical progress")!=std::string::npos;
 }catch(const std::exception& ex){report["setup_error"]=ex.what();report["passed"]=false;report["after"]=r.export_snapshot();}
 std::ofstream out(fs::path(argv[1])/"report.json");out<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';return report.at("passed").get<bool>()?0:1;
}
