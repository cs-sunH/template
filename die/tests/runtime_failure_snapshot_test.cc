#include "test_support.hh"
#include <filesystem>
using namespace coarse_die;using namespace coarse_die_test;
Activity timed(std::uint64_t id,std::size_t resource,int exponent) {
 auto a=activity(id,resource==COMPUTE_GATE?Role::Compute:Role::Load,resource,std::ldexp(1.,exponent));
 a.private_time_s=std::ldexp(1.,exponent);a.canonical_q=1;a.vec_eqops=1;
 a.credit_curve={{0,{{0,0}}},{1,{{0,0}}}};return a;
}
int main(int argc,char**argv) {
 if(argc!=3)return 2;namespace fs=std::filesystem;fs::create_directories(argv[1]);
 std::array<double,resource_count> rates{};rates[COMPUTE_GATE]=rates[HBM0]=1;
 auto hw=manual_hardware(rates);hw.fingerprint="synthetic_current_core_clock_snapshot_review:"+stable_hash(json(rates).dump());
 Epoch a;a.name="positive_tiny_terminal";a.activities={timed(1,COMPUTE_GATE,-85)};
 Epoch b;b.name="positive_tiny_clock_anchor";b.activities={timed(2,HBM0,-84)};
 Epoch c;c.name="unrepresentable_clock_tail";c.activities={timed(3,HBM0,-160)};
 Runtime r(hw);json report={{"label",argv[2]},{"scope","independent synthetic failclosed clock snapshot control"}};
 try {
  auto ah=r.submit(manual_plan(hw,{a},"terminal"),owner(1,1,1),0);
  auto bh=r.submit(manual_plan(hw,{b,c},"tail"),owner(2,2,2),0);
  report["before"]=r.export_snapshot();std::string error;bool threw=false;
  try {r.advance_to(1);}catch(const std::exception& ex){threw=true;error=ex.what();}
  report["after"]=r.export_snapshot();report["error"]=error;report["threw"]=threw;
  report["a"]=r.query_progress(ah);report["b"]=r.query_progress(bh);
  bool imported=false;std::string import_error;
  try {auto fork=Runtime::import_snapshot(hw,r.export_snapshot());imported=true;}catch(const std::exception& ex){import_error=ex.what();}
  report["snapshot_reimported"]=imported;report["snapshot_import_error"]=import_error;
  report["passed"]=threw&&error.find("represented clock")!=std::string::npos&&imported;
 }catch(const std::exception& ex){report["setup_error"]=ex.what();report["passed"]=false;report["after"]=r.export_snapshot();}
 std::ofstream out(fs::path(argv[1])/"report.json");out<<report.dump(2)<<'\n';
 std::cout<<report.dump(2)<<'\n';return report.at("passed").get<bool>()?0:1;
}
