#include "test_support.hh"
#include <filesystem>
using namespace coarse_die;
using namespace coarse_die_test;

int main(int argc,char**argv) {
  if(argc!=3)return 2;
  namespace fs=std::filesystem;fs::create_directories(argv[1]);
  std::array<double,resource_count> rates{};rates[COMPUTE_GATE]=1;rates[HBM0]=1;
  auto hw=manual_hardware(rates);hw.fingerprint="synthetic_current_core_credit_or_review:"+stable_hash(json(rates).dump());
  const double unit=std::ldexp(1.,-60),mass=std::ldexp(1.,40);
  Epoch e;e.name="consumer_offset_producer_zero";e.credit_capacity={{4*unit,0}};e.credit_initial={{unit,0}};
  auto consumer=activity(1,Role::Compute,COMPUTE_GATE,1);consumer.private_time_s=1;consumer.canonical_q=1;consumer.vec_eqops=1000000000000ULL;
  consumer.credit_delta={{-mass,0}};consumer.credit_curve={{0,{{0,0}}},{.5,{{0,0}}},{1,consumer.credit_delta}};
  auto producer=activity(2,Role::Load,HBM0,1);producer.private_time_s=1;producer.canonical_q=1;producer.hbm_bytes[0][0]=1000000000000ULL;
  producer.credit_delta={{mass,0}};producer.credit_curve={{0,{{0,0}}},{1,producer.credit_delta}};
  e.activities={consumer,producer};Runtime r(hw);r.update_capacity(HBM0,0,0);
  auto h=r.submit(manual_plan(hw,{e},"credit_or_representability_review"),owner(1,1,1),0);
  json report={{"label",argv[2]},{"scope","synthetic current-Core independent numerical review; not hardware measurement"}};
  try {
    r.advance_to(500000000);r.update_capacity(HBM0,1,500000000);
    const auto before=r.export_snapshot();report["before"]=before;report["before_query"]=r.query_progress(h);
    bool threw=false;std::string error;try{r.advance_to(500000001);}catch(const std::exception& ex){threw=true;error=ex.what();}
    report["after"]=r.export_snapshot();report["after_query"]=r.query_progress(h);report["error"]=error;report["threw"]=threw;
    report["state_unchanged"]=before==r.export_snapshot();
    report["passed"]=threw && error.find("represented physical progress")!=std::string::npos && before==r.export_snapshot();
  }catch(const std::exception& ex){report["setup_error"]=ex.what();report["passed"]=false;report["after"]=r.export_snapshot();}
  std::ofstream out(fs::path(argv[1])/"report.json");out<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';return report.at("passed").get<bool>()?0:1;
}
