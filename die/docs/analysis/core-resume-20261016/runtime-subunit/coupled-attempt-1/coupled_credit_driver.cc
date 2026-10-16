// Independent public-API observations for coupled_credit_oracle.py.
// All work is explicit synthetic fluid work on current-Core manual_hardware.
#include "test_support.hh"
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;
using json = nlohmann::json;
namespace {
constexpr std::uint64_t second = 1000000000ULL;

std::vector<Activity> pipeline(double u) {
  auto l=activity(1,Role::Load,HBM0,1024);
  l.hbm_bytes[0][0]=1024;
  auto c=activity(2,Role::Compute,COMPUTE_GATE,1);
  c.vec_eqops=1024;
  auto s=activity(3,Role::Store,HBM1,1024);
  s.hbm_bytes[1][1]=1024;
  l.credit_delta={u,0};c.credit_delta={-u,u};s.credit_delta={0,-u};
  std::vector<Activity> result{l,c,s};
  for(auto& a:result) {
    a.canonical_q=1024;a.private_time_s=1;
    a.credit_curve={{0,{{0,0}}},{1,a.credit_delta}};
  }
  return result;
}

json observe(const Runtime& runtime,const std::vector<std::uint64_t>& handles) {
  json result={{"now_ns",runtime.now_ns()},{"drained",runtime.drained()},
               {"counters",runtime.counters()},{"executions",json::array()}};
  const auto event=runtime.next_event();
  result["next_event_ns"]=event?json(*event):json(nullptr);
  const auto snapshot=runtime.export_snapshot();
  for(auto h:handles) {
    auto q=runtime.query_progress(h);
    for(const auto& e:snapshot.at("executions"))
      if(e.at("handle")==h) {
        q["snapshot_credit"]=e.at("credit");
        q["snapshot_credit_capacity"]=e.at("credit_capacity");
      }
    result["executions"].push_back(std::move(q));
  }
  return result;
}

json probe(const Hardware& hw,unsigned owners,int exponent,const std::string& mode,
           bool split) {
  const double u=std::ldexp(1.,exponent);
  json out={{"family",split?"same_owner_split":"coupled"},{"owners",owners},
            {"unit_exponent",exponent},{"boundary",mode},{"plans",json::array()},
            {"hardware_fingerprint",hw.fingerprint},
            {"hardware_input",hw.input},{"synthetic_rates",hw.rates}};
  std::string stage="submit";
  try {
    Runtime runtime(hw);
    std::vector<std::uint64_t> handles;
    auto submit=[&](Epoch epoch,Owner who) {
      auto plan=manual_plan(hw,{epoch},"independent-coupled-unit-fixture");
      out["plans"].push_back(plan_to_json(*plan));
      handles.push_back(runtime.submit(plan,who,0));
    };
    if(!split) {
      for(unsigned n=1;n<=owners;++n) {
        Epoch e;e.name="LCS_shared_signed_credit";
        e.credit_capacity={2*u,2*u};
        const double initial=mode=="empty"?0:mode=="full"?2*u:u;
        e.credit_initial={initial,initial};e.activities=pipeline(u);
        submit(e,owner(n,n,1));
      }
    } else {
      // Three executions share ONE owner/Q denominator. They deliberately have
      // separate inventories; no claim that credit flows between executions.
      const auto work=pipeline(u);
      for(unsigned a=0;a<3;++a) {
        Epoch e;e.name="same_owner_separate_inventory";
        e.credit_capacity={2*u,2*u};
        e.credit_initial=a==0?std::array<double,2>{0,0}:
                         a==1?std::array<double,2>{u,0}:
                              std::array<double,2>{0,u};
        e.activities={work[a]};submit(e,owner(11+a,1,1));
      }
      // Each competitor has Q=3072, equal to A's total Q=3*1024.
      // Thus all progress rates are 1/J. Treating split handles as owners would
      // mint weight and make their progress three times the competitors'.
      for(unsigned n=2;n<=owners;++n) {
        auto a=activity(4,Role::Compute,COMPUTE_GATE,1);
        a.private_time_s=1;a.canonical_q=3072;a.vec_eqops=3072;
        Epoch e;e.name="independent_compute_competitor";e.activities={a};
        submit(e,owner(n+20,n,1));
      }
    }
    stage="initial";out[stage]=observe(runtime,handles);
    const std::uint64_t quarter=owners*second/4,finish=owners*second;
    stage="quarter";runtime.advance_to(quarter);out[stage]=observe(runtime,handles);
    stage="snapshot_import";
    const auto saved=runtime.export_snapshot();
    out["quarter_snapshot"]=saved;
    auto fork=Runtime::import_snapshot(hw,saved);
    out["snapshot_roundtrip_exact"]=fork->export_snapshot()==saved;
    auto cancelled=Runtime::import_snapshot(hw,saved);
    stage="cancel";
    for(auto h:handles)cancelled->cancel(h,quarter);
    cancelled->advance_to(quarter+second);
    out[stage]=observe(*cancelled,handles);
    stage="completion";
    runtime.advance_to(finish);fork->advance_to(finish);
    out[stage]=observe(runtime,handles);
    out["continuation_snapshot_exact"]=runtime.export_snapshot()==fork->export_snapshot();
    out["driver_status"]="observed";
  } catch(const std::exception& error) {
    out["driver_status"]="exception";out["failed_stage"]=stage;out["error"]=error.what();
  }
  return out;
}
}
int main() {
  std::array<double,resource_count> capacities{};
  capacities[COMPUTE_GATE]=1;capacities[HBM0]=capacities[HBM1]=1024;
  auto hw=manual_hardware(capacities);
  // A new manual hardware identity explicitly includes changed synthetic rates.
  hw.fingerprint=stable_hash(json{{"base_current_core",hw.fingerprint},
      {"synthetic_fixture","coupled_credit"},{"rates",capacities}}.dump());
  unsigned errors=0;
  for(int exponent:{0,-10,-60})for(unsigned owners:{1U,2U,8U,32U}) {
    for(const std::string boundary:{"empty","full","interior"}) {
      auto result=probe(hw,owners,exponent,boundary,false);
      errors+=result.at("driver_status")!="observed";
      std::cout<<result.dump()<<'\n';
    }
    auto result=probe(hw,owners,exponent,"declared_separate_inventories",true);
    errors+=result.at("driver_status")!="observed";
    std::cout<<result.dump()<<'\n';
  }
  return errors?1:0;
}
