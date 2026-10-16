// Reuse only the public observation helper; this is a different analytic input.
#define main coupled_fixture_unused_main
#include "coupled_credit_driver.cc"
#undef main

int main() {
  std::array<double,resource_count> capacity{};
  capacity[COMPUTE_GATE]=1;capacity[HBM0]=capacity[HBM1]=1024;
  auto hw=manual_hardware(capacity);
  hw.fingerprint=stable_hash(json{{"base_current_core",hw.fingerprint},
      {"synthetic_fixture","asymmetric_credit"},{"rates",capacity}}.dump());
  unsigned errors=0;
  for(int exponent:{0,-10,-60})for(unsigned owners:{1U,2U,8U,32U})
      for(const std::string boundary:{"empty","full","interior"}) {
    const double u=std::ldexp(1.,exponent);
    json out={{"family","asymmetric_coupled"},{"owners",owners},
              {"unit_exponent",exponent},{"boundary",boundary},{"plans",json::array()},
              {"hardware_input",hw.input},{"hardware_fingerprint",hw.fingerprint},
              {"synthetic_rates",capacity}};
    std::string stage="submit";
    try {
      Runtime runtime(hw);std::vector<std::uint64_t> handles;
      for(unsigned n=1;n<=owners;++n) {
        Epoch e;e.name="asymmetric_LCS_signed_tangent";
        e.credit_capacity={2*u,2*u};
        double initial=boundary=="empty"?0:boundary=="full"?2*u:u;
        e.credit_initial={initial,initial};e.activities=pipeline(u);
        // Load/Store can each progress twice as fast as Compute. Empty output
        // must constrain S to C, full input must constrain L to C.
        e.activities[0].work[HBM0]=512;e.activities[0].hbm_bytes[0][0]=512;
        e.activities[2].work[HBM1]=512;e.activities[2].hbm_bytes[1][1]=512;
        for(unsigned a:{0U,2U}) {
          e.activities[a].canonical_q=512;e.activities[a].private_time_s=.5;
        }
        auto plan=manual_plan(hw,{e},"independent-asymmetric-coupled-fixture");
        out["plans"].push_back(plan_to_json(*plan));
        handles.push_back(runtime.submit(plan,owner(n,n,1),0));
      }
      const std::uint64_t quarter=owners*second/4,half=owners*second/2,finish=owners*second;
      stage="initial";out[stage]=observe(runtime,handles);
      stage="quarter";runtime.advance_to(quarter);out[stage]=observe(runtime,handles);
      const auto saved=runtime.export_snapshot();out["quarter_snapshot"]=saved;
      std::unique_ptr<Runtime> restored;
      try {
        restored=Runtime::import_snapshot(hw,saved);
        out["snapshot_roundtrip_exact"]=restored->export_snapshot()==saved;
        auto cancelled=Runtime::import_snapshot(hw,saved);
        for(auto h:handles)cancelled->cancel(h,quarter);
        cancelled->advance_to(quarter+second);
        out["cancel"]=observe(*cancelled,handles);
      } catch(const std::exception& error) {
        out["snapshot_error"]=error.what();++errors;
      }
      stage="half";runtime.advance_to(half);out[stage]=observe(runtime,handles);
      stage="completion";runtime.advance_to(finish);out[stage]=observe(runtime,handles);
      if(restored) {
        restored->advance_to(half);restored->advance_to(finish);
        out["continuation_snapshot_exact"]=runtime.export_snapshot()==restored->export_snapshot();
      }
      out["driver_status"]="observed";
    } catch(const std::exception& error) {
      out["driver_status"]="exception";out["failed_stage"]=stage;out["error"]=error.what();++errors;
    }
    std::cout<<out.dump()<<'\n';
  }
  return errors?1:0;
}
