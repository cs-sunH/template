// Read-only production candidate diagnostics. No compiler/solver shortcut.
#define HOTPATH_PROFILE_UTILITIES_ONLY
#include "phase_profile.cc"

int main(int argc,char** argv) {
  try {
    const auto args=parse_args(argc,argv);const auto config=read_json_file(args.config_path);
    const auto hw=resolve_hardware(config);
    const auto profile=config.at("performance_profile").at("execution_profile").get<std::string>();
    Json output={{"scope","candidate path and JSON allocation diagnostics only, no budget acceptance"},{"cases",Json::array()}};
    for(auto rows:{1u,128u}) {
      const auto spec=gemm_task(rows,128,128,profile),normalized=detail::normalize_task(hw,spec);
      const auto candidates=detail::enumerate_mappings_for_compile(hw,normalized);
      Json paths=Json::array(),copies=Json::array(),serialization=Json::array();
      const auto winner=compile(hw,spec);
      for(std::size_t sample=0;sample<args.samples;++sample) {
        std::vector<Json> task_copies;task_copies.reserve(candidates.size());
        copies.push_back(measure([&]{for(std::size_t i=0;i<candidates.size();++i)task_copies.push_back(normalized);}));
        Plan serializable=*winner;serializable.fingerprint.clear();
        Json wire;
        serialization.push_back(measure([&]{wire=plan_to_json(serializable);}));
      }
      const auto shared=std::make_shared<const Hardware>(hw);
      for(const auto& candidate:candidates) {
        // Seal an exact copy solely to use the public Runtime diagnostic API.
        auto p=std::make_shared<Plan>(*candidate);
        p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();
        Runtime runtime(hw);const auto handle=runtime.submit(p,owner(1,1,1),0);
        Json counters=Json::object(),details=Json::array();
        auto capture=[&]{
          const auto a=runtime.last_arbitration();const std::string name=arbitration_path_name(a.path);
          counters[name]=counters.value(name,0)+1;
          details.push_back({{"time_ns",runtime.now_ns()},{"path",name},{"activities",a.activities},
              {"rows",a.constraint_rows},{"pivots",a.pivots},{"tangent_rounds",a.tangent_rounds}});
        };
        capture();
        for(unsigned events=0;!runtime.drained();++events) {
          if(events>=100)throw std::runtime_error("candidate exceeded bounded Runtime events");
          const auto next=runtime.next_event();if(!next)throw std::runtime_error("candidate has no positive service");
          runtime.advance_to(*next);capture();
        }
        const auto latency=detail::estimate_candidate_latency_ns(shared,candidate);
        if(runtime.terminal_finish_ns(handle)!=latency)throw std::runtime_error("sealed diagnostic and internal candidate latency differ");
        paths.push_back({{"tm",candidate->summary.at("tm")},{"tn",candidate->summary.at("tn")},
            {"tk",candidate->summary.at("tk")},{"dataflow",candidate->summary.at("dataflow_id")},
            {"latency_ns",latency},{"allocation_paths",counters},{"event_diagnostics",details}});
      }
      output["cases"].push_back({{"m",rows},{"candidate_count",candidates.size()},
          {"task_identity_copy_samples",copies},{"winner_public_serialization_samples",serialization},{"candidate_paths",paths}});
    }
    std::ofstream file(args.output_path);if(!file)throw std::runtime_error("cannot open candidate diagnostics output");
    file<<output.dump(2)<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<"candidate_paths: "<<e.what()<<'\n';return 1;}
}
