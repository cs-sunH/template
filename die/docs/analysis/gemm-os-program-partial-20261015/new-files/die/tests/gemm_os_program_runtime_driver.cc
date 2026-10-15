// Actual opt-in Program measurements. The ordinary E5 mapping is retained
// solely as an independent ledger/geometry input for the offline oracle.
#define main compiled_gemm_driver_entry
#include "compiled_gemm_oracle_driver.cc"
#undef main

int main(int argc, char** argv) {
  try {
    const auto shape = parse_shape(argc,argv);
    const auto hardware = synthetic_hardware();
    auto task = gemm_task(shape);
    const auto baseline = compile(hardware,task);
    task["runtime_program"] = {{"kind",coarse_die::gemm_os_k_readiness_kind},
                               {"max_expanded_waves",4096}};
    const auto program = compile(hardware,task);
    if (!program->gemm_os_program || !program->epochs.empty())
      throw std::runtime_error("compiler did not produce the opted-in Program");
    for (const auto field : {"tm","tn","tk","dataflow_id",
                             "consumption_wave_count","output_round_count"})
      if (program->summary.at(field)!=baseline->summary.at(field))
        throw std::runtime_error(std::string("oracle mapping differs: ")+field);
    json concurrent = json::array();
    for (const auto offset : {0ULL,50000000ULL})
      concurrent.push_back(run_two_comp(hardware,program,offset));
    std::cout << json{
        {"schema_version",2},
        {"measurement_source","bounded_cpp_gemm_os_program_v1"},
        {"diagnostic_only",true},
        {"baseline_compiled_plan_executed",false},
        {"compiled_program_plan_executed",true},
        {"validation_scope","bounded expanded Program; no production efficiency certificate"},
        {"synthetic_hardware",synthetic_capacity_json(hardware)},
        {"compiled_plan",plan_json(*baseline)},
        {"executed_program_plan",plan_to_json(*program)},
        {"standalone",run_standalone(hardware,program)},
        {"two_comp",std::move(concurrent)}}.dump(2) << '\n';
  } catch (const std::exception& error) {
    std::cerr << "bounded OS Program diagnostic: " << error.what() << '\n';
    return 1;
  }
}
