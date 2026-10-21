#include "test_support.hh"
#include "coarse_die/Prediction.hh"
#include <fstream>
#include <iterator>

using namespace coarse_die;
using namespace coarse_die_test;
namespace {
void rejected(const Plan& p,const char* reason) {
  bool failed=false;
  try {require_complete_prediction(p);}catch(const std::invalid_argument& e) {
    failed=std::string(e.what()).find(reason)!=std::string::npos;
  }
  require(failed,"an explicitly incomplete or malformed prediction must not execute");
}
void verify() {
  Plan legacy;require_complete_prediction(legacy);
  legacy.summary={{"heldout_status","not_available"},{"target_calibration","not_available"}};
  require_complete_prediction(legacy);
  for(const char* field:{"prediction_status","alone_estimate_status"}) {
    auto p=legacy;p.summary[field]="ok_assumed";require_complete_prediction(p);
    p.summary[field]="not_available";rejected(p,"prediction_not_available");
    p.summary[field]=false;rejected(p,"invalid_prediction");
  }
  const auto hw=example_hardware();
  const auto task=json{{"op_kind","residual"},{"tensor_shapes",{{"m",1},{"hidden",257}}},
      {"storage_dtype","bf16"},{"layer_instances",3},{"weight_passes",1}};
  const auto body=compile(hw,task);
  require(body->summary.at("prediction_status")=="not_available","fixture must be a real incomplete repeated region");
  rejected(*body,"prediction_not_available");
  // Diagnostic execution remains available through the public model itself.
  Runtime runtime(hw);const auto h=runtime.submit(body,owner(1,1,1),0);runtime.cancel(h,0);
  require(runtime.query_progress(h).at("state")=="cancelled","consumer policy must preserve public body diagnostics");

  const auto text=[](const std::string& path) {
    std::ifstream file(path);require(file.good(),"prediction policy source must exist");
    return std::string(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());
  };
  const auto canonical=text(std::string(COARSE_DIE_SOURCE_DIR)+"/include/coarse_die/Prediction.hh");
  for(const char* repo:{"astra-sim-face","astra-sim-face-LRU","astra-sim-wscllm","astra-sim-wscllm-LRU","astra-sim-joint"})
    require(text(std::string(COARSE_DIE_SOURCE_DIR)+"/../"+repo+"/astra-sim/workload/CoarseDiePrediction.hh")==canonical,
        "offline compatibility headers must be exact copies of the shared prediction policy");
}
}
int main(){return run_test("prediction_availability_test",verify);}
