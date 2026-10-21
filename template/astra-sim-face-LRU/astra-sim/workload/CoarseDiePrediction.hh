#ifndef COARSE_DIE_PREDICTION_AVAILABILITY_HH
#define COARSE_DIE_PREDICTION_AVAILABILITY_HH

#include "coarse_die/Plan.hh"

namespace coarse_die {
// Canonical policy in template/die/include/coarse_die/Prediction.hh. Thin
// adapter copies support offline builds with older SDKs lacking this header.
// Public compile/Runtime retain body diagnostics; production consumers call
// this before acquiring resources, advancing clocks, or publishing completion.
inline void require_complete_prediction(const Plan& plan) {
  // Legacy Plans need not contain prediction diagnostics. Structural and
  // hardware validation remain the responsibility of the existing validators.
  if(!plan.summary.is_object()) return;
  for(const char* field:{"prediction_status","alone_estimate_status"}) {
    const auto value=plan.summary.find(field);
    if(value==plan.summary.end()) continue;
    if(!value->is_string())
      throw std::invalid_argument(std::string("invalid_prediction: ")+field+" must be a string");
    if(value->get<std::string>()=="not_available")
      throw std::invalid_argument(std::string("prediction_not_available: ")+field+
          " marks body-only timing; a complete latency prediction is required");
  }
}
}  // namespace coarse_die
#endif
