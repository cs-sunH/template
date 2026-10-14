#pragma once

#include <array>
#include <string>

#include <json/json.hpp>

namespace coarse_die {

struct Hardware {
  nlohmann::json input;
  nlohmann::json derived;
  std::array<double, 13> rates{};
  std::string fingerprint;
};

Hardware resolve_hardware(const nlohmann::json& input);
nlohmann::json search_hardware(const nlohmann::json& input);

}  // namespace coarse_die
