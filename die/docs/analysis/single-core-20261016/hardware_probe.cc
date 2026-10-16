#include "coarse_die/Hardware.hh"
#include <iostream>
int main() {
  try {
    nlohmann::json cases; std::cin >> cases;
    nlohmann::json results = nlohmann::json::array();
    for (const auto& input : cases) {
      const auto hw = coarse_die::resolve_hardware(input);
      results.push_back({{"derived", hw.derived}, {"rates", hw.rates},
                         {"fingerprint", hw.fingerprint}});
    }
    std::cout << results.dump(2) << '\n';
    return 0;
  } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
