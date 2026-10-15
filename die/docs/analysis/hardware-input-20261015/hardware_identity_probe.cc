#include "coarse_die/Hardware.hh"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  nlohmann::json j;std::ifstream in(argv[1]);in>>j;
  const auto hw=coarse_die::resolve_hardware(j);
  std::cout<<nlohmann::json{{"fingerprint",hw.fingerprint},{"derived",hw.derived},{"rates",hw.rates}}.dump()<<'\n';
}
