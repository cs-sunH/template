#include "../src/RegionDelay.hh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using u64=std::uint64_t;
namespace delay=coarse_die::detail::region_delay;
constexpr u64 maximum=std::numeric_limits<u64>::max();
unsigned checks=0;

void require(bool value,const char* message) {
  if(!value)throw std::runtime_error(message);
}
void exact(double a,double b,u64 count,u64 whole,u64 ceil) {
  const auto forward=delay::sum(a,b,count),reverse=delay::sum(b,a,count);
  require(forward.whole==whole && forward.ceil==ceil,"incorrect exact region delay floor/ceil");
  require(reverse.whole==whole && reverse.ceil==ceil,"region delay result must be commutative");
  require(forward.integral()==(whole==ceil),"integral query disagrees with exact floor/ceil");
  ++checks;
}
void overflow(double a,double b,u64 count) {
  bool rejected=false;
  try {delay::sum(a,b,count);}catch(const std::overflow_error&){rejected=true;}
  require(rejected,"delay exceeding uint64 floor or ceil must reject explicitly");++checks;
}
void invalid(double a,double b,u64 count=1) {
  bool rejected=false;
  try {delay::sum(a,b,count);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"negative or nonfinite coefficient must reject explicitly");++checks;
}

void hand_calculated_boundaries() {
  const double tiny=std::numeric_limits<double>::denorm_min();
  const double below_one=std::nextafter(1.0,0.0);
  const double below_two=std::nextafter(2.0,0.0);
  const double infinity=std::numeric_limits<double>::infinity();
  exact(0,0,0,0,0);
  exact(-0.0,0,maximum,0,0);
  exact(std::numeric_limits<double>::max(),tiny,0,0,0);
  exact(0.25,0.5,1,0,1);
  exact(0.25,0.5,3,2,3);
  exact(0.25,0.75,3,3,3);
  exact(0.5,std::ldexp(1.0,-65),1,0,1);
  exact(tiny,0,1,0,1);
  exact(tiny,tiny,maximum,0,1);
  exact(std::numeric_limits<double>::min(),tiny,maximum,0,1);
  exact(1,tiny,1,1,2);
  exact(below_one,std::ldexp(1.0,-54),1,0,1);
  exact(below_one,std::ldexp(1.0,-53),1,1,1);
  exact(below_one,std::nextafter(std::ldexp(1.0,-53),infinity),1,1,2);
  exact(std::nextafter(1.0,infinity),0,1,1,2);
  exact(below_two,std::ldexp(1.0,-53),1,1,2);
  exact(below_two,std::ldexp(1.0,-52),1,2,2);
  exact(std::ldexp(1.0,61),0.5,3,3*(u64{1}<<61)+1,3*(u64{1}<<61)+2);
  exact(std::ldexp(1.0,63),std::nextafter(std::ldexp(1.0,63),0.0),1,
      maximum-1023,maximum-1023);
  exact(1,0,maximum,maximum,maximum);
  exact(0.5,0.5,maximum,maximum,maximum);
  exact(below_one,std::ldexp(1.0,-53),maximum,maximum,maximum);
  exact(0.5,std::ldexp(1.0,-65),maximum,(u64{1}<<63)-1,u64{1}<<63);
  exact(0.5,std::ldexp(1.0,-64),maximum,u64{1}<<63,(u64{1}<<63)+1);
  exact(std::ldexp(1.0,-64),0,maximum,0,1);
  exact(std::ldexp(1.0,-63),0,maximum,1,2);
  exact(std::ldexp(1.0,-64),std::ldexp(1.0,-64),maximum,1,2);
  overflow(1,tiny,maximum);  // Floor fits, but the required ceil does not.
  overflow(1,1,maximum);
  overflow(std::ldexp(1.0,63),std::ldexp(1.0,63),1);
  overflow(std::ldexp(1.0,64),0,1);
  overflow(std::numeric_limits<double>::max(),0,1);
  invalid(-1,0);
  invalid(0,-tiny);
  invalid(infinity,0);
  invalid(0,std::numeric_limits<double>::quiet_NaN());
  invalid(infinity,0,0);
}

// Independent Python Fraction probes use the exact binary64 bit patterns,
// avoiding decimal parser behavior around subnormals. Each input line is:
// <a uint64 hex bits> <b uint64 hex bits> <count uint64 decimal>
// Output is '<floor> <ceil>', 'overflow', or 'invalid_input', one per line.
int fraction_driver() {
  static_assert(sizeof(double)==sizeof(u64),"binary64 bit probe needs 8-byte double");
  std::string line;
  while(std::getline(std::cin,line)) {
    if(line.empty())continue;
    std::istringstream input(line);u64 a_bits=0,b_bits=0,count=0;
    if(!(input>>std::hex>>a_bits>>b_bits>>std::dec>>count)) {
      std::cerr<<"invalid Fraction probe input\n";return 1;
    }
    double a=0,b=0;std::memcpy(&a,&a_bits,sizeof(a));std::memcpy(&b,&b_bits,sizeof(b));
    try {
      const auto result=delay::sum(a,b,count);
      std::cout<<result.whole<<' '<<result.ceil<<'\n';
    }catch(const std::overflow_error&){std::cout<<"overflow\n";}
    catch(const std::invalid_argument&){std::cout<<"invalid_input\n";}
  }
  return 0;
}
}  // namespace

int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="--fraction-driver")return fraction_driver();
  try {
    require(argc==1,"use no arguments for tests or --fraction-driver for exact bit probes");
    hand_calculated_boundaries();
    std::cout<<"PASS region_delay_math_test "<<checks<<" hand-calculated cases\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL region_delay_math_test: "<<e.what()<<'\n';return 1;}
}
