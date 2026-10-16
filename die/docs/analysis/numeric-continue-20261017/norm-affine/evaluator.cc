// Standalone arithmetic only. No Runtime, Plan compiler, or allocation code.
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <json/json.hpp>
using nlohmann::json;
using Real=long double;
std::string hex(Real x){std::ostringstream s;s<<std::hexfloat<<x;return s.str();}
std::string dec(Real x){std::ostringstream s;s<<std::scientific<<std::setprecision(std::numeric_limits<Real>::max_digits10)<<x;return s.str();}
json number(Real x){return {{"hex",hex(x)},{"decimal",dec(x)}};}
Real parsed_double(const json& j){const auto s=j.get<std::string>();char* e=nullptr;const double x=std::strtod(s.c_str(),&e);if(e!=s.c_str()+s.size()||!std::isfinite(x))throw std::runtime_error("invalid coefficient");return x;}
Real parsed_progress(const json& j){const auto s=j.get<std::string>();char* e=nullptr;const Real x=std::strtold(s.c_str(),&e);if(e!=s.c_str()+s.size()||!std::isfinite(x))throw std::runtime_error("invalid progress");return x;}
Real plain(const std::vector<Real>& terms){Real sum=0;for(Real x:terms)sum+=x;return sum;}
Real compensated(const std::vector<Real>& terms){Real sum=0,c=0;for(Real x:terms){const Real next=sum+x;if(std::abs(sum)>=std::abs(x))c+=(sum-next)+x;else c+=(x-next)+sum;sum=next;}return sum+c;}
json sums(const std::vector<Real>& terms){return {{"plain",number(plain(terms))},{"compensated",number(compensated(terms))}};}
int main(int argc,char** argv){
 if(argc!=3)return 2;std::ifstream in(argv[1]);json cases;in>>cases;std::ofstream out(argv[2]);
 out<<json{{"kind","abi"},{"long_double_digits",std::numeric_limits<Real>::digits},{"long_double_min_exponent",std::numeric_limits<Real>::min_exponent},{"long_double_max_exponent",std::numeric_limits<Real>::max_exponent},{"rounding","nearest, no fast math; ffp-contract=off"}}.dump()<<'\n';
 for(const auto& row:cases.at("cases")){
  json result={{"kind","case"},{"id",row.at("id")}};
  try {
   const Real initial=parsed_double(row.at("initial")),capacity=parsed_double(row.at("capacity"));
   const std::size_t n=row.at("coefficients").size();if(n<1||n>3||row.at("progress").size()!=n)throw std::runtime_error("fixed activity count must be 1..3");
   std::array<Real,3> d{},p{};Real endpoint=initial;
   for(std::size_t i=0;i<n;++i){d[i]=parsed_double(row.at("coefficients")[i]);p[i]=parsed_progress(row.at("progress")[i]);endpoint+=d[i];}
   if(row.contains("neighbor")){const auto k=row.at("neighbor").at("index").get<std::size_t>();const int direction=row.at("neighbor").at("direction").get<int>();p[k]=std::nextafter(p[k],direction<0?-std::numeric_limits<Real>::infinity():std::numeric_limits<Real>::infinity());}
   for(std::size_t i=0;i<n;++i)if(!(p[i]>=0&&p[i]<=1))throw std::runtime_error("neighbor outside legal progress domain");
   result["initial"]=number(initial);result["capacity"]=number(capacity);result["endpoint"]=number(endpoint);result["head_initial"]=number(capacity-initial);result["head_endpoint"]=number(capacity-endpoint);
   result["coefficients"]=json::array();result["progress"]=json::array();
   for(std::size_t i=0;i<n;++i){result["coefficients"].push_back(number(d[i]));result["progress"].push_back(number(p[i]));}
   std::vector<Real> q{initial},h{capacity-initial};for(std::size_t i=0;i<n;++i){q.push_back(d[i]*p[i]);h.push_back(-d[i]*p[i]);}
   result["direct_q"]=sums(q);result["direct_h"]=sums(h);result["anchors"]=json::array();
   for(std::size_t j=0;j<n;++j){
    std::vector<Real> aq{initial*(1-p[j]),endpoint*p[j]},ah{(capacity-initial)*(1-p[j]),(capacity-endpoint)*p[j]};
    for(std::size_t i=0;i<n;++i)if(i!=j){aq.push_back(d[i]*(p[i]-p[j]));ah.push_back(-d[i]*(p[i]-p[j]));}
    result["anchors"].push_back({{"pivot",j},{"q",sums(aq)},{"h",sums(ah)}});
   }
   std::size_t q_pivot=0,h_pivot=0;for(std::size_t i=1;i<n;++i){if(d[i]<d[q_pivot])q_pivot=i;if(d[i]>d[h_pivot])h_pivot=i;}
   result["q_pivot"]=q_pivot;result["h_pivot"]=h_pivot;
  }catch(const std::exception& e){result["error"]=e.what();}
  out<<result.dump()<<'\n';
 }
}
