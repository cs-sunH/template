// Standalone arithmetic only. No Runtime, Plan compiler, or allocation code.
#include <array>
#include <chrono>
#include <cfenv>
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
struct Dot { Real value=0;bool finite=true,normal_domain=true;std::size_t product_terms=0,nonzero_residuals=0;std::array<Real,3> high{},low{}; };
bool normal_or_zero(Real x){return x==0 || std::fpclassify(x)==FP_NORMAL;}
Dot fma_dot(Real constant,const std::array<Real,3>& d,const std::array<Real,3>& p,std::size_t n) {
 Dot result;std::array<Real,7> terms{};std::size_t count=1;terms[0]=constant;
 result.finite=std::isfinite(constant);result.normal_domain=normal_or_zero(constant);
 for(std::size_t i=0;i<n;++i){
  const Real hi=d[i]*p[i];const Real lo=std::fma(d[i],p[i],-hi);
  result.high[i]=hi;result.low[i]=lo;
  terms[count++]=hi;terms[count++]=lo;++result.product_terms;result.nonzero_residuals+=lo!=0;
  result.finite=result.finite&&std::isfinite(hi)&&std::isfinite(lo);
  result.normal_domain=result.normal_domain&&normal_or_zero(hi)&&normal_or_zero(lo);
  // Conservative lower bound on the exact binary product quantum. It avoids
  // treating a zero fma residual as an error-free product when underflow could
  // have erased it. No exponent relaxation is selected from the oracle.
  if(d[i]!=0&&p[i]!=0){const int quantum=std::ilogb(std::abs(d[i]))+std::ilogb(std::abs(p[i]))-2*(std::numeric_limits<Real>::digits-1);if(quantum<std::numeric_limits<Real>::min_exponent-1)result.normal_domain=false;}
 }
 Real sum=0,correction=0;
 for(std::size_t i=0;i<count;++i){const Real x=terms[i],next=sum+x;const Real residual=std::abs(sum)>=std::abs(x)?(sum-next)+x:(x-next)+sum;correction+=residual;sum=next;result.finite=result.finite&&std::isfinite(sum)&&std::isfinite(correction);result.normal_domain=result.normal_domain&&normal_or_zero(sum)&&normal_or_zero(correction)&&normal_or_zero(residual);}
 result.value=sum+correction;result.finite=result.finite&&std::isfinite(result.value);result.normal_domain=result.normal_domain&&normal_or_zero(result.value);return result;
}
struct Sample {Real initial,capacity;std::array<Real,3>d,p;std::size_t n;};
Real direct_cost(const Sample& s,bool head){Real sum=head?s.capacity-s.initial:s.initial;for(std::size_t i=0;i<s.n;++i)sum+=(head?-s.d[i]:s.d[i])*s.p[i];return sum;}
Real anchor_cost(const Sample& s,bool head){std::size_t j=0;Real endpoint=s.initial;for(std::size_t i=0;i<s.n;++i){endpoint+=s.d[i];if(head?s.d[i]>s.d[j]:s.d[i]<s.d[j])j=i;}std::array<Real,4>terms{};std::size_t n=2;terms[0]=(head?s.capacity-s.initial:s.initial)*(1-s.p[j]);terms[1]=(head?s.capacity-endpoint:endpoint)*s.p[j];for(std::size_t i=0;i<s.n;++i)if(i!=j)terms[n++]=(head?-s.d[i]:s.d[i])*(s.p[i]-s.p[j]);Real sum=0,c=0;for(std::size_t i=0;i<n;++i){const Real next=sum+terms[i];if(std::abs(sum)>=std::abs(terms[i]))c+=(sum-next)+terms[i];else c+=(terms[i]-next)+sum;sum=next;}return sum+c;}
Real fma_cost(const Sample& s,bool head){auto d=s.d;if(head)for(std::size_t i=0;i<s.n;++i)d[i]=-d[i];return fma_dot(head?s.capacity-s.initial:s.initial,d,s.p,s.n).value;}
int main(int argc,char** argv){
 if(argc!=3)return 2;std::ifstream in(argv[1]);json cases;in>>cases;std::ofstream out(argv[2]);
 if(std::fegetround()!=FE_TONEAREST)throw std::runtime_error("requires round-to-nearest");
 out<<json{{"kind","abi"},{"long_double_digits",std::numeric_limits<Real>::digits},{"long_double_min_exponent",std::numeric_limits<Real>::min_exponent},{"long_double_max_exponent",std::numeric_limits<Real>::max_exponent},{"rounding","verified FE_TONEAREST; no fast math; ffp-contract=off"}}.dump()<<'\n';
 std::vector<Sample> benchmark_samples;
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
   auto hd=d;for(std::size_t i=0;i<n;++i)hd[i]=-hd[i];const auto fq=fma_dot(initial,d,p,n),fh=fma_dot(capacity-initial,hd,p,n);
   result["fma_dot_q"]={{"value",number(fq.value)},{"finite",fq.finite},{"normal_domain",fq.normal_domain},{"nonzero_product_residuals",fq.nonzero_residuals}};
   result["fma_dot_h"]={{"value",number(fh.value)},{"finite",fh.finite},{"normal_domain",fh.normal_domain},{"nonzero_product_residuals",fh.nonzero_residuals}};
   result["fma_dot_q"]["products"]=json::array();result["fma_dot_h"]["products"]=json::array();for(std::size_t i=0;i<n;++i){result["fma_dot_q"]["products"].push_back({{"high",number(fq.high[i])},{"low",number(fq.low[i])}});result["fma_dot_h"]["products"].push_back({{"high",number(fh.high[i])},{"low",number(fh.low[i])}});}
   if(fq.finite&&fh.finite&&fq.normal_domain&&fh.normal_domain)benchmark_samples.push_back({initial,capacity,d,p,n});
  }catch(const std::exception& e){result["error"]=e.what();}
  out<<result.dump()<<'\n';
 }
 // This is only an offline arithmetic micro-cost. Parsing/JSON/oracle and all
 // Runtime operations are outside it. Methods see the same eligible inputs.
 json cost={{"kind","benchmark"},{"scope","tiny arithmetic evaluator only; not Runtime or API speed budget"},{"eligible_samples",benchmark_samples.size()},{"batches",5},{"rounds_per_batch",100},{"methods",json::object()}};
 volatile Real sink=0;
 for(const auto& method:{std::string("direct"),std::string("structural_anchor_compensated"),std::string("fma_product_residual_compensated")}){
  const auto fn=method=="direct"?direct_cost:method=="structural_anchor_compensated"?anchor_cost:fma_cost;
  for(const auto& s:benchmark_samples){sink+=fn(s,false);sink+=fn(s,true);}
  json times=json::array();for(unsigned batch=0;batch<5;++batch){const auto begin=std::chrono::steady_clock::now();for(unsigned round=0;round<100;++round)for(const auto& s:benchmark_samples){sink+=fn(s,false);sink+=fn(s,true);}const auto end=std::chrono::steady_clock::now();const double ns=std::chrono::duration<double,std::nano>(end-begin).count();times.push_back(ns/(200.0*benchmark_samples.size()));}cost["methods"][method]=times;
 }
 cost["observable_sink"]=number(sink);out<<cost.dump()<<'\n';
}
