#include "prepared_readiness_driver.cc"
#include <iomanip>

std::string scalar(long double x) {
  std::ostringstream out; out << std::setprecision(std::numeric_limits<long double>::max_digits10) << x; return out.str();
}
json pair_study(const Hardware& hw, k_readiness::Kernel& kernel, std::uint64_t ordinal, std::uint64_t start) {
  Runtime runtime(hw); runtime.advance_to(start);
  const auto ctoken=kernel.token(ordinal), ltoken=kernel.token(ordinal+1);
  if(ctoken.first||ctoken.final||ltoken.first||ltoken.final||ctoken.q!=ltoken.q||ctoken.class_index!=ltoken.class_index)
    throw std::runtime_error("not a homogeneous nonfinal middle pair");
  json cycles=json::array();
  for(unsigned i=0;i<32;++i) {
    auto before=runtime.export_snapshot();auto begin=runtime.now_ns();
    auto ch=runtime.submit(kernel.plans(ctoken)[1],owner(1),begin);
    auto lh=runtime.submit(kernel.plans(ltoken)[0],owner(1),begin);
    auto cp=runtime.query_progress(ch),lp=runtime.query_progress(lh);
    json steps=json::array();
    while(!runtime.terminal_finish_ns(ch)||!runtime.terminal_finish_ns(lh)) {
      auto next=runtime.next_event();if(!next)throw std::runtime_error("no event");runtime.advance_to(*next);
      steps.push_back({{"at_ns",runtime.now_ns()-begin},{"compute_terminal",runtime.terminal_finish_ns(ch).has_value()},{"load_terminal",runtime.terminal_finish_ns(lh).has_value()}});
    }
    auto after=runtime.export_snapshot();
    auto delta=[&](const char* key){return std::stold(after["stats"][key].get<std::string>())-std::stold(before["stats"][key].get<std::string>());};
    cycles.push_back({{"period_ns",runtime.now_ns()-begin},{"compute_finish_ns",*runtime.terminal_finish_ns(ch)-begin},{"load_finish_ns",*runtime.terminal_finish_ns(lh)-begin},{"compute_rate",cp["activities"][0]["rate"]},{"load_rate",lp["activities"][0]["rate"]},{"compute_alpha",cp["activities"][0]["alpha"]},{"load_alpha",lp["activities"][0]["alpha"]},{"hbm_busy_ns",scalar(delta("hbm_busy_ns"))},{"compute_busy_ns",scalar(delta("compute_busy_ns"))},{"shared_ns",scalar(delta("hbm_shared_ns"))},{"steps",steps}});
    runtime.destroy_execution(ch);runtime.destroy_execution(lh);
  }
  return {{"start_ns",start},{"first",cycles.front()},{"all_cycles",cycles},{"all_cycle_records_equal",std::all_of(cycles.begin(),cycles.end(),[&](const auto& c){return c==cycles.front();})}};
}
int main(){try {
  const auto hw=synthetic_hardware(); const auto plan=compile(hw,gemm_task(Shape{}));k_readiness::Kernel kernel(hw,k_readiness::geometry_input(hw,*plan),false);
  json classes=json::array();const auto& geom=kernel.geometry();
  for(std::size_t i=0;i<geom.class_count();++i) {
    const auto& cls=geom.access_class(i);if(cls.first_k||cls.final_k||cls.k_tile_count<2)continue;
    auto q=cls.m_tile_begin*geom.n_tiles()+cls.n_tile_begin;auto ordinal=q*geom.k_tiles()+cls.k_tile_begin;
    json starts=json::array();for(auto start:{0ULL,50000000ULL,1000000000000ULL,9007199254740991ULL,9223372036854775808ULL})starts.push_back(pair_study(hw,kernel,ordinal,start));
    classes.push_back({{"m",cls.m},{"n",cls.n},{"k",cls.k},{"starts",starts}});
  }
  std::cout<<json{{"scope","actual public Runtime repeated isolated nonfinal pair; no fast-forward applied; phase/clock numerical study only"},{"classes",classes}}.dump(2)<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
