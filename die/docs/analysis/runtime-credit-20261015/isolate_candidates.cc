#include "coarse_die/Execution.hh"
#include <fstream>
#include <iostream>
#include <filesystem>
using nlohmann::json;
namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&,const json&);
}
json read(const std::string& path) {std::ifstream stream(path);json j;stream>>j;return j;}
void write(const std::filesystem::path& path,const json& value) {std::ofstream(path)<<value.dump(2)<<"\n";}
int main(int argc,char** argv) {
 if(argc!=4)return 2;
 const auto hw=coarse_die::resolve_hardware(read(argv[1]));
 const auto task=read(argv[2]).at("specs").at(0);
 const auto normalized=coarse_die::compile_workspace_capacity_bound(hw,task).normalized_task;
 const auto candidates=coarse_die::detail::enumerate_mappings(hw,normalized);
 const std::filesystem::path out(argv[3]);std::filesystem::create_directories(out);
 json report=json::array();std::size_t index=0;
 for(const auto& plan:candidates) {
  coarse_die::Runtime runtime(hw);
  const auto handle=runtime.submit(plan,{1,1,1,1},0);
  json before=runtime.export_snapshot();std::uint64_t requested=0;std::size_t steps=0;
  try {
   while(!runtime.drained()) {
    before=runtime.export_snapshot();const auto next=runtime.next_event();
    if(!next)throw std::runtime_error("no event");requested=*next;
    runtime.advance_to(*next);if(++steps>100000)throw std::runtime_error("guard");
   }
   report.push_back({{"index",index},{"status","finished"},{"steps",steps},{"summary",plan->summary},
                    {"counters",runtime.counters()}});
  }catch(const std::exception& error) {
   const auto prefix=std::to_string(index);
   write(out/(prefix+"-before.json"),before);
   write(out/(prefix+"-plan.json"),coarse_die::plan_to_json(*plan));
   try {write(out/(prefix+"-after.json"),runtime.export_snapshot());}catch(...){}
   report.push_back({{"index",index},{"status","failed"},{"steps",steps},{"requested_ns",requested},
                    {"error",error.what()},{"summary",plan->summary}});
   std::cout<<"candidate "<<index<<" failed after "<<steps<<" steps at requested ns "<<requested<<": "<<error.what()<<"\n";
  }
  ++index;
 }
 write(out/"report.json",report);
 std::cout<<"candidates="<<index<<"\n";
}
