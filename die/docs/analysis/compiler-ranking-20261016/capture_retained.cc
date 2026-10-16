#include "coarse_die/Plan.hh"
#include "coarse_die/Memory.hh"
#include <fstream>
#include <iostream>
using namespace coarse_die;using J=nlohmann::json;
void visit(const J& j,const std::string& path,J& out){
 if(j.is_string())out[path]={{"value",j},{"capacity",j.get_ref<const std::string&>().capacity()}};
 if(j.is_object())for(auto i=j.begin();i!=j.end();++i){out[path+"/key:"+i.key()]=i.key().capacity();visit(i.value(),path+"/"+i.key(),out);}
 if(j.is_array()){out[path+"/@capacity"]=j.get_ref<const J::array_t&>().capacity();for(std::size_t i=0;i<j.size();++i)visit(j[i],path+"/"+std::to_string(i),out);}
}
int main(int argc,char**argv){J cfg,tasks;std::ifstream("die/configs/example_hardware_explicit.json")>>cfg;std::ifstream("die/configs/example_tasks.json")>>tasks;auto hw=resolve_hardware(cfg);J out=J::array();for(const auto& task:tasks["specs"]){auto p=compile(hw,task);J row;visit(p->task,"task",row);visit(p->summary,"summary",row);row["@retained_plan_bytes"]=retained_plan_bytes(*p);row["@fingerprint_capacity"]=p->fingerprint.capacity();row["@hardware_fingerprint_capacity"]=p->hardware_fingerprint.capacity();out.push_back(row);}std::ofstream(argv[1])<<out.dump(2)<<'\n';}
