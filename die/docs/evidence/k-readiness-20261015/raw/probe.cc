// Diagnostic only: reuse the exact baseline fixture and output schema.
#define main existing_compiled_driver_entry
#include "/workspace/template/die/tests/compiled_gemm_oracle_driver.cc"
#undef main
#include "/workspace/template/die/src/GemmOSResourceFacts.hh"
#include <deque>
#include <map>
#include <optional>
#include <tuple>

namespace diagnostic {
namespace access = coarse_die::detail::gemm_os_access_counter;
using Key=std::tuple<std::uint64_t,std::uint64_t,std::uint64_t,bool,bool>;
struct Token {std::uint64_t output=0,k=0;bool final=false;Key key;};
struct Job {Token token;std::uint64_t handle=0;};
struct Request {
 std::uint64_t arrival=0,id=0,next_read=0,stored=0,finish=0;
 bool started=false,done=false;
 std::deque<Token> inputs,outputs;
 std::optional<Job> load,compute,store;
};
struct Kernel {
 const Hardware& hw;const Plan& mapping;
 std::uint64_t M,N,K,tm,tn,tk,mt,nt,kt,output_count,wave_count;
 std::map<Key,std::array<std::shared_ptr<const Plan>,3>> facts;
 Kernel(const Hardware& h,const Plan& p):hw(h),mapping(p) {
  auto shape=p.task.at("tensor_shapes");auto summary=p.summary;
  if(summary.at("dataflow_id")!="output_stationary_v2")throw std::runtime_error("diagnostic requires selected OS mapping");
  M=shape.at("m");N=shape.at("n");K=shape.at("k");tm=summary.at("tm");tn=summary.at("tn");tk=summary.at("tk");
  mt=(M+tm-1)/tm;nt=(N+tn-1)/tn;kt=(K+tk-1)/tk;output_count=mt*nt;wave_count=output_count*kt;
 }
 Token token(std::uint64_t ordinal) {
  const auto output=ordinal/kt,k=ordinal%kt;
  const auto row=output/nt,column=output%nt;
  return {output,k,k+1==kt,Key{std::min(tm,M-row*tm),std::min(tn,N-column*tn),std::min(tk,K-k*tk),k==0,k+1==kt}};
 }
 const std::array<std::shared_ptr<const Plan>,3>& plans(const Token& token) {
  auto found=facts.find(token.key);if(found!=facts.end())return found->second;
  access::Input in;std::tie(in.m,in.n,in.k,in.first_k,in.final_k)=token.key;
  in.a_storage=in.w_storage=in.output_storage=access::StorageDType::Fp32;
  in.matrix_path=access::MatrixPath::StrictFp32Vec;
  in.kappa_quant=hw.input.at("performance_profile").value("kappa_quant_eqops",4.0);
  in.kappa_acc=hw.input.at("performance_profile").value("kappa_acc_convert_eqops",2.0);
  const auto activities=coarse_die::detail::gemm_os_single_core_resource_facts(hw,in);
  std::array<std::shared_ptr<const Plan>,3> result;
  for(std::size_t role=0;role<3;++role) {
   auto plan=std::make_shared<Plan>();plan->hardware_fingerprint=hw.fingerprint;
   plan->task={{"name","diagnostic_k_ready_fact"},{"m",in.m},{"n",in.n},{"k",in.k},{"first",in.first_k},{"final",in.final_k},{"role",role}};
   plan->summary={{"status","diagnostic_fact"}};
   coarse_die::Epoch epoch;epoch.name="exact_k_stage_fact";epoch.activities={activities[role]};
   // Slot reservation and full-terminal tokens enforce readiness outside the
   // Runtime. The returned facts already have no signed fluid credit.
   if(epoch.activities[0].credit_delta!=std::array<double,2>{0.0,0.0})throw std::runtime_error("resource fact unexpectedly has fluid credit");
   plan->epochs={epoch};plan->fingerprint=coarse_die::plan_to_json(*plan).at("fingerprint");result[role]=plan;
  }
  return facts.emplace(token.key,std::move(result)).first->second;
 }
};

json run(const Hardware& hw,const Plan& mapping,const std::vector<std::uint64_t>& arrivals) {
 Kernel kernel(hw,mapping);Runtime runtime(hw);std::vector<Request> requests;
 for(std::size_t i=0;i<arrivals.size();++i){Request r;r.arrival=arrivals[i];r.id=i+1;requests.push_back(r);}
 std::uint64_t now=0,events=0,stage_completions=0;
 const auto submit=[&](Request& r,const Token& token,std::size_t role) {
  return Job{token,runtime.submit(kernel.plans(token)[role],owner(r.id),now)};
 };
 while(!std::all_of(requests.begin(),requests.end(),[](const auto& r){return r.done;})) {
  // Terminal facts release reservations before any same-tick admissions.
  for(auto& r:requests) {
   if(!r.started&&r.arrival<=now)r.started=true;
   for(auto role:{0,1,2}) {
    auto& job=role==0?r.load:(role==1?r.compute:r.store);
    if(!job)continue;const auto finish=runtime.terminal_finish_ns(job->handle);if(!finish)continue;
    if(role==0)r.inputs.push_back(job->token);
    else if(role==1&&job->token.final)r.outputs.push_back(job->token);
    else if(role==2){++r.stored;if(r.stored==kernel.output_count){r.done=true;r.finish=*finish;}}
    runtime.destroy_execution(job->handle);job.reset();++stage_completions;
   }
  }
  bool changed=true;
  while(changed) {
   changed=false;
   for(auto& r:requests) {
    if(!r.started||r.done)continue;
    const auto input_used=r.inputs.size()+bool(r.load)+bool(r.compute);
    const auto output_used=r.outputs.size()+bool(r.store)+(r.compute&&r.compute->token.final);
    if(!r.load&&r.next_read<kernel.wave_count&&input_used<2) {
     auto token=kernel.token(r.next_read++);r.load=submit(r,token,0);changed=true;
    }
    if(!r.compute&&!r.inputs.empty()&&output_used<2) {
     auto token=r.inputs.front();r.inputs.pop_front();r.compute=submit(r,token,1);changed=true;
    }
    if(!r.store&&!r.outputs.empty()) {
     auto token=r.outputs.front();r.outputs.pop_front();r.store=submit(r,token,2);changed=true;
    }
   }
  }
  if(std::all_of(requests.begin(),requests.end(),[](const auto& r){return r.done;}))break;
  auto next=runtime.next_event();
  for(const auto& r:requests)if(!r.started&&(!next||r.arrival<*next))next=r.arrival;
  if(!next||*next<=now)throw std::runtime_error("diagnostic readiness deadlock");
  runtime.advance_to(*next);now=*next;
  if(++events>1000000)throw std::runtime_error("diagnostic exhausted unchanged event budget");
 }
 for(const auto& r:requests)if(r.load||r.compute||r.store||!r.inputs.empty()||!r.outputs.empty())throw std::runtime_error("diagnostic ended with pending stages");
 json counters=counters_json(runtime);
 json extra={{"diagnostic_only",true},{"stage_completions",stage_completions},{"fact_class_count",kernel.facts.size()},
             {"input_slots",2},{"output_slots",2},{"time_quantization","public Runtime integer-nanosecond calls"},
             {"complexity","uncompressed O(events); no production efficiency claim"}};
 if(arrivals.size()==1)return {{"request","solo"},{"arrival_ns",arrivals[0]},{"finish_ns",requests[0].finish},
  {"finish_s",double(requests[0].finish)/kNsPerSecond},{"event_advances",events},{"counters",counters},{"readiness_diagnostic",extra}};
 return {{"request_names",{"comp0","comp1"}},{"phase_offset_ns",arrivals[1]},{"phase_offset_s",double(arrivals[1])/kNsPerSecond},
  {"finish_ns_by_request",{{"comp0",requests[0].finish},{"comp1",requests[1].finish}}},
  {"finish_s_by_request",{{"comp0",double(requests[0].finish)/kNsPerSecond},{"comp1",double(requests[1].finish)/kNsPerSecond}}},
  {"event_advances",events},{"counters",counters},{"readiness_diagnostic",extra}};
}
}
int main(int argc,char** argv) {
 try {
  const auto shape=parse_shape(argc,argv);const auto hardware=synthetic_hardware();const auto plan=compile(hardware,gemm_task(shape));
  json concurrent=json::array();for(auto offset:{0ULL,50000000ULL})concurrent.push_back(diagnostic::run(hardware,*plan,{0,offset}));
  std::cout<<json{{"schema_version",1},{"measurement_source","diagnostic_cpp_k_ready_facts_and_actual_public_runtime"},
   {"baseline_compiled_plan_executed",false},{"synthetic_hardware",synthetic_capacity_json(hardware)},
   {"compiled_plan",plan_json(*plan)},{"standalone",diagnostic::run(hardware,*plan,{0})},{"two_comp",concurrent}}.dump(2)<<"\n";
 }catch(const std::exception& e){std::cerr<<"k-ready diagnostic: "<<e.what()<<"\n";return 1;}
}
