#include "test_support.hh"
#include "../src/CompilerInternal.hh"
#include <array>
#include <algorithm>
namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&,const nlohmann::json&);
}
using namespace coarse_die;
using namespace coarse_die_test;
namespace {
json task(std::uint64_t m,std::uint64_t n,std::uint64_t k,std::uint64_t repeats=1) {
  return {{"op_kind","gemm"},{"tensor_shapes",{{"m",m},{"n",n},{"k",k}}},
    {"storage_dtype",{{"activations","int4"},{"weights","int4"},{"output","bf16"}}},
    {"execution_profile","int4_matrix_vec_fp32_assumed_v1"},
    {"layer_instances",repeats},{"weight_passes",1}};
}
std::uint64_t divup(std::uint64_t x,std::uint64_t y){return x/y+(x%y!=0);}
void verify() {
  std::size_t checked=0;
  for(const auto side:{1ULL,2ULL}) {
    auto cfg=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware.json");
    cfg["architecture"]["core_columns_nx"]=side;cfg["architecture"]["core_rows_ny"]=side;
    cfg["architecture"]["die_geometry"]="isotropic";
    const auto hw=resolve_hardware(cfg);
    for(const auto shape:{std::array<std::uint64_t,3>{{3,5,7}},{{17,65,129}}}) {
      const auto spec=detail::normalize_task(hw,task(shape[0],shape[1],shape[2]));
      for(const auto& p:detail::enumerate_mappings(hw,spec)) {
        if(p->summary.at("dataflow_id")!="output_stationary_v2")continue;
        const auto tm=p->summary.at("tm").get<std::uint64_t>();
        const auto tn=p->summary.at("tn").get<std::uint64_t>();
        const auto tk=p->summary.at("tk").get<std::uint64_t>();
        const auto blocks=divup(shape[0],tm)*divup(shape[1],tn);
        const auto inputs=blocks*divup(shape[2],tk);
        require(p->epochs.size()>=3&&p->epochs.size()<=4&&service_description_count(*p)<=8,
                "all OS shapes use a fixed total description budget");
        const auto& audit=p->summary.at("service_curves").at(0);
        require(audit.at("description_count")==service_description_count(*p)&&
                audit.at("description_budget")== (service_description_count(*p)<=5 ? 5 : 8),
                "select smallest supported description budget including independent terminal switches");
        require(audit.at("input_units")==inputs&&audit.at("output_units")==blocks,
                "input K consumption and output blocks use actual tile counts, not waves or bytes");
        require(audit.at("first_load_units")==std::min<std::uint64_t>(side*side,blocks),
                "first load contains one K input for each active output tile");
        const auto first_bytes=p->epochs[0].activities[0].hbm_bytes;
        std::uint64_t first_payload=0;
        for(const auto& row:first_bytes)first_payload+=row[0]+row[1];
        if(divup(shape[2],tk)>1)
          require(first_payload<p->summary.at("hbm_read_bytes_total").get<std::uint64_t>(),
                  "multi-K does not place an entire output's K inputs in first SRAM load");
        const auto first_output_epoch=audit.at("first_output_epoch").get<std::size_t>();
        require(first_output_epoch==1,"first output threshold is the real whole-K prefix terminal");
        long double prefix_input=0;
        bool first_compute=false;
        for(const auto& a:p->epochs[first_output_epoch].activities) {
          if(a.role!=Role::Compute)continue;
          first_compute=true;
          prefix_input-=a.credit_delta[0];
          require(a.credit_delta[1]==0&&credit_at(a,0.5)[1]==0,
                  "no output is released before first cohort whole-K compute completes");
        }
        require(first_compute&&prefix_input==audit.at("first_output_input_units").get<double>(),
                "first whole-K compute consumes the exact first cohort K units");
        require(p->epochs[first_output_epoch+1].credit_initial[1]==
                    audit.at("first_load_units").get<double>(),
                "first complete cohort output is released at its analytic work threshold");
        std::uint64_t read=0,write=0,pe=0;
        long double in_delta=0,out_delta=0;
        for(const auto& e:p->epochs) {
          out_delta+=e.credit_initial[1];
          for(const auto& a:e.activities) {
            for(const auto& row:a.hbm_bytes){read+=row[0];write+=row[1];}
            pe+=a.int4_ops;in_delta+=a.credit_delta[0];out_delta+=a.credit_delta[1];
            if(a.role==Role::Compute && a.credit_curve.size()>2) {
              const auto f=a.credit_curve[1].progress;
              require(credit_at(a,f)[0]==-audit.at("first_output_input_units").get<double>(),
                      "first output consumes the actual cohort K prefix, including tail work");
              require(credit_at(a,f/2)[1]==0&&credit_at(a,f)[1]==0,
                      "first output cannot be retired before its compute-work threshold");
            }
          }
        }
        require(read==p->summary.at("hbm_read_bytes_total")&&
                write==p->summary.at("hbm_write_bytes_total")&&
                pe==2*shape[0]*shape[1]*shape[2],"actual primitive bytes/operations are conserved exactly");
        require(in_delta==0&&out_delta==0,
                "each unit has matching load/consume or output/final-release/store endpoint");
        ++checked;
      }
    }
  }
  require(checked>20,"full/tail and multicore checks exercised real candidate enumeration");
}
}
int main(){return run_test("os_service_curve_test",verify);}
