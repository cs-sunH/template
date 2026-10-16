/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "DieModel.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace {
bool ok = true;
void check(bool condition, const std::string& label) {
  if (!condition) { ok = false; std::fprintf(stderr, "FAIL: %s\n", label.c_str()); }
}
void near(double a, double b, const std::string& label) {
  check(std::abs(a-b) <= 1e-12 * std::max(1.0, std::abs(b)), label);
}
nlohmann::json node(double sram=0.75, unsigned width=512, unsigned nx=23,
                    unsigned ny=23, double area=800) {
  return {{"sram-mb-per-core",sram},{"mesh-nx",nx},{"mesh-ny",ny},
          {"compute-area-mm2",area},{"hbm-per-side",4u},
          {"hbm-sides",1u},{"noc-width-bits",width}};
}
void reject(const nlohmann::json& j,const std::string& text) {
  try { DieModel::make_hardware(j); check(false,"accepted invalid input: "+text); }
  catch(const DieModel::DieModelError& e) {
    check(std::string(e.what()).find(text)!=std::string::npos,"diagnostic: "+std::string(e.what()));
  }
}
}
int main() {
  // Independent transcription of the authoritative DOCX's 32 integer rows.
  const int vec[4][8]={{2,15,40,77,126,204,280,368},
    {2,15,40,77,126,198,273,360},{2,15,40,70,117,180,252,336},
    {2,12,32,56,99,156,210,272}};
  const int pe[4][8]={{40,297,791,1523,2492,4035,5538,7278},
    {40,297,791,1523,2492,3916,5399,7120},{40,297,791,1384,2314,3560,4984,6645},
    {40,237,633,1108,1958,3085,4153,5379}};
  const unsigned widths[]={64,128,256,512};
  const auto rows=DieModel::core_table();check(rows.size()==32,"32 Core rows");
  for(unsigned b=0;b<4;++b) for(unsigned s=0;s<8;++s) {
    const double mib=(s+1)*0.25, side=(s+1)*0.40662;
    const auto hw=DieModel::make_hardware(node(mib,widths[b],1,1));
    check(hw.cores==1 && hw.mesh_n==1,"explicit single Core count");
    near(hw.acore_mm2,side*side,"SRAM long side fixes outline");
    near(hw.pe_peak_die,2e9*pe[b][s],"INT4 installed PE peak");
    near(hw.vec_peak_die,3.9554e9*vec[b][s],"FP32 equivalent Vec peak");
    near(hw.sram_usable_bytes_per_core,0.9*mib*1048576.0,"MiB SRAM bytes");
    check(rows[8*b+s].noc_width_bits==widths[b],"NoC row identity");
    near(rows[8*b+s].pe_int4_gops_per_core,2.0*pe[b][s],"PE row mirror");
  }
  const auto hw=DieModel::make_hardware(node());
  check(hw.mesh_n==23 && hw.cores==529,"user-specified 23x23 fixture");
  near(hw.pe_peak_die,529.0*1266e9,"fixture INT4 peak");
  near(hw.vec_peak_die,529.0*32*3.9554e9,"fixture FP32 Vec equivalent peak");
  near(hw.sram_usable_bytes_per_core,707788.8,"fixture usable SRAM");
  near(hw.link_peak_bytes,64e9,"NoC link peak");
  near(hw.hbm_capacity_bytes,64e9,"explicit legacy HBM capacity");
  near(hw.hbm_bandwidth_bytes,1640e9,"explicit legacy HBM bandwidth");
  auto summary=DieModel::hardware_summary_json(hw);
  check(!summary.at("die").contains("core-type"),"no Core model selector in summary");
  check(summary.at("die").at("mesh-nx")==23 && summary.at("die").at("mesh-ny")==23,"grid summary");
  near(summary.at("derived").at("pe_peak_int4_tops"),529.0*1.266,"INT4 unit");
  check(summary.at("derived").contains("pe_peak_tflops_deprecated_unit"),"compatibility scalar unit disclosure");
  check(summary.at("derived").contains("hbm_profile"),"legacy HBM provenance");
  check(DieModel::hardware_summary_json(DieModel::make_hardware(nlohmann::json::parse(summary.dump())))==summary,"summary round trip");
  check(summary.at("source_identity").at("core_resource_table").at("id")=="core32_v1","runtime Core32 identity");
  check(summary.at("source_identity").at("source_document").contains("sha256"),"runtime DOCX identity");
  auto forged=nlohmann::json::parse(summary.dump());
  forged["source_identity"]["core_resource_table"]["id"]="unrecognized_table";
  reject(forged,"source_identity");
  forged=nlohmann::json::parse(summary.dump());
  forged["source_identity"]["source_document"]["sha256"]="forged";
  reject(forged,"source_identity");
  auto j=node();j.erase("mesh-nx");reject(j,"explicit mesh-nx and mesh-ny");
  reject(node(0.75,512,23,22),"must be equal positive integers");
  reject(node(0.75,512,0,0),"must be equal positive integers");
  reject(node(0.75,512,35,35),"exceeds compute-area-mm2");
  reject(node(0.75,1024),"noc-width-bits");
  reject(node(3.0,256,1,1),"core table");
  for(const char* value:{"small","large","core"}) {
    j=node();j["core-type"]=value;reject(j,"core-type");
  }
  // Preserve legacy HBM layout legality; no new SKU or capacity is inferred.
  const unsigned layouts[][3]={{300,2,1},{300,2,2},{500,2,1},{500,3,1},
    {500,2,2},{500,3,2},{800,3,1},{800,4,1},{800,3,2},{800,4,2}};
  for(const auto& layout:layouts) {
    j=node(0.25,256,1,1,layout[0]);j["hbm-per-side"]=layout[1];j["hbm-sides"]=layout[2];
    const auto h=DieModel::make_hardware(j);
    check(h.hbm_count==layout[1]*layout[2],"explicit legacy HBM layout");
    near(h.hbm_capacity_bytes,16e9*layout[1]*layout[2],"legacy stack capacity");
    near(h.hbm_bandwidth_bytes,410e9*layout[1]*layout[2],"legacy stack bandwidth");
  }
  for(const auto& layout:std::array<std::array<unsigned,3>,3>{{{{300,3,1}},{{800,2,2}},{{700,3,2}}}}) {
    j=node(0.25,256,1,1,layout[0]);j["hbm-per-side"]=layout[1];j["hbm-sides"]=layout[2];reject(j,"illegal");
  }
  j=node();j["turbo-mode"]=true;reject(j,"turbo-mode");
  j=node();j.erase("noc-width-bits");reject(j,"noc-width-bits");
  j=node(0.75,512,10,10);j["die-width-mm"]=25.0;j["die-height-mm"]=32.0;
  const auto explicit_shape=DieModel::make_hardware(j);
  check(explicit_shape.explicit_die_dimensions && explicit_shape.cores==100,"authorized explicit dimensions");
  near(explicit_shape.pe_peak_die,126.6e12,"authorized INT4 engineering peak");
  auto explicit_summary=DieModel::hardware_summary_json(explicit_shape);
  check(DieModel::hardware_summary_json(DieModel::make_hardware(nlohmann::json::parse(explicit_summary.dump())))==explicit_summary,"explicit dimensions wire round trip");
  check(explicit_summary.at("die").at("die-width-mm")==25.0 && explicit_summary.at("die").at("die-height-mm")==32.0,"dimensions carried on wire");
  j.erase("die-height-mm");reject(j,"provided together");
  j["die-height-mm"]=33.0;reject(j,"must equal");
  j["die-width-mm"]=10.0;j["die-height-mm"]=80.0;reject(j,"independently");
  // Width changes installed units, never the user grid or the Core outline.
  const auto narrow=DieModel::make_hardware(node(0.75,64));
  check(narrow.cores==hw.cores,"NoC does not silently change Core count");
  near(narrow.acore_mm2,hw.acore_mm2,"NoC does not change Core outline");
  check(narrow.pe_peak_die>hw.pe_peak_die,"NoC affects installed compute units");
  j={{"schema_version",1u},{"profile","tx8_reference_v1"},{"die",node()}};
  const auto tx8=DieModel::make_hardware(j);
  near(tx8.tx8_matrix_peak_flop_s,DieModel::calibration::matrix_peak_flop_s(),"frozen TX8 conditional reference");
  j["profile"]="mesh_die_v9";reject(j,"profile");
  if(ok) std::puts("[hardware_test] ALL PASS");return ok?0:1;
}
