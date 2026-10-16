#pragma once

// Included inside Mapping.cc's private namespace, after its checked workload,
// routing and Flow helpers. No separate timing engine or runtime state.
// One row per Core permits statistics to remain in 16 B of SRAM across two
// ordered kernels. Multiple rows per Core require another explicit mapping.
bool add_norm_long_single_cohort(Plan& plan,Metrics& m,const Hardware& hw,
    const Topology& topo,u64 rows,u64 hidden,u64 executions,
    const std::string& ad,const std::string& pd,const std::string& od,bool layernorm) {
  if(executions!=1 || rows>du(hw,"die","core_count") || hidden<=512)return false;
  const u64 active=rows,tile=256,tiles=ceildiv(hidden,tile),tail=hidden-(tiles-1)*tile;
  const u64 parameters=layernorm ? 2 : 1;
  const auto param_bytes=[&](u64 n){return mul128(parameters,bytes(n,pd),"Norm long parameter bytes");};
  const u128 input_tile=add128(bytes(tile,ad),param_bytes(tile),"Norm long input tile");
  const u128 working=add128(mul128(tile,4,"Norm long work tile"),
      pd=="fp32" ? 0 : mul128(parameters,mul128(tile,4,"Norm long parameter tile"),"Norm long parameter tiles"),
      "Norm long working buffers");
  const u128 peak=add128(add128(mul128(2,input_tile,"Norm long input slots"),
      mul128(2,bytes(tile,od),"Norm long output slots"),"Norm long raw slots"),
      add128(working,80,"Norm long statistics and metadata"),"Norm long SRAM peak");
  if(static_cast<long double>(peak)>perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core"))return false;
  for(u64 units:{hidden,tiles,tiles-1,tiles-2,tiles-3})
    if(static_cast<long double>(static_cast<double>(units))!=static_cast<long double>(units))
      fail("overflow: Norm long cumulative credit is not exactly representable");
  const auto private_bytes=[&](u128 n,std::size_t resource) {
    const long double rate=hw.rates[resource]/du(hw,"die","core_count");
    if(!(rate>0))fail("invalid_hardware: Norm long per-Core byte rate is zero");
    return static_cast<long double>(as_u64(n,"Norm long serial bytes"))/rate;
  };
  const auto io=[&](u128 per_core,bool write) {
    Flow f;const u128 payload=mul128(active,per_core,"Norm long IO payload");
    add_hbm(f,hw,topo,payload,active,write,eta(hw,"eta_hbm_vector"));
    add_bytes_work(f,write ? SRAM_READ : SRAM_WRITE,payload,"Norm long IO SRAM");
    f.private_s=private_bytes(per_core,write ? SRAM_READ : SRAM_WRITE);return f;
  };
  const auto conversion=[&](u128 n,const std::string& dt) {
    return dt=="fp32" ? EqWork{} : weighted_ops(n,dt=="int4" ? perf(hw,"kappa_quant_eqops",4) : 1,"Norm long conversion");
  };
  const auto comp=[&](u128 sr,u128 sw,const EqWork& per_core_ops) {
    Flow f;add_bytes_work(f,SRAM_READ,mul128(active,sr,"Norm long Compute reads"),"Norm long Compute reads");
    add_bytes_work(f,SRAM_WRITE,mul128(active,sw,"Norm long Compute writes"),"Norm long Compute writes");
    add_bytes_work(f,LOCAL,mul128(active,add128(sr,sw,"Norm long Compute LOCAL"),"Norm long Compute LOCAL"),"Norm long Compute LOCAL");
    f.vec=per_core_ops.multiply(active);f.compute_s=compute_time(hw,0,per_core_ops,1);
    f.private_s=std::max({static_cast<long double>(f.compute_s),private_bytes(sr,SRAM_READ),
        private_bytes(sw,SRAM_WRITE),private_bytes(add128(sr,sw,"Norm long serial LOCAL"),LOCAL)});
    return f;
  };
  const u128 stat_converted=ad=="fp32" ? 0 : mul128(4,hidden,"Norm long statistics conversion");
  const u128 stat_sr=add128(bytes(hidden,ad),stat_converted,"Norm long statistics reads");
  const u128 stat_sw=add128(stat_converted,16,"Norm long retained statistics write");
  // Fixed 6/8 equivalent-op templates are partitioned 2+4 / 3+5 between
  // statistics and application; these remain engineering SFU assumptions.
  Flow statistics=comp(stat_sr,stat_sw,
      EqWork(mul128(layernorm ? 3 : 2,hidden,"Norm long statistics math"))+conversion(hidden,ad));
  const auto apply_bytes=[&](u64 n,bool first) {
    const u128 xc=ad=="fp32" ? 0 : mul128(4,n,"Norm long converted input");
    const u128 pc=mul128(parameters,mul128(4,n,"Norm long FP32 parameters"),"Norm long FP32 parameters");
    const u128 converted_p=pd=="fp32" ? 0 : pc;
    const u128 converted_y=od=="fp32" ? 0 : mul128(4,n,"Norm long output cast");
    u128 sr=add128(add128(bytes(n,ad),xc,"Norm long apply X"),
        add128(pc,converted_y,"Norm long parameter/output reads"),"Norm long apply reads");
    if(pd!="fp32")sr=add128(sr,param_bytes(n),"Norm long parameter conversion reads");
    if(first)sr=add128(sr,16,"Norm long retained statistics read");
    u128 sw=add128(add128(xc,converted_p,"Norm long conversion writes"),
        mul128(4,n,"Norm long FP32 output"),"Norm long apply writes");
    if(od!="fp32")sw=add128(sw,bytes(n,od),"Norm long cast output writes");
    return std::array<u128,2>{{sr,sw}};
  };
  const auto apply=[&](u64 n,bool first) {
    const auto memory=apply_bytes(n,first);
    auto f=comp(memory[0],memory[1],EqWork(mul128(layernorm ? 5 : 4,n,"Norm long apply math"))+
        conversion(n,ad)+conversion(mul128(parameters,n,"Norm long parameter elements"),pd)+conversion(n,od));
    return f;
  };
  Flow first_apply=apply(tile,true),last_apply=apply(tail,false),application=apply(hidden,true);
  // Serial tile bounds use the same per-tile maximum, including the one-time
  // SRAM statistics read. This is an analytic count, never a tile loop.
  application.private_s=first_apply.private_s+(tiles-2)*apply(tile,false).private_s+last_apply.private_s;
  const auto begin=plan.epochs.size(),description_begin=service_description_count(plan);
  const auto append=[&](const char* suffix,Flow l,Flow c,Flow s,
      std::array<double,2> initial,std::array<double,2> capacity) {
    Epoch e;e.name=std::string("norm_two_pass_single_cohort_v1")+suffix;
    e.credit_initial=initial;e.credit_capacity=capacity;
    for(auto item:{std::pair<Role,Flow*>{Role::Load,&l},{Role::Compute,&c},{Role::Store,&s}}) {
      Activity a;a.role=item.first;a.id=plan.epochs.size()*3+e.activities.size()+1;
      seal_activity(a,*item.second,hw,active);a.credit_curve={{0,{{0,0}}},{1,a.credit_delta}};
      if(std::any_of(a.work.begin(),a.work.end(),[](double v){return v>0;}))e.activities.push_back(std::move(a));
    }
    plan.epochs.push_back(std::move(e));
  };
  Flow first_stat_load=io(bytes(tile,ad),false),stat_load=io(bytes(hidden,ad),false);
  first_stat_load.credit={{tile,0}};
  append("_statistics_first_input",first_stat_load,{}, {},{{0,0}},{{2*tile,1}});
  Flow remaining_stat_load=subtract_io_flow(stat_load,first_stat_load);
  remaining_stat_load.credit={{static_cast<long double>(hidden-tile),0}};
  statistics.credit={{-static_cast<long double>(hidden),0}};
  append("_statistics_stream",remaining_stat_load,statistics,{},{{tile,0}},{{2*tile,1}});
  const auto statistics_activity=plan.epochs.back().activities.back().id;
  const auto apply_begin=plan.epochs.size(),apply_description_begin=service_description_count(plan);
  Flow load=io(add128(bytes(hidden,ad),param_bytes(hidden),"Norm long apply input"),false);
  Flow first_load=io(input_tile,false),second_load=first_load;
  Flow store=io(bytes(hidden,od),true),last_store=io(bytes(tail,od),true);
  Flow penultimate_store=io(bytes(tile,od),true);
  first_load.credit={{1,0}};
  append("_apply_first_input",first_load,{}, {},{{0,0}},{{2,1}});
  second_load.credit={{1,0}};first_apply.credit={{-1,0}};
  append("_apply_first_output_and_prefetch",second_load,first_apply,{},{{1,0}},{{2,1}});
  Flow remaining_load=subtract_io_flow(subtract_io_flow(load,first_load),second_load);
  Flow middle_apply=subtract_compute_flow(subtract_compute_flow(application,first_apply),last_apply);
  Flow early_store=subtract_io_flow(subtract_io_flow(store,last_store),penultimate_store);
  remaining_load.credit={{static_cast<long double>(tiles-3),0}};
  middle_apply.credit={{-static_cast<long double>(tiles-2),static_cast<long double>(tiles-3)}};
  early_store.credit={{0,-static_cast<long double>(tiles-2)}};
  append("_apply_middle_reserved_tail",remaining_load,middle_apply,early_store,{{1,1}},{{1,1}});
  last_apply.credit={{-1,0}};penultimate_store.credit={{0,-1}};
  append("_apply_last_compute_and_penultimate_store",{},last_apply,penultimate_store,{{1,1}},{{1,1}});
  last_store.credit={{0,-1}};
  append("_apply_last_store",{}, {},last_store,{{0,1}},{{1,1}});
  const auto descriptions=service_description_count(plan)-description_begin;
  if(descriptions>16)fail("mapping_invariant: Norm long description budget exceeded");
  plan.summary["service_curves"].push_back({{"model","analytical_cumulative_norm_long_statistics_v1"},
      {"epoch_begin",begin},{"epoch_count",2},{"description_count",3},{"description_budget",5},
      {"input_unit","one_input_element_at_each_consumer_core"},{"input_units",hidden},
      {"input_capacity_units",2*tile},{"statistics_ready_epoch",begin+1},
      {"statistics_activity_id",statistics_activity},{"statistics_retention","16_bytes_per_core_across_kernel_barrier"}});
  const auto first_apply_activity=plan.epochs.at(apply_begin+1).activities.back().id;
  plan.summary["service_curves"].push_back({{"model","analytical_cumulative_norm_long_apply_v1"},
      {"epoch_begin",apply_begin},{"epoch_count",5},
      {"description_count",service_description_count(plan)-apply_description_begin},{"description_budget",16},
      {"input_unit","one_256_element_block_at_each_consumer_core"},{"output_unit","one_output_block_at_each_consumer_core"},
      {"input_units",tiles},{"output_units",tiles},{"input_capacity_units",1},{"output_capacity_units",1},
      {"first_output_epoch",apply_begin+1},{"first_output_activity_id",first_apply_activity},
      {"first_output_release","first_apply_compute_terminal_after_complete_row_statistics"},
      {"first_output_input_bytes",as_u64(mul128(active,input_tile,"Norm long first apply input"),"Norm long first input")},
      {"first_output_compute_work_s",as_double(first_apply.compute_s,"Norm long first compute")},
      {"reserved_last_input_slots_per_core",1},{"reserved_penultimate_output_slots_per_core",1},
      {"reserved_input_terminal_transfer_units",1},{"reserved_input_terminal_transfer_epoch",apply_begin+3},
      {"readiness_approximation","bounded_prefetch_middle_affine_and_reserved_tail"}});
  const u128 x=bytes(hidden,ad),y=bytes(hidden,od),p=param_bytes(hidden);
  const u128 reads=add128(mul128(2,x,"Norm long two HBM input passes"),p,"Norm long reads per Core");
  const auto apply_memory=apply_bytes(hidden,true);
  m.hbm_read=add128(m.hbm_read,mul128(active,reads,"Norm long HBM reads"),"HBM reads");
  m.hbm_write=add128(m.hbm_write,mul128(active,y,"Norm long HBM writes"),"HBM writes");
  m.logical=add128(m.logical,mul128(active,add128(add128(x,y,"Norm long logical tensors"),p,"Norm long logical tensors"),"Norm long logical tensors"),"logical bytes");
  m.source_vops=add128(m.source_vops,mul128(active,mul128(layernorm ? 8 : 6,hidden,"Norm long math"),"Norm long math"),"source Vec ops");
  m.vecops=m.vecops+statistics.vec+application.vec;
  const u128 converted=add128(ad=="fp32" ? 0 : mul128(2,hidden,"Norm long input conversions"),
      add128(pd=="fp32" ? 0 : mul128(parameters,hidden,"Norm long parameter conversions"),
          od=="fp32" ? 0 : hidden,"Norm long conversions"),"Norm long conversions");
  m.conversion_elements=add128(m.conversion_elements,mul128(active,converted,"Norm long conversions"),"conversion elements");
  const u128 sr=add128(add128(stat_sr,apply_memory[0],"Norm long Compute reads"),y,"Norm long reads with Store");
  const u128 sw=add128(add128(stat_sw,apply_memory[1],"Norm long Compute writes"),reads,"Norm long writes with Load");
  m.sram_read=add128(m.sram_read,mul128(active,sr,"Norm long SRAM reads"),"SRAM reads");
  m.sram_write=add128(m.sram_write,mul128(active,sw,"Norm long SRAM writes"),"SRAM writes");
  m.local=add128(m.local,mul128(active,add128(add128(stat_sr,stat_sw,"Norm long stats LOCAL"),
      add128(apply_memory[0],apply_memory[1],"Norm long apply LOCAL"),"Norm long LOCAL"),"Norm long LOCAL"),"LOCAL bytes");
  m.sram_peak=std::max(m.sram_peak,peak);m.active_cores=active;m.template_id="norm_two_pass_single_cohort_v1";
  m.output_rounds=add128(m.output_rounds,tiles,"Norm long output rounds");
  m.waves=add128(m.waves,mul128(2,tiles,"Norm long pass tiles"),"Norm long waves");
  m.assumptions.push_back("one_row_per_core_two_hbm_passes_with_private_retained_statistics");
  m.assumptions.push_back("reference_norm_math_split_not_measured_sfu_cycles");
  plan.summary["norm_long_row_ledger"]={{"mapping","one_row_per_core; same_core_statistics_then_apply"},
      {"rows",rows},{"hidden",hidden},{"input_hbm_visits",2},{"parameter_hbm_loads",active},
      {"parameter_cache","streamed_once_no_repeated_row_reuse"},{"parameter_vectors",parameters},
      {"statistics_bytes_per_core",16},{"statistics_hbm_workspace_bytes",0},
      {"allocated_input_slots_per_core",2},{"allocated_output_slots_per_core",2},
      {"input_slot_bytes_per_core",as_u64(input_tile,"Norm long input slot")},
      {"output_slot_bytes_per_core",as_u64(bytes(tile,od),"Norm long output slot")},
      {"working_bytes_per_core",as_u64(working,"Norm long working buffers")},
      {"metadata_bytes_per_core",64},{"statistics_epoch",begin+1},{"apply_epoch_begin",apply_begin},
      {"scope","R1_and_rows_not_exceeding_core_count; multi_row_per_core_not_implemented"}};
  return true;
}
