#include "coarse_die/Capi.h"
#include "coarse_die/Execution.hh"
#include "coarse_die/Memory.hh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <list>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace {
using namespace coarse_die;
using json = nlohmann::json;
struct Model {
    Hardware hardware;
    struct CacheEntry { std::shared_ptr<const Plan> plan; std::size_t bytes; std::list<std::string>::iterator recency; };
    std::map<std::string, CacheEntry> cache;
    std::list<std::string> lru;
    std::size_t cache_bytes=0;
};
struct PlanRef { std::shared_ptr<Model> model; std::shared_ptr<const Plan> plan; };
struct RuntimeRef { std::shared_ptr<Model> model; std::unique_ptr<Runtime> world; };
struct ExecRef { DieHandle runtime; std::uint64_t local; };
std::map<DieHandle,std::shared_ptr<Model>> models;
std::map<DieHandle,PlanRef> plans;
std::map<DieHandle,RuntimeRef> runtimes;
std::map<DieHandle,ExecRef> executions;
DieHandle next_id=1;
std::recursive_mutex abi_mutex;
thread_local std::string error;
DieHandle id() {
    if(next_id==std::numeric_limits<DieHandle>::max()) throw std::overflow_error("overflow: handle space exhausted");
    return next_id++;
}
template<class M> auto& require(M& map, DieHandle handle) {
    auto i=map.find(handle);
    if(i==map.end()) throw std::runtime_error("stale_handle: unknown or released handle");
    return i->second;
}
json parse(const char* input) {
    if(!input) throw std::invalid_argument("invalid_input: null JSON");
    return json::parse(input);
}
int code_for(const std::string& m) {
    if(m.find("stale_handle")!=std::string::npos) return DIE_STALE_HANDLE;
    if(m.find("numerical_failure")!=std::string::npos) return DIE_NUMERICAL_FAILURE;
    if(m.find("unsupported_legacy")!=std::string::npos) return DIE_UNSUPPORTED_LEGACY_STATE;
    if(m.find("missing_topology")!=std::string::npos) return DIE_MISSING_TOPOLOGY;
    if(m.find("incompatible_fingerprint")!=std::string::npos) return DIE_INCOMPATIBLE_FINGERPRINT;
    if(m.find("schema")!=std::string::npos) return DIE_INCOMPATIBLE_SCHEMA;
    if(m.find("invalid_time")!=std::string::npos) return DIE_INVALID_TIME;
    if(m.find("overflow")!=std::string::npos) return DIE_OVERFLOW;
    if(m.find("busy")!=std::string::npos) return DIE_BUSY;
    if(m.find("unsupported_projection_state")!=std::string::npos) return DIE_UNSUPPORTED_PROJECTION_STATE;
    if(m.find("unsupported_workspace")!=std::string::npos) return DIE_UNSUPPORTED_WORKSPACE;
    if(m.find("unsupported_head_mapping")!=std::string::npos) return DIE_UNSUPPORTED_HEAD_MAPPING;
    if(m.find("unsupported_mapping")!=std::string::npos) return DIE_UNSUPPORTED_MAPPING;
    if(m.find("unsupported")!=std::string::npos && m.find("dtype")!=std::string::npos) return DIE_UNSUPPORTED_DTYPE;
    if(m.find("unsupported")!=std::string::npos && (m.find("op_kind")!=std::string::npos || m.find("unsupported_op")!=std::string::npos)) return DIE_UNSUPPORTED_OP;
    if(m.find("unsupported")!=std::string::npos) return DIE_UNSUPPORTED;
    if(m.find("hardware")!=std::string::npos || m.find("architecture.")==0 ||
       m.find("performance_profile.")==0 || m.find("search_profile.")==0 ||
       m.find("source_constants.")==0) return DIE_INVALID_HARDWARE;
    return DIE_INVALID_INPUT;
}
int fail(const std::exception& e) {
    int c=code_for(e.what());
    std::string reason=e.what();
    error=json{{"schema_version",2},{"status_code",c},{"reason",reason},
               {"field_path",reason.substr(0,reason.find(':'))}}.dump();
    return c;
}
template<class F> int guarded(F f) {
    std::lock_guard<std::recursive_mutex> lock(abi_mutex);
    try { f(); error.clear(); return DIE_OK; }
    catch(const std::exception& e) { return fail(e); }
    catch(...) { std::runtime_error e("numerical_failure: unknown C++ exception"); return fail(e); }
}
char* output(const json& j) {
    auto s=j.dump();
    auto p=static_cast<char*>(std::malloc(s.size()+1));
    if(!p) throw std::bad_alloc();
    std::memcpy(p,s.c_str(),s.size()+1); return p;
}
template<class F> const char* result(F f) {
    const char* out=nullptr;
    if(guarded([&]{out=output(f());})!=DIE_OK) return nullptr;
    return out;
}
Owner owner(const DieOwnerV2* p) {
    if(!p || p->struct_size<sizeof(DieOwnerV2) || p->schema_version!=2 || !p->generation)
        throw std::invalid_argument("incompatible_schema: DieOwnerV2 header/generation");
    return {p->request_id,p->node_id,p->execution_id,p->generation};
}
void out_handle(DieHandle* out) {
    if(!out) throw std::invalid_argument("invalid_input: null output handle");
    *out=0;
}
std::shared_ptr<const Plan> cached(Model& m, const json& task) {
    auto key=stable_hash(m.hardware.fingerprint+task.dump());
    auto i=m.cache.find(key);
    if(i!=m.cache.end()) { m.lru.splice(m.lru.begin(),m.lru,i->second.recency); return i->second.plan; }
    auto p=compile(m.hardware,task);
    std::size_t bytes=retained_plan_bytes(*p)+key.size()+512;
    if(bytes<=64u*1024u*1024u) {
        m.lru.push_front(key);
        m.cache.emplace(key,Model::CacheEntry{p,bytes,m.lru.begin()});
        m.cache_bytes+=bytes;
        while(m.cache.size()>4096 || m.cache_bytes>64u*1024u*1024u) {
            auto k=m.lru.back(); m.lru.pop_back();
            auto e=m.cache.find(k); m.cache_bytes-=e->second.bytes; m.cache.erase(e);
        }
    }
    return p;
}
json batch(DieHandle model,const char* in,bool retain) {
    auto m=require(models,model); auto b=parse(in);
    if(!b.contains("specs") || !b["specs"].is_array()) throw std::invalid_argument("invalid_input: specs must be an array");
    json rs=json::array(), es=json::array();
    // Each failure is isolated; successful siblings remain reviewable.
    for(const auto& task:b["specs"]) {
        try {
            auto p=cached(*m,task);
            if(retain) {
                auto h=id(); plans.emplace(h,PlanRef{m,p});
                rs.push_back({{"plan_handle",h},{"plan_hash",p->fingerprint},{"summary",p->summary}});
                if(has_fractional_work(*p)) {
                    rs.back()["plan_schema"]="coarse_die_plan_v3";
                    rs.back()["equivalent_work_schema"]="dyadic_equivalent_work_v1";
                }
            } else rs.push_back(estimate_alone(m->hardware,*p));
            es.push_back(nullptr);
        } catch(const std::exception& e) {
            std::string reason=e.what();
            rs.push_back(nullptr); es.push_back({{"status_code",code_for(reason)},{"reason",reason},
                {"field_path",reason.substr(0,reason.find(':'))}});
        }
    }
    return {{"schema_version",2},{"results",rs},{"errors",es},
            {"cache_entries",m->cache.size()},{"cache_accounted_bytes",m->cache_bytes},
            {"cache_accounting","retained_container_capacities_with_allocator_allowances"}};
}
}
extern "C" {
uint32_t die_coarse_abi_version(void) { return 2; }
const char* die_coarse_last_error_v2(void) { return error.c_str(); }
void die_coarse_free_result_v2(const char* p) { std::free(const_cast<char*>(p)); }
int die_coarse_create_model_v2(const char* in,DieHandle* out) {
    return guarded([&]{out_handle(out); auto m=std::make_shared<Model>(); m->hardware=resolve_hardware(parse(in));
        auto h=id(); models.emplace(h,std::move(m)); *out=h;});
}
const char* die_coarse_hardware_summary_v2(DieHandle h) {
    return result([&]{auto& m=require(models,h); return json{{"schema_version",2},{"input",m->hardware.input},
        {"derived_hardware",m->hardware.derived},{"hardware_fingerprint",m->hardware.fingerprint}};});
}
const char* die_coarse_compile_batch_v2(DieHandle m,const char* s) { return result([&]{return batch(m,s,true);}); }
const char* die_coarse_estimate_batch_v2(DieHandle m,const char* s) { return result([&]{return batch(m,s,false);}); }
int die_coarse_create_runtime_v2(DieHandle m,DieHandle* out) {
    return guarded([&]{out_handle(out); auto model=require(models,m); auto w=std::make_unique<Runtime>(model->hardware);
        auto h=id(); runtimes.emplace(h,RuntimeRef{model,std::move(w)}); *out=h;});
}
int die_coarse_submit_plan_v2(DieHandle r,DieHandle p,const DieOwnerV2* o,uint64_t now,DieHandle* out) {
    return guarded([&]{out_handle(out); auto& w=require(runtimes,r); auto& plan=require(plans,p); auto ow=owner(o);
        if(w.model->hardware.fingerprint!=plan.model->hardware.fingerprint)
            throw std::runtime_error("incompatible_fingerprint: plan/runtime hardware");
        auto h=id(); auto local=w.world->submit(plan.plan,ow,now);
        executions.emplace(h,ExecRef{r,local}); *out=h;});
}
int die_coarse_submit_endpoint_v2(DieHandle r,const char* spec,const DieOwnerV2* o,uint64_t now,DieHandle* out) {
    return guarded([&]{out_handle(out); auto& w=require(runtimes,r); auto task=parse(spec); auto ow=owner(o);
        auto h=id(); auto local=w.world->submit_endpoint(task,ow,now);
        executions.emplace(h,ExecRef{r,local}); *out=h;});
}
int die_coarse_advance_to_v2(DieHandle r,uint64_t now) { return guarded([&]{require(runtimes,r).world->advance_to(now);}); }
const char* die_coarse_next_event_v2(DieHandle r) {
    return result([&]{auto n=require(runtimes,r).world->next_event(); return json{{"next_event_ns",n?json(*n):json(nullptr)}};});
}
int die_coarse_cancel_execution_v2(DieHandle h,uint64_t now) {
    return guarded([&]{auto& e=require(executions,h); require(runtimes,e.runtime).world->cancel(e.local,now);});
}
int die_coarse_update_capacity_v2(DieHandle r,uint32_t resource,double rate,uint64_t now) {
    return guarded([&]{require(runtimes,r).world->update_capacity(resource,rate,now);});
}
const char* die_coarse_query_progress_v2(DieHandle h) {
    return result([&]{auto& e=require(executions,h); return require(runtimes,e.runtime).world->query_progress(e.local);});
}
const char* die_coarse_query_counters_v2(DieHandle r) {
    return result([&]{auto& w=require(runtimes,r); auto c=w.world->counters();
        c["cache_entries"]=w.model->cache.size(); c["cache_accounted_bytes"]=w.model->cache_bytes; return c;});
}
const char* die_coarse_list_executions_v2(DieHandle r) {
    return result([&]{auto& w=require(runtimes,r); json rows=json::array();
        for(auto local:w.world->execution_handles()) {
            DieHandle handle=0;
            for(const auto& e:executions) if(e.second.runtime==r && e.second.local==local) {handle=e.first; break;}
            if(!handle) {handle=id(); executions.emplace(handle,ExecRef{r,local});}
            rows.push_back({{"execution_handle",handle},{"local_handle",local},{"progress",w.world->query_progress(local)}});
        }
        return json{{"schema_version",2},{"executions",rows}};});
}
const char* die_coarse_export_snapshot_v2(DieHandle r) {
    return result([&]{return require(runtimes,r).world->export_snapshot();});
}
int die_coarse_import_snapshot_v2(DieHandle m,const char* in,DieHandle* out) {
    return guarded([&]{out_handle(out); auto model=require(models,m); auto snapshot=parse(in);
        auto w=Runtime::import_snapshot(model->hardware,snapshot); auto h=id();
        runtimes.emplace(h,RuntimeRef{model,std::move(w)}); *out=h;});
}
int die_coarse_fork_runtime_v2(DieHandle r,DieHandle* out) {
    return guarded([&]{out_handle(out); auto& base=require(runtimes,r);
        auto w=Runtime::import_snapshot(base.model->hardware,base.world->export_snapshot());
        auto h=id(); runtimes.emplace(h,RuntimeRef{base.model,std::move(w)}); *out=h;});
}
const char* die_coarse_estimate_on_snapshot_v2(DieHandle r,DieHandle p,const DieOwnerV2* o,const char* overlay) {
    return result([&]{auto& base=require(runtimes,r); auto& pr=require(plans,p); auto ow=owner(o);
        if(base.model->hardware.fingerprint!=pr.model->hardware.fingerprint)
            throw std::runtime_error("incompatible_fingerprint: projection plan/runtime hardware");
        auto w=Runtime::import_snapshot(base.model->hardware,base.world->export_snapshot());
        auto candidate=w->submit(pr.plan,ow,w->now_ns());
        json events=overlay?parse(overlay):json::array();
        if(!events.is_array()) throw std::invalid_argument("invalid_input: known_overlay array required");
        auto tick=w->now_ns(); std::vector<std::pair<uint64_t,uint64_t>> seq;
        for(std::size_t i=0;i<events.size();++i) {
            auto t=events[i].at("event_tick").get<uint64_t>();
            if(t<tick) throw std::runtime_error("invalid_time: overlay event in past");
            seq.emplace_back(t,i);
        }
        std::stable_sort(seq.begin(),seq.end(),[&](const auto& a,const auto& b){
            const auto& x=events[a.second]; const auto& y=events[b.second];
            return std::make_tuple(a.first,x.value("priority",0),x.value("producer",uint64_t(0)),x.value("sequence",a.second),
                       x.value("kind",std::string()),x.value("owner",json::object()).dump())
                 < std::make_tuple(b.first,y.value("priority",0),y.value("producer",uint64_t(0)),y.value("sequence",b.second),
                       y.value("kind",std::string()),y.value("owner",json::object()).dump());
        });
        for(auto ti:seq) {
            const auto& e=events[ti.second]; w->advance_to(ti.first);
            std::string k=e.at("kind");
            if(k=="capacity") w->update_capacity(e.at("resource"),e.at("rate"),ti.first);
            else if(k=="cancel") {
                uint64_t local=0;
                if(e.contains("local_handle")) local=e.at("local_handle");
                else {
                    const auto& target=require(executions,e.at("execution_handle").get<DieHandle>());
                    if(target.runtime!=r) throw std::runtime_error("stale_handle: overlay cancel belongs to another world");
                    local=target.local;
                }
                w->cancel(local,ti.first);
            }
            else if(k=="endpoint") {
                const auto& v=e.at("owner"); Owner eo{v.value("request_id",uint64_t(0)),v.at("node_id"),v.at("execution_id"),v.at("generation")};
                w->submit_endpoint(e.at("endpoint"),eo,ti.first);
            } else if(k=="plan") {
                const auto& v=e.at("owner"); Owner eo{v.value("request_id",uint64_t(0)),v.at("node_id"),v.at("execution_id"),v.at("generation")};
                w->submit(compile(base.model->hardware,e.at("task")),eo,ti.first);
            } else throw std::runtime_error("unsupported_projection_state: external network/pool overlay requires adapter clone");
        }
        std::size_t guard=0;
        while(!w->drained()) {
            if(++guard>100000) throw std::runtime_error("numerical_failure: projection event budget");
            auto n=w->next_event(); if(!n) throw std::runtime_error("numerical_failure: no event for active world");
            w->advance_to(*n);
        }
        return json{{"schema_version",2},{"candidate",w->query_progress(candidate)},
            {"counters",w->counters()},{"projection_scope","known_local_world"},
            {"unknown_future_included",false}};});
}
int die_coarse_release_plan_v2(DieHandle p) { return guarded([&]{require(plans,p); plans.erase(p);}); }
int die_coarse_destroy_execution_v2(DieHandle h) {
    return guarded([&]{auto e=require(executions,h); require(runtimes,e.runtime).world->destroy_execution(e.local); executions.erase(h);});
}
int die_coarse_destroy_runtime_v2(DieHandle r) {
    return guarded([&]{auto& w=require(runtimes,r); if(!w.world->drained()) throw std::runtime_error("busy: runtime has active executions");
        for(auto i=executions.begin();i!=executions.end();) { if(i->second.runtime==r) i=executions.erase(i); else ++i; }
        runtimes.erase(r);});
}
int die_coarse_destroy_model_v2(DieHandle h) {
    return guarded([&]{auto& m=require(models,h);
        for(const auto& x:plans) if(x.second.model==m) throw std::runtime_error("busy: model has live plan handles");
        for(const auto& x:runtimes) if(x.second.model==m) throw std::runtime_error("busy: model has live runtimes");
        models.erase(h);});
}
}
