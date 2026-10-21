#include "coarse_die/Capi.h"
#include "coarse_die/PlanCache.hh"
#include "coarse_die/Execution.hh"
#include "coarse_die/Memory.hh"
#include "CompilerInternal.hh"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {
using namespace coarse_die;
using json = nlohmann::json;
struct Model {
    Hardware hardware;
    struct ResponseTemplate {
        std::string plan_hash_json;
        std::string summary_json;
        bool fractional = false;
        bool curves = false;
    };
    struct CacheEntry {
        std::shared_ptr<const Plan> plan;
        std::shared_ptr<const ResponseTemplate> response;
        std::size_t bytes;
        std::list<std::string>::iterator recency;
    };
    struct BatchMemo {
        std::string raw_request;
        std::vector<CacheEntry*> entries;
        std::size_t bytes=0;
    };
    std::map<std::string, CacheEntry> cache;
    std::list<std::string> lru;
    // Two recent verified raw batches bypass JSON parsing and normalization.
    // Pointers refer only to current map nodes; cache structure changes clear it.
    std::array<std::unique_ptr<BatchMemo>,2> batch_memos;
    std::size_t cache_bytes=0;
};
struct CachedPlan {
    std::shared_ptr<const Plan> plan;
    std::shared_ptr<const Model::ResponseTemplate> response;
    std::string key;
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
    auto field=m.substr(0,m.find(':'));
    if(field.find("input.")==0)field.erase(0,6);
    const auto hardware_field=[&](const char* prefix) {
        const std::string name=prefix;
        return field==name || field.find(name+".")==0 || field.find(name+"[")==0;
    };
    if(m.find("hardware")!=std::string::npos || hardware_field("architecture") ||
       hardware_field("performance_profile") || hardware_field("search_profile") ||
       hardware_field("validation_constraints") || hardware_field("source_constants") ||
       hardware_field("source_revision") || hardware_field("core_resource_table"))
        return DIE_INVALID_HARDWARE;
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
char* output(std::string_view s) {
    if(s.size()==std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("overflow: JSON result size");
    auto p=static_cast<char*>(std::malloc(s.size()+1));
    if(!p) throw std::bad_alloc();
    if(!s.empty()) std::memcpy(p,s.data(),s.size());
    p[s.size()]='\0';
    return p;
}
char* output(const json& j) {
    const auto s=j.dump();
    return output(std::string_view(s));
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
void add_cache_bytes(std::size_t& total, std::size_t bytes) {
    if(bytes>std::numeric_limits<std::size_t>::max()-total)
        throw std::overflow_error("overflow: cache retained bytes");
    total+=bytes;
}
void add_response_bytes(std::size_t& total, std::size_t bytes) {
    if(bytes>std::numeric_limits<std::size_t>::max()-total)
        throw std::overflow_error("overflow: C ABI batch response size");
    total+=bytes;
}
std::size_t retained_string_capacity(const std::string& value) {
    std::size_t result=value.capacity();
    add_cache_bytes(result,1);
    return result;
}
std::uint64_t estimated_latency_ns(const Plan& plan) {
    const auto latency=plan.summary.find("estimated_latency_ns");
    if(latency==plan.summary.end())
        throw std::runtime_error("numerical_failure: compiled Plan lacks estimated_latency_ns");
    if(latency->is_number_unsigned())
        return latency->get<std::uint64_t>();
    if(latency->is_number_integer()) {
        const auto signed_latency=latency->get<std::int64_t>();
        if(signed_latency<0)
            throw std::runtime_error("numerical_failure: compiled Plan has negative estimated_latency_ns");
        return static_cast<std::uint64_t>(signed_latency);
    }
    throw std::runtime_error("numerical_failure: compiled Plan has non-integer estimated_latency_ns");
}
const std::string& cost_status(const Plan& plan) {
    const auto status=plan.summary.find("alone_estimate_status");
    if(status==plan.summary.end() || !status->is_string())
        throw std::runtime_error("numerical_failure: compiled Plan lacks alone_estimate_status");
    return status->get_ref<const std::string&>();
}
std::shared_ptr<const Model::ResponseTemplate> make_response_template(const Plan& plan) {
    auto response=std::make_shared<Model::ResponseTemplate>();
    response->plan_hash_json=json(plan.fingerprint).dump();
    response->summary_json=plan.summary.dump();
    (void)estimated_latency_ns(plan);
    (void)cost_status(plan);
    response->fractional=has_fractional_work(plan);
    response->curves=has_credit_curves(plan);
    return response;
}
std::size_t retained_response_template_bytes(const Model::ResponseTemplate& response) {
    // Include the retained object/control-block allowance and every owned
    // string's allocation capacity, including the NUL terminator.
    std::size_t bytes=sizeof(Model::ResponseTemplate)+64;
    add_cache_bytes(bytes,retained_string_capacity(response.plan_hash_json));
    add_cache_bytes(bytes,retained_string_capacity(response.summary_json));
    return bytes;
}
std::size_t retained_cache_entry_bytes(const Plan& plan, const std::string& key,
                                       const Model::ResponseTemplate& response) {
    std::size_t bytes=retained_plan_bytes(plan);
    add_cache_bytes(bytes,key.size());
    add_cache_bytes(bytes,sizeof(Model::CacheEntry)+512);
    add_cache_bytes(bytes,retained_response_template_bytes(response));
    return bytes;
}
void discard_batch_memo(Model& model,std::size_t index) {
    auto& memo=model.batch_memos[index];
    if(!memo) return;
    model.cache_bytes-=memo->bytes;
    memo.reset();
}
void invalidate_batch_memos(Model& model) {
    for(std::size_t i=0;i<model.batch_memos.size();++i) discard_batch_memo(model,i);
}
std::size_t retained_batch_memo_bytes(const Model::BatchMemo& memo) {
    // Account the owned memo object and the capacities of both retained buffers,
    // including allocator allowances for the string and pointer vector.
    std::size_t bytes=sizeof(Model::BatchMemo)+64;
    add_cache_bytes(bytes,retained_string_capacity(memo.raw_request));
    add_cache_bytes(bytes,memo.entries.capacity()*sizeof(Model::CacheEntry*));
    add_cache_bytes(bytes,64);
    return bytes;
}
constexpr std::size_t kBatchMemoMaxRawBytes=64u*1024u;
constexpr std::size_t kBatchMemoMaxRows=256;
constexpr std::size_t kCacheByteLimit=64u*1024u*1024u;
constexpr std::size_t kCacheEntryLimit=4096;
const Model::BatchMemo* matching_batch_memo(Model& model,const char* input) {
    if(!input) return nullptr;
    const std::string_view request(input);
    for(std::size_t i=0;i<model.batch_memos.size();++i) {
        const auto& memo=model.batch_memos[i];
        if(memo && std::string_view(memo->raw_request)==request) {
            if(i) std::swap(model.batch_memos[0],model.batch_memos[i]);
            return model.batch_memos[0].get();
        }
    }
    return nullptr;
}
void install_batch_memo(Model& model,const char* input,
                        const std::vector<std::string>& keys) {
    // Empty batches are lightweight cache diagnostics. Retaining a raw memo
    // for them adds no reusable Plan and would mutate an otherwise empty
    // cache merely by inspecting it.
    if(!input || keys.empty() || keys.size()>kBatchMemoMaxRows) return;
    const std::size_t raw_size=std::char_traits<char>::length(input);
    if(raw_size>kBatchMemoMaxRawBytes) return;

    auto memo=std::make_unique<Model::BatchMemo>();
    memo->raw_request.assign(input,raw_size);
    memo->entries.reserve(keys.size());
    for(const auto& key:keys) {
        const auto entry=model.cache.find(key);
        if(entry==model.cache.end()) return;
        memo->entries.push_back(&entry->second);
    }
    memo->bytes=retained_batch_memo_bytes(*memo);

    // Each memo shares the existing retained-item and byte limits. Keep Plan
    // entries intact: only evict older memos to make room for this shortcut.
    if(model.cache.size()>=kCacheEntryLimit) return;
    std::size_t retained_memos=0;
    for(const auto& old:model.batch_memos) if(old) add_cache_bytes(retained_memos,old->bytes);
    const auto plan_bytes=model.cache_bytes-retained_memos;
    if(memo->bytes>kCacheByteLimit-plan_bytes) return;
    discard_batch_memo(model,1);
    if(model.cache.size()+2>kCacheEntryLimit || memo->bytes>kCacheByteLimit-model.cache_bytes)
        discard_batch_memo(model,0);
    add_cache_bytes(model.cache_bytes,memo->bytes);
    model.batch_memos[1]=std::move(model.batch_memos[0]);
    model.batch_memos[0]=std::move(memo);
}

CachedPlan cached(Model& m, const json& task) {
    const auto normalized=detail::normalize_task(m.hardware,task);
    const std::string hardware_identity=m.hardware.fingerprint.empty()
        ? stable_hash(m.hardware.input.dump()) : m.hardware.fingerprint;
    // One public definition of "same compile input up to tracing
    // identities": the key text comes from semantic_task_key() so this C ABI
    // cache and host-side compile caches share a single normalization
    // contract instead of two spelling-tolerant key schemes.
    auto key=stable_hash(hardware_identity+"|coarse_die_compile_cache_v1|"
                         +semantic_task_key(m.hardware,task));
    auto i=m.cache.find(key);
    if(i!=m.cache.end()) {
        if(i->second.plan->task!=normalized)
            throw std::runtime_error("numerical_failure: normalized compile cache key collision");
        m.lru.splice(m.lru.begin(),m.lru,i->second.recency);
        return {i->second.plan,i->second.response,std::move(key)};
    }
    // Opt-in disk persistence (COARSE_DIE_PLAN_CACHE_DIR): consult and
    // repair the on-disk plan cache only when the variable is set; with it
    // unset this function stays byte-identical to the memory-only path and
    // never touches the filesystem.
    const char* plan_cache_dir=std::getenv("COARSE_DIE_PLAN_CACHE_DIR");
    bool disk_enabled=plan_cache_dir && *plan_cache_dir;
    std::shared_ptr<const Plan> p;
    if(disk_enabled)
        p=try_load_disk_plan(m.hardware,task);
    if(!p) {
        p=detail::compile_normalized(m.hardware,normalized);
        if(disk_enabled) store_disk_plan_for(m.hardware,task,*p);
    }
    auto response=make_response_template(*p);
    const std::size_t bytes=retained_cache_entry_bytes(*p,key,*response);
    if(bytes<=kCacheByteLimit) {
        // A map insertion can invalidate cached node references through later
        // eviction, so discard the shortcut before changing cache structure.
        invalidate_batch_memos(m);
        m.lru.push_front(key);
        const auto recency=m.lru.begin();
        std::map<std::string,Model::CacheEntry>::iterator inserted;
        try {
            auto result=m.cache.emplace(key,Model::CacheEntry{p,response,bytes,recency});
            if(!result.second) {
                m.lru.erase(recency);
                m.lru.splice(m.lru.begin(),m.lru,result.first->second.recency);
                return {result.first->second.plan,result.first->second.response};
            }
            inserted=result.first;
        } catch(...) {
            m.lru.erase(recency);
            throw;
        }
        try {
            add_cache_bytes(m.cache_bytes,bytes);
        } catch(...) {
            m.cache.erase(inserted);
            m.lru.erase(recency);
            throw;
        }
        while(m.cache.size()>kCacheEntryLimit || m.cache_bytes>kCacheByteLimit) {
            auto victim=std::prev(m.lru.end());
            auto e=m.cache.find(*victim);
            if(e!=m.cache.end()) {
                m.cache_bytes-=e->second.bytes;
                m.cache.erase(e);
            }
            m.lru.erase(victim);
        }
    }
    return {std::move(p),std::move(response),std::move(key)};
}
struct ResolvedBatchRow {
    bool success=false;
    std::shared_ptr<const Plan> plan;
    std::shared_ptr<const Model::ResponseTemplate> response;
    std::string cache_key;
    int status_code=0;
    std::string reason;
};
std::vector<ResolvedBatchRow> resolve_batch(Model& model,const char* input) {
    if(const auto* memo=matching_batch_memo(model,input)) {
        std::vector<ResolvedBatchRow> rows;
        rows.reserve(memo->entries.size());
        for(const auto* entry:memo->entries) {
            model.lru.splice(model.lru.begin(),model.lru,entry->recency);
            rows.push_back({true,entry->plan,entry->response,{},0,{}});
        }
        return rows;
    }

    const auto batch=parse(input);
    if(!batch.contains("specs") || !batch["specs"].is_array())
        throw std::invalid_argument("invalid_input: specs must be an array");
    const auto& specs=batch["specs"];
    std::vector<ResolvedBatchRow> rows;
    rows.reserve(specs.size());
    std::vector<std::string> keys;
    keys.reserve(specs.size());
    bool all_success=true;
    for(const auto& task:specs) {
        ResolvedBatchRow row;
        try {
            auto value=cached(model,task);
            row.success=true;
            row.plan=std::move(value.plan);
            row.response=std::move(value.response);
            row.cache_key=std::move(value.key);
            keys.push_back(row.cache_key);
        } catch(const std::exception& e) {
            row.reason=e.what();
            row.status_code=code_for(row.reason);
            all_success=false;
        }
        rows.push_back(std::move(row));
    }
    if(all_success) install_batch_memo(model,input,keys);
    return rows;
}
json estimate_batch(DieHandle model,const char* in) {
    auto m=require(models,model);
    auto rows=resolve_batch(*m,in);
    json rs=json::array(), es=json::array();
    // Each failure is isolated; successful siblings remain reviewable.
    for(const auto& row:rows) {
        if(row.success) {
            try {
                rs.push_back(estimate_alone(m->hardware,*row.plan));
                es.push_back(nullptr);
            } catch(const std::exception& e) {
                std::string reason=e.what();
                rs.push_back(nullptr); es.push_back({{"status_code",code_for(reason)},{"reason",reason},
                    {"field_path",reason.substr(0,reason.find(':'))}});
            }
        } else {
            rs.push_back(nullptr); es.push_back({{"status_code",row.status_code},{"reason",row.reason},
                {"field_path",row.reason.substr(0,row.reason.find(':'))}});
        }
    }
    return {{"schema_version",2},{"results",rs},{"errors",es},
            {"cache_entries",m->cache.size()},{"cache_accounted_bytes",m->cache_bytes},
            {"cache_accounting","retained_container_capacities_with_allocator_allowances"}};
}
json estimate_cost_batch(DieHandle model,const char* in) {
    auto m=require(models,model);
    auto rows=resolve_batch(*m,in);
    json rs=json::array(), es=json::array();
    // The compiler already ran its isolated Runtime trials to rank candidates.
    // Return that trusted selected cost without starting another Runtime or
    // constructing estimate_alone's full diagnostic report.
    for(const auto& row:rows) {
        if(row.success) {
            try {
                rs.push_back({{"estimated_latency_ns",estimated_latency_ns(*row.plan)},
                              {"plan_hash",row.plan->fingerprint},
                              {"status",cost_status(*row.plan)}});
                es.push_back(nullptr);
            } catch(const std::exception& e) {
                std::string reason=e.what();
                rs.push_back(nullptr); es.push_back({{"status_code",code_for(reason)},
                    {"reason",reason},{"field_path",reason.substr(0,reason.find(':'))}});
            }
        } else {
            rs.push_back(nullptr); es.push_back({{"status_code",row.status_code},
                {"reason",row.reason},{"field_path",row.reason.substr(0,row.reason.find(':'))}});
        }
    }
    return {{"schema_version",2},{"results",rs},{"errors",es},
            {"cache_entries",m->cache.size()},{"cache_accounted_bytes",m->cache_bytes},
            {"cache_accounting","retained_container_capacities_with_allocator_allowances"}};
}
struct CompileBatchRow {
    bool success=false;
    std::shared_ptr<const Plan> plan;
    std::shared_ptr<const Model::ResponseTemplate> response;
    DieHandle plan_handle=0;
    int status_code=0;
    std::string reason;
};
void append_json_string(std::string& out, const std::string& value) {
    out+=json(value).dump();
}
std::string compile_batch_text(Model& model, const char* input,
                               std::vector<CompileBatchRow>& rows) {
    auto resolved=resolve_batch(model,input);
    rows.reserve(resolved.size());
    for(auto& item:resolved) {
        CompileBatchRow row;
        if(item.success) {
          try {
            row.plan_handle=id();
            row.plan=std::move(item.plan);
            row.response=std::move(item.response);
            row.success=true;
          } catch(const std::exception& e) {
            row.reason=e.what();
            row.status_code=code_for(row.reason);
          }
        } else {
            row.reason=std::move(item.reason);
            row.status_code=item.status_code;
        }
        rows.push_back(std::move(row));
    }

    std::size_t reserve=256;
    for(const auto& row:rows) {
        add_response_bytes(reserve,160);
        if(row.success) {
            add_response_bytes(reserve,row.response->summary_json.size());
            add_response_bytes(reserve,row.response->plan_hash_json.size());
        } else {
            if(row.reason.size()>std::numeric_limits<std::size_t>::max()/6)
                throw std::overflow_error("overflow: C ABI batch response size");
            add_response_bytes(reserve,row.reason.size()*6);
        }
    }
    std::string out;
    out.reserve(reserve);
    out+="{\"cache_accounted_bytes\":";
    out+=std::to_string(model.cache_bytes);
    out+=",\"cache_accounting\":\"retained_container_capacities_with_allocator_allowances\",\"cache_entries\":";
    out+=std::to_string(model.cache.size());
    out+=",\"errors\":[";
    for(std::size_t i=0;i<rows.size();++i) {
        if(i) out.push_back(',');
        const auto& row=rows[i];
        if(row.success) {
            out+="null";
        } else {
            const auto colon=row.reason.find(':');
            const std::string field_path=row.reason.substr(0,colon);
            out+="{\"field_path\":";
            append_json_string(out,field_path);
            out+=",\"reason\":";
            append_json_string(out,row.reason);
            out+=",\"status_code\":";
            out+=std::to_string(row.status_code);
            out.push_back('}');
        }
    }
    out.push_back(']');
    out+=",\"results\":[";
    for(std::size_t i=0;i<rows.size();++i) {
        if(i) out.push_back(',');
        const auto& row=rows[i];
        if(!row.success) {
            out+="null";
            continue;
        }
        const auto& response=*row.response;
        out.push_back('{');
        if(response.fractional)
            out+="\"equivalent_work_schema\":\"dyadic_equivalent_work_v1\",";
        out+="\"plan_handle\":";
        out+=std::to_string(row.plan_handle);
        out+=",\"plan_hash\":";
        out+=response.plan_hash_json;
        if(response.curves)
            out+=",\"plan_schema\":\"coarse_die_plan_v4\"";
        else if(response.fractional)
            out+=",\"plan_schema\":\"coarse_die_plan_v3\"";
        out+=",\"summary\":";
        out+=response.summary_json;
        out.push_back('}');
    }
    out+= "],\"schema_version\":2}";
    return out;
}
const char* compile_batch_result(DieHandle model_handle,const char* input) {
    char* returned=nullptr;
    const int status=guarded([&] {
        auto model=require(models,model_handle);
        std::vector<CompileBatchRow> rows;
        auto text=compile_batch_text(*model,input,rows);
        std::unique_ptr<char,decltype(&std::free)> buffer(output(std::string_view(text)),&std::free);
        std::vector<DieHandle> inserted;
        inserted.reserve(rows.size());
        try {
            for(const auto& row:rows) {
                if(!row.success) continue;
                const auto result=plans.emplace(row.plan_handle,PlanRef{model,row.plan});
                if(!result.second)
                    throw std::logic_error("numerical_failure: duplicate plan handle");
                inserted.push_back(row.plan_handle);
            }
        } catch(...) {
            for(const auto handle:inserted) plans.erase(handle);
            throw;
        }
        returned=buffer.release();
    });
    return status==DIE_OK?returned:nullptr;
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
const char* die_coarse_workspace_envelope_v2(DieHandle h,const char* input) {
    return result([&]{auto& m=require(models,h); const auto task=parse(input);
        const auto envelope=compile_workspace_envelope(m->hardware,task);
        return json{{"schema_version",2},{"workspace_envelope_schema_version",1},
            {"candidate_count",envelope.candidate_count},
            {"max_hbm_workspace_peak_bytes",envelope.max_hbm_workspace_peak_bytes},
            {"hardware_fingerprint",envelope.hardware_fingerprint},
            {"normalized_task_fingerprint",envelope.normalized_task_fingerprint},
            {"op_kind",envelope.op_kind},{"phase",envelope.phase},
            {"normalized_task",envelope.normalized_task}};});
}
const char* die_coarse_workspace_capacity_bound_v2(DieHandle h,const char* input) {
    return result([&]{auto& m=require(models,h); const auto task=parse(input);
        const auto bound=compile_workspace_capacity_bound(m->hardware,task);
        return json{{"schema_version",2},
            {"workspace_capacity_bound_schema_version",1},
            {"capacity_model_revision","workspace_lifetimes_v2"},
            {"capacity_bound_kind","all_template_lifetimes_without_sram_filter"},
            {"hbm_workspace_upper_bound_bytes",bound.hbm_workspace_upper_bound_bytes},
            {"hardware_fingerprint",bound.hardware_fingerprint},
            {"normalized_task_fingerprint",bound.normalized_task_fingerprint},
            {"op_kind",bound.op_kind},{"phase",bound.phase},
            {"normalized_task",bound.normalized_task}};});
}
const char* die_coarse_compile_batch_v2(DieHandle m,const char* s) { return compile_batch_result(m,s); }
const char* die_coarse_estimate_batch_v2(DieHandle m,const char* s) { return result([&]{return estimate_batch(m,s);}); }
const char* die_coarse_estimate_cost_batch_v2(DieHandle m,const char* s) { return result([&]{return estimate_cost_batch(m,s);}); }
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

