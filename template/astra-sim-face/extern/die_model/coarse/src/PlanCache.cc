
// Opt-in disk plan cache implementation (roofline stage 3, WP4). The load/
// store internals and the host-facing hooks live in the static coarse_die
// library so host compile caches can link them without the C ABI shared
// object; Capi.cc's cached() uses the same hooks.
#include "coarse_die/PlanCache.hh"
#include "CompilerInternal.hh"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace coarse_die;
using json = nlohmann::json;

// Opt-in disk-persistent plan cache. COARSE_DIE_PLAN_CACHE_DIR names a
// directory whose compiled-plan entries survive process exits so repeated
// experiments reuse Plans across processes. Unset or empty keeps this C ABI
// byte-identical to the memory-only behavior and performs no filesystem
// access at all. A disk hit still reads the file, parses the canonical
// payload, allocates the Plan, and verifies the stable_hash fingerprint plus
// the envelope version; it is a load/verify shortcut, not "zero compile".
constexpr int kPlanCacheSchemaVersion = 1;
constexpr const char* kPlanCacheFormat = "coarse_die_plan_cache_v1";
constexpr std::uintmax_t kPlanCacheMaxFileBytes = 32u*1024u*1024u;
std::string plan_cache_source_identity() {
    // Deterministic compile-time identity of the translation unit that owns
    // this cache format: __DATE__/__TIME__ pin the exact build and
    // __VERSION__ separates toolchains, so the same binary shares entries
    // across processes while any rebuild or toolchain change invalidates
    // fail-closed. Deliberate envelope changes additionally bump
    // kPlanCacheSchemaVersion. Residual limit: a partial rebuild that changes
    // compiler semantics without recompiling this file keeps the old stamp;
    // host workflows rebuild the vendored library as a unit after sync, and
    // payload integrity is still enforced by plan_from_json's canonical
    // fingerprint recomputation.
    return std::string(__DATE__)+" "+__TIME__+" | "+__VERSION__;
}
bool plan_cache_safe_key(const std::string& key) {
    // stable_hash() emits 64 lowercase hex characters; anything else would
    // mean an unexpected hash format, so refuse to build a path from it.
    if(key.size()!=64) return false;
    for(const char c:key)
        if(!((c>='0'&&c<='9')||(c>='a'&&c<='f'))) return false;
    return true;
}
std::shared_ptr<const Plan> load_disk_plan(const std::string& dir,const std::string& key,
                                           const std::string& hardware_identity,
                                           const json& normalized) {
    const std::string path=dir+"/"+key+".json";
    struct stat st{};
    if(::stat(path.c_str(),&st)!=0 || st.st_size<=0 ||
       static_cast<std::uintmax_t>(st.st_size)>kPlanCacheMaxFileBytes)
        return nullptr;
    std::ifstream in(path,std::ios::binary);
    if(!in) return nullptr;
    std::string text(static_cast<std::size_t>(st.st_size),'\0');
    if(!in.read(text.data(),static_cast<std::streamsize>(st.st_size))) return nullptr;
    std::shared_ptr<const Plan> plan;
    try {
        const json envelope=json::parse(text);
        const auto& schema=envelope.at("cache_schema_version");
        const auto& format=envelope.at("cache_format");
        const auto& identity=envelope.at("source_identity");
        if(!schema.is_number_integer() ||
           schema.get<int>()!=kPlanCacheSchemaVersion ||
           !format.is_string() || kPlanCacheFormat!=format.get_ref<const std::string&>() ||
           !identity.is_string() ||
           identity.get_ref<const std::string&>()!=plan_cache_source_identity())
            throw std::runtime_error("incompatible_schema: plan cache envelope version mismatch");
        plan=plan_from_json(envelope.at("plan"));
        if(plan->task!=normalized)
            throw std::runtime_error("numerical_failure: disk plan cache key collision");
        if(plan->hardware_fingerprint!=hardware_identity)
            throw std::runtime_error("incompatible_fingerprint: disk plan cache hardware");
    } catch(...) {
        // Fail closed on corrupt, foreign-version, or mismatched entries:
        // delete the unusable file and let the caller compile normally; a
        // later successful compile rewrites a fresh entry.
        std::remove(path.c_str());
        return nullptr;
    }
    return plan;
}
void store_disk_plan(const std::string& dir,const std::string& key,const Plan& plan) {
    // Best effort only: a store failure must never turn a successful compile
    // into an ABI error. All callers run under abi_mutex, so the plain
    // function-local counter is race-free inside this process; the pid in
    // the temporary name keeps concurrent processes distinct.
    static unsigned long long temp_counter=0;
    try {
        const json envelope{{"cache_schema_version",kPlanCacheSchemaVersion},
                            {"cache_format",kPlanCacheFormat},
                            {"source_identity",plan_cache_source_identity()},
                            {"plan",plan_to_json(plan)}};
        (void)::mkdir(dir.c_str(),0777);  // leaf directory only; failures surface below
        const std::string path=dir+"/"+key+".json";
        const std::string temp=dir+"/.tmp-"+std::to_string(::getpid())+"-"+
                               std::to_string(++temp_counter)+".json";
        {
            std::ofstream out(temp,std::ios::binary|std::ios::trunc);
            if(!out) return;
            const std::string text=envelope.dump();
            out.write(text.data(),static_cast<std::streamsize>(text.size()));
            out.flush();
            if(!out) { out.close(); std::remove(temp.c_str()); return; }
        }
        // Atomic publish: a concurrent reader observes either the previous
        // complete entry or this one, never a partial file.
        if(std::rename(temp.c_str(),path.c_str())!=0) std::remove(temp.c_str());
    } catch(...) {
    }
}

}  // namespace

namespace coarse_die {

std::shared_ptr<const Plan> try_load_disk_plan(const Hardware& hardware,
                                               const nlohmann::json& task) {
    const char* dir=std::getenv("COARSE_DIE_PLAN_CACHE_DIR");
    if(!dir||!*dir) return nullptr;
    const auto normalized=detail::normalize_task(hardware,task);
    const std::string hardware_identity=hardware.fingerprint.empty()
        ? stable_hash(hardware.input.dump()) : hardware.fingerprint;
    const std::string key=stable_hash(hardware_identity
        +"|coarse_die_compile_cache_v1|"+semantic_task_key(hardware,task));
    if(!plan_cache_safe_key(key)) return nullptr;
    return load_disk_plan(dir,key,hardware_identity,normalized);
}

void store_disk_plan_for(const Hardware& hardware,
                         const nlohmann::json& task, const Plan& plan) {
    const char* dir=std::getenv("COARSE_DIE_PLAN_CACHE_DIR");
    if(!dir||!*dir) return;
    const std::string hardware_identity=hardware.fingerprint.empty()
        ? stable_hash(hardware.input.dump()) : hardware.fingerprint;
    const std::string key=stable_hash(hardware_identity
        +"|coarse_die_compile_cache_v1|"+semantic_task_key(hardware,task));
    if(!plan_cache_safe_key(key)) return;
    store_disk_plan(dir,key,plan);
}

}  // namespace coarse_die
