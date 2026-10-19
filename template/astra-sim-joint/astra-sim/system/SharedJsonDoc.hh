/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SHARED_JSON_DOC_HH__
#define __SHARED_JSON_DOC_HH__

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <json/json.hpp>

namespace AstraSim {

// Process-level shared read-only JSON documents (batch 4, plan 5.2): every
// Sys parses the same configuration files (the system.json, the resolved
// die-model config, and the communicator-group file are identical across
// ranks), so the parsed document is cached per path and handed out as
// shared_ptr<const json>. Only the parse RESULT is shared: every consumer
// keeps its own per-rank key-walk, validation, and derived-object
// construction, and no document is ever mutated after parse (const
// propagation all the way down).
//
// Contract: an unopenable file returns nullptr (each caller keeps its own
// fail-closed open-error message); a parse exception propagates unchanged
// so each caller keeps its own catch/exit path. Failures are never
// memoized -- only successfully parsed documents enter the registry.
//
// The simulation is single-threaded (one event loop drives all Sys, same
// discipline as shared_die_plan_cache in Sys.cc), so the registry carries
// no lock; if a multi-threaded runtime is ever introduced, guard this
// function and the cached documents with a mutex.
inline std::shared_ptr<const nlohmann::json> shared_json_document(
    const std::string& path) {
    static std::map<std::string, std::shared_ptr<const nlohmann::json>>
        registry;
    auto it = registry.find(path);
    if (it == registry.end()) {
        std::ifstream in(path);
        if (!in.is_open()) {
            return nullptr;
        }
        auto doc = std::make_shared<nlohmann::json>();
        in >> *doc;
        it = registry.emplace(path, std::move(doc)).first;
    }
    return it->second;
}

}  // namespace AstraSim

#endif /* __SHARED_JSON_DOC_HH__ */
