/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_ONLINE_CLI_FAMILY_HH
#define EXECUTION_DRIVEN_ONLINE_CLI_FAMILY_HH

#include <string>

namespace AstraSim {
namespace ExecutionDriven {

// The single source of truth for the online CLI family prefixes
// (2026-09-30, R2 audit): OnlineCli.cc owns the family's hard-error
// contract and CmdLineParser.cc exempts exactly these prefixes from its
// unmatched-token fail-closed guard. The two lists used to be hand-synced
// copies in anonymous namespaces with no test locking them together -- a
// one-sided edit would silently reject (or silently swallow) a family flag.
inline constexpr const char* kOnlineFamilyPrefixes[] = {
    "--request-", "--bridge-",  "--close-", "--online-",
    "--command-", "--sensing-", "--idle-",
};

inline bool is_online_family_token(const std::string& token) {
    for (const char* prefix : kOnlineFamilyPrefixes) {
        if (token.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif /* EXECUTION_DRIVEN_ONLINE_CLI_FAMILY_HH */
