/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CmdLineParser.hh"

using namespace AstraSimAnalytical;

CmdLineParser::CmdLineParser(const char* const argv0) noexcept
    : options(argv0, "ASTRA-sim") {
    parsed = {};

    // define options
    define_options();
}

void CmdLineParser::define_options() noexcept {
    options.set_width(70).allow_unrecognised_options().add_options()(
        // Optional with an empty default (M21, face-LRU audit 2026-09-29):
        // the online-only entry of this repo never consumes the
        // workload file, so forcing it only trapped upstream-habit
        // invocations with a raw terminate.
        "workload-configuration", "Workload configuration file",
        cxxopts::value<std::string>()->default_value(""))(
        "comm-group-configuration", "Communicator group configuration file",
        cxxopts::value<std::string>()->default_value("empty"))(
        "system-configuration", "System configuration file",
        cxxopts::value<std::string>())("remote-memory-configuration",
                                       "Remote memory configuration file",
                                       cxxopts::value<std::string>())(
        "network-configuration", "Network configuration file",
        cxxopts::value<std::string>())(
        "logging-configuration", "Logging configuration file",
        cxxopts::value<std::string>()->default_value("empty"))(
        "logging-folder", "Logging folder",
        cxxopts::value<std::string>()->default_value("log"))(
        "num-queues-per-dim", "Number of queues per each dimension",
        cxxopts::value<int>()->default_value("1"))(
        "comm-scale", "Communication scale",
        cxxopts::value<double>()->default_value("1"))(
        "injection-scale", "Injection scale",
        cxxopts::value<double>()->default_value("1"))(
        "rendezvous-protocol", "Whether to enable rendezvous protocol",
        cxxopts::value<bool>()->default_value("false"))(
        "metrics-configuration", "Metrics manifest configuration file",
        cxxopts::value<std::string>()->default_value("empty"))(
        "metrics-detail", "Metrics detail level (off|summary|full)",
        cxxopts::value<std::string>()->default_value("off"));
}

void CmdLineParser::parse(int argc, char* argv[]) noexcept {
    try {
        // try parsing command line options
        parsed = options.parse(argc, argv);
    } catch (const cxxopts::OptionException& e) {
        // error occurred
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "Error parsing options: " << e.what() << std::endl;
        // (2026-10-01) exit code unification: -1 (255) vs the online
        // family's EXIT_FAILURE (1) left no single decodable $? for
        // scripts; use the same standard failure code.
        std::exit(EXIT_FAILURE);
    }
    // (2026-10-01) Typo fail-closed for shared options:
    // allow_unrecognised_options() exists so the --online family
    // (parsed fail-closed by parse_online_cli) can pass through, but it
    // also silently swallowed ANY typo'd shared flag and fell back to its
    // default -- worst case "--metrics-detai full" degraded to "off" and a
    // healthy run produced zero [METRIC] output with exit 0. Reject any
    // unmatched "--" token that does not belong to the online family; the
    // family's own parser rejects its unknown members. Keep this prefix
    // list in sync with is_online_family in OnlineCli.cc.
    static const char* const kOnlineFamilyPrefixes[] = {
        "--request-", "--bridge-", "--close-", "--online-",
        "--command-", "--sensing-", "--idle-"};
    for (const std::string& token : parsed.unmatched()) {
        if (token.rfind("--", 0) != 0) {
            continue;  // positional token; not this parser's business
        }
        bool online_family = false;
        for (const char* const prefix : kOnlineFamilyPrefixes) {
            if (token.rfind(prefix, 0) == 0) {
                online_family = true;
                break;
            }
        }
        if (online_family) {
            continue;  // consumed (or rejected) by parse_online_cli
        }
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "Unrecognized option (a typo'd shared option would "
                     "silently fall back to its default): "
                  << token << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
