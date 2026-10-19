/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CmdLineParser.hh"

#include <cmath>
#include <vector>

using namespace AstraSimAnalytical;

namespace {

// P2-7 fix (2026-09-28): this cxxopts vintage gives every bool-typed option
// an implicit value of "true" and its checked_parse_arg NEVER consumes the
// next argv token for an implicit-value option. The separated form
//   --rendezvous-protocol false
// therefore parsed the implicit "true" and left "false" as an unmatched
// positional (silently dropped) -- the simulation ran with the rendezvous
// protocol ON while the operator explicitly asked for OFF, zero warnings.
// The pre-merge below rewrites the separated form into the inline
// "--rendezvous-protocol=<literal>" form cxxopts' bool parser handles
// correctly, and fails closed on a non-literal value token:
//   --rendezvous-protocol true|True|false|False|1|0 -> merged inline
//   --rendezvous-protocol <any other bare token>    -> usage error, exit(-1)
//   --rendezvous-protocol <-another option> / <EOF> -> bare flag (true)
// If a future bool-typed option is added to define_options(), it MUST be
// listed in option_takes_separated_bool() too (or declared as a string with
// explicit parsing), or it re-opens this silent inversion.
bool option_takes_separated_bool(const std::string& token) {
    return token == "--rendezvous-protocol";
}

// The exact lexicon cxxopts' bool parser accepts inline (IsTrueText /
// IsFalseText): true/True/1 -> true, false/False/0 -> false.
bool is_bool_literal(const std::string& token) {
    return token == "true" || token == "True" || token == "false" ||
           token == "False" || token == "1" || token == "0";
}

}  // namespace

CmdLineParser::CmdLineParser(const char* const argv0) noexcept
    : options(argv0 == nullptr ? "astra-sim" : argv0, "ASTRA-sim") {
    parsed = {};

    // define options
    define_options();
}

void CmdLineParser::define_options() noexcept {
    // Unknown options fail closed (net-frontend-E2 fix, 2026-09-27): a
    // misspelled option carrying a default value used to be silently
    // ignored here and the run proceeded on defaults (e.g. --injection-scle
    // 2.0 ran at injection scale 1); without allow_unrecognised_options()
    // cxxopts rejects it via option_not_exists_exception. The ONLINE entry
    // (main_online.cc) strips its own already-validated family
    // (strip_online_family_argv) before calling this parser, so an online
    // token reaching here is a genuine unknown option.
    options.set_width(70).add_options()(
        "workload-configuration", "Workload configuration file",
        cxxopts::value<std::string>())(
        "comm-group-configuration", "Communicator group configuration file",
        cxxopts::value<std::string>()->default_value("empty"))(
        "system-configuration", "System configuration file",
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
    // P2-7 fix (2026-09-28): one pass over argv that folds the separated
    // boolean form into the inline form (see option_takes_separated_bool).
    // The tokens live in a local vector because an inline value cannot be
    // written back into the caller's argv buffers in place (the merged
    // token is longer than the option token).
    std::vector<std::string> tokens;
    tokens.reserve(static_cast<size_t>(argc));
    std::vector<char*> arg_ptrs;
    arg_ptrs.reserve(static_cast<size_t>(argc));
    if (argc > 0 && argv[0] != nullptr) {
        tokens.emplace_back(argv[0]);  // program name: never rewritten
    }
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == nullptr) {
            // POSIX execve may pass argc==0 (argv[0]==NULL); a NULL inside
            // the loop is not a conforming layout but must not crash the
            // parser either -- treat it as an empty token so the usual
            // validation reports it instead of std::string(NULL) UB.
            tokens.emplace_back();
            continue;
        }
        std::string token(argv[i]);
        if (option_takes_separated_bool(token) && i + 1 < argc) {
            const std::string next(argv[i + 1]);
            if (next.empty() || next[0] != '-') {
                if (!is_bool_literal(next)) {
                    std::cerr << "[Error] (AstraSim/analytical/common) "
                              << "option --rendezvous-protocol requires a "
                                 "boolean value (true/1/false/0), got: "
                              << next << std::endl;
                    exit(-1);
                }
                token += "=" + next;
                ++i;  // consume the explicit value token
            }
        }
        tokens.push_back(std::move(token));
    }
    for (const auto& token : tokens) {
        arg_ptrs.push_back(const_cast<char*>(token.c_str()));
    }

    try {
        // try parsing command line options
        parsed = options.parse(static_cast<int>(arg_ptrs.size()),
                               arg_ptrs.data());
    } catch (const std::exception& e) {
        // error occurred. Broadened from cxxopts::OptionException (same
        // broadening as the wscllm-LRU M5-03 fix) so no other exception
        // type can escape this noexcept function into std::terminate;
        // every failure mode still fails closed with a message and a
        // nonzero exit.
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "Error parsing options: " << e.what() << std::endl;
        exit(-1);
    }

    // E2 completion fix (2026-09-29): with no parse_positional() target,
    // this cxxopts vintage collects bare tokens matching no option into
    // ParseResult's unmatched list and then DROPS them silently -- e.g.
    // `injection-scale 2.0` with the leading -- typo'd away ran at the
    // default injection scale 1 with zero diagnostics, the exact
    // silent-default shape the net-frontend-E2 fix (2026-09-27) closed
    // for unknown --prefixed options. Any unmatched token is a usage
    // error; cxxopts already consumes the separated value of a recognized
    // option, so legitimate invocations have an empty list.
    const std::vector<std::string> unmatched = parsed.unmatched();
    if (!unmatched.empty()) {
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "unrecognized stray argument(s):";
        for (const auto& token : unmatched) {
            std::cerr << " '" << token << "'";
        }
        std::cerr << " (every option must be --prefixed and carry its own "
                     "value; bare positionals are not part of the grammar)"
                  << std::endl;
        exit(-1);
    }

    // C3 fix (2026-09-28): sign/range validation for the shared numeric
    // options. cxxopts converts lexically, so an inline negative or NaN
    // value used to flow straight into Sys: --num-queues-per-dim=-2 and
    // --injection-scale=-1.5 were accepted by the option parser and only
    // the first was caught downstream (Sys ctor queues-per-dim >= 1 panic;
    // comm-scale != 1 panic). injection-scale had NO downstream guard at
    // all -- a negative value scaled the endpoint delay negative and
    // llround(10 * negative) produced negative event times. Fail closed
    // here at the shared boundary with the same exit(-1) convention as the
    // bool pre-merge above; Sys keeps its own guards (defense in depth).
    const auto num_queues_per_dim =
        parsed["num-queues-per-dim"].as<int>();
    if (num_queues_per_dim < 1) {
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "option --num-queues-per-dim must be >= 1, got: "
                  << num_queues_per_dim << std::endl;
        exit(-1);
    }
    const auto comm_scale = parsed["comm-scale"].as<double>();
    if (!std::isfinite(comm_scale)) {
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "option --comm-scale must be a finite number, got: "
                  << comm_scale << std::endl;
        exit(-1);
    }
    const auto injection_scale = parsed["injection-scale"].as<double>();
    if (!std::isfinite(injection_scale) || injection_scale <= 0.0) {
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "option --injection-scale must be a finite number "
                     "> 0, got: "
                  << injection_scale << std::endl;
        exit(-1);
    }
}
