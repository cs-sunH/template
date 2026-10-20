/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CmdLineParser.hh"

#include "astra-sim/workload/execution_driven/OnlineCliFamily.hh"
#include <cmath>
#include <string>
#include <vector>

using namespace AstraSimAnalytical;

namespace {

// (2026-09-30, R2 audit) The hand-synced prefix copy is gone: the table
// lives in workload/execution_driven/OnlineCliFamily.hh
// (kOnlineFamilyPrefixes) and is shared verbatim with OnlineCli.cc, which
// owns the family's hard-error contract. The online entry parses its own
// family first and hands the SAME argv to this parser, so those tokens
// legitimately land in ParseResult::unmatched() below and are exempt by
// that shared prefix test. Every OTHER unmatched flag is a typo of a shared
// option and is rejected instead of being silently swallowed into the
// option's default value.

// P2-7 fix (2026-09-30, ported from the joint repo / wscllm parser): this
// cxxopts vintage gives every bool-typed option an implicit value of "true"
// and its checked_parse_arg NEVER consumes the next argv token for an
// implicit-value option. The separated form
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
    : options(argv0, "ASTRA-sim") {
    parsed = {};

    // define options
    define_options();
}

void CmdLineParser::define_options() noexcept {
    options.set_width(70).allow_unrecognised_options().add_options()(
        // Optional with an empty default (M21, mirrored from the joint
        // repo): the online-only entry of this repo never consumes the
        // workload file, so forcing it only trapped upstream-habit
        // invocations with a raw terminate; the value stays unused either
        // way.
        "workload-configuration", "Workload configuration file",
        cxxopts::value<std::string>()->default_value(""))(
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
        "comm-scale",
        "Communication scale (no-op: accepted for CLI compatibility but "
        "ignored by the simulator)",
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
    // P2-7 fix: one pass over argv that folds the separated boolean form
    // into the inline form (see option_takes_separated_bool). The tokens
    // live in a local vector because an inline value cannot be written back
    // into the caller's argv buffers in place (the merged token is longer
    // than the option token).
    std::vector<std::string> tokens;
    tokens.reserve(static_cast<size_t>(argc));
    std::vector<char*> arg_ptrs;
    arg_ptrs.reserve(static_cast<size_t>(argc));
    if (argc > 0 && argv[0] != nullptr) {
        tokens.emplace_back(argv[0]);  // program name: never rewritten
    }
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == nullptr) {
            // A NULL inside the loop is not a conforming argv layout but
            // must not crash the parser either -- treat it as an empty
            // token so the usual validation reports it instead of
            // std::string(NULL) UB.
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
    } catch (const cxxopts::OptionException& e) {
        // error occurred
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "Error parsing options: " << e.what() << std::endl;
        exit(-1);
    }

    // Fail closed on unrecognised flags. allow_unrecognised_options() above
    // keeps cxxopts from throwing on the foreign online-family tokens, but a
    // misspelled shared option (e.g. --workload-configurationn) must not be
    // silently dropped and fall back to its default value.
    for (const auto& token : parsed.unmatched()) {
        if (token.size() < 2 || token[0] != '-') {
            continue;  // a value token of a foreign option, or a positional
        }
        if (AstraSim::ExecutionDriven::is_online_family_token(token)) {
            continue;  // already validated (hard errors) by parse_online_cli
        }
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "Unrecognised option: " << token
                  << " (only the shared options and the online CLI family "
                     "are accepted)"
                  << std::endl;
        exit(-1);
    }

    // C3 alignment (2026-09-30, ported from the wscllm CmdLineParser; the
    // switch-manifest row already documents this contract): sign/range
    // validation for the shared numeric options. cxxopts converts
    // lexically, so an inline negative or NaN --injection-scale used to
    // flow straight into Sys: a negative value scaled the endpoint delay
    // negative and produced negative event times. Fail closed here at the
    // shared boundary with the same exit(-1) convention as the unmatched
    // guard above. (2026-10-01 comment fix: the old main_online-side
    // non-negative guard was removed the same day -- this parser is the
    // single validation point for the flag, and the accepted domain is
    // strictly positive, not merely non-negative.)
    const double injection_scale = parsed["injection-scale"].as<double>();
    if (!std::isfinite(injection_scale) || injection_scale <= 0.0) {
        std::cerr << "[Error] (AstraSim/analytical/common) "
                  << "option --injection-scale must be a finite positive "
                     "number, got: "
                  << injection_scale << std::endl;
        exit(-1);
    }
}
