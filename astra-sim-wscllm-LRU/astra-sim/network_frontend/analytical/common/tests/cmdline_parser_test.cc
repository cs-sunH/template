/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

cmdline_parser_test.cc -- P2-7 regression (M-23 移植,2026-09-30;母本
wscllm 仓 2026-09-28 同名测试): the shared
CmdLineParser's boolean option --rendezvous-protocol must honor the
separated value form.

Defect (report P2-7): this cxxopts vintage gives every bool-typed option an
implicit value of "true" and checked_parse_arg NEVER consumes the next argv
token for an implicit-value option, so

    --rendezvous-protocol false

parsed the implicit "true" and left "false" as an unmatched positional
(silently dropped): the simulation ran with the rendezvous protocol ON
while the operator explicitly asked for OFF -- a silent semantic inversion
with zero warnings (and the inversion is what arms the dormant
uninitialized rcv_req / notification-stream defects downstream).

Pinned forms (CmdLineParser.cc P2-7 pre-merge):
  B1  --rendezvous-protocol false   -> false (the reported inversion)
  B2  --rendezvous-protocol true    -> true
  B3  bare --rendezvous-protocol    -> true (implicit)
  B4  inline =false / =true / =0 / =1 -> parsed by cxxopts' bool lexicon
  B5  separated 0 / 1               -> false / true
  B6  --rendezvous-protocol false --injection-scale 2.0: the value token is
      consumed exactly once and the NEXT option still parses (the merge
      must not swallow the following flag or its value)
  B7  illegal separated value ("--rendezvous-protocol yes") -> fail-closed
      exit(-1) (fork + waitpid: parse() exits, it cannot return)
  B8  illegal inline value ("--rendezvous-protocol=yes") -> cxxopts
      argument_incorrect_type -> fail-closed exit(-1)
  B9  absent option keeps the declared default false
  B10 "--rendezvous-protocol --injection-scale 2.0": a "--"-prefixed next
      token is another option, not a value -> bare flag (true), and the
      following option still parses
  B11 C3 fix (2026-09-28): shared numeric sign/range validation at the
      parse boundary -- --num-queues-per-dim < 1, --injection-scale <= 0
      or non-finite, and --comm-scale non-finite fail closed (exit(-1));
      legal values still parse through (B12)

Build: cmake target AstraSim_Analytical_CmdLineParserTest (recipe of
ChunkIdGeneratorTest; links only common/CmdLineParser.cc). Run from the
build tree: bin/AstraSim_Analytical_CmdLineParserTest, exit 0 = ALL PASS.
*******************************************************************************/

#include "common/CmdLineParser.hh"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace AstraSimAnalytical;

namespace {

bool g_ok = true;

void expect(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "[cmdline-parser-test] FAIL: %s\n", message);
        g_ok = false;
    }
}

struct Args {
    std::vector<std::string> tokens;
    int argc = 0;
    std::vector<char*> argv;

    explicit Args(const std::vector<std::string>& toks) : tokens(toks) {
        tokens.insert(tokens.begin(), "fake_analytical_bin");
        argc = static_cast<int>(tokens.size());
        argv.reserve(tokens.size());
        for (auto& t : tokens) {
            argv.push_back(const_cast<char*>(t.c_str()));
        }
    }
};

bool rendezvous_is(const std::vector<std::string>& toks,
                   const bool expected, const char* const what) {
    Args a(toks);
    CmdLineParser parser(a.argv[0]);
    parser.parse(a.argc, a.argv.data());
    const bool value = parser.get<bool>("rendezvous-protocol");
    expect(value == expected, what);
    return value == expected;
}

// The fail-closed paths exit(-1) from inside parse(); run them in a forked
// child and require a normal EXIT_FAILURE-style exit (any non-zero code).
template <typename Fn>
void expect_fail_closed(Fn&& fn, const char* const what) {
    const pid_t pid = ::fork();
    if (pid < 0) {
        expect(false, "fork failed in fail-closed assertion");
        return;
    }
    if (pid == 0) {
        std::freopen("/dev/null", "w", stderr);
        fn();
        ::_exit(0);  // reached only when the parse did NOT exit
    }
    int status = 0;
    const pid_t waited = ::waitpid(pid, &status, 0);
    expect(waited == pid && WIFEXITED(status) && WEXITSTATUS(status) != 0,
           what);
}

}  // namespace

int main() {
    // B1: the reported inversion shape must parse to FALSE.
    rendezvous_is({"--rendezvous-protocol", "false"}, false,
                  "B1: separated 'false' parses to false");

    // B2: separated 'true' parses to true.
    rendezvous_is({"--rendezvous-protocol", "true"}, true,
                  "B2: separated 'true' parses to true");

    // B3: the bare flag keeps its implicit-true semantics.
    rendezvous_is({"--rendezvous-protocol"}, true,
                  "B3: bare flag is true (implicit)");

    // B4: inline forms parse via the cxxopts bool lexicon.
    rendezvous_is({"--rendezvous-protocol=false"}, false,
                  "B4: inline =false is false");
    rendezvous_is({"--rendezvous-protocol=true"}, true,
                  "B4: inline =true is true");
    rendezvous_is({"--rendezvous-protocol=0"}, false,
                  "B4: inline =0 is false");
    rendezvous_is({"--rendezvous-protocol=1"}, true,
                  "B4: inline =1 is true");

    // B5: separated numeric literals.
    rendezvous_is({"--rendezvous-protocol", "0"}, false,
                  "B5: separated '0' is false");
    rendezvous_is({"--rendezvous-protocol", "1"}, true,
                  "B5: separated '1' is true");

    // B6: the merge consumes exactly the value token; the next option and
    // its own separated value still parse.
    {
        Args a({"--rendezvous-protocol", "false", "--injection-scale",
                "2.0"});
        CmdLineParser parser(a.argv[0]);
        parser.parse(a.argc, a.argv.data());
        expect(parser.get<bool>("rendezvous-protocol") == false,
               "B6: rendezvous false with a following option");
        expect(parser.get<double>("injection-scale") == 2.0,
               "B6: the following option keeps its value");
    }

    // B7: an illegal separated value token fails closed.
    expect_fail_closed(
        [&]() {
            Args a({"--rendezvous-protocol", "yes"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B7: separated 'yes' fail-closes");

    // B8: an illegal inline value fails closed through cxxopts.
    expect_fail_closed(
        [&]() {
            Args a({"--rendezvous-protocol=yes"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B8: inline '=yes' fail-closes");

    // B9: absent option keeps the declared default false.
    rendezvous_is({}, false, "B9: absent option defaults to false");

    // B10: a "--"-prefixed token after the bare flag is another option.
    {
        Args a({"--rendezvous-protocol", "--injection-scale", "2.0"});
        CmdLineParser parser(a.argv[0]);
        parser.parse(a.argc, a.argv.data());
        expect(parser.get<bool>("rendezvous-protocol") == true,
               "B10: bare flag before another option is true");
        expect(parser.get<double>("injection-scale") == 2.0,
               "B10: the following option parses normally");
    }

    // B11 (C3 fix, 2026-09-28): shared numeric sign/range validation.
    // The old parser handed any lexically-valid number straight to Sys;
    // --injection-scale had no downstream guard at all (a negative value
    // scaled the endpoint delay negative), and the queue/count and
    // comm-scale guards lived two layers deeper.
    expect_fail_closed(
        [&]() {
            Args a({"--num-queues-per-dim=-2"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B11: negative --num-queues-per-dim fail-closes");
    expect_fail_closed(
        [&]() {
            Args a({"--num-queues-per-dim=0"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B11: zero --num-queues-per-dim fail-closes");
    expect_fail_closed(
        [&]() {
            Args a({"--injection-scale=-1.5"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B11: negative --injection-scale fail-closes");
    expect_fail_closed(
        [&]() {
            Args a({"--injection-scale=0"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B11: zero --injection-scale fail-closes");
    expect_fail_closed(
        [&]() {
            Args a({"--injection-scale=nan"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B11: nan --injection-scale fail-closes");
    expect_fail_closed(
        [&]() {
            Args a({"--comm-scale=nan"});
            CmdLineParser parser(a.argv[0]);
            parser.parse(a.argc, a.argv.data());
        },
        "B11: nan --comm-scale fail-closes");

    // B12: legal values still parse through untouched (defaults included).
    {
        Args a({"--num-queues-per-dim=3", "--injection-scale=0.5",
                "--comm-scale=1"});
        CmdLineParser parser(a.argv[0]);
        parser.parse(a.argc, a.argv.data());
        expect(parser.get<int>("num-queues-per-dim") == 3,
               "B12: num-queues-per-dim=3 parses");
        expect(parser.get<double>("injection-scale") == 0.5,
               "B12: injection-scale=0.5 parses");
        expect(parser.get<double>("comm-scale") == 1.0,
               "B12: comm-scale=1 parses");
    }

    if (!g_ok) {
        std::fprintf(stderr, "[cmdline-parser-test] FAIL: see messages "
                             "above\n");
        return 1;
    }
    std::printf("[cmdline-parser-test] ALL PASS: --rendezvous-protocol "
                "separated/inline/bare forms + fail-closed lexicon\n");
    return 0;
}
