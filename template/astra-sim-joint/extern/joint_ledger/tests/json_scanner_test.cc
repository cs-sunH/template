// json_scanner_test.cc -- structural scanner: boundaries, escapes, splices.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "joint_ledger/JsonScanner.hh"
#include "joint_ledger/LedgerTypes.hh"
#include "json/json.hpp"

using jointledger::LedgerError;
using jointledger::scan_request;

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } \
} while (0)

static void check_splice(const std::string& raw) {
    const auto result = scan_request(raw.data(), raw.size());
    const std::string outer = raw.substr(0, result.member_start)
        + raw.substr(result.member_end);
    // The spliced outer frame must parse as a JSON object, not contain a
    // work_ledger member, and equal the full parse minus that member.
    const nlohmann::json full = nlohmann::json::parse(raw);
    const nlohmann::json outer_json = nlohmann::json::parse(outer);
    CHECK(outer_json.is_object());
    CHECK(outer_json.find("work_ledger") == outer_json.end());
    nlohmann::json expected = full;
    expected.erase("work_ledger");
    CHECK(outer_json == expected);
    // The value slice must equal the member value.
    const std::string value = raw.substr(
        result.value_start, result.value_end - result.value_start);
    CHECK(nlohmann::json::parse(value) == full.at("work_ledger"));
}

int main() {
    // Simple mid-object member with trailing comma.
    check_splice(R"({"a":1,"work_ledger":{"x":[1,2,{"y":"z"}]},"b":"c"})");
    // Leading member (comma after).
    check_splice(R"({"work_ledger":[1,2,3],"a":1,"b":2})");
    // Trailing member (comma before).
    check_splice(R"({"a":1,"b":2,"work_ledger":{"k":"v"}})");
    // Escapes inside the big value must not confuse boundary tracking.
    check_splice(R"({"a":"x}y\"z{","work_ledger":{"s":"a\"b{}[]","n":null}})");
    // Escaped member name spelling of work_ledger must be located too.
    check_splice(
        R"({"a":1,"work_\u006cedger":{"t":1},"b":2})");
    // Numbers with exponents, unicode and nested arrays.
    check_splice(
        R"({"t":-1.5e+10,"work_ledger":{"u":"é😀","deep":[[["}"]]]},"z":true})");
    // Whitespace-tolerant input.
    check_splice("{ \"a\" : 1 , \"work_ledger\" : { \"x\" : 2 } , \"b\" : 3 }");

    // Rejections.
    auto throws = [](const std::string& raw, const char* why) {
        try {
            scan_request(raw.data(), raw.size());
            std::fprintf(stderr, "FAIL no-throw: %s\n", why);
            ++failures;
        } catch (const LedgerError&) {
        }
    };
    throws(R"({"a":1})", "no work_ledger");
    throws(R"({"work_ledger":{"a":1}})", "only work_ledger");
    throws(R"({"a":1,"work_ledger":{"x":1},"work_ledger":{"x":2}})",
           "duplicate member");
    throws(R"([1,2,3])", "not an object");
    throws("{\"a\":1,\"work_ledger\":{\"x\":1", "unterminated value");
    throws(R"({"a":1 "work_ledger":1})", "missing comma");

    if (failures == 0) {
        std::printf("json_scanner_test PASS\n");
        return 0;
    }
    std::printf("json_scanner_test FAIL (%d)\n", failures);
    return 1;
}
