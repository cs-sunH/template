// JsonScanner.cc -- structural scan implementation.
#include "joint_ledger/JsonScanner.hh"

#include <cstdint>

#include "joint_ledger/LedgerTypes.hh"

namespace jointledger {
namespace {

inline bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

[[noreturn]] void fail(const std::string& what, std::uint64_t offset) {
    throw LedgerError(what + " at byte " + std::to_string(offset));
}

// Skip whitespace; returns the offset of the next non-ws byte (or length).
std::size_t skip_ws(const char* data, std::size_t length, std::size_t pos) {
    while (pos < length && is_ws(data[pos])) ++pos;
    return pos;
}

// Scan one JSON string starting at the opening quote; returns the offset one
// past the closing quote.  Handles every escape (including \uXXXX; the hex
// digits can never terminate or open a string) without decoding.
std::size_t scan_string(const char* data, std::size_t length,
                        std::size_t start) {
    std::size_t pos = start + 1;
    while (pos < length) {
        const char c = data[pos];
        if (c == '\\') {
            pos += 2;  // escaped char (\" \\ \/ \b \f \n \r \t \uXXXX)
            continue;
        }
        if (c == '"') return pos + 1;
        ++pos;
    }
    fail("unterminated JSON string", start);
}

// Decode one JSON string (without surrounding quotes) into UTF-8 text;
// returns false when a raw backslash is absent (pure-ASCII fast path).
bool key_needs_decode(const char* data, std::size_t start, std::size_t end) {
    for (std::size_t pos = start; pos < end; ++pos) {
        if (data[pos] == '\\') return true;
    }
    return false;
}

void append_utf8(std::string& out, std::uint32_t code) {
    if (code < 0x80) {
        out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (code >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
}

std::uint32_t hex4(const char* data, std::size_t pos) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = data[pos + static_cast<std::size_t>(i)];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= static_cast<std::uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') value |= static_cast<std::uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= static_cast<std::uint32_t>(c - 'A' + 10);
        else fail("bad \\u escape", pos);
    }
    return value;
}

std::string decode_string(const char* data, std::size_t start,
                          std::size_t end) {
    std::string out;
    out.reserve(end - start);
    std::size_t pos = start;
    while (pos < end) {
        const char c = data[pos];
        if (c != '\\') {
            out.push_back(c);
            ++pos;
            continue;
        }
        if (pos + 1 >= end) fail("dangling escape", pos);
        const char esc = data[pos + 1];
        switch (esc) {
            case '"': out.push_back('"'); pos += 2; break;
            case '\\': out.push_back('\\'); pos += 2; break;
            case '/': out.push_back('/'); pos += 2; break;
            case 'b': out.push_back('\b'); pos += 2; break;
            case 'f': out.push_back('\f'); pos += 2; break;
            case 'n': out.push_back('\n'); pos += 2; break;
            case 'r': out.push_back('\r'); pos += 2; break;
            case 't': out.push_back('\t'); pos += 2; break;
            case 'u': {
                if (pos + 6 > end) fail("truncated \\u escape", pos);
                std::uint32_t code = hex4(data, pos + 2);
                pos += 6;
                if (code >= 0xD800 && code <= 0xDBFF && pos + 6 <= end
                        && data[pos] == '\\' && data[pos + 1] == 'u') {
                    const std::uint32_t low = hex4(data, pos + 2);
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        code = 0x10000 + ((code - 0xD800) << 10)
                             + (low - 0xDC00);
                        pos += 6;
                    }
                }
                append_utf8(out, code);
                break;
            }
            default:
                fail("unknown escape", pos);
        }
    }
    return out;
}

// Skip one JSON value (any type); returns the offset one past its last
// byte.  Structural skipping only: strings/escapes, nesting, numbers and
// literals.
std::size_t skip_value(const char* data, std::size_t length,
                       std::size_t start) {
    if (start >= length) fail("value expected", start);
    const char c = data[start];
    if (c == '"') return scan_string(data, length, start);
    if (c == '{' || c == '[') {
        const bool object = c == '{';
        int depth = 0;
        std::size_t pos = start;
        while (pos < length) {
            const char d = data[pos];
            if (d == '"') {
                pos = scan_string(data, length, pos);
                continue;
            }
            if (d == '{' || d == '[') {
                ++depth;
                ++pos;
                continue;
            }
            if (d == '}' || d == ']') {
                --depth;
                ++pos;
                if (depth == 0) {
                    if (object != (d == '}')) {
                        fail("mismatched closing bracket", pos - 1);
                    }
                    return pos;
                }
                continue;
            }
            if (d == ',' || d == ':' || is_ws(d)) {
                ++pos;
                continue;
            }
            // number or literal fragment
            while (pos < length && !is_ws(data[pos]) && data[pos] != ','
                   && data[pos] != '}' && data[pos] != ']' && data[pos] != ':'
                   && data[pos] != '"') {
                ++pos;
            }
        }
        fail("unterminated composite value", start);
    }
    if (c == 't' || c == 'f' || c == 'n') {
        const char* literals[3] = {"true", "false", "null"};
        for (const char* lit : literals) {
            const std::string text(lit);
            if (length - start >= text.size()
                    && std::string(data + start, text.size()) == text) {
                return start + text.size();
            }
        }
        fail("invalid literal", start);
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        std::size_t pos = start + 1;
        while (pos < length) {
            const char d = data[pos];
            if ((d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E'
                    || d == '+' || d == '-') {
                ++pos;
                continue;
            }
            break;
        }
        return pos;
    }
    fail("unexpected value byte", start);
}

}  // namespace

ScanResult scan_request(const char* data, std::size_t length) {
    if (data == nullptr) fail("null input", 0);
    std::size_t pos = skip_ws(data, length, 0);
    if (pos >= length || data[pos] != '{') {
        fail("request must be a JSON object", pos);
    }
    ++pos;
    bool found = false;
    ScanResult result;
    std::size_t member_index = 0;
    std::size_t last_member_end = 0;
    while (true) {
        pos = skip_ws(data, length, pos);
        if (pos < length && data[pos] == '}') {
            ++pos;
            break;
        }
        if (member_index > 0) {
            // A comma must separate members.
            if (pos >= length || data[pos] != ',') {
                fail("expected ',' between object members", pos);
            }
            ++pos;
            pos = skip_ws(data, length, pos);
        }
        if (pos >= length || data[pos] != '"') {
            fail("object member name must be a string", pos);
        }
        const std::size_t name_start = pos;
        const std::size_t name_end = scan_string(data, length, pos);
        pos = skip_ws(data, length, name_end);
        if (pos >= length || data[pos] != ':') {
            fail("expected ':' after member name", pos);
        }
        pos = skip_ws(data, length, pos + 1);
        const std::size_t value_start = pos;
        const std::size_t value_end = skip_value(data, length, pos);
        std::string name;
        if (key_needs_decode(data, name_start + 1, name_end - 1)) {
            name = decode_string(data, name_start + 1, name_end - 1);
        } else {
            name.assign(data + name_start + 1, name_end - 1 - (name_start + 1));
        }
        if (name == "work_ledger") {
            if (found) {
                fail("duplicate top-level work_ledger member", name_start);
            }
            found = true;
            result.member_start = name_start;
            result.value_start = value_start;
            result.value_end = value_end;
        }
        last_member_end = value_end;
        pos = value_end;
        ++member_index;
    }
    pos = skip_ws(data, length, pos);
    if (pos != length) {
        fail("trailing bytes after request object", pos);
    }
    if (!found) {
        fail("request carries no work_ledger member", 0);
    }
    if (member_index < 2) {
        // Only the work_ledger member: nothing would remain after splicing.
        fail("request must carry outer fields besides work_ledger", 0);
    }
    // Extend the splice window over one adjacent separator so the remaining
    // bytes stay a valid JSON object: prefer the comma after the member,
    // otherwise the comma before it.
    result.member_end = result.value_end;
    std::size_t splice_end = static_cast<std::size_t>(result.member_end);
    std::size_t splice_start = static_cast<std::size_t>(result.member_start);
    std::size_t probe = skip_ws(data, length, splice_end);
    if (probe < length && data[probe] == ',') {
        splice_end = probe + 1;
    } else {
        std::size_t back = splice_start;
        while (back > 0 && is_ws(data[back - 1])) --back;
        if (back == 0 || data[back - 1] != ',') {
            fail("work_ledger member is not comma-separated correctly",
                 splice_start);
        }
        splice_start = back - 1;
    }
    result.member_start = splice_start;
    result.member_end = splice_end;
    (void)last_member_end;
    return result;
}

}  // namespace jointledger
