#include "checked_rational256.hh"

#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using os_fixedwidth::Rational256;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Rational256 parse(const std::string& text) {
  Rational256 value;
  std::string error;
  require(Rational256::from_fraction(text, &value, &error), error.c_str());
  return value;
}

void expect_string(const Rational256& value, const std::string& expected,
                   const char* message) {
  require(value.to_string() == expected, message);
}

void self_test() {
  std::string error;
  Rational256 a, b, out;

  require(Rational256::from_fraction("-12/18", &a, &error), "parse -12/18");
  expect_string(a, "-2/3", "fraction normalization and sign");
  require(Rational256::from_fraction("0/999", &a, &error), "parse zero");
  expect_string(a, "0/1", "zero canonical form");
  require(Rational256::from_fraction("-1/6", &a, &error), "parse negative");
  require(Rational256::from_fraction("1/3", &b, &error), "parse positive");
  require(Rational256::add(a, b, &out, &error), "signed addition");
  expect_string(out, "1/6", "signed addition result");
  require(Rational256::subtract(a, b, &out, &error), "signed subtraction");
  expect_string(out, "-1/2", "signed subtraction result");
  require(Rational256::from_fraction("-5/7", &a, &error), "parse multiply lhs");
  require(Rational256::from_fraction("14/15", &b, &error), "parse multiply rhs");
  require(Rational256::multiply(a, b, &out, &error), "signed multiplication");
  expect_string(out, "-2/3", "signed multiplication result");
  require(Rational256::from_fraction("10/21", &b, &error), "parse divide rhs");
  require(Rational256::divide(a, b, &out, &error), "signed division");
  expect_string(out, "-3/2", "signed division result");
  int comparison = 0;
  require(Rational256::compare(parse("-3/2"), parse("-4/3"),
                               &comparison, &error) && comparison < 0,
          "signed comparison");

  const std::string p255 =
      "57896044618658097711785492504343953926634992332820282019728792003956564819968";
  const std::string p255m1 =
      "57896044618658097711785492504343953926634992332820282019728792003956564819967";
  const std::string p255m2 =
      "57896044618658097711785492504343953926634992332820282019728792003956564819966";
  const Rational256 wide_a = parse(p255m1 + "/" + p255);
  const Rational256 wide_b = parse(p255m2 + "/" + p255);
  require(Rational256::compare(wide_a, wide_b, &comparison, &error) &&
              comparison > 0,
          "comparison with >256-bit raw cross products");
  require(Rational256::add(wide_a, wide_a, &out, &error),
          "addition with >256-bit unreduced cross products");
  expect_string(out, p255m1 + "/" +
                     "28948022309329048855892746252171976963317496166410141009864396001978282409984",
                 "wide add normalization");
  const Rational256 reciprocal = parse(p255 + "/" + p255m1);
  require(Rational256::multiply(wide_a, reciprocal, &out, &error),
          "multiply with >256-bit raw products");
  expect_string(out, "1/1", "wide product normalization");

  require(!Rational256::from_fraction("1/0", &out, &error),
          "zero denominator must fail closed");
  require(!Rational256::from_fraction("1/-3", &out, &error),
          "negative denominator must fail closed");
  require(!Rational256::from_fraction(p255 + "0/1", &out, &error),
          "257-bit state input must fail closed");
  require(Rational256::from_integer(p255, &a, &error), "parse 2^255");
  require(Rational256::from_integer("2", &b, &error), "parse two");
  require(!Rational256::multiply(a, b, &out, &error),
          "257-bit normalized result must fail closed");
  require(!Rational256::divide(a, parse("0/1"), &out, &error),
          "division by zero must fail closed");

  const std::string repeat = "170141183460469231731687303715884105851";
  require(Rational256::from_fraction("5/7", &a, &error), "parse repeat base");
  require(Rational256::multiply_repeat(a, repeat, &out, &error),
          "checked 128-bit repeat multiplier");
  expect_string(out,
      "850705917302346158658436518579420529255/7",
      "large repeated-ledger multiplication");
  require(!Rational256::multiply_repeat(a,
      "340282366920938463463374607431768211456", &out, &error),
      "129-bit repeat multiplier must fail closed");

  require(Rational256::from_double(0.1, &out, &error),
          "finite binary64 conversion");
  expect_string(out, "3602879701896397/36028797018963968",
                "binary64 must be represented exactly");
  const double subnormal = std::numeric_limits<double>::denorm_min();
  require(!Rational256::from_double(subnormal, &out, &error),
          "over-budget subnormal binary64 must fail closed");
  require(!Rational256::from_double(std::numeric_limits<double>::min(),
                                    &out, &error),
          "over-budget minimum-normal binary64 must fail closed");
  require(!Rational256::from_double(std::numeric_limits<double>::infinity(),
                                    &out, &error),
          "infinite binary64 must fail closed");
  require(!Rational256::from_double(std::numeric_limits<double>::quiet_NaN(),
                                    &out, &error),
          "NaN binary64 must fail closed");
}

bool parse_arg(const std::string& text, Rational256* value,
               std::string* error) {
  return Rational256::from_fraction(text, value, error);
}

int command_line() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream input(line);
    std::string op, lhs_text, rhs_text;
    input >> op >> lhs_text >> rhs_text;
    Rational256 lhs, rhs, result;
    std::string error;
    bool ok = parse_arg(lhs_text, &lhs, &error);
    if (ok && op != "repeat") ok = parse_arg(rhs_text, &rhs, &error);
    if (!ok) { std::cout << "ERR\n"; continue; }
    if (op == "add") ok = Rational256::add(lhs, rhs, &result, &error);
    else if (op == "sub") ok = Rational256::subtract(lhs, rhs, &result, &error);
    else if (op == "mul") ok = Rational256::multiply(lhs, rhs, &result, &error);
    else if (op == "div") ok = Rational256::divide(lhs, rhs, &result, &error);
    else if (op == "cmp") {
      int comparison = 0;
      if (!Rational256::compare(lhs, rhs, &comparison, &error)) {
        std::cout << "ERR\n";
      } else {
        std::cout << comparison << '\n';
      }
      continue;
    } else if (op == "eq") {
      std::cout << (lhs.equal(rhs) ? 1 : 0) << '\n';
      continue;
    } else if (op == "repeat") {
      ok = Rational256::multiply_repeat(lhs, rhs_text, &result, &error);
    } else {
      ok = false;
    }
    std::cout << (ok ? result.to_string() : "ERR") << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      self_test();
      std::cout << "fixed_limb_rational_self_test=passed\n";
      return 0;
    }
    return command_line();
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
