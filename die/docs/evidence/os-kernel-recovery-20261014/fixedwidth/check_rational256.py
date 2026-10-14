#!/usr/bin/env python3
"""Compile and differentially test the bounded C++ rational primitive."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "rational256_test.cc"
HEADER = HERE / "checked_rational256.hh"
BINARY = HERE / "rational256_test"


def check(result: bool, message: str) -> None:
    if not result:
        raise AssertionError(message)


def fits_state(value: Fraction) -> bool:
    return value.numerator.bit_length() <= 256 and value.denominator.bit_length() <= 256


def random_fraction(rng: random.Random) -> Fraction:
    numerator_bits = rng.randrange(0, 257)
    denominator_bits = rng.randrange(1, 257)
    numerator = rng.getrandbits(numerator_bits)
    if rng.randrange(2):
        numerator = -numerator
    denominator = rng.getrandbits(denominator_bits) or 1
    return Fraction(numerator, denominator)


def expected(cmd: str, a: Fraction, b_text: str) -> str:
    op = cmd.split()[0]
    if op == "repeat":
        count = int(b_text)
        result = a * count
    elif op == "cmp":
        b = Fraction(b_text)
        return str((a > b) - (a < b))
    elif op == "eq":
        b = Fraction(b_text)
        return "1" if a == b else "0"
    else:
        b = Fraction(b_text)
        if op == "add": result = a + b
        elif op == "sub": result = a - b
        elif op == "mul": result = a * b
        elif op == "div":
            if b == 0: return "ERR"
            result = a / b
        else: raise AssertionError(f"unknown test operation {op}")
    return str(result) if fits_state(result) else "ERR"


def main() -> int:
    compile_cmd = [
        "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
        "-O2", str(SOURCE), "-o", str(BINARY),
    ]
    build = subprocess.run(compile_cmd, cwd=ROOT, text=True,
                           capture_output=True, check=False)
    check(build.returncode == 0, f"compile failed:\n{build.stdout}\n{build.stderr}")
    self_test = subprocess.run([str(BINARY), "--self-test"], cwd=ROOT,
                               text=True, capture_output=True, check=False)
    check(self_test.returncode == 0 and
          "fixed_limb_rational_self_test=passed" in self_test.stdout,
          f"self-test failed:\n{self_test.stdout}\n{self_test.stderr}")

    rng = random.Random(20261014)
    commands: list[str] = []
    answers: list[str] = []
    for _ in range(320):
        a, b = random_fraction(rng), random_fraction(rng)
        a_text, b_text = str(a), str(b)
        for op in ("add", "sub", "mul", "div", "cmp", "eq"):
            command = f"{op} {a_text} {b_text}"
            commands.append(command)
            answers.append(expected(command, a, b_text))
        count = rng.getrandbits(rng.randrange(0, 129))
        if rng.randrange(2): count = -count
        # Repeat counts are unsigned u128, so negatives must fail closed.
        command = f"repeat {a_text} {count}"
        commands.append(command)
        answers.append("ERR" if count < 0 or count.bit_length() > 128
                       else expected(command, a, str(count)))

    # Exercise equality/ordering and normalization with ~255-bit raw products,
    # plus parser and repeat-counter failures at their checked boundaries.
    p255 = 1 << 255
    wide_a = Fraction(p255 - 1, p255)
    wide_b = Fraction(p255 - 2, p255)
    commands.extend([
        f"cmp {wide_a} {wide_b}",
        f"add {wide_a} {wide_a}",
        f"mul {wide_a} {Fraction(p255, p255 - 1)}",
        f"mul {p255} 2",
        f"repeat 1/3 {1 << 128}",
        f"div 1/9 0/1",
        f"mul {1 << 768} 1",
    ])
    answers.extend(["1", str(wide_a + wide_a), "1", "ERR", "ERR", "ERR", "ERR"])

    run = subprocess.run([str(BINARY)], input="\n".join(commands) + "\n",
                         cwd=ROOT, text=True, capture_output=True, check=False)
    check(run.returncode == 0, f"differential process failed: {run.stderr}")
    actual = run.stdout.splitlines()
    check(len(actual) == len(answers),
          f"output count mismatch: got {len(actual)}, expected {len(answers)}")
    for index, (got, want) in enumerate(zip(actual, answers, strict=True)):
        op = commands[index].split()[0]
        if op in ("add", "sub", "mul", "div", "repeat") and want != "ERR":
            check(got != "ERR" and Fraction(got) == Fraction(want),
                  f"case {index} mismatch: {commands[index]} -> {got}, expected {want}")
        else:
            check(got == want,
                  f"case {index} mismatch: {commands[index]} -> {got}, expected {want}")

    log = {
        "status": "passed",
        "compile_command": compile_cmd,
        "self_test": self_test.stdout.strip(),
        "differential_cases": len(commands),
        "random_seed": 20261014,
        "random_rational_pairs": 320,
        "checked_widths": {"canonical_num_den_bits": 256, "scratch_bits": 768,
                           "repeat_count_bits": 128},
        "source_sha256": {
            SOURCE.name: hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            HEADER.name: hashlib.sha256(HEADER.read_bytes()).hexdigest(),
            Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "This tests the standalone arithmetic primitive only; no Runtime, timing, production schedule, or descriptor integration is claimed.",
            "The independent Python Fraction is the exact differential oracle; the C++ implementation contains no arbitrary-precision dependency.",
        ],
    }
    (HERE / "rational256_test_results.json").write_text(__import__("json").dumps(
        log, indent=2, sort_keys=True) + "\n")
    print(__import__("json").dumps(log, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
