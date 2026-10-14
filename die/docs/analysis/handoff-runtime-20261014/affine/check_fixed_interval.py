#!/usr/bin/env python3
"""Independent Fraction/Q128 checks for the C++ interval affine prototype."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)

spec = importlib.util.spec_from_file_location("nested_k_output", HERE / "nested_k_output.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import nested K/output reference")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)


def read_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for row in path.read_text().splitlines():
        if not row:
            continue
        key, value = row.split("=", 1)
        if key in fields:
            raise AssertionError(f"duplicate descriptor field {key}")
        fields[key] = value
    return fields


def parse_interval(text: str) -> tuple[F, F]:
    lo, hi = text.split("|", 1)
    return F(lo), F(hi)


def contains(bounds: tuple[F, F], value: F) -> bool:
    return bounds[0] <= value <= bounds[1]


def q128_bounds(x: ref.Interval) -> tuple[F, F]:
    return F(x.lo, ref.SCALE), F(x.hi, ref.SCALE)


def compose_exact(first: tuple, second: tuple) -> tuple:
    # Each tuple is (a,b,c[],d[]); apply first, then second.
    a1, b1, c1, d1 = first
    a2, b2, c2, d2 = second
    return (
        a2 * a1,
        a2 * b1 + b2,
        [c1[i] + c2[i] * a1 for i in range(len(c1))],
        [d1[i] + c2[i] * b1 + d2[i] for i in range(len(d1))],
    )


def power_exact(base: tuple, count: int) -> tuple:
    dims = len(base[2])
    result = (F(1), F(0), [F(0) for _ in range(dims)],
              [F(0) for _ in range(dims)])
    while count:
        if count & 1:
            result = compose_exact(result, base)
        count >>= 1
        if count:
            base = compose_exact(base, base)
    return result


def apply_exact(transform: tuple, x0: F) -> tuple[F, list[F]]:
    a, b, c, d = transform
    return a * x0 + b, [c[i] * x0 + d[i] for i in range(len(c))]


def main() -> None:
    descriptor = HERE / "nested-k-output.affine"
    output = HERE / "fixed-interval.out"
    fields = read_fields(descriptor)
    observed: dict[str, str] = {}
    for row in output.read_text().splitlines():
        key, value = row.split("=", 1)
        if key in observed:
            raise AssertionError(f"duplicate C++ output key {key}")
        observed[key] = value
    if observed.get("status") != "PASS":
        raise AssertionError("C++ prototype did not report PASS")

    names = ref.LEDGERS
    a = F(fields["a"])
    b = F(fields["b"])
    x0 = F(fields["x0"])
    slopes = [F(fields[f"ledger.{name}.slope"]) for name in names]
    offsets = [F(fields[f"ledger.{name}.intercept"]) for name in names]
    exact_base = (a, b, slopes, offsets)
    q128_base = ref.transform_from_exact(a, b, tuple(zip(slopes, offsets)))

    samples = (1, 10, 100, 995, 1_000_000, 1_000_000_000_000)
    for n in samples:
        scope = "lifecycle-bounded" if n <= int(fields["tail_half_bound"]) else "numeric-only"
        if observed[f"case.{n}.scope"] != scope:
            raise AssertionError(f"wrong bounded/numeric scope for repeat {n}")
        composition_ops = int(observed[f"case.{n}.composition_ops"])
        if composition_ops > 2 * max(1, n.bit_length()):
            raise AssertionError(f"powering work is not logarithmic for repeat {n}")

        q128_power = ref.power_transform(q128_base, n)
        q128_x, q128_y = ref.apply_transform(q128_power,
                                              ref.Interval.from_fraction(x0))
        cpp_x = parse_interval(observed[f"case.{n}.state"])
        if cpp_x[0] > q128_bounds(q128_x)[0] or cpp_x[1] < q128_bounds(q128_x)[1]:
            raise AssertionError(f"C++ state interval fails to enclose Q128 interval for repeat {n}")
        for i, name in enumerate(names):
            cpp_y = parse_interval(observed[f"case.{n}.ledger.{name}"])
            qy = q128_bounds(q128_y[i])
            if cpp_y[0] > qy[0] or cpp_y[1] < qy[1]:
                raise AssertionError(f"C++ {name} interval fails to enclose Q128 for repeat {n}")

        if n <= 995:
            exact_x, exact_y = apply_exact(power_exact(exact_base, n), x0)
            if not contains(cpp_x, exact_x):
                raise AssertionError(f"C++ state interval misses exact Fraction result at repeat {n}")
            for i, name in enumerate(names):
                cpp_y = parse_interval(observed[f"case.{n}.ledger.{name}"])
                if not contains(cpp_y, exact_y[i]):
                    raise AssertionError(f"C++ {name} interval misses exact Fraction result at repeat {n}")

    tail_time = parse_interval(observed["case.995.ledger.time_s"])
    if tail_time[1] - tail_time[0] >= F(1, 1_000_000_000):
        raise AssertionError("995-segment elapsed interval is wider than one nanosecond")

    stop = F(int(fields["external_stop_ns"]), 1_000_000_000)
    exact123 = apply_exact(power_exact(exact_base, 123), x0)[1][names.index("time_s")]
    exact124 = apply_exact(power_exact(exact_base, 124), x0)[1][names.index("time_s")]
    if not exact123 < stop < exact124:
        raise AssertionError("exact external stop is not strictly between segment 123 and 124")
    if int(observed["external_stop.count"]) != 123:
        raise AssertionError("C++ stop search returned the wrong segment count")
    cpp_stop = parse_interval(observed["external_stop.stop_s"])
    cpp_safe = parse_interval(observed["external_stop.elapsed_safe_s"])
    cpp_next = parse_interval(observed["external_stop.elapsed_next_s"])
    if not contains(cpp_stop, stop):
        raise AssertionError("C++ stop interval misses the exact integer-nanosecond stop")
    if not (cpp_safe[1] <= stop and cpp_next[0] > stop):
        raise AssertionError("C++ stop intervals do not prove predecessor <= stop < successor")
    if not contains(cpp_safe, exact123) or not contains(cpp_next, exact124):
        raise AssertionError("C++ stop intervals miss exact Fraction elapsed values")

    if (int(observed["state_dimension"]) != 1 or
            int(observed["ledger_count"]) != len(names) or
            int(observed["event_template_steps_per_segment"]) != 32 or
            int(observed["runtime_expanded_events"]) != 0):
        raise AssertionError("state/event representation size changed with repeat scale")
    if int(fields["tail_half_bound"]) != 995:
        raise AssertionError("unexpected fixture lifecycle tail bound")

    binary = HERE / "fixed_interval_affine"
    source_lines = descriptor.read_text().splitlines()
    with TemporaryDirectory(prefix="affine-overflow-") as scratch:
        bad = Path(scratch) / "bad.affine"
        overwide = str(1 << 256)
        for mutation, label in ((f"a={overwide}/1", "256-bit input overflow"),
                                ("a=1/0", "zero denominator")):
            bad.write_text("\n".join(
                [mutation if row.startswith("a=") else row for row in source_lines]) + "\n")
            result = subprocess.run([str(binary), str(bad)], check=False,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            if result.returncode == 0 or "FAIL:" not in result.stderr:
                raise AssertionError(f"C++ prototype did not fail closed on {label}")

    print("PASS: C++ interval contains exact Fraction results for 1, 10, 100, 995")
    print("PASS: C++ intervals enclose independent Python Q128 powers through 10^12")
    print("PASS: 995-segment elapsed enclosure is < 1 ns wide")
    print("PASS: integer-ns external stop certifies 123 safe, 124 next")
    print("PASS: 1 state, 10 ledger values, 32 template steps, 0 expanded runtime events at all scales")
    print("PASS: 256-bit input overflow and zero denominator fail closed")


if __name__ == "__main__":
    main()
