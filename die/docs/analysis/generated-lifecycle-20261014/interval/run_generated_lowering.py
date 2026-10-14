#!/usr/bin/env python3
"""Lower and independently check one generated guarded affine return.

Run from the repository root:
  python3 work/generated-interval-lowering/run_generated_lowering.py
"""
from __future__ import annotations

import json
import copy
import subprocess
import tempfile
from fractions import Fraction as F
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'die/src/Execution.cc').is_file())
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "work/symbolic-lifecycle-discovery/generated-K4097.json"
SOURCE = HERE / "fixed_interval_generated.cpp"
DESCRIPTOR = HERE / "generated-return.affine"
OUTPUT = HERE / "generated-return.out"
BINARY = HERE / "fixed_interval_generated"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def affine_compose(first: tuple, second: tuple) -> tuple:
    """Apply first, then second; tuple is (a,b,c[],d[])."""
    a1, b1, c1, d1 = first
    a2, b2, c2, d2 = second
    return (
        a2 * a1,
        a2 * b1 + b2,
        [c1[i] + c2[i] * a1 for i in range(len(c1))],
        [d1[i] + c2[i] * b1 + d2[i] for i in range(len(c1))],
    )


def affine_power(base: tuple, count: int) -> tuple:
    dimensions = len(base[2])
    result = (F(1), F(0), [F(0)] * dimensions, [F(0)] * dimensions)
    while count:
        if count & 1:
            result = affine_compose(result, base)
        count >>= 1
        if count:
            base = affine_compose(base, base)
    return result


def affine_apply(transform: tuple, x0: F) -> tuple[F, list[F]]:
    a, b, c, d = transform
    return a * x0 + b, [c[i] * x0 + d[i] for i in range(len(c))]


def max_reference(life: dict) -> int:
    waves = [life["next_load"]]
    waves.extend(item["wave"] for item in life["active"].values())
    waves.extend(life["loaded"])
    waves.extend(life["computed"])
    require(bool(waves), "lifecycle record has no reference waves")
    return max(waves)


def lifecycle_bound(lifecycles: list[dict], shifts: list[int],
                    first_nonfull: int) -> int:
    bounds = []
    for life, shift in zip(lifecycles, shifts, strict=True):
        reference = max_reference(life)
        require(shift > 0, "lifecycle wave translation must be positive")
        require(reference < first_nonfull,
                "lifecycle already references the first non-full wave")
        bounds.append((first_nonfull - 1 - reference) // shift)
    return min(bounds)


def audit_guards(record: dict) -> tuple[F, int]:
    """Recompute the rational interval intersection and check every guard."""
    guards = record.get("pulled_back_guards")
    require(isinstance(guards, list) and len(guards) == record["guard_count"],
            "pulled-back guard array is absent or disagrees with guard_count")
    lower: F | None = None
    upper: F | None = None
    lower_strict = False
    upper_strict = False

    def set_lower(value: F, strict: bool) -> None:
        nonlocal lower, lower_strict
        if lower is None or value > lower:
            lower, lower_strict = value, strict
        elif value == lower:
            lower_strict = lower_strict or strict

    def set_upper(value: F, strict: bool) -> None:
        nonlocal upper, upper_strict
        if upper is None or value < upper:
            upper, upper_strict = value, strict
        elif value == upper:
            upper_strict = upper_strict or strict

    parsed: list[tuple[F, F, str]] = []
    for index, guard in enumerate(guards):
        kind = guard["kind"]
        require(kind in ("ge", "gt", "eq"), f"guard {index} has unknown kind")
        slope = F(guard["affine"]["slope"])
        intercept = F(guard["affine"]["intercept"])
        parsed.append((slope, intercept, kind))
        if kind == "eq":
            if slope == 0:
                require(intercept == 0, f"constant equality guard {index} is false")
            else:
                point = -intercept / slope
                set_lower(point, False)
                set_upper(point, False)
            continue
        if slope == 0:
            require(intercept >= 0 if kind == "ge" else intercept > 0,
                    f"constant guard {index} is false")
            continue
        boundary = -intercept / slope
        strict = kind == "gt"
        if slope > 0:
            set_lower(boundary, strict)
        else:
            set_upper(boundary, strict)

    require(lower is None or upper is None or lower < upper or
            (lower == upper and not (lower_strict or upper_strict)),
            "exact pulled-back guards have an empty aggregate domain")
    declared = record["domain"]
    exact_domain = [str(lower) if lower is not None else None,
                    str(upper) if upper is not None else None]
    require(declared == exact_domain,
            "recorded aggregate domain differs from independently recomputed facets")
    require(record["domain_endpoint_strict"] ==
            [lower_strict, upper_strict],
            "recorded domain strictness differs from guard-derived facets")

    a = F(record["map"]["slope"])
    b = F(record["map"]["intercept"])
    x0 = F(record["initial_state"])
    fixed = F(record["fixed_point"])
    require(0 < a < 1, "guard audit requires a positive contraction")
    require(b / (1 - a) == fixed,
            "exact fixed point differs from b/(1-a)")
    orbit_lo, orbit_hi = min(x0, fixed), max(x0, fixed)
    require(record["invariant_orbit"] == [str(orbit_lo), str(orbit_hi)],
            "recorded invariant orbit differs from exact endpoint hull")
    strict_margins: list[F] = []
    for index, (slope, intercept, kind) in enumerate(parsed):
        values = (slope * orbit_lo + intercept,
                  slope * orbit_hi + intercept)
        if kind == "eq":
            require(values[0] == 0 and values[1] == 0,
                    f"equality guard {index} does not hold across orbit")
        elif kind == "ge":
            require(min(values) >= 0,
                    f"non-strict guard {index} fails on orbit")
        else:
            margin = min(values)
            require(margin > 0, f"strict guard {index} fails on orbit")
            strict_margins.append(margin)
    min_strict_margin = min(strict_margins) if strict_margins else F(0)
    return min_strict_margin, sum(kind == "eq" for _, _, kind in parsed)


def audit_geometry(certificate: dict, record: dict) -> dict:
    geometry = certificate["geometry"]
    m, n, k = geometry["M"], geometry["N"], geometry["K"]
    tm, tn, tk = geometry["tm"], geometry["tn"], geometry["tk"]
    mt = (m + tm - 1) // tm
    nt = (n + tn - 1) // tn
    kt = (k + tk - 1) // tk
    tails = []
    if n % tn:
        tails.append((nt - 1) * kt)
    if m % tm:
        tails.append((mt - 1) * nt * kt)
    first_nonfull = min(tails, default=mt * nt * kt)
    require(first_nonfull ==
            record["all_full_MN_bound"]["first_nonfull_wave"],
            "geometry-derived first non-full wave differs from return record")
    require(record["shift_waves"] == [kt] * len(record["shift_waves"]),
            "return translation does not match the full-K wave count per owner")
    return {"M": m, "N": n, "K": k, "mt": mt, "nt": nt, "kt": kt,
            "first_nonfull_wave": first_nonfull}


def check_equality_guard_edges(record: dict,
                               expected_margin: F) -> None:
    """Exercise the equality guard path with synthetic 0=0 and reject 0=1."""
    valid = copy.deepcopy(record)
    valid["pulled_back_guards"].append({
        "affine": {"slope": "0", "intercept": "0"}, "kind": "eq"})
    valid["guard_count"] += 1
    margin, count = audit_guards(valid)
    require(margin == expected_margin and count == 1,
            "zero equality guard was not recognized as an exact identity")
    invalid = copy.deepcopy(record)
    invalid["pulled_back_guards"].append({
        "affine": {"slope": "0", "intercept": "1"}, "kind": "eq"})
    invalid["guard_count"] += 1
    try:
        audit_guards(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("false constant equality guard was accepted")


def validate_lifecycle(record: dict) -> tuple[int, int, int, int]:
    initial = record["initial_lifecycle"]
    ending = record["end_lifecycle"]
    shifts = record["shift_waves"]
    retired_outputs = record["retired_outputs"]
    require(len(initial) == len(ending) == len(shifts) == len(retired_outputs),
            "lifecycle owner vectors have different lengths")
    for owner, (before, after, shift, retired) in enumerate(
            zip(initial, ending, shifts, retired_outputs, strict=True)):
        require(after["next_load"] == before["next_load"] + shift,
                f"owner {owner} next_load does not advance by its wave translation")
        require(after["retired"] == before["retired"] + retired,
                f"owner {owner} retired cursor does not match generated return")
        require(list(before["active"]) == list(after["active"]),
                f"owner {owner} active stage order changed")
        for stage in before["active"]:
            old, new = before["active"][stage], after["active"][stage]
            require(new["wave"] == old["wave"] + shift,
                    f"owner {owner} active stage {stage} wave does not translate")
        for category in ("loaded", "computed"):
            require(len(before[category]) == len(after[category]),
                    f"owner {owner} {category} queue shape changed")
            require(after[category] == [wave + shift for wave in before[category]],
                    f"owner {owner} {category} queue does not translate")

    first_nonfull = record["all_full_MN_bound"]["first_nonfull_wave"]
    initial_max = max(max_reference(owner) for owner in initial)
    end_max = max(max_reference(owner) for owner in ending)
    initial_bound = lifecycle_bound(initial, shifts, first_nonfull)
    end_bound = lifecycle_bound(ending, shifts, first_nonfull)
    declared_end = record["all_full_MN_bound"]["additional_complete_returns"]
    require(end_bound == declared_end,
            "derived end-state lookahead differs from discovery metadata")
    declared_end_next_load = record["all_full_MN_bound"][
        "maximum_reference_wave_at_end"]
    require(declared_end_next_load == [owner["next_load"] for owner in ending],
            "end-state next_load values differ from discovery metadata")
    require(any(max_reference(owner) + (initial_bound + 1) * shift >=
                first_nonfull for owner, shift in zip(initial, shifts, strict=True)),
            "initial lookahead bound is not maximal")
    require(any(max_reference(owner) + (end_bound + 1) * shift >=
                first_nonfull for owner, shift in zip(ending, shifts, strict=True)),
            "end-state lookahead bound is not maximal")
    return initial_max, end_max, initial_bound, end_bound


def fields_for_lifecycle(fields: list[str], phase: str,
                         lifecycles: list[dict]) -> None:
    for owner, life in enumerate(lifecycles):
        prefix = f"lifecycle.{phase}.{owner}."
        active = sorted(life["active"].items(), key=lambda item: int(item[0]))
        fields.extend([
            f"{prefix}next_load={life['next_load']}",
            f"{prefix}retired={life['retired']}",
            f"{prefix}active_count={len(active)}",
            f"{prefix}loaded_count={len(life['loaded'])}",
            f"{prefix}computed_count={len(life['computed'])}",
        ])
        for i, (stage, value) in enumerate(active):
            item = f"{prefix}active.{i}."
            fields.extend([
                f"{item}stage={stage}",
                f"{item}wave={value['wave']}",
            ])
        for category in ("loaded", "computed"):
            for i, wave in enumerate(life[category]):
                fields.append(f"{prefix}{category}.{i}.wave={wave}")


def descriptor_for(record: dict, geometry: dict) -> tuple[str, dict]:
    initial_max, end_max, tail, additional = validate_lifecycle(record)
    min_guard_margin, equality_guards = audit_guards(record)
    map_a = F(record["map"]["slope"])
    map_b = F(record["map"]["intercept"])
    x0 = F(record["initial_state"])
    fixed = F(record["fixed_point"])
    domain_lo, domain_hi = map(F, record["domain"])
    require(record["domain_endpoint_strict"] == [True, True],
            "generated guard domain endpoints are not strict")
    require(0 < map_a < 1, "generated state map is not a positive contraction")
    require(map_b / (1 - map_a) == fixed,
            "reported fixed point differs from exact b/(1-a)")
    require(domain_lo < x0 < domain_hi and domain_lo < fixed < domain_hi,
            "initial point or fixed point is outside strict guard domain")
    orbit_lo, orbit_hi = sorted((x0, fixed))
    require(domain_lo < orbit_lo <= orbit_hi < domain_hi,
            "exact invariant orbit hull is outside strict guard domain")
    x1 = map_a * x0 + map_b
    if x0 < fixed:
        require(x0 < x1 < fixed, "exact orbit does not increase toward fixed point")
    elif x0 > fixed:
        require(fixed < x1 < x0, "exact orbit does not decrease toward fixed point")

    coordinate = record["coordinate"]
    selected_stage = str(coordinate["stage"])
    selected_owner = coordinate["owner"]
    initial_progress = record["initial_lifecycle"][selected_owner]["active"][
        selected_stage]["progress"]
    end_progress = record["end_lifecycle"][selected_owner]["active"][
        selected_stage]["progress"]
    require(F(initial_progress["slope"]) == 1 and
            F(initial_progress["intercept"]) == 0,
            "declared state coordinate is not the initial active progress")
    require(F(end_progress["slope"]) == map_a and
            F(end_progress["intercept"]) == map_b,
            "end lifecycle does not return the generated affine state map")

    ledger_names = list(record["ledger"])
    require(0 < len(ledger_names) <= 256,
            "generated ledger dimension is outside the bounded descriptor cap")
    require("elapsed" in ledger_names, "generated ledgers have no elapsed-time value")
    ledger_slope = [F(record["ledger"][name]["slope"]) for name in ledger_names]
    ledger_intercept = [F(record["ledger"][name]["intercept"])
                        for name in ledger_names]
    exact_base = (map_a, map_b, ledger_slope, ledger_intercept)

    # Elapsed time is the only ledger that must be strictly positive for stop
    # search. Other ledgers permit zero increments and negative slopes.
    for name, slope, intercept in zip(
            ledger_names, ledger_slope, ledger_intercept, strict=True):
        ends = [slope * x0 + intercept, slope * fixed + intercept]
        require(min(ends) >= 0,
                f"ledger {name} has a negative increment on the invariant orbit")
        if name == "elapsed":
            require(min(ends) > 0,
                    "elapsed-time increment is not strictly positive")

    # Pick an integer-nanosecond stop strictly between exact cumulative 123/124.
    t123 = affine_apply(affine_power(exact_base, 123), x0)[1][ledger_names.index("elapsed")]
    t124 = affine_apply(affine_power(exact_base, 124), x0)[1][ledger_names.index("elapsed")]
    require(t123 < t124, "exact elapsed time is not increasing")
    midpoint_ns = (t123 + t124) * 1_000_000_000 / 2
    stop_ns = midpoint_ns.numerator // midpoint_ns.denominator
    if F(stop_ns, 1_000_000_000) <= t123:
        stop_ns = (t123.numerator * 1_000_000_000) // t123.denominator + 1
    require(t123 < F(stop_ns, 1_000_000_000) < t124,
            "no integer-nanosecond stop strictly fits between 123 and 124")

    fields = [
        "version=1",
        "state_dimension=1",
        f"ledger_count={len(ledger_names)}",
        f"event_template_steps={record['representative_steps']}",
        f"guard_count={record['guard_count']}",
        f"lifecycle_owner_count={len(record['initial_lifecycle'])}",
        f"first_nonfull_wave={record['all_full_MN_bound']['first_nonfull_wave']}",
        "declared_additional_returns="
        f"{record['all_full_MN_bound']['additional_complete_returns']}",
        "time_ledger_name=elapsed",
        f"external_stop_ns={stop_ns}",
        f"a={record['map']['slope']}",
        f"b={record['map']['intercept']}",
        f"x0={record['initial_state']}",
        f"fixed={record['fixed_point']}",
        f"domain_lo={record['domain'][0]}",
        f"domain_hi={record['domain'][1]}",
        "domain_lo_strict=1",
        "domain_hi_strict=1",
    ]
    for owner, (shift, retired) in enumerate(zip(
            record["shift_waves"], record["retired_outputs"], strict=True)):
        fields.extend([
            f"owner.{owner}.shift={shift}",
            f"owner.{owner}.retired_output={retired}",
        ])
    fields_for_lifecycle(fields, "initial", record["initial_lifecycle"])
    fields_for_lifecycle(fields, "end", record["end_lifecycle"])
    for i, name in enumerate(ledger_names):
        fields.extend([
            f"ledger.{i}.name={name}",
            f"ledger.{i}.slope={record['ledger'][name]['slope']}",
            f"ledger.{i}.intercept={record['ledger'][name]['intercept']}",
        ])
    return "\n".join(fields) + "\n", {
        "record": record,
        "tail": tail,
        "additional": additional,
        "initial_max": initial_max,
        "end_max": end_max,
        "stop_ns": stop_ns,
        "stop_t123": t123,
        "stop_t124": t124,
        "exact_base": exact_base,
        "x0": x0,
        "ledger_names": ledger_names,
        "min_guard_margin": min_guard_margin,
        "equality_guards": equality_guards,
        "geometry": geometry,
    }


def read_output(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        require(key not in output, f"duplicate C++ output key {key}")
        output[key] = value
    return output


def parse_interval(text: str) -> tuple[F, F]:
    low, high = text.split("|", 1)
    return F(low), F(high)


def exact_contains(interval: tuple[F, F], value: F) -> bool:
    return interval[0] <= value <= interval[1]


def check_result(metadata: dict, observed: dict[str, str]) -> None:
    record = metadata["record"]
    tail = metadata["tail"]
    require(observed.get("status") == "PASS", "C++ lowering did not pass")
    require(int(observed["state_dimension"]) == 1,
            "state dimension changed")
    require(int(observed["ledger_count"]) == len(metadata["ledger_names"]),
            "ledger dimension changed")
    require(int(observed["event_template_steps_per_return"]) ==
            record["representative_steps"] <= 256,
            "event template count changed or exceeds fixed cap")
    require(int(observed["guard_count"]) == record["guard_count"],
            "guard count changed")
    require(int(observed["runtime_expanded_events"]) == 0,
            "lowering expanded runtime events")
    require(int(observed["initial_max_reference_wave"]) == metadata["initial_max"],
            "initial lifecycle maximum reference wave changed")
    require(int(observed["end_max_reference_wave"]) == metadata["end_max"],
            "end lifecycle maximum reference wave changed")
    require(int(observed["initial_lifecycle_tail_bound"]) == tail,
            "initial-state lifecycle tail bound is wrong")
    require(int(observed["end_additional_complete_returns"]) ==
            metadata["additional"],
            "end-state additional return bound is wrong")
    require(int(observed["tail_successor_count"]) == tail + 1,
            "tail successor count is wrong")

    names = metadata["ledger_names"]
    exact_base = metadata["exact_base"]
    x0 = metadata["x0"]
    for n in (1, 10, 100, tail, 1_000_000, 1_000_000_000_000):
        if f"case.{n}.scope" not in observed:
            continue
        scope = "lifecycle-bounded" if n <= tail else "numeric-only"
        require(observed[f"case.{n}.scope"] == scope,
                f"wrong lifecycle scope at repeat {n}")
        operations = int(observed[f"case.{n}.composition_ops"])
        require(operations <= 2 * max(1, n.bit_length()),
                f"affine powering is not logarithmic at repeat {n}")
        if n > tail:
            continue
        exact_x, exact_y = affine_apply(affine_power(exact_base, n), x0)
        cpp_x = parse_interval(observed[f"case.{n}.state"])
        require(exact_contains(cpp_x, exact_x),
                f"state interval misses exact Fraction at repeat {n}")
        for i, name in enumerate(names):
            cpp_y = parse_interval(observed[f"case.{n}.ledger.{name}"])
            require(exact_contains(cpp_y, exact_y[i]),
                    f"{name} interval misses exact Fraction at repeat {n}")

    width = parse_interval(observed["tail.elapsed_width_s"])
    require(width[1] < F(1, 1_000_000_000),
            "tail elapsed interval width is at least one nanosecond")

    require(int(observed["external_stop.count"]) == 123,
            "external stop did not select 123 full segments")
    exact_stop = F(metadata["stop_ns"], 1_000_000_000)
    stop = parse_interval(observed["external_stop.stop_s"])
    safe = parse_interval(observed["external_stop.elapsed_safe_s"])
    next_time = parse_interval(observed["external_stop.elapsed_next_s"])
    residual = parse_interval(observed["external_stop.partial_residual_s"])
    require(exact_contains(stop, exact_stop), "stop interval misses exact integer ns")
    require(safe[1] <= stop[0] and next_time[0] > stop[1],
            "interval comparison does not prove safe <= stop < next")
    require(exact_contains(safe, metadata["stop_t123"]),
            "safe elapsed interval misses exact Fraction t123")
    require(exact_contains(next_time, metadata["stop_t124"]),
            "next elapsed interval misses exact Fraction t124")
    require(exact_contains(residual, exact_stop - metadata["stop_t123"]),
            "partial residual interval misses exact Fraction residual")
    require(observed["external_stop.partial_handler"] == "ordinary_integrator",
            "partial residual was not routed to the ordinary integrator")
    require(int(observed["external_stop.search_composition_ops"]) <=
            4 * max(1, tail.bit_length()),
            "external-stop search exceeded logarithmic composition work")


def check_zero_ledger_is_permitted(metadata: dict) -> None:
    """Exercise a genuinely zero additive ledger without changing source data."""
    rows = DESCRIPTOR.read_text().splitlines()
    names = metadata["ledger_names"]
    zero_name = next(name for name in names if name != "elapsed")
    zero_index = names.index(zero_name)
    fields = {}
    for row in rows:
        key, value = row.split("=", 1)
        fields[key] = value
    fields[f"ledger.{zero_index}.slope"] = "0"
    fields[f"ledger.{zero_index}.intercept"] = "0"
    with tempfile.TemporaryDirectory(prefix="zero-ledger-", dir=HERE) as scratch:
        zero_descriptor = Path(scratch) / "zero-ledger-probe.affine"
        zero_output = Path(scratch) / "zero-ledger-probe.out"
        zero_descriptor.write_text(
            "".join(f"{key}={value}\n" for key, value in fields.items()))
        with zero_output.open("w") as stream:
            subprocess.run([str(BINARY), str(zero_descriptor)],
                           check=True, stdout=stream, text=True)
        probe = read_output(zero_output)
        require(probe.get("status") == "PASS",
                "C++ rejected an exactly zero non-time ledger")
        zero_interval = parse_interval(
            probe[f"case.{metadata['tail']}.ledger.{zero_name}"])
        require(exact_contains(zero_interval, F(0)),
                "zero-ledger interval does not contain exact zero")


def main() -> None:
    raw = json.loads(INPUT.read_text())
    require(raw["status"] == "failed_closed",
            "discovery status changed; this task lowers only its generated return")
    require(raw["events"] == 256,
            "expected the fail-closed 256-event discovery snapshot")
    returns = raw.get("generated_returns", [])
    require(bool(returns), "discovery did not emit a generated return")
    record = returns[0]
    require(record["kind"] == "generated_guarded_output_return_not_lowered",
            "first record is not the requested guarded generated return")
    certificate = json.loads(
        (ROOT / "work/symbolic-lifecycle-discovery/first-return-certificate.json")
        .read_text())
    require(certificate.get("record") == record,
            "standalone first-return certificate differs from generated record")
    geometry = audit_geometry(certificate, record)
    descriptor, metadata = descriptor_for(record, geometry)
    check_equality_guard_edges(record, metadata["min_guard_margin"])
    DESCRIPTOR.write_text(descriptor)
    subprocess.run([
        "c++", "-std=c++17", "-O2", "-frounding-math",
        "-fno-fast-math", "-ffp-contract=off",
        str(SOURCE), "-o", str(BINARY),
    ], check=True)
    with OUTPUT.open("w") as stream:
        subprocess.run([str(BINARY), str(DESCRIPTOR)], check=True,
                       stdout=stream, text=True)
    observed = read_output(OUTPUT)
    check_result(metadata, observed)
    require(any(F(value["slope"]) < 0 for value in record["ledger"].values()),
            "generated return no longer exercises negative ledger slopes")
    check_zero_ledger_is_permitted(metadata)

    print("PASS: generated record is a strict one-coordinate contraction return")
    print(f"PASS: Fraction rebuilt all {record['guard_count']} guard facets; "
          f"checked every guard over invariant orbit (eq={metadata['equality_guards']})")
    print(f"PASS: geometry M/N/K={geometry['M']}/{geometry['N']}/{geometry['K']}, "
          f"K waves/return={geometry['kt']}, first non-full wave="
          f"{geometry['first_nonfull_wave']}")
    print(f"PASS: minimum strict guard margin is "
          f"{metadata['min_guard_margin']} ({float(metadata['min_guard_margin']):.9g})")
    print(f"PASS: lifecycle derived initial bound {metadata['tail']}; successor crosses "
          f"wave {record['all_full_MN_bound']['first_nonfull_wave']}")
    print(f"PASS: end-state additional-complete-return bound independently derives to "
          f"{metadata['additional']}")
    print(f"PASS: exact Fraction containment for state plus all {len(metadata['ledger_names'])} "
          f"ledgers at 1, 10, 100, {metadata['tail']}")
    print(f"PASS: {metadata['tail']}-segment elapsed enclosure width is "
          f"{float(parse_interval(observed['tail.elapsed_width_s'])[1] * 1e9):.6g} ns")
    print(f"PASS: integer stop {metadata['stop_ns']} ns proves exact 123 safe, 124 next")
    print("PASS: partial residual is bounded and assigned to the ordinary integrator")
    print("PASS: repeat powering and external-stop search use logarithmic composition counts")
    print("PASS: 10^6 and 10^12 outputs are numeric-only beyond the lifecycle bound")
    print("PASS: negative ledger slopes retained; an exact zero non-time ledger is accepted")
    print("PASS: one state, "
          f"{len(metadata['ledger_names'])} ledgers, "
          f"{record['representative_steps']} template steps, "
          f"{record['guard_count']} guard records, zero expanded runtime events")
    print(f"Descriptor: {DESCRIPTOR.relative_to(ROOT)}")
    print(f"Output: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
