#!/usr/bin/env python3
"""Exact symbolic validation of one 520-event grouped-qlocal output cycle."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path

sys.set_int_max_str_digits(0)

HERE = Path(__file__).resolve().parent
EVENT_TRACE = HERE / "affine-trace-complete-cycle.txt"
BOUNDARY_TRACE = HERE / "scale-r1e6-cycle-trace.txt"


def frac(s: str) -> F:
    if "/" not in s:
        return F(int(s), 1)
    a, b = s.split("/", 1)
    return F(int(a), int(b))


def fstr(x: F) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def fbits(x: F) -> tuple[int, int]:
    return (abs(x.numerator).bit_length(), x.denominator.bit_length())


@dataclass(frozen=True)
class Aff:
    a: F = F(0)
    b: F = F(0)

    def __add__(self, other: "Aff | F | int") -> "Aff":
        if isinstance(other, Aff):
            return Aff(self.a + other.a, self.b + other.b)
        q = F(other)
        return Aff(self.a, self.b + q)

    __radd__ = __add__

    def __sub__(self, other: "Aff | F | int") -> "Aff":
        if isinstance(other, Aff):
            return Aff(self.a - other.a, self.b - other.b)
        q = F(other)
        return Aff(self.a, self.b - q)

    def __rsub__(self, other: "Aff | F | int") -> "Aff":
        if isinstance(other, Aff):
            return other.__sub__(self)
        q = F(other)
        return Aff(-self.a, q - self.b)

    def __mul__(self, other: F | int) -> "Aff":
        q = F(other)
        return Aff(self.a * q, self.b * q)

    __rmul__ = __mul__

    def __truediv__(self, other: F | int) -> "Aff":
        q = F(other)
        return Aff(self.a / q, self.b / q)

    def at(self, x: F) -> F:
        return self.a * x + self.b

    def as_json(self) -> dict[str, str]:
        return {"slope": fstr(self.a), "intercept": fstr(self.b)}


def parse_tokens(line: str) -> dict[str, str]:
    return dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)


def boundary_states() -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for line in BOUNDARY_TRACE.read_text().splitlines():
        if not line.startswith("trace probe="):
            continue
        row = parse_tokens(line)
        if row.get("scenario") == "scale_two_offset_005" and row.get("probe") in {"5", "9", "13"}:
            found[row["probe"]] = row
    if set(found) != {"5", "9", "13"}:
        raise AssertionError(f"missing boundary probes: {sorted(found)}")
    return found


def progress_at(row: dict[str, str], key: str) -> F:
    value = row[key]
    return frac(value.split("@", 1)[1].split("|", 1)[0])


def fit_line(xs: list[F], ys: list[F]) -> Aff:
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    intercept = ys[0] - slope * xs[0]
    if any(slope * x + intercept != y for x, y in zip(xs, ys)):
        raise AssertionError("three exact boundary samples are not collinear")
    return Aff(slope, intercept)


ACTIVITY_RE = re.compile(r"^a(\d+)\.(\d+)\.(\d+)=([^| ]+)\|([^| ]+)\|([^| ]+)\|q([^ ]+)$")

# The affine trace was generated from this full-category segment of the strict
# FP32 GEMM fixture. Keep its geometry explicit so rate/Q validation rebuilds
# wave work from dimensions, never from the trace's logged Q/rates.
FIXTURE_M, FIXTURE_N, FIXTURE_K = 15985, 63937, 4097
TM, TN, TK = 16, 64, 64
COMPUTE_DENOM = 158216000
CAPACITY = (F(1), F(50_000_000), F(50_000_000))
K_TILES = (FIXTURE_K + TK - 1) // TK
N_TILES = (FIXTURE_N + TN - 1) // TN
M_TILES = (FIXTURE_M + TM - 1) // TM
WAVE_COUNT = M_TILES * N_TILES * K_TILES


def read_event_rows() -> tuple[list[dict], dict[str, str]]:
    rows = []
    endpoint = None
    for line in EVENT_TRACE.read_text().splitlines():
        if line.startswith("traceevent "):
            t = parse_tokens(line)
            acts = {}
            for token in line.split()[1:]:
                match = ACTIVITY_RE.match(token)
                if not match:
                    continue
                owner, stage, wave, p, rate, deadline, q = match.groups()
                ident = (int(owner), int(stage), int(wave))
                acts[ident] = {"p": frac(p), "rate": frac(rate),
                               "deadline": frac(deadline), "q": frac(q)}
            selected = [] if not t.get("selected") else [
                tuple(map(int, s.split("."))) for s in t["selected"].split(",")
            ]
            rows.append({"tokens": t, "activities": acts, "selected": selected,
                         "dt": frac(t["dt"]), "basis": t["basis"]})
        elif line.startswith("traceendpoint "):
            endpoint = parse_tokens(line)
    if endpoint is None:
        raise AssertionError("missing trace endpoint")
    return rows, endpoint


def row0_partition(total: int, prefix: int) -> int:
    """Independent integer form of ordered row splitting."""
    return (prefix + total + 1) // 2 - (prefix + 1) // 2


def wave_facts(wave_index: int) -> dict:
    if not 0 <= wave_index < WAVE_COUNT:
        raise AssertionError(f"wave outside fixture: {wave_index}")
    output, k_index = divmod(wave_index, K_TILES)
    mt, nt = divmod(output, N_TILES)
    m = min(TM, FIXTURE_M - mt * TM)
    n = min(TN, FIXTURE_N - nt * TN)
    k = min(TK, FIXTURE_K - k_index * TK)
    final_k = k_index + 1 == K_TILES
    read = 4 * k * (m + n)
    write = 4 * m * n if final_k else 0
    vec_ops = 2 * m * n * k
    accumulator = 4 * m * n
    local = read + accumulator + (accumulator if k_index > 0 else 0)

    # Correct complete-M-row prefix. The C++ prototype currently contains
    # TN*TM here instead of N_TILES*TM. For this FP32 fixture all payloads and
    # prefixes are multiples of four, so the row split is exactly half either
    # way; the separate report records the source limitation.
    n_before = nt * TN
    area_before = mt * (N_TILES * TM + FIXTURE_N) + nt * (m + TN)
    read_prefix = 4 * FIXTURE_K * area_before + 4 * (m + n) * (k_index * TK)
    write_prefix = 4 * (mt * TM * FIXTURE_N + m * n_before)
    read0 = row0_partition(read, read_prefix)
    write0 = row0_partition(write, write_prefix)
    return {
        "output": output, "k_index": k_index, "m": m, "n": n, "k": k,
        "final_k": final_k, "read": read, "write": write,
        "read_rows": (read0, read - read0),
        "write_rows": (write0, write - write0),
        "vec_ops": vec_ops, "local": local,
    }


def independently_solve_grouped_lp(activities: dict) -> tuple[dict, dict, list[str]]:
    """Exact progressive max-min LP, rebuilt from each wave's physical work/Q.

    At each lexicographic tier, raise every unfrozen activity by the same
    owner-alpha increment divided by that owner's active canonical-Q sum.
    Resource and compute-private constraints are solved as exact Fraction LP
    bounds. This solver receives only activity IDs and wave geometry; it does
    not consume any trace rate, Q, basis, or deadline.
    """
    q_by_id: dict[tuple[int, int, int], F] = {}
    work_by_id: dict[tuple[int, int, int], tuple[F, F, F]] = {}
    private_by_id: dict[tuple[int, int, int], F] = {}
    owner_q: dict[int, F] = {}
    for ident in activities:
        owner, stage, wave = ident
        facts = wave_facts(wave)
        compute = F(facts["vec_ops"], COMPUTE_DENOM)
        if stage == 0:
            q = F(facts["read"])
            work = (F(0), F(facts["read_rows"][0]), F(facts["read_rows"][1]))
            private = F(0)
        elif stage == 1:
            q = F(facts["local"])
            work = (compute, F(0), F(0))
            private = compute
        elif stage == 2:
            q = F(facts["write"])
            work = (F(0), F(facts["write_rows"][0]), F(facts["write_rows"][1]))
            private = F(0)
        else:
            raise AssertionError(f"invalid stage {stage}")
        if q <= 0:
            raise AssertionError(f"zero canonical Q for traced activity {ident}")
        q_by_id[ident] = q
        work_by_id[ident] = work
        private_by_id[ident] = private
        owner_q[owner] = owner_q.get(owner, F(0)) + q

    rates = {ident: F(0) for ident in activities}
    free = set(activities)
    basis = []
    for _ in range(len(activities)):
        if not free:
            break
        used = [sum(work_by_id[i][r] * rates[i] for i in activities)
                for r in range(3)]
        candidates: dict[tuple[str, object], F] = {}
        for resource in range(3):
            slope = sum(work_by_id[i][resource] / owner_q[i[0]] for i in free)
            if slope > 0:
                candidates[("resource", resource)] = (CAPACITY[resource] - used[resource]) / slope
        for ident in free:
            private_time = private_by_id[ident]
            if private_time > 0:
                candidates[("private", ident)] = (
                    (1 / private_time - rates[ident]) * owner_q[ident[0]])
        if not candidates:
            raise AssertionError("independent LP has no finite tier")
        step = min(candidates.values())
        if step < 0:
            raise AssertionError("independent LP starts over a constraint")
        limiting_resources = {key[1] for key, value in candidates.items()
                              if key[0] == "resource" and value == step}
        private_limiters = set()
        for key, value in candidates.items():
            if key[0] == "private" and value == step:
                private_limiters.add(key[1])
        for ident in free:
            rates[ident] += step / owner_q[ident[0]]
        freeze = set(private_limiters)
        tier_names = []
        names = ("COMPUTE_GATE", "HBM0", "HBM1")
        for resource in sorted(limiting_resources):
            tier_names.append(names[resource])
            freeze.update(i for i in free if work_by_id[i][resource] > 0)
        if private_limiters:
            tier_names.append("PRIVATE")
        if not freeze:
            raise AssertionError("independent LP tier freezes no activity")
        basis.append("+".join(tier_names))
        free.difference_update(freeze)
    if free:
        raise AssertionError(f"independent LP left free activities: {free}")
    for resource in range(3):
        served = sum(work_by_id[i][resource] * rates[i] for i in activities)
        if served > CAPACITY[resource]:
            raise AssertionError(f"independent LP violates resource {resource}: {served}")
    for ident, private_time in private_by_id.items():
        if private_time > 0 and rates[ident] > 1 / private_time:
            raise AssertionError(f"independent LP violates private compute limit: {ident}")
    return rates, q_by_id, basis


class LifecycleOwner:
    def __init__(self, next_load: int, retired: int, active: dict[int, int],
                 loaded: list[int], computed: list[int]):
        self.next_load = next_load
        self.retired = retired
        self.active = active
        self.loaded = loaded
        self.computed = computed


def lifecycle_occupancy(owner: LifecycleOwner) -> tuple[int, int]:
    input_slots = len(owner.loaded) + int(0 in owner.active) + int(1 in owner.active)
    active_final_compute = (int(owner.active[1] % K_TILES == K_TILES - 1)
                            if 1 in owner.active else 0)
    output_slots = len(owner.computed) + int(2 in owner.active) + active_final_compute
    return input_slots, output_slots


def check_lifecycle_state(states: list[LifecycleOwner], row_no: int,
                          expected_active: set[tuple[int, int, int]],
                          expected_retired: tuple[int, int]) -> None:
    actual_active = {(owner, stage, wave)
                     for owner, state in enumerate(states)
                     for stage, wave in state.active.items()}
    if actual_active != expected_active:
        raise AssertionError(f"event {row_no}: lifecycle membership mismatch: "
                             f"{actual_active} != {expected_active}")
    if tuple(state.retired for state in states) != expected_retired:
        raise AssertionError(f"event {row_no}: lifecycle retired mismatch")
    for owner_id, state in enumerate(states):
        if len(state.active) > 3 or len(state.loaded) > 2 or len(state.computed) > 2:
            raise AssertionError(f"event {row_no}: owner {owner_id} exceeds fixed slot bound")
        input_slots, output_slots = lifecycle_occupancy(state)
        if input_slots > 2 or output_slots > 2:
            raise AssertionError(f"event {row_no}: owner {owner_id} exceeds slot occupancy")


def advance_lifecycle(states: list[LifecycleOwner], selected: list[tuple[int, int, int]]) -> None:
    selected_set = set(selected)
    # Match the L, then C, then S completion queues used by the runtime model.
    for stage in (0, 1, 2):
        for owner_id, state in enumerate(states):
            ident = next((item for item in selected_set if item[0] == owner_id and item[1] == stage), None)
            if ident is None:
                continue
            _, _, wave = ident
            if state.active.get(stage) != wave:
                raise AssertionError(f"selected completion absent from owner state: {ident}")
            del state.active[stage]
            if stage == 0:
                state.loaded.append(wave)
            elif stage == 1:
                if wave % K_TILES == K_TILES - 1:
                    state.computed.append(wave)
            else:
                if wave % K_TILES != K_TILES - 1:
                    raise AssertionError(f"non-K-final store retires output: {wave}")
                state.retired += 1

    # Settle the zero-progress pipeline starts to the source's fixed point.
    while True:
        changed = False
        for state in states:
            input_slots, output_slots = lifecycle_occupancy(state)
            if 0 not in state.active and state.next_load < WAVE_COUNT and input_slots < 2:
                state.active[0] = state.next_load
                state.next_load += 1
                changed = True
            input_slots, output_slots = lifecycle_occupancy(state)
            if 1 not in state.active and state.loaded and output_slots < 2:
                state.active[1] = state.loaded.pop(0)
                changed = True
            if 2 not in state.active and state.computed:
                state.active[2] = state.computed.pop(0)
                changed = True
        if not changed:
            return


def add_geq_zero(domain: list[F | None], expr: Aff) -> None:
    """Intersect [lo,hi] with expr(x)>=0; None denotes infinity."""
    if expr.a == 0:
        if expr.b < 0:
            raise AssertionError(f"empty feasible trace region: {expr}")
        return
    cut = -expr.b / expr.a
    if expr.a > 0:
        domain[0] = cut if domain[0] is None else max(domain[0], cut)
    else:
        domain[1] = cut if domain[1] is None else min(domain[1], cut)
    if domain[0] is not None and domain[1] is not None and domain[0] > domain[1]:
        raise AssertionError("event inequalities have empty intersection")


def main() -> None:
    bstates = boundary_states()
    probes = ("5", "9", "13")
    xs = [progress_at(bstates[p], "owner1_C") for p in probes]
    owner0_cs = [progress_at(bstates[p], "owner0_C") for p in probes]
    owner1_ls = [progress_at(bstates[p], "owner1_L") for p in probes]
    phase_map = fit_line(xs[:2], xs[1:])
    # fit_line on adjacent sample pairs gives the exact cycle map; verify both.
    phase_map_2 = Aff((xs[2] - xs[1]) / (xs[1] - xs[0]),
                      xs[1] - ((xs[2] - xs[1]) / (xs[1] - xs[0])) * xs[0])
    if phase_map != phase_map_2:
        raise AssertionError("phase recurrence differs across the first two cycles")
    owner0_line = fit_line(xs, owner0_cs)
    owner1_load_line = fit_line(xs, owner1_ls)
    fixed = phase_map.b / (1 - phase_map.a)
    x0 = xs[0]

    rows, endpoint = read_event_rows()
    if not rows:
        raise AssertionError("empty event trace")
    first = rows[0]
    if first["tokens"]["retired"] != "3,2":
        raise AssertionError("trace does not start at probe-5 boundary")
    initial_ids = set(first["activities"])
    expected_start = {(0, 1, 195), (1, 0, 135), (1, 1, 134)}
    if initial_ids != expected_start:
        raise AssertionError(f"unexpected start activities: {initial_ids}")
    progress: dict[tuple[int, int, int], Aff] = {
        (0, 1, 195): owner0_line,
        (1, 0, 135): owner1_load_line,
        (1, 1, 134): Aff(F(1), F(0)),
    }

    # The probe-5 state is reconstructed from the trace's stage membership and
    # the bounded two-slot lifecycle: wave 196 is loaded behind C195; owner 1
    # has L135 and C134 active. These queues are then replayed independently.
    lifecycle = [
        LifecycleOwner(197, 3, {1: 195}, [196], []),
        LifecycleOwner(136, 2, {0: 135, 1: 134}, [], []),
    ]
    initial_lifecycle = [
        LifecycleOwner(s.next_load, s.retired, dict(s.active), list(s.loaded), list(s.computed))
        for s in lifecycle
    ]

    domain: list[F | None] = [F(0), F(1)]
    constraints = 0
    simultaneous_events = 0
    basis_counts: dict[str, int] = {}
    independent_lp_checks = 0
    max_active_per_owner = 0
    max_input_slots = 0
    max_output_slots = 0
    max_loaded_queue = 0
    max_computed_queue = 0
    identity_tie_count = 0
    min_progress_endpoint: F | None = None
    max_progress_endpoint: F | None = None
    min_positive_dt: F | None = None
    min_strict_deadline_margin: F | None = None
    per_event_time = Aff()
    max_prediction_error = F(0)
    for step, event in enumerate(rows, 1):
        acts = event["activities"]
        active = set(acts)
        expected_retired = tuple(map(int, event["tokens"]["retired"].split(",")))
        check_lifecycle_state(lifecycle, step, active, expected_retired)
        max_active_per_owner = max(max_active_per_owner,
                                   *(len(s.active) for s in lifecycle))
        for state in lifecycle:
            input_slots, output_slots = lifecycle_occupancy(state)
            max_input_slots = max(max_input_slots, input_slots)
            max_output_slots = max(max_output_slots, output_slots)
            max_loaded_queue = max(max_loaded_queue, len(state.loaded))
            max_computed_queue = max(max_computed_queue, len(state.computed))

        fresh = active - set(progress)
        for ident in active:
            if ident not in progress:
                progress[ident] = Aff(F(0), F(0))
        for ident in fresh:
            if progress[ident] != Aff(F(0), F(0)):
                raise AssertionError(f"step {step}: new activity did not start at zero: {ident}")
        # All event-state progress values must follow the proposed one-dimensional line.
        for ident, info in acts.items():
            pred = progress[ident].at(x0)
            error = abs(pred - info["p"])
            max_prediction_error = max(max_prediction_error, error)
            if pred != info["p"]:
                raise AssertionError(f"step {step}: state mismatch {ident}: {pred} != {info['p']}")
            for orbit_x in (fixed, x0):
                p = progress[ident].at(orbit_x)
                if not 0 <= p <= 1:
                    raise AssertionError(f"step {step}: progress outside [0,1] for {ident} at {orbit_x}")
                min_progress_endpoint = p if min_progress_endpoint is None else min(min_progress_endpoint, p)
                max_progress_endpoint = p if max_progress_endpoint is None else max(max_progress_endpoint, p)

        # Rebuild wave Q/work and solve the exact grouped max-min tier LP from
        # scratch. This is an independent check of every trace rate and basis.
        lp_rates, lp_q, lp_basis = independently_solve_grouped_lp(acts)
        if "+".join(lp_basis) != event["basis"]:
            raise AssertionError(f"step {step}: independently solved basis mismatch: "
                                 f"{lp_basis} != {event['basis']}")
        for ident, info in acts.items():
            if lp_rates[ident] != info["rate"]:
                raise AssertionError(f"step {step}: independent rate mismatch {ident}: "
                                     f"{lp_rates[ident]} != {info['rate']}")
            if lp_q[ident] != info["q"]:
                raise AssertionError(f"step {step}: independent Q mismatch {ident}: "
                                     f"{lp_q[ident]} != {info['q']}")
        independent_lp_checks += 1
        deadlines: dict[tuple[int, int, int], Aff] = {}
        for ident, info in acts.items():
            d = (Aff(F(0), F(1)) - progress[ident]) / info["rate"]
            deadlines[ident] = d
            if d.at(x0) != info["deadline"]:
                raise AssertionError(f"step {step}: deadline mismatch for {ident}")
        selected = event["selected"]
        if len(selected) > 1:
            simultaneous_events += 1
        if not selected or any(i not in active for i in selected):
            raise AssertionError(f"step {step}: malformed selected event set {selected}")
        selected_deadline = deadlines[selected[0]]
        identity_ties = {ident for ident, deadline in deadlines.items()
                         if deadline == selected_deadline}
        if identity_ties != set(selected):
            raise AssertionError(f"step {step}: selected set omits/adds an affine identity tie: "
                                 f"{identity_ties} != {set(selected)}")
        identity_tie_count += max(0, len(identity_ties) - 1)
        for ident in selected[1:]:
            if deadlines[ident] != selected_deadline:
                raise AssertionError(
                    f"step {step}: simultaneous completion is only a point tie: {selected[0]} vs {ident}")
        if selected_deadline.at(x0) != event["dt"]:
            raise AssertionError(f"step {step}: selected dt mismatch")
        for orbit_x in (fixed, x0):
            dt_value = selected_deadline.at(orbit_x)
            if dt_value <= 0:
                raise AssertionError(f"step {step}: dt is not strictly positive at orbit endpoint")
            min_positive_dt = dt_value if min_positive_dt is None else min(min_positive_dt, dt_value)
        # The selected event must be no later than every other activity deadline.
        for ident, deadline in deadlines.items():
            if ident in selected:
                continue
            margin = deadline - selected_deadline
            if margin.at(x0) < 0:
                raise AssertionError(f"step {step}: observed event was not earliest")
            if margin == Aff(F(0), F(0)):
                raise AssertionError(f"step {step}: unselected exact tie omitted from selected set: {ident}")
            if margin.at(fixed) <= 0 or margin.at(x0) <= 0:
                raise AssertionError(f"step {step}: nonselected deadline margin lacks strict endpoint "
                                     f"clearance for {ident}: {margin.at(fixed)}, {margin.at(x0)}")
            for orbit_x in (fixed, x0):
                value = margin.at(orbit_x)
                min_strict_deadline_margin = (value if min_strict_deadline_margin is None else
                                              min(min_strict_deadline_margin, value))
            add_geq_zero(domain, margin)
            constraints += 1
        # Also preserve a nonnegative selected interval.
        if selected_deadline.at(x0) <= 0:
            raise AssertionError(f"step {step}: nonpositive dt")
        add_geq_zero(domain, selected_deadline)
        constraints += 1
        per_event_time += selected_deadline
        basis_counts[event["basis"]] = basis_counts.get(event["basis"], 0) + 1

        for ident, info in acts.items():
            progress[ident] = progress[ident] + info["rate"] * selected_deadline
        for ident in selected:
            if progress[ident] != Aff(F(0), F(1)):
                raise AssertionError(f"step {step}: selected progress not identically 1 for {ident}")
            del progress[ident]
        # Entries absent from a subsequent state are retired; keep no stale progress.
        if step < len(rows):
            next_ids = set(rows[step]["activities"])
            for ident in list(progress):
                if ident not in next_ids:
                    del progress[ident]
        advance_lifecycle(lifecycle, selected)

    endpoint_ids: dict[tuple[int, int], tuple[int, F]] = {}
    for owner in (0, 1):
        for stage in (0, 1, 2):
            value = endpoint.get(f"o{owner}_s{stage}")
            if not value or value == "idle":
                continue
            wave, p = value.split("@", 1)
            endpoint_ids[(owner, stage)] = (int(wave), frac(p))
    expected_endpoint = {(0, 1): 325, (1, 0): 265, (1, 1): 264}
    if {k: v[0] for k, v in endpoint_ids.items()} != expected_endpoint:
        raise AssertionError(f"unexpected endpoint wave IDs: {endpoint_ids}")
    endpoint_active = {(owner, stage, wave)
                       for (owner, stage), (wave, _) in endpoint_ids.items()}
    expected_final_retired = (5, 4)
    check_lifecycle_state(lifecycle, len(rows) + 1, endpoint_active, expected_final_retired)
    for owner, state in enumerate(lifecycle):
        if state.next_load != int(endpoint[f"o{owner}_next"]):
            raise AssertionError(f"endpoint next-load mismatch for owner {owner}")
        for stage in (0, 1, 2):
            expected_wave = endpoint_ids.get((owner, stage))
            actual_wave = state.active.get(stage)
            if (expected_wave is None) != (actual_wave is None):
                raise AssertionError(f"endpoint activity occupancy mismatch owner={owner}, stage={stage}")
            if expected_wave is not None and expected_wave[0] != actual_wave:
                raise AssertionError(f"endpoint wave mismatch owner={owner}, stage={stage}")
    endpoint_map: dict[tuple[int, int], Aff] = {}
    for (owner, stage), (wave, actual) in endpoint_ids.items():
        ident = (owner, stage, wave)
        got = progress[ident]
        if got.at(x0) != actual:
            raise AssertionError(f"endpoint state mismatch for {ident}")
        endpoint_map[(owner, stage)] = got

    start_time = frac(first["tokens"]["now"])
    endpoint_time = frac(endpoint["now"])
    traced_elapsed = endpoint_time - start_time
    if per_event_time.at(x0) != traced_elapsed:
        raise AssertionError("sum of selected event dt does not match endpoint time")

    xprime = endpoint_map[(1, 1)]
    if xprime != phase_map:
        raise AssertionError(f"event-composed phase map {xprime} != boundary map {phase_map}")
    if endpoint_map[(0, 1)] != owner0_line.a * xprime + owner0_line.b:
        raise AssertionError("owner0 compute coordinate does not return to the invariant line")
    if endpoint_map[(1, 0)] != owner1_load_line.a * xprime + owner1_load_line.b:
        raise AssertionError("owner1 load coordinate does not return to the invariant line")

    lo, hi = domain
    if lo is None or hi is None:
        raise AssertionError("unexpected unbounded legal trace interval")
    if not (lo <= fixed <= x0 <= hi):
        raise AssertionError("fixed point-to-initial orbit interval is outside trace region")
    if not (0 < phase_map.a < 1 and fixed < x0):
        raise AssertionError("phase map is not a strict contraction toward the fixed point")

    # The start/end queues, wave cursors and stage memberships must be the same
    # modulo the exact two-output shift. This checks the complete bounded
    # lifecycle state, not only the three progress coordinates.
    for owner, (start, finish) in enumerate(zip(initial_lifecycle, lifecycle)):
        if finish.next_load != start.next_load + 130 or finish.retired != start.retired + 2:
            raise AssertionError(f"owner {owner}: lifecycle cursors do not shift by one joint cycle")
        if finish.loaded != [wave + 130 for wave in start.loaded]:
            raise AssertionError(f"owner {owner}: loaded-slot boundary is not cycle-equivalent")
        if finish.computed != [wave + 130 for wave in start.computed]:
            raise AssertionError(f"owner {owner}: computed-slot boundary is not cycle-equivalent")
        if finish.active != {stage: wave + 130 for stage, wave in start.active.items()}:
            raise AssertionError(f"owner {owner}: active pipeline boundary is not cycle-equivalent")
        paired = ([*start.loaded, *start.computed, *start.active.values()] +
                  [*finish.loaded, *finish.computed, *finish.active.values()])
        for wave in paired:
            facts = wave_facts(wave)
            shifted = wave_facts(wave + 130) if wave + 130 < WAVE_COUNT else None
            if shifted is not None and (facts["m"], facts["n"], facts["k"], facts["final_k"]) != (
                    shifted["m"], shifted["n"], shifted["k"], shifted["final_k"]):
                raise AssertionError(f"owner {owner}: +130 crosses a wave-family boundary at {wave}")

    cycle_index = 497  # Stop at output 997; the next cycle reaches the N-tail.
    # Exact closed form x_n = x* + a^n (x_0 - x*).
    an = phase_map.a ** cycle_index
    xn = fixed + an * (x0 - fixed)
    elapsed_n = cycle_index * (per_event_time.a * fixed + per_event_time.b) + (
        per_event_time.a * (x0 - fixed) * (1 - an) / (1 - phase_map.a))
    elapsed_n_limit = cycle_index * (per_event_time.a * fixed + per_event_time.b) + (
        per_event_time.a * (x0 - fixed) / (1 - phase_map.a))
    elapsed_tail_error = elapsed_n_limit - elapsed_n
    if abs(elapsed_tail_error) != abs(per_event_time.a * (x0 - fixed) * an / (1 - phase_map.a)):
        raise AssertionError("geometric tail error identity mismatch")
    if not an < F(1, 2 ** 1335):
        raise AssertionError("exact a^497 < 2^-1335 comparison failed")
    if not abs(x0 - fixed) < F(1, 8):
        raise AssertionError("exact |x0-x*| < 1/8 comparison failed")
    phase_error = abs(xn - fixed)
    if not phase_error < F(1, 2 ** 1338):
        raise AssertionError("exact |x497-x*| < 2^-1338 comparison failed")
    tail_prefactor = abs(per_event_time.a * (x0 - fixed) / (1 - phase_map.a))
    if not tail_prefactor < F(1, 2 ** 13):
        raise AssertionError("exact time-tail prefactor < 2^-13 seconds comparison failed")
    if not abs(elapsed_tail_error) < F(1, 2 ** 1348):
        raise AssertionError("exact cumulative-time tail error < 2^-1348 seconds comparison failed")

    first_lp_rates, first_lp_q, _ = independently_solve_grouped_lp(first["activities"])
    first_owner_q: dict[int, F] = {}
    for ident, q in first_lp_q.items():
        first_owner_q[ident[0]] = first_owner_q.get(ident[0], F(0)) + q
    first_alpha = {
        f"{owner}.{stage}.{wave}": fstr(first_owner_q[owner] * rate)
        for (owner, stage, wave), rate in first_lp_rates.items()
    }
    all_infos = [info for row in rows for info in row["activities"].values()]
    rate_bit_max = (max(fbits(info["rate"])[0] for info in all_infos),
                    max(fbits(info["rate"])[1] for info in all_infos))
    progress_bit_max = (max(fbits(info["p"])[0] for info in all_infos),
                        max(fbits(info["p"])[1] for info in all_infos))

    result = {
        "source_trace": EVENT_TRACE.name,
        "sample_boundary_probes": list(probes),
        "outputs_per_joint_cycle": 2,
        "wave_shift_per_owner_per_cycle": 130,
        "fixture": {"M": FIXTURE_M, "N": FIXTURE_N, "K": FIXTURE_K,
                    "tm": TM, "tn": TN, "tk": TK, "Mtiles": M_TILES,
                    "Ntiles": N_TILES, "Ktiles": K_TILES},
        "physical_capacities": {"COMPUTE_GATE": "1", "HBM0_bytes_per_second": "50000000",
                                "HBM1_bytes_per_second": "50000000"},
        "time_unit": "seconds (compute work is vec_ops / 158216000 ops/s; HBM work is bytes / 50000000 bytes/s)",
        "independent_grouped_lp": {
            "events_recomputed": independent_lp_checks,
            "rate_q_and_basis_exact_match": True,
            "first_event_active_q": {f"{o}.{s}.{w}": fstr(q) for (o, s, w), q in first_lp_q.items()},
            "first_event_owner_q_sum": {str(o): fstr(q) for o, q in first_owner_q.items()},
            "first_event_owner_alpha_coordinate_q_owner_times_activity_rate": first_alpha,
            "maximum_rate_numerator_denominator_bits": list(rate_bit_max),
            "maximum_progress_numerator_denominator_bits_in_event_trace": list(progress_bit_max),
        },
        "bounded_lifecycle_replay": {
            "event_states_checked": len(rows) + 1,
            "K_final_only_store_retirement": True,
            "start_and_end_slots_queues_and_activity_match_modulo_wave_shift": True,
            "next_load_shift_per_owner": 130,
            "retired_output_shift_per_owner": 2,
            "max_active_per_owner": max_active_per_owner,
            "max_input_slot_occupancy": max_input_slots,
            "max_output_slot_occupancy": max_output_slots,
            "max_loaded_queue_length": max_loaded_queue,
            "max_computed_queue_length": max_computed_queue,
            "newly_started_activities_have_zero_progress": True,
        },
        "fully_logged_interval_count": len(rows),
        "intervals_before_probe5": 662,
        "intervals_at_endpoint": 1182,
        "event_basis_counts": basis_counts,
        "phase_progress_map": phase_map.as_json(),
        "phase_fixed_point": fstr(fixed),
        "phase_start": fstr(x0),
        "phase_after_497_formula": "x* + a^497*(x0-x*)",
        "owner0_compute_boundary_line": owner0_line.as_json(),
        "owner1_load_boundary_line": owner1_load_line.as_json(),
        "event_order_feasible_interval": {"lo": fstr(lo), "hi": fstr(hi)},
        "cycle_start_orbit_interval": {"lo": fstr(fixed), "hi": fstr(x0)},
        "progress_domain_over_orbit_interval": {
            "min_endpoint_value": fstr(min_progress_endpoint),
            "max_endpoint_value": fstr(max_progress_endpoint),
            "all_active_progress_in_closed_unit_interval": True,
        },
        "event_deadline_inequalities_checked": constraints,
        "all_inequalities_hold_over_orbit_interval": True,
        "strict_margin_checks": {
            "selected_dt_minimum_at_orbit_endpoints": fstr(min_positive_dt),
            "nonselected_deadline_margin_minimum_at_orbit_endpoints": fstr(min_strict_deadline_margin),
            "all_nonselected_margins_strict_at_both_endpoints": True,
            "all_identity_ties_selected": True,
        },
        "simultaneous_completion_event_count": simultaneous_events,
        "simultaneous_extra_activity_count": identity_tie_count,
        "all_simultaneous_completions_are_identity_ties": True,
        "phase_map_composed_from_logged_event_steps": True,
        "boundary_subspace_invariant": True,
        "cycle_elapsed_time_affine": per_event_time.as_json(),
        "cycle_elapsed_time_at_fixed_point": fstr(per_event_time.a * fixed + per_event_time.b),
        "elapsed_time_497_formula": "497*T(x*) + u*(x0-x*)*(1-a^497)/(1-a)",
        "elapsed_time_497_unit": "seconds",
        "elapsed_time_497_numerator_denominator_bits": list(fbits(elapsed_n)),
        "elapsed_time_497_geometric_tail_bound": {
            "exact_a_497_lt_2^-1335_assertion": True,
            "exact_abs_x0_minus_fixed_lt_1_over_8_assertion": True,
            "exact_time_tail_prefactor_lt_2^-13_seconds_assertion": True,
            "exact_error_comparison": "Fraction(abs(T_slope*(x0-x*)*a**497/(1-a))) < Fraction(1, 2**1348)",
            "error_numerator_denominator_bits": list(fbits(abs(elapsed_tail_error))),
            "bound": "|error from replacing (1-a^497) by 1| < 2^-1348 seconds",
        },
        "elapsed_time_497_limit_approx": float(elapsed_n_limit),
        "phase_after_497_numerator_denominator_bits": list(fbits(xn)),
        "phase_error_after_497_bound": {
            "exact_comparison": "Fraction(abs(x497-x*)) < Fraction(1, 2**1338)",
            "error_numerator_denominator_bits": list(fbits(phase_error)),
            "bound": "|x_497-x*| < 2^-1338",
        },
        "phase_a_power_497_bound": {
            "exact_comparison": "Fraction(a)**497 < Fraction(1, 2**1335)",
            "a_power_numerator_denominator_bits": list(fbits(an)),
            "bound": "a^497 < 2^-1335",
        },
        "map_a_num_den_bits": list(fbits(phase_map.a)),
        "map_b_num_den_bits": list(fbits(phase_map.b)),
        "max_event_progress_prediction_error": fstr(max_prediction_error),
        "qualification": (
            "Exact trace proof for this qlocal 2-owner offset-0.05 full M/N category segment. "
            "It does not establish arbitrary offsets, tail transitions, production capacity-change or cancellation handling."
        ),
    }
    (HERE / "affine-cycle-proof.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
