#!/usr/bin/env python3
"""Independent exact Fraction oracle for tiny grouped-fair Runtime LPs.

Variables are canonical service x_i = Q_i * progress_rate_i.  This mirrors
only the documented LP definition; it does not call or parse production code.
For exact owner copies, a second LP projects each activity class to one
variable, multiplies only the physical rows, and compares against the full LP.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations

@dataclass(frozen=True)
class Activity:
    owner: str
    name: str
    q: F
    work: tuple
    private_time: F = F(0)
    delta: tuple = (F(0), F(0))
    symmetry: str = ""
    progress: F = F(0)  # Deliberately absent from all LP rows.


def f(value):
    return value if isinstance(value, F) else F(value)


def solve_square(rows, rhs):
    n = len(rhs)
    a = [[f(v) for v in rows[i]] + [f(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for r in range(n):
            if r == col or a[r][col] == 0:
                continue
            scale = a[r][col]
            a[r] = [a[r][c] - scale * a[col][c] for c in range(n + 1)]
    return tuple(a[r][-1] for r in range(n))


def lp_max(dim, rows, objective):
    """Enumerate exact vertices of a bounded polytope A*x<=b."""
    best_value = None
    best_point = None
    for active in combinations(range(len(rows)), dim):
        point = solve_square([rows[i][0] for i in active],
                             [rows[i][1] for i in active])
        if point is None:
            continue
        if any(sum(a * x for a, x in zip(coeff, point)) > bound
               for coeff, bound in rows):
            continue
        value = sum(a * x for a, x in zip(objective, point))
        if best_value is None or value > best_value:
            best_value, best_point = value, point
    if best_point is None:
        raise AssertionError("exact vertex enumeration found no feasible vertex")
    return best_value, best_point


def add_eq(rows, dim, vector, value):
    rows.append((tuple(vector), value))
    rows.append((tuple(-x for x in vector), -value))


def base_constraints(activities, capacities, boundary_states, excluded,
                     symmetric_projection):
    n = len(activities)
    rows = []
    # One row per physical resource. The uncompressed LP naturally sums every
    # owner's coefficient; projection onto equal variables yields m*work/Q.
    for resource, cap in enumerate(capacities):
        coeff = tuple(a.work[resource] / a.q for a in activities)
        if any(coeff):
            rows.append((coeff, cap))
    for i, a in enumerate(activities):
        if a.private_time > 0:
            coeff = [F(0)] * n
            coeff[i] = 1 / a.q
            rows.append((tuple(coeff), 1 / a.private_time))
        nonnegative = [F(0)] * n
        nonnegative[i] = -1
        rows.append((tuple(nonnegative), F(0)))
        if i in excluded:
            upper = [F(0)] * n
            upper[i] = 1
            rows.append((tuple(upper), F(0)))

    # Each execution has its own signed-credit boundary rows.  Duplicate rows
    # become identical after exact symmetry projection, so one representative
    # row is sufficient in the reduced LP.
    by_owner = {}
    for i, a in enumerate(activities):
        by_owner.setdefault(a.owner, []).append(i)
    for owner, indices in by_owner.items():
        state = boundary_states.get(owner, ("interior", "interior"))
        for buffer, side in enumerate(state):
            signs = []
            if side in ("low", "both"):
                signs.append(-1)
            if side in ("high", "both"):
                signs.append(1)
            for sign in signs:
                coeff = [F(0)] * n
                for i in indices:
                    coeff[i] = sign * activities[i].delta[buffer] / activities[i].q
                if any(coeff):
                    rows.append((tuple(coeff), F(0)))

    if symmetric_projection:
        classes = {}
        for i, a in enumerate(activities):
            if a.symmetry:
                classes.setdefault(a.symmetry, []).append(i)
        for members in classes.values():
            rep = members[0]
            for i in members[1:]:
                eq = [F(0)] * n
                eq[i] = 1
                eq[rep] = -1
                add_eq(rows, n, eq, F(0))
    return rows


def lexicographic_max_min(activities, capacities, boundary_states=None,
                          symmetric_projection=False):
    boundary_states = boundary_states or {}
    n = len(activities)
    excluded = set()
    final_alpha = [F(0)] * n
    final_x = [F(0)] * n
    restart_count = 0
    while True:
        restart_count += 1
        assert restart_count <= n + 1
        owner_q = {}
        for i, a in enumerate(activities):
            if i not in excluded:
                owner_q[a.owner] = owner_q.get(a.owner, F(0)) + a.q
        if not owner_q:
            return final_x, final_alpha, excluded
        alpha = [F(0)] * n
        for i, a in enumerate(activities):
            if i not in excluded:
                alpha[i] = a.q / owner_q[a.owner]
        rows = base_constraints(activities, capacities, boundary_states,
                                excluded, symmetric_projection)
        frozen = {}
        restart_with_zero = False

        while True:
            unfilled = [i for i in range(n)
                        if i not in excluded and i not in frozen]
            if not unfilled:
                break
            dim = n + 1
            tier_rows = [(tuple(coeff) + (F(0),), bound)
                         for coeff, bound in rows]
            for i, utility in frozen.items():
                eq = [F(0)] * dim
                eq[i] = 1
                add_eq(tier_rows, dim, eq, alpha[i] * utility)
            for i in unfilled:
                lower = [F(0)] * dim
                lower[i] = -1
                lower[n] = alpha[i]
                tier_rows.append((tuple(lower), F(0)))
            t_nonnegative = [F(0)] * dim
            t_nonnegative[n] = -1
            tier_rows.append((tuple(t_nonnegative), F(0)))
            objective = [F(0)] * dim
            objective[n] = 1
            floor, _ = lp_max(dim, tier_rows, objective)

            if floor == 0:
                zero_rows = list(rows)
                for i, utility in frozen.items():
                    eq = [F(0)] * n
                    eq[i] = 1
                    add_eq(zero_rows, n, eq, alpha[i] * utility)
                forced = []
                for i in unfilled:
                    objective = [F(0)] * n
                    objective[i] = 1
                    maximum, _ = lp_max(n, zero_rows, objective)
                    if maximum == 0:
                        forced.append(i)
                if not forced:
                    raise AssertionError("zero fairness floor had no forced-zero activity")
                excluded.update(forced)
                restart_with_zero = True
                break

            face_rows = list(rows)
            for i, utility in frozen.items():
                eq = [F(0)] * n
                eq[i] = 1
                add_eq(face_rows, n, eq, alpha[i] * utility)
            for i in unfilled:
                lower = [F(0)] * n
                lower[i] = -1
                face_rows.append((tuple(lower), -alpha[i] * floor))
            forced = []
            maxima = []
            for i in unfilled:
                objective = [F(0)] * n
                objective[i] = 1
                maximum, _ = lp_max(n, face_rows, objective)
                maxima.append((i, maximum / alpha[i]))
                if maximum / alpha[i] == floor:
                    forced.append(i)
            if not forced:
                raise AssertionError(f"positive fairness floor had no forced activity: floor={floor}, alpha={alpha}, maxima={maxima}")
            for i in forced:
                frozen[i] = floor

        if restart_with_zero:
            continue
        for i, utility in frozen.items():
            final_x[i] = alpha[i] * utility
            final_alpha[i] = alpha[i]
        return final_x, final_alpha, excluded


def make(name, owner, rows, q, private=0, delta=(0, 0), symmetry="", progress=0):
    return Activity(owner, name, f(q), tuple(map(f, rows)), f(private),
                    tuple(map(f, delta)), symmetry, f(progress))


def assert_feasible(label, activities, x, capacities, boundary_states, excluded):
    rates = [x[i] / a.q for i, a in enumerate(activities)]
    for r, capacity in enumerate(capacities):
        usage = sum(a.work[r] * rates[i] for i, a in enumerate(activities))
        assert usage <= capacity, (label, "physical", r, usage, capacity)
    for i, a in enumerate(activities):
        if a.private_time:
            assert rates[i] <= 1 / a.private_time, (label, "PRIVATE", i, rates[i])
        if i in excluded:
            assert x[i] == 0
    for owner, members in _group_by_owner(activities).items():
        state = boundary_states.get(owner, ("interior", "interior"))
        for b, side in enumerate(state):
            flow = sum(activities[i].delta[b] * rates[i] for i in members)
            if side in ("low", "both"):
                assert flow >= 0, (label, "credit-low", owner, b, flow)
            if side in ("high", "both"):
                assert flow <= 0, (label, "credit-high", owner, b, flow)


def _group_by_owner(activities):
    groups = {}
    for i, a in enumerate(activities):
        groups.setdefault(a.owner, []).append(i)
    return groups


def compress_exact_owner_blocks(activities, boundary_states):
    """Project exact same-name blocks onto shared variables with physical m."""
    groups = {}
    for i, a in enumerate(activities):
        key = ("sym", a.symmetry) if a.symmetry else ("single", i)
        groups.setdefault(key, []).append(i)
    reduced = []
    original_to_reduced = [None] * len(activities)
    for key, members in groups.items():
        first = activities[members[0]]
        for i in members[1:]:
            other = activities[i]
            assert (first.q, first.work, first.private_time, first.delta) == \
                   (other.q, other.work, other.private_time, other.delta), \
                   ("invalid symmetry fixture", key, first, other)
            assert boundary_states.get(first.owner, ("interior", "interior")) == \
                   boundary_states.get(other.owner, ("interior", "interior")), \
                   ("invalid boundary phase fixture", key)
        index = len(reduced)
        reduced.append(Activity(
            owner=first.owner,
            name=first.name,
            q=first.q,
            work=tuple(w * len(members) for w in first.work),
            private_time=first.private_time,
            delta=first.delta,
            symmetry="",
            progress=first.progress,
        ))
        for i in members:
            original_to_reduced[i] = index
    return reduced, original_to_reduced


def run_case(label, activities, capacities, boundary_states, expect,
             closed_form_full=False):
    if closed_form_full:
        full_x = [f(v) for v in expect["x"]]
        full_alpha = [f(v) for v in expect["alpha"]]
        full_excluded = set(expect.get("excluded", []))
    else:
        full_x, full_alpha, full_excluded = lexicographic_max_min(
            activities, tuple(map(f, capacities)), boundary_states, False)
    reduced, mapping = compress_exact_owner_blocks(activities, boundary_states)
    reduced_x, reduced_alpha, reduced_excluded = lexicographic_max_min(
        reduced, tuple(map(f, capacities)), boundary_states, False)
    projected_x = [reduced_x[mapping[i]] for i in range(len(activities))]
    projected_alpha = [reduced_alpha[mapping[i]] for i in range(len(activities))]
    projected_excluded = {i for i in range(len(activities))
                          if mapping[i] in reduced_excluded}
    assert full_x == projected_x, (label, "projection changed exact solution", full_x, projected_x)
    assert full_alpha == projected_alpha, (label, "alpha changed", full_alpha, projected_alpha)
    assert full_excluded == projected_excluded, (label, "zero set changed", full_excluded, projected_excluded)
    assert_feasible(label, activities, full_x, tuple(map(f, capacities)),
                    boundary_states, full_excluded)
    for i, j in combinations(range(len(activities)), 2):
        if activities[i].symmetry and activities[i].symmetry == activities[j].symmetry:
            assert full_x[i] == full_x[j], (label, "exact owners did not mirror", i, j)
    for i, value in enumerate(expect.get("x", [])):
        if value is not None:
            assert full_x[i] == f(value), (label, "unexpected exact service", i, full_x[i], value)
    for i, value in enumerate(expect.get("alpha", [])):
        if value is not None:
            assert full_alpha[i] == f(value), (label, "unexpected alpha", i, full_alpha[i], value)
    expected_excluded = set(expect.get("excluded", []))
    assert full_excluded == expected_excluded, (label, "unexpected exclusion", full_excluded)
    rates = [full_x[i] / a.q for i, a in enumerate(activities)]
    print(f"{label}: exact x={full_x}; rates={rates}; alpha={full_alpha}; excluded={sorted(full_excluded)}")


def main():
    # PRIVATE is per owner and must not receive physical multiplicity.
    private = [make("only", "A", [1], 7, private=F(1, 2), symmetry="only"),
               make("only", "B", [1], 7, private=F(1, 2), symmetry="only")]
    run_case("private-not-multiplied", private, [100], {},
             {"x": [14, 14], "alpha": [1, 1]})

    # Different canonical Q values retain each owner's alpha; equal physical
    # work and signed-credit constraints produce a closed-form rate 3/5.
    pipeline = [
        make("load", "A", [1, 0], 2, private=1, delta=(1, 0), symmetry="load", progress=F(1, 3)),
        make("compute", "A", [0, 4], 5, private=1, delta=(-1, 0), symmetry="compute", progress=F(1, 3)),
        make("load", "B", [1, 0], 2, private=1, delta=(1, 0), symmetry="load", progress=0),
        make("compute", "B", [0, 4], 5, private=1, delta=(-1, 0), symmetry="compute", progress=0),
    ]
    # Resource caps make the first-tier rates both 3/5; α values stay 2/7,5/7.
    run_case("q-alpha-physical-multiplicity-phase-offset", pipeline, [F(6, 5), F(24, 5)],
             {"A": ("low", "interior"), "B": ("low", "interior")},
             {"x": [F(6, 5), F(3), F(6, 5), F(3)],
              "alpha": [F(2, 7), F(5, 7), F(2, 7), F(5, 7)]})

    # B is forced to zero by its zero-capacity physical row. At a low credit
    # boundary that forces A=0 too. The mirrored C remains feasible and must
    # be re-normalized to alpha=1 after the zero pair is excluded.
    zero = []
    for owner in ("A", "B"):
        zero.extend([
            make("consume", owner, [1, 0, 0], 1, delta=(-1, 0), symmetry="consume"),
            make("supply", owner, [0, 1, 0], 1, delta=(1, 0), symmetry="supply"),
            make("free", owner, [0, 0, 1], 1, symmetry="free"),
        ])
    run_case("zero-exclusion-renormalizes", zero, [2, 0, 10],
             {"A": ("low", "interior"), "B": ("low", "interior")},
             {"x": [0, 0, 5, 0, 0, 5],
              "alpha": [0, 0, 1, 0, 0, 1], "excluded": [0, 1, 3, 4]},
             closed_form_full=True)

    # A heterogeneous third owner exercises mixed-tier allocation around an
    # exact symmetric pair; its activity is not in the symmetry projection.
    mixed = [make("a", "A", [1, 0], 2, symmetry="a"),
             make("a", "B", [1, 0], 2, symmetry="a"),
             make("other", "C", [0, 2], 3)]
    run_case("symmetric-pair-with-heterogeneous-owner", mixed, [4, 6], {}, {})

    # One fair owner can span multiple Runtime executions.  Its two activities
    # share a single Q denominator (alpha=1/2 each); the separate owner keeps
    # alpha=1.  This is the case production deliberately leaves uncompressed.
    split_owner = [make("part-a", "same-owner", [1], 1),
                   make("part-b", "same-owner", [1], 1),
                   make("other", "other-owner", [1], 1)]
    run_case("same-owner-across-executions-keeps-Q-denominator", split_owner,
             [3], {}, {"x": [F(3, 4), F(3, 4), F(3, 2)],
                       "alpha": [F(1, 2), F(1, 2), 1]})

    print("PASS: 5 exact Fraction LP adversarial cases; full LP vs exact block projection")

if __name__ == "__main__":
    main()
