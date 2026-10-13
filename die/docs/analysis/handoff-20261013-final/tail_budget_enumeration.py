#!/usr/bin/env python3
"""Scratch proof-by-enumeration for GEMM slow-core compute budgets."""
from __future__ import annotations
import itertools
import math


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def setup(p):
    m, n, k, tm, tn, tk = (p[x] for x in ("m", "n", "k", "tm", "tn", "tk"))
    t_m, t_n = ceildiv(m, tm), ceildiv(n, tn)
    mtail, ntail = m % tm, n % tn
    mfull = m // tm
    nfull = n // tn
    has_mt = bool(mtail and mfull)
    has_nt = bool(ntail and nfull)
    m_full_len = tm if mfull else m
    n_full_len = tn if nfull else n
    if not has_mt:
        mfull = t_m
        mtail = 0
    if not has_nt:
        nfull = t_n
        ntail = 0
    return t_m, t_n, mfull, nfull, mtail, ntail, has_mt, has_nt, m_full_len, n_full_len


def conversion_eq_per_element(dtype: str, int4_path: bool, kappa_quant: float) -> float:
    execution_dtype = "int4" if int4_path else "fp32"
    if dtype == execution_dtype:
        return 0.0
    if int4_path:
        return kappa_quant
    return kappa_quant if dtype == "int4" else 1.0


def output_cast_eq_per_element(dtype: str, kappa_output_int4: float) -> float:
    if dtype == "fp32":
        return 0.0
    if dtype == "int4":
        return kappa_output_int4
    return 1.0


def tile_parts(p, m_len: int, n_len: int, include_weight_conversion: bool = True):
    """Return serial PE seconds, Vec seconds, input-W setup Vec seconds, op ledger."""
    k, tk = p["k"], p["tk"]
    x = p["executions"]
    int4_path = p["int4_path"]
    ca = conversion_eq_per_element(p["ad"], int4_path, p["kappa_quant"])
    cw = conversion_eq_per_element(p["wd"], int4_path, p["kappa_quant"])
    cast = output_cast_eq_per_element(p["od"], p["kappa_output_int4"])
    matrix_ops = 2.0 * m_len * n_len * k * x
    a_conv_ops = m_len * k * ca * x
    w_conv_ops = n_len * k * cw * x
    output_cast_ops = m_len * n_len * cast * x
    if int4_path:
        acc_convert_ops = m_len * n_len * tk * p["kappa_acc"] * x
        acc_add_ops = m_len * n_len * (tk - 1) * x
        pe_ops = matrix_ops
        vec_ops = a_conv_ops + (w_conv_ops if include_weight_conversion else 0.0)
        vec_ops += acc_convert_ops + acc_add_ops + output_cast_ops
    else:
        pe_ops = 0.0
        vec_ops = matrix_ops + a_conv_ops + (w_conv_ops if include_weight_conversion else 0.0)
        vec_ops += output_cast_ops
        acc_convert_ops = acc_add_ops = 0.0
    pe_s = pe_ops / (p["p_pe"] * p["eta_pe"]) if pe_ops else 0.0
    vec_s = vec_ops / (p["p_vec"] * p["eta_vec"])
    w_setup_s = w_conv_ops / (p["p_vec"] * p["eta_vec"])
    return pe_s, vec_s, w_setup_s, {
        "matrix_ops": matrix_ops,
        "a_input_conversion_eqops": a_conv_ops,
        "w_input_conversion_eqops": w_conv_ops,
        "int32_acc_convert_eqops": acc_convert_ops,
        "int32_acc_add_eqops": acc_add_ops,
        "output_cast_eqops": output_cast_ops,
    }


def tile_cost(p, m_len, n_len, include_weight_conversion=True):
    pe, vec, _, _ = tile_parts(p, m_len, n_len, include_weight_conversion)
    return pe + vec


def tile_classes(p):
    t_m, t_n, mf, nf, mt, nt, has_mt, has_nt, m_full, n_full = setup(p)
    ms = [(m_full, mf)] + ([(mt, 1)] if has_mt else [])
    ns = [(n_full, nf)] + ([(nt, 1)] if has_nt else [])
    return [(mlen, nlen, mc * nc)
            for mlen, mc in ms for nlen, nc in ns if mc and nc]


def os_coverage_count(p, threshold: float) -> int:
    """O(1) cohorts touched by qualifying full/tail cost classes."""
    t_m, t_n, mf, nf, mt, nt, has_mt, has_nt, m_full, n_full = setup(p)
    q = t_m * t_n
    c = p["ncore"]
    full_full = tile_cost(p, m_full, n_full)
    if full_full < threshold:
        return 0
    m_all = not has_mt or tile_cost(p, mt, n_full) >= threshold
    n_all = not has_nt or tile_cost(p, m_full, nt) >= threshold

    bottom_right_too_slow = (has_mt and has_nt and
                             tile_cost(p, mt, nt) < threshold)
    if c == 1:
        if m_all and n_all:
            return q - int(bottom_right_too_slow)
        if m_all:
            return t_m * nf
        if n_all:
            return mf * t_n
        return mf * nf

    if m_all and n_all:
        if bottom_right_too_slow:
            return ceildiv(q - 1, c)
        return ceildiv(q, c)
    if m_all:  # all M rows, full-N prefix only; one N-tail gap per row
        buckets = ceildiv(q, c)
        if has_nt and (q - 1) % c == 0:
            buckets -= 1  # the final partial cohort consists solely of the gap tile
        return buckets
    if n_all:  # full-M rows, all N tiles
        return ceildiv(mf * t_n, c)

    # full-M x full-N prefixes: a one-tile N-tail gap at the end of each row.
    prefix = mf * t_n
    buckets = ceildiv(prefix, c)
    if has_nt and prefix % c == 1:
        buckets -= 1
    return buckets


def os_analytic(p):
    t_m, t_n, _, _, _, _, _, _, _, _ = setup(p)
    q = t_m * t_n
    c = p["ncore"]
    cohorts = ceildiv(q, c)
    costs = sorted({tile_cost(p, m, n) for m, n, count in tile_classes(p) if count})
    total = costs[0] * cohorts
    for prev, cur in zip(costs, costs[1:]):
        total += (cur - prev) * os_coverage_count(p, cur)
    active_last = q - (cohorts - 1) * c
    start_last = (cohorts - 1) * c
    candidates = []
    for mlen, nlen, count in tile_classes(p):
        # The final index of each class is derived from the rectangle boundaries.
        # Test all four category endpoints only; no Core/tile-sized loop is needed.
        if mlen == setup(p)[8] and nlen == setup(p)[9]:
            q_last = (setup(p)[2] - 1) * t_n + (setup(p)[3] - 1)
        elif mlen == setup(p)[8] and setup(p)[7] and nlen == setup(p)[5]:
            q_last = (setup(p)[2] - 1) * t_n + setup(p)[3]
        elif setup(p)[6] and mlen == setup(p)[4] and nlen == setup(p)[9]:
            q_last = setup(p)[2] * t_n + (setup(p)[3] - 1)
        else:
            q_last = q - 1
        if count and q_last >= start_last:
            candidates.append(tile_cost(p, mlen, nlen))
    return total, active_last, max(candidates)


def os_brute(p):
    t_m, t_n, _, _, _, _, _, _, _, _ = setup(p)
    costs = []
    for i in range(t_m):
        ml = min(p["tm"], p["m"] - i * p["tm"])
        for j in range(t_n):
            nl = min(p["tn"], p["n"] - j * p["tn"])
            costs.append(tile_cost(p, ml, nl))
    c = p["ncore"]
    cohort_max = [max(costs[s:s + c]) for s in range(0, len(costs), c)]
    return sum(cohort_max), len(costs) - (len(cohort_max) - 1) * c, cohort_max[-1]


def wr_analytic(p):
    t_m, t_n, mf, nf, mt, nt, has_mt, has_nt, m_full, n_full = setup(p)
    c = p["ncore"]
    full_groups = ceildiv(nf, c)
    tail_only_groups = int(has_nt and nf % c == 0)
    m_lengths = [(m_full, mf)] + ([(mt, 1)] if has_mt else [])

    def sum_m(n_len):
        return sum(count * tile_cost(p, m_len, n_len, include_weight_conversion=False)
                   for m_len, count in m_lengths)

    # A converted W tile is a once-per-column-group setup. Every Core in a group
    # converts its W tile in parallel, so the setup cost is the slowest such tile.
    _, _, w_full, _ = tile_parts(p, 1, n_full, include_weight_conversion=True)
    _, _, w_tail, _ = tile_parts(p, 1, nt, include_weight_conversion=True)
    w_full_setup = w_full
    w_tail_setup = w_tail
    total = full_groups * (sum_m(n_full) + w_full_setup)
    total += tail_only_groups * (sum_m(nt) + w_tail_setup)
    return total, t_n - (ceildiv(t_n, c) - 1) * c, (sum_m(nt) + w_tail_setup if tail_only_groups else sum_m(n_full) + w_full_setup)


def wr_brute(p):
    t_m, t_n, _, _, _, _, _, _, _, _ = setup(p)
    c = p["ncore"]
    total = 0.0
    last_active = t_n - (ceildiv(t_n, c) - 1) * c
    last_cost = 0.0
    groups = [list(range(s, min(s + c, t_n))) for s in range(0, t_n, c)]
    for gi, cols in enumerate(groups):
        group_cost = 0.0
        m_cost = 0.0
        for i in range(t_m):
            ml = min(p["tm"], p["m"] - i * p["tm"])
            m_cost += max(tile_cost(p, ml,
                                    min(p["tn"], p["n"] - j * p["tn"]),
                                    include_weight_conversion=False)
                          for j in cols)
        weight_setup = max(tile_parts(
            p, 1, min(p["tn"], p["n"] - j * p["tn"]),
            include_weight_conversion=True)[2] for j in cols)
        group_cost = m_cost + weight_setup
        total += group_cost
        if gi == len(groups) - 1:
            last_cost = group_cost
    return total, last_active, last_cost


def configs():
    # Exhaustive bounded shapes/tilings/cores cover absent and present full/tail classes.
    for m, n, k, tm, tn, tk, c in itertools.product(
            range(1, 7), range(1, 7), range(1, 6),
            range(1, 5), range(1, 5), range(1, 4), range(1, 6)):
        for int4_path in (False, True):
            for ad, wd, od in (("fp32", "fp32", "fp32"),
                                ("bf16", "int4", "bf16"),
                                ("int4", "bf16", "int4"),
                                ("bf16", "bf16", "int4")):
                yield {
                    "m": m, "n": n, "k": k, "tm": tm, "tn": tn, "tk": tk,
                    "ncore": c, "int4_path": int4_path,
                    "ad": ad, "wd": wd, "od": od,
                    "kappa_quant": 0.625, "kappa_acc": 1.375,
                    "kappa_output_int4": 0.625,
                    "p_pe": 79.0, "p_vec": 113.0,
                    "eta_pe": 0.83, "eta_vec": 0.71,
                    "executions": 2.5,
                }


def main():
    cases = 0
    for p in configs():
        a = os_analytic(p)
        b = os_brute(p)
        if any(not math.isclose(x, y, rel_tol=2e-12, abs_tol=2e-12)
               for x, y in zip(a, b)):
            raise AssertionError(("OS mismatch", p, a, b))
        wa = wr_analytic(p)
        wb = wr_brute(p)
        if any(not math.isclose(x, y, rel_tol=2e-12, abs_tol=2e-12)
               for x, y in zip(wa, wb)):
            raise AssertionError(("WR mismatch", p, wa, wb))
        cases += 1
    print({"status": "passed", "cases": cases,
           "coverage": "OS threshold-count formula and last-cohort active/cost; WR group-max sum and once-per-group W conversion",
           "coefficients": "fractional nonnegative kappa_quant=0.625, kappa_acc=1.375, kappa_output_int4=0.625; executions=2.5 for algebra only",
           "limits": "bounded exhaustive model check only; no compiler/runtime/granularity gate claim"})


if __name__ == "__main__":
    main()
