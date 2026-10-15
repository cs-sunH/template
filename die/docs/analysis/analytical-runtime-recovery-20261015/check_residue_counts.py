#!/usr/bin/env python3
"""Small independent arithmetic check; not a project test or production code."""
from math import ceil
from random import Random


def floor_sum(n: int, m: int, a: int, b: int) -> int:
    """AtCoder-style sum floor((a*i+b)/m), for nonnegative arguments."""
    ans = 0
    while True:
        if a >= m:
            ans += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            ans += n * (b // m)
            b %= m
        y = a * n + b
        if y < m:
            return ans
        n = y // m
        b = y % m
        m, a = a, m


def nlt(n: int, c: int, d: int, b: int, t: int) -> int:
    return n - floor_sum(n, c, d, b + c - t) + floor_sum(n, c, d, b)


def ap_lane_count(n: int, c: int, d: int, b: int, lo: int, hi: int) -> int:
    return nlt(n, c, d, b, hi) - nlt(n, c, d, b, lo)


def add_interval(terms, lo, hi, sign=1):
    if hi > lo:
        terms.append((lo, 1, hi - lo, sign))


def add_ap(terms, b, d, n, sign=1):
    if n > 0:
        terms.append((b, d, n, sign))


def class_terms(nt, full_n, m_begin, m_count, n_tail, qlo, qhi):
    """Return signed APs for one rectangular M/N class clipped to q range."""
    terms = []
    m_end = m_begin + m_count
    if n_tail:
        # The tail-column class is q=m*Nt+full_n.
        first = max(m_begin, max(0, (qlo - full_n + nt - 1) // nt))
        end = min(m_end, max(0, (qhi - full_n + nt - 1) // nt))
        add_ap(terms, first * nt + full_n, nt, end - first)
        return terms

    width = full_n
    if width == 0:
        return terms
    if width == nt:
        lo = max(qlo, m_begin * nt)
        hi = min(qhi, m_end * nt)
        add_interval(terms, lo, hi)
        return terms

    # Full-N class when there is an N-tail: complete rows minus one tail q
    # per row, plus at most the two q-range boundary rows.
    mid_begin = max(m_begin, (qlo + nt - 1) // nt)
    mid_end = min(m_end, qhi // nt)
    if mid_end > mid_begin:
        add_interval(terms, mid_begin * nt, mid_end * nt)
        add_ap(terms, mid_begin * nt + width, nt, mid_end - mid_begin, -1)

    boundary_rows = set()
    if qlo < qhi and qlo % nt:
        boundary_rows.add(qlo // nt)
    if qlo < qhi and qhi % nt:
        boundary_rows.add((qhi - 1) // nt)
    for m in boundary_rows:
        if not (m_begin <= m < m_end):
            continue
        lo = max(qlo, m * nt)
        hi = min(qhi, m * nt + width)
        add_interval(terms, lo, hi)
    return terms


def term_lane_count(terms, c, lane_lo, lane_hi):
    return sum(sign * ap_lane_count(n, c, d, b, lane_lo, lane_hi)
               for b, d, n, sign in terms)


def class_brute(nt, full_n, m_begin, m_count, n_tail, qlo, qhi):
    m_end = m_begin + m_count
    n_values = [full_n] if n_tail else list(range(full_n))
    return [m * nt + n for m in range(m_begin, m_end) for n in n_values
            if qlo <= m * nt + n < qhi]


def main():
    rng = Random(0x05102026)
    checks = 0
    for c in range(1, 20):
        for _ in range(100):
            n = rng.randrange(0, 50)
            d = rng.randrange(0, 50)
            b = rng.randrange(0, 50)
            lo = rng.randrange(0, c + 1)
            hi = rng.randrange(lo, c + 1)
            got = ap_lane_count(n, c, d, b, lo, hi)
            want = sum(lo <= (b + i * d) % c < hi for i in range(n))
            assert got == want, (n, c, d, b, lo, hi, got, want)
            checks += 1

    for nt in range(1, 13):
        for full_n in range(0, nt + 1):
            if full_n == nt:
                n_cases = (False,)
            elif full_n + 1 == nt:
                n_cases = (False, True)
            else:
                continue
            for n_tail in n_cases:
                for m_begin in range(3):
                    for m_count in range(1, 6):
                        total = (m_begin + m_count) * nt
                        for _ in range(20):
                            qlo = rng.randrange(total + 1)
                            qhi = rng.randrange(qlo, total + 1)
                            q_values = class_brute(nt, full_n, m_begin,
                                                  m_count, n_tail, qlo, qhi)
                            terms = class_terms(nt, full_n, m_begin,
                                                m_count, n_tail, qlo, qhi)
                            # Full-N with a hardware N-tail is represented as
                            # full rows minus tail AP; compare the signed lane
                            # histogram rather than expanding signed terms.
                            for c in range(1, 14):
                                for lane in range(c):
                                    got = term_lane_count(terms, c, lane, lane + 1)
                                    want = sum(q % c == lane for q in q_values)
                                    assert got == want, (nt, full_n, m_begin,
                                        m_count, n_tail, qlo, qhi, c, lane,
                                        terms, got, want)
                                    checks += 1
    print(f"PASS independent arithmetic cases={checks}")


if __name__ == "__main__":
    main()
