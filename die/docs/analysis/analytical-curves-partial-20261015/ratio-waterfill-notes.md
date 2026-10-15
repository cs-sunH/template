# Signed-credit ratio closure and owner-Q water filling

This is a read-only derivation for a possible fast path between the existing nonnegative `PackingAllocation` path and the general signed-credit simplex path. It does not change the allocator or its fairness rule.

## Fairness coordinates

For the current non-excluded activity set, let `Q_i` be the activity's canonical quantity and let `W_o = sum(Q_i)` over the participating activities belonging to owner `o`. The existing LP sets `alpha_i = Q_i / W_o` and uses a common tier variable with rates proportional to `1 / W_o`. Its lexicographic utility is therefore

```text
u_i = W_owner(i) * r_i = (Q_i * r_i) / alpha_i.
```

The fast path must lexicographically maximize the sorted `u_i` values. It must not replace them with raw `r_i`, and a common scalar applied to all rates cannot repair a violated homogeneous credit row.

In these coordinates, a physical packing row and a private-time limit become

```text
sum_i multiplicity_i * work[i][resource] / W_owner(i) * u_i <= capacity[resource]
u_i <= W_owner(i) / private_time_i       (when private_time_i > 0).
```

For a zero-bound signed row with exactly one positive coefficient `c_p` and one negative coefficient `c_n`,

```text
c_p * r_p + c_n * r_n <= 0
u_p <= rho * u_n
rho = (-c_n / c_p) * W_owner(p) / W_owner(n) > 0.
```

Thus each supported signed row is a directed ratio edge. A row with only nonpositive coefficients is redundant for nonnegative rates. A row with only positive coefficients forces its supported activities to zero; it can only stay on the fast path if zero exclusion and owner-weight recomputation exactly match the existing LP. Otherwise it falls back.

## Closure and progressive water filling

Credit rows are local to an execution. For each owner/execution activity block, convert supported signed rows to directed edges `u_i <= rho_ij * u_j`; a typical OS block has at most the load, compute, and store activities. Multiple executions for one owner contribute to that owner's `W_o`, while their ratio graphs remain separate. The global water fill combines the closed lower bounds from all blocks in each physical resource row.

At a lexicographic tier, partition activities into `A` (not yet frozen) and `F` (frozen at exact tier values `v_f`). Require `u_i >= lambda` for every `i` in `A`. An edge `i -> j` implies the lower-bound propagation `u_j >= u_i / rho_ij`. Propagate these lower bounds to closure, starting with `lambda` on `A` and `v_f` on `F`:

```text
L_j(lambda) = max over active paths i -> ... -> j of lambda * product(1 / rho)
              and frozen paths f -> ... -> j of v_f * product(1 / rho).
```

Include zero-length paths. If closure raises a frozen activity above its fixed value, that `lambda` is infeasible. Also propagate finite upper bounds in the reverse direction: an edge `i -> j` and an upper bound `U_j` imply `U_i <= rho_ij * U_j`. Check `L_i <= U_i` for every node.

For a ratio graph without a gain-amplifying cycle, each active lower bound has the form `L_j(lambda) = max(d_j, g_j * lambda)`, where `d_j` is the strongest frozen-path floor and `g_j` is the strongest active-path gain. This is the componentwise least vector satisfying the tier floors and all ratio edges. Because physical row coefficients are nonnegative, this vector is also the minimum-resource choice: a tier is feasible exactly when its closed lower vector passes all physical rows, private caps, fixed-value checks, and ratio upper bounds.

Each physical-row use is a monotone piecewise-linear function of `lambda`. With at most three activities in a local OS ratio block, closure has a small number of path alternatives and breakpoints; solve each linear segment's row intersection and private-cap intersection directly. Sum the per-block contributions across all owners before choosing the global `lambda`.

To identify which activities are forced at the maximum tier `lambda*`, test each remaining activity `i`: replace its floor by a trial value `t >= lambda*`, retain `lambda*` on the other active activities, redo closure, and find the largest feasible `t`. If that maximum is `lambda*`, freeze `i` at this tier. The feasibility function is monotone and piecewise linear, so the same segment-root calculation applies. At least one activity must be forced at a finite optimum; failure to identify one robustly is a fallback condition. Repeat until all activities are frozen.

If `lambda* == 0`, the fast path must identify the exact zero-rate bottlenecks, exclude them, recompute each affected `W_o`, and restart, matching `Execution.cc`. If it does not implement this restart, it must fall back to the existing LP. A directed ratio cycle with product `rho < 1` forces its strongly connected activities to zero; predecessor activities that must be no greater than those zero targets are zero as well. Product-one cycles encode ratio equalities and can be closed normally. Detect cycle products with extended precision; ambiguous products near one should use the LP.

After solving, restore `r_i = u_i / W_owner(i)` and `alpha_i = Q_i / W_owner(i)`. Certify the original physical, private, and signed rows. To preserve runtime's one-sided rounding behavior, only round a positive signed-row contributor down when correcting a floating-point violation, then recertify all rows. If a correction changes the result beyond the existing precision envelope or certification is inconclusive, use the general LP.

## Counterexample to common-rate scaling

Take two activities of one owner with `Q_1 = Q_2 = 1`, so `W_o = 2` and `u_i = 2 r_i`. Let a signed boundary row be `2 r_1 - r_2 <= 0`, hence `u_1 <= 0.5 u_2`, and let the physical row be `r_1 + r_2 <= 1`, hence `u_1 + u_2 <= 2`. The lexicographic max-min solution is

```text
u_1 = 2/3, u_2 = 4/3;   r_1 = 1/3, r_2 = 2/3.
```

The minimum tier is bounded by `u_1 = lambda`, `u_2 >= 2 lambda`, and `3 lambda <= 2`. At that tier the second activity can rise to `4/3`. The nonnegative-only equal-rate point `r_1 = r_2 = 1/2` violates the signed row; multiplying both rates by a positive scalar leaves the ratio violation unchanged. The solution needs ratio closure, followed by lexicographic water filling.

## Safe fast-path boundary

Keep `PackingAllocation` for all-nonnegative rows, try ratio closure only when every nontrivial signed row has zero bound and at most one positive plus one negative coefficient, and otherwise use the existing simplex path. Other fallback cases are:

- invalid or out-of-range `Q_i`, `W_o`, `rho`, path gain, or resource coefficient;
- a one-sided positive row unless its zero-exclusion/restart behavior is implemented;
- a gain-amplifying ratio cycle unless its forced-zero closure and restart are implemented;
- multiple same-sign coefficients in a signed row, nonzero signed-row bounds, or a non-packing physical row;
- inability to identify a forced lexicographic bottleneck, a zero tier without owner-Q restart, or ambiguous floating-point ties/cycle products;
- failure to certify the final candidate against the original rows.

The local ratio closure need not cap the total active reference count at three: a global water fill can aggregate per-owner/execution closure blocks through the physical rows. If future code preserves that decomposition, many independent OS owners can share a resource without falling back solely because the global `refs` vector is larger than three. The current generic LP remains the semantic fallback for every unsupported case.
