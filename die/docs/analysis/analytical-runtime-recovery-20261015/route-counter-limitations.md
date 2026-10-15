# OS route-counter remaining limits

`GemmOSRouteCounter` remains an unintegrated partial helper. Its fixed-size
class-to-q decomposition and Euclidean residue floor-sum provide exact
byte-weighted counts, x/y first moments, and x/y cut counts without walking
tiles or cores. The existing small enumerating test is an offline oracle for
that partial API; this note adds no test result or production precision claim.

The x second moment has a bounded route: for an AP `z_i = a*i+b`,
`x_i = z_i mod nx = z_i - nx*floor(z_i/nx)`, so `sum(x_i^2)` needs only
`sum(floor)`, `sum(i*floor)`, and `sum(floor^2)`. A fixed-degree Euclidean
floor-moment reducer can provide those in logarithmic modulus depth. The y
second moment is the unresolved part. With `C=nx*ny`,

```text
y_i = floor(z_i/nx) - ny*floor(z_i/C)
```

and therefore `sum(y_i^2)` needs the cross moment
`sum(floor(z_i/nx)*floor(z_i/C))` in addition to the two single-modulus
second moments. The current one-modulus floor-sum reducer does not compute
that mixed moment. Summing y-threshold counts one row at a time costs
`O(ny log C)` and is not a permitted production route. A complete helper
needs a proved fixed-degree Euclidean reciprocity for the nested quotient
pair (or an equivalent bounded periodic-polynomial reducer), with checked
intermediate bounds and exhaustive small oracle coverage before it can expose
second moments. No such implementation is claimed here.

Two-row odd-byte apportionment also needs one additive rule shared by every
service subset. Independently applying `ceil(bytes/2)` to each class or
subset does not conserve row totals under subtraction. A possible model
convention is a stable global event-byte order, with each event block `[P,P+B)`
assigned

```text
row0 = ceil((P+B)/2) - ceil(P/2)
row1 = floor((P+B)/2) - floor(P/2).
```

This conserves both rows for every partition. If event payload parity is
constant within each `(K role, access class)` block, its row-0 correction can
be counted from the parity of the block's class-local event rank using fixed
AP/prefix counts. Before using this, the complete ordering across execution,
K roles, access classes, and q must be specified and used by all first-load,
prefix, remaining, final-wave, and drain selections. It is only a stable
aggregate conservation policy; the current model has no physical per-byte
HBM-row address evidence. No odd-row exact-subflow implementation is claimed.

Consequently the helper must retain its explicit partial status and must not
replace the current route approximation or feed a production HOP calculation.
