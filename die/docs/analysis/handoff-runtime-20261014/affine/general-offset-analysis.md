# What blocks a general offset-to-affine compiler

The fixed-offset result reduces this fixture to one continuous boundary coordinate only because the exact repeated lifecycle places the other active progresses on affine lines in that coordinate. The current exact one-output map has `owner0_C = u(x)`, `owner1_L = v(x)`, and a constant `owner1_C`; those relations were derived and checked for offset `1/20`. They do not follow from “two owners” alone.

## State a generic descriptor has to preserve

For the three-stage L/C/S lifecycle analyzed here, a conservative continuous boundary state has up to `3J` active progress coordinates for `J` owners, one per active stage and owner. A return map can use fewer coordinates only after proving exact affine relations or another invariant subspace. The one-coordinate fixture map is therefore a proven quotient for this orbit, not a generic state shape.

The bounded discrete state also has to retain, at minimum:

- active stage and tile identity relative to the current K/output boundary;
- owner load/retirement cursors, including phase modulo `Ktiles` and the first/final-K flags;
- the ordered loaded/computed queue contents and input/output slot occupancy;
- output row-major position and full-M/full-N versus tail class;
- the active grouped LOCAL-Q allocation class and selected LP basis, plus any resource-capacity or service-rate class that changes the rate map;
- the exact tie-breaking order for simultaneous event deadlines.

Time, operation, byte, and busy-union totals are additive affine outputs, not generally scheduler state. They still need fixed ledger slots in a composed descriptor. Current fixture output tracks ten such values, including separate HBM payloads and their busy union. LOCAL-Q bytes are a mapped C-stage ledger here; they are not an extra physical LP capacity.

For a fixed integer geometry and fixed `J`, lifecycle slot bounds and K phase make the *discrete* possibilities finite. A simple unminimized state bound grows at least with K phases and the combinations of per-owner stage/queue occupancy; a useful compiler needs a symbolic bound for its selected quotient, not the assertion that a finite trace happened to repeat. Continuous progress creates an additional exact partition problem.

## Category generation that is still missing

At one discrete state, each active grouped activity has an exact completion-time affine function of the boundary progress vector. The next event is the minimum of those functions, with the runtime's deterministic tie rule. A discovery algorithm must form the exact guard for every candidate winner, intersect those guards with the current rational polyhedral region, and compose the resulting progress and ledger maps. After a completion it must update queues/cursors and re-solve the grouped LOCAL-Q rates for the new active set.

That operation generates a graph whose nodes include discrete lifecycle state and progress-region cell. A bounded compiler must then prove that the graph closes under a finite set of event templates or return-map categories, including exact ties, and that repeated affine returns stay in their guard cells. Merely observing a repeated active set or a repeated trace suffix does not prove that closure. A changed offset can move a boundary state across a deadline guard, select another LP basis, or change the K phase; those changes can split a cell or replace the whole cycle.

The nested fixture established a small category table from the saved trace: K-first, a 58-step middle identity run, and six K/output-boundary categories, each represented by four event intervals. It independently re-solved 32 representative grouped-LP intervals and matched those templates to the saved positions. That is valuable compression of this trace, but representative selection and category discovery are not yet a general symbolic algorithm. It does not establish a global category-count bound for other offsets, owners, capacities, or lifecycle queues.

## Route to a bounded descriptor

For one fixed rational offset and fixed geometry, a credible compiler path is:

1. Construct the exact initial lifecycle state, owner phase, queue contents, and active progress vector from the runtime descriptor.
2. Enumerate bounded lifecycle/phase states symbolically, canonicalizing tile IDs by translation while retaining K/output/tail class and the absolute cursors needed for external-stop bounds.
3. For each candidate cell, independently solve the grouped LOCAL-Q LP for every active set and retain the exact basis inequalities that make its rates valid.
4. Split each cell by exact deadline guards, apply the production tie rule, and derive rational affine progress/ledger transforms for each selected event.
5. Find a finite return in the discrete graph; intersect all pulled-back guards to obtain one strict return-map domain; prove its exact orbit or invariant set remains inside that domain.
6. Derive a symbolic repeated-middle run and a first-tail/cancellation/stop bound from cursors and queue slots. Only then lower the bounded graph into a fixed-size descriptor and interval-powered evaluator.

In the general case, each event guard is an affine inequality in at most `3J` progress variables, so its region is rational polyhedral. The number of cells can grow rapidly with active deadline competitors and LP bases. A compiler needs an explicit upper bound or a fail-closed descriptor-size cap; otherwise a fixed runtime representation is not established. If the domain is a multi-dimensional polytope, checking a handful of trajectory probes is insufficient: strict inequalities need exact vertex/LP bounds or a proved invariant set.

An arbitrary real-valued offset adds a parameter dimension and guard boundaries where event order or LP basis changes. Supporting a parameter range then means compiling a finite partition of offset space, including tie surfaces; it cannot be inferred by interpolating two fixed-offset maps. A first practical target should be one declared rational offset and geometry, followed by a fixed two-owner phase class. Until that symbolic closure is implemented, no “flat 256” or arbitrary-offset compiler claim is justified.
