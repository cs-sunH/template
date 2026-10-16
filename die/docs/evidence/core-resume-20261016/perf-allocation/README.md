# Current Core allocation diagnosis

The diagnostic rebuilt the existing tracked phase probe against this session's
fresh baseline and the isolated three-patch candidate. Every build, command,
input, static/shared library and resulting executable is identified. Only five
instrumented samples were taken; unrelated functional builds/tests could run.
These are allocation counts and thread-CPU observations, never replacement
performance acceptance samples. The earlier normal30 wall-clock failures stand.

The C++ `new` counts and requested bytes are identical across the five samples
of each phase. They do not include every possible libc allocation. Complete
compiler winners have equal hashes in both versions and configurations.

For 128-cubed, complete-compile C++ allocations decrease from 14864 to 13352
on example hardware, and 15343 to 13621 on the engineering point. At OS J32,
example requested bytes decrease from 31112 to 16048; the engineering point
decreases from 32768 to 18152. Counts establish the intended storage reduction.
Measured thread-CPU changes are mixed: these observations do not establish a
tail-latency cure. In particular, the engineering-point OS path is ratio, so
packing scratch reuse is not credited for that path.

Reproduction tool: `../../../analysis/core-resume-20261016/run_allocation_diagnostic.py`.
Its source inputs are the separately identified baseline and isolated candidate,
not production changes or old-account executables. `comparison.json` summarizes
the phases; raw samples and original command logs are retained here.
