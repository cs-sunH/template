#!/usr/bin/env python3
"""Offline RR enumeration vs fixed prefix/middle/drain slot bounds."""
import argparse
import json
import pathlib

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
scale, checks = 20, 0
violations = []
for cores in range(1, 9):
    for blocks in range(2 * cores + 1, 6 * cores + 1):
        assignment = [block % cores for block in range(blocks)]
        rounds = (blocks + cores - 1) // cores
        last_begin = (rounds - 1) * cores
        # Count physical blocks in each fixed partition independently.
        remaining = [assignment[2 * cores:].count(core) for core in range(cores)]
        middle = [assignment[cores:last_begin].count(core) for core in range(cores)]
        prefix_store = [assignment[:last_begin - cores].count(core) for core in range(cores)]
        last = [assignment[last_begin:].count(core) for core in range(cores)]
        for compute in range(scale + 1):
            for load in range(scale + 1):
                middle_input = scale + (rounds - 3) * load - (rounds - 2) * compute
                if not 0 <= middle_input <= scale:
                    continue
                for core in range(cores):
                    actual = scale + remaining[core] * load - middle[core] * compute
                    checks += 1
                    if not 0 <= actual <= 2 * scale:
                        violations.append(['input', cores, blocks, core, load, compute])
            for store in range(scale + 1):
                visible = scale + (rounds - 3) * compute - (rounds - 2) * store
                if not 0 <= visible <= scale:
                    continue
                for core in range(cores):
                    actual_visible = scale + (middle[core] - 1) * compute - prefix_store[core] * store
                    actual_output = actual_visible + compute  # withheld penultimate
                    checks += 1
                    if not (0 <= actual_visible <= scale and 0 <= actual_output <= 2 * scale):
                        violations.append(['output', cores, blocks, core, store, compute])
        # All Last inputs and penultimate outputs exist at the bulk terminal.
        for core in range(cores):
            terminal_input = 1 + remaining[core] - middle[core]
            terminal_output = 1 + middle[core] - prefix_store[core]
            checks += 1
            if terminal_input != last[core] or terminal_output != 1:
                violations.append(['terminal', cores, blocks, core])
result = {'scope': 'offline physical slot invariants; not latency/hardware calibration',
          'cores': [1, 8], 'blocks': '2*cores+1 through 6*cores', 'executions': 1,
          'progress_grid_denominator': scale, 'checks': checks,
          'violations': violations, 'passed': not violations}
pathlib.Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
raise SystemExit(bool(violations))
