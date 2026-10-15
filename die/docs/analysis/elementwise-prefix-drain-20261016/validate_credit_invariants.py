#!/usr/bin/env python3
"""Offline integer-grid check against explicit invocation-local RR blocks.

This checks a buffer invariant, not model accuracy or independent hardware data.
No block enumeration is used by the production compiler/runtime.
"""
import argparse
import json
import pathlib

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
scale = 20
checks = 0
violations = []
for cores in range(1, 9):
    for blocks in range(1, 6 * cores + 1):
        active = min(cores, blocks)
        for repeats in range(1, 6):
            counts = [0] * active
            last = [0] * active
            for execution in range(repeats):
                for block in range(blocks):
                    counts[block % active] += 1
                    if execution == repeats - 1 and block >= (blocks - 1) // active * active:
                        last[block % active] = 1
            maximum = max(counts)
            if maximum < 2:
                continue
            for compute in range(scale + 1):
                for load in range(scale + 1):
                    if not 0 <= (maximum - 1) * (load - compute) <= 2 * scale:
                        continue
                    for core, count in enumerate(counts):
                        checks += 1
                        if not 0 <= (count - 1) * (load - compute) <= 2 * scale:
                            violations.append(['input', cores, blocks, repeats, core, load, compute])
                for store in range(scale + 1):
                    most_loaded_visible = scale + (maximum - 2) * compute - (maximum - 1) * store
                    if not 0 <= most_loaded_visible <= scale:
                        continue
                    for core, count in enumerate(counts):
                        visible = scale + (count - 1 - last[core]) * compute - (count - last[core]) * store
                        hidden = last[core] * compute
                        checks += 1
                        if not (0 <= visible <= scale and 0 <= visible + hidden <= 2 * scale):
                            violations.append(['output', cores, blocks, repeats, core, store, compute])
result = {'scope': 'integer-grid per-Core physical slot invariants; no latency/hardware accuracy claim',
          'cores': [1, 8], 'blocks_max': '6*cores', 'repeats': [1, 5],
          'progress_grid_denominator': scale, 'checks': checks,
          'violations': violations, 'passed': not violations}
pathlib.Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
raise SystemExit(bool(violations))
