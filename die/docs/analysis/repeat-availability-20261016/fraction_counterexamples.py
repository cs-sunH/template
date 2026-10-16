#!/usr/bin/env python3
"""Offline exact semantics examples; not a production repeat interpreter."""
from fractions import Fraction as F
import json


def run(spec):
    pending = {k: [F(t), list(stages)] for k, (t, stages) in spec.items()}
    live = {}
    done = {}
    trace = []
    t = F(0)
    while pending or live:
        for k, (arrival, stages) in list(pending.items()):
            if arrival <= t:
                resource, work = stages.pop(0)
                live[k] = [resource, F(work), stages]
                del pending[k]
        groups = {}
        for k, (resource, work, stages) in live.items():
            if resource is not None:
                groups.setdefault(resource, []).append(k)
        rates = {
            k: F(1) if resource is None else F(1, len(groups[resource]))
            for k, (resource, work, stages) in live.items()
        }
        choices = [arrival - t for arrival, stages in pending.values()]
        choices += [work / rates[k] for k, (resource, work, stages) in live.items()]
        dt = min(choices)
        trace.append((str(t), str(t + dt), {
            k: str(rates[k]) for k in live if live[k][0] is not None
        }))
        for k in list(live):
            live[k][1] -= rates[k] * dt
        t += dt
        for k in list(live):
            resource, work, stages = live[k]
            if work == 0:
                if stages:
                    next_resource, next_work = stages.pop(0)
                    live[k] = [next_resource, F(next_work), stages]
                else:
                    done[k] = str(t)
                    del live[k]
    return {"finish": done, "trace": trace}


cases = {
    "wait_true": {
        "A": (0, [(None, 1), ("X", 1), (None, 1), ("X", 1)]),
        "B": (2, [("X", 1)]),
    },
    "wait_front": {
        "A": (0, [(None, 2), ("X", 2)]),
        "B": (2, [("X", 1)]),
    },
    "barrier_true": {
        "A": (0, [("C", 1), ("S", 1), ("C", 1), ("S", 1)]),
        "B": (1, [("C", 1)]),
    },
    "barrier_bulk": {
        "A": (0, [("C", 2), ("S", 2)]),
        "B": (1, [("C", 1)]),
    },
}
print(json.dumps({name: run(spec) for name, spec in cases.items()}, sort_keys=True))
