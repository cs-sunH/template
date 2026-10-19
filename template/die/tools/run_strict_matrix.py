#!/usr/bin/env python3
"""Reproducible entry for the 84-case strict matrix (baseline v3 P1.4).

Compiles both drivers against a Release build of libcoarse_die.a, runs them,
runs the exact-Fraction oracles, runs the P1 counterfactual replay, and emits a
single verdict:

  expected: 78 passed + 6 named (c)-exemption cases
            (same_owner_split x owners{2,8} x unit_exponent{0,-10,-60})
  verdict "ok" iff coupled failures == the six expected cases, asymmetric has
  zero failures, and the replay confirms single-source linear propagation for
  every same_owner_split case. Anything else -> verdict "unexpected" (exit 1).

Usage: python3 die/tools/run_strict_matrix.py --output-dir <dir> [--build-dir die/build-release]
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUB = os.path.join(ROOT, "die/docs/analysis/core-resume-20261016/runtime-subunit")
REPLAY = os.path.join(ROOT, "die/docs/analysis/legacy-closeout-20261017/P1-REPLAY/replay_counterfactual.py")
EXPECTED_FAILED = {("same_owner_split", o, e) for o in (2, 8) for e in (0, -10, -60)}

def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "die/build-release"))
    args = ap.parse_args()
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    lib = os.path.join(args.build_dir, "libcoarse_die.a")
    if not os.path.exists(lib):
        print("missing %s; build Release first" % lib)
        return 2
    identity = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "lib_sha256": sha(lib), "platform": platform.platform(), "started_unix": time.time()}
    families = []
    for name, oracle in [("coupled_credit_driver", "coupled_credit_oracle.py"),
                          ("asymmetric_credit_driver", "asymmetric_credit_oracle.py")]:
        exe = os.path.join(out, name)
        cmd = ["g++", "-std=c++17", "-O3", "-DNDEBUG",
               "-I", os.path.join(ROOT, "die/tests"), "-I", os.path.join(ROOT, "die/include"),
               "-I", os.path.join(ROOT, "astra-sim-wscllm/extern/helper"),
               "-DCOARSE_DIE_SOURCE_DIR=" + chr(34) + ROOT + "/die" + chr(34),
               "-DCOARSE_DIE_ROOT_DIR=" + chr(34) + ROOT + chr(34),
               os.path.join(SUB, "tests", name + ".cc"), lib, "-o", exe]
        build = subprocess.run(cmd, capture_output=True, text=True)
        errors = (build.stdout + build.stderr).count("error:")
        if errors or build.returncode != 0:
            print("compile failed for %s (%d errors)" % (name, errors))
            return 2
        identity[name + "_sha256"] = sha(exe)
        jsonl = os.path.join(out, name + ".jsonl")
        with open(jsonl, "w") as f:
            run = subprocess.run([exe], stdout=f, stderr=subprocess.PIPE, text=True, timeout=600)
        if run.returncode != 0:
            print("driver %s exit %d" % (name, run.returncode))
            return 2
        res = os.path.join(out, name + "-results.json")
        orc = subprocess.run([sys.executable, os.path.join(SUB, "runners", oracle), jsonl, res],
                             capture_output=True, text=True)
        print(oracle, "->", orc.stdout.strip())
        families.append((name, json.load(open(res))))
    replay_out = os.path.join(out, "replay-results.json")
    rp = subprocess.run([sys.executable, REPLAY,
                         os.path.join(out, "coupled_credit_driver.jsonl"), replay_out],
                        capture_output=True, text=True)
    print("replay ->", rp.stdout.strip())
    replay = json.load(open(replay_out))
    identity["finished_unix"] = time.time()
    with open(os.path.join(out, "identity.json"), "w") as f:
        json.dump(identity, f, indent=2)
    failed = {(c["family"], c["owners"], c["unit_exponent"])
              for c in families[0][1]["cases"] if c["status"] == "failed"}
    verdict = {
        "coupled": {"passed": families[0][1]["passed"], "failed": families[0][1]["failed"]},
        "asymmetric": {"passed": families[1][1]["passed"], "failed": families[1][1]["failed"]},
        "failed_set": sorted(map(list, failed)),
        "replay_single_source": replay["single_source_confirmed"],
        "replay_case_count": replay["case_count"],
    }
    verdict["ok"] = (failed == EXPECTED_FAILED
                     and families[1][1]["failed"] == 0
                     and replay["single_source_confirmed"] == replay["case_count"] == 12)
    verdict["verdict"] = ("78-passed + 6 named (c)-exemption (single-source proven)"
                          if verdict["ok"] else "unexpected")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
