#!/usr/bin/env python3
"""P8 large-R repeat budget measurement (research track, 2026-10-17).

Reproducible entry: compiles tools/p8_repeat_budget_driver.cc against a
Release libcoarse_die.a, first runs a small cross-check probe (the ranking
reconstruction must match compile()'s winner latency and candidate count),
then the full interleaved R-level measurement, and writes raw JSON + identity
into an evidence directory.

Machine-independent exact event counts are the primary metric; wall times are
same-window interleaved samples on this single machine (absolute wall levels
await the second machine, batched with P15-L2).

Usage: python3 die/tools/run_p8_budget.py --output-dir <dir>
          [--build-dir die/build-release] [--hardware die/configs/example_hardware.json]
          [--levels 16,64,256,1024,4096] [--rounds 7] [--timeout 1800]
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


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def compile_driver(build_dir, output_dir):
    lib = os.path.join(build_dir, "libcoarse_die.a")
    if not os.path.exists(lib):
        raise SystemExit("missing %s; build Release first" % lib)
    exe = os.path.join(output_dir, "p8_repeat_budget_driver")
    cmd = ["g++", "-std=c++17", "-O3", "-DNDEBUG",
           "-I", os.path.join(ROOT, "die/src"),
           "-I", os.path.join(ROOT, "die/include"),
           "-I", os.path.join(ROOT, "astra-sim-wscllm/extern/helper"),
           os.path.join(HERE, "p8_repeat_budget_driver.cc"), lib, "-o", exe]
    build = subprocess.run(cmd, capture_output=True, text=True)
    errors = (build.stdout + build.stderr).count("error:")
    if errors or build.returncode != 0:
        sys.stderr.write(build.stdout + build.stderr)
        raise SystemExit("driver compile failed (%d errors)" % errors)
    return exe, lib


def run_driver(exe, hardware, rounds, levels, reconstruct, out_path, timeout):
    cmd = [exe, hardware, str(rounds), "1" if reconstruct else "0"] + [
        str(level) for level in levels]
    with open(out_path, "w") as handle:
        run = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE,
                             text=True, timeout=timeout)
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        raise SystemExit("driver exit %d (see %s)" % (run.returncode, out_path))
    with open(out_path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-dir", default=os.path.join(ROOT, "die/build-release"))
    parser.add_argument("--hardware", default=os.path.join(ROOT, "die/configs/example_hardware.json"))
    parser.add_argument("--levels", default="16,64,256,1024,4096")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    levels = [int(piece) for piece in args.levels.split(",") if piece.strip()]

    exe, lib = compile_driver(args.build_dir, out)

    identity = {"head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "lib_sha256": sha(lib),
                "driver_source_sha256": sha(os.path.join(
                    HERE, "p8_repeat_budget_driver.cc")),
                "platform": platform.platform(),
                "hardware": os.path.relpath(args.hardware, ROOT),
                "levels": levels,
                "rounds": args.rounds,
                "started_unix": time.time()}

    if not args.skip_probe:
        probe = run_driver(exe, args.hardware, 1, [2, 16], True,
                           os.path.join(out, "probe.json"), 600)
        identity["probe_driver_sha256"] = sha(exe)
        checked = [row for row in probe["rows"] if "cross_check_ok" in row]
        if not checked or not all(row["cross_check_ok"] for row in checked):
            raise SystemExit("probe cross-check failed: the reconstructed "
                             "ranking trial stream does not match compile()")

    measure = run_driver(exe, args.hardware, args.rounds, levels, True,
                         os.path.join(out, "measure.json"), args.timeout)
    identity["finished_unix"] = time.time()
    identity["measure_driver_sha256"] = sha(exe)
    with open(os.path.join(out, "identity.json"), "w") as handle:
        json.dump(identity, handle, indent=2)

    recon = [row for row in measure["rows"] if "ranking_events_total" in row]
    print("levels:", levels, "rounds:", args.rounds)
    for row in recon:
        print("R=%d candidates=%d ranking_events=%d advance_events=%d "
              "compile_wall_ms=%.1f advance_wall_ms=%.1f" % (
                  row["level_layers"], row["reconstructed_candidates"],
                  row["ranking_events_total"], row["advance_events"],
                  row["compile_wall_ns"] / 1e6, row["advance_wall_ns"] / 1e6))
    if not all(row.get("cross_check_ok", False) for row in recon):
        raise SystemExit("measure cross-check failed")
    print("cross-check ok; evidence in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
