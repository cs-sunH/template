#!/usr/bin/env python3
"""Build and run the corrected diagnostic, keeping raw and oracle files apart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_CASES = ((80, 128, 4096), (80, 128, 4097), (65, 65, 4097))


def run_to_files(command: list[str], cwd: Path, stdout_path: Path,
                 stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        return subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr,
                              check=False).returncode


def main() -> int:
    script = Path(__file__).resolve()
    default_root = script.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--build-dir", type=Path, default=Path("work/build-k-readiness"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("work/k-readiness-corrected"))
    parser.add_argument("--driver", type=Path,
                        help="use this built executable instead of the CMake target")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--skip-build", action="store_true",
                        help="run an already-built driver")
    parser.add_argument("--skip-oracle", action="store_true",
                        help="save raw C++ output without running the grouped oracle")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    build = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    build = build.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output = output.resolve()
    work_root = (root / "work").resolve()
    try:
        output.relative_to(work_root)
    except ValueError as error:
        raise SystemExit("output directory must be inside repo-root/work") from error
    output.mkdir(parents=True, exist_ok=True)

    shape_flags = (args.m, args.n, args.k)
    if any(value is not None for value in shape_flags):
        if any(value is None for value in shape_flags):
            parser.error("--m, --n, and --k must be supplied together")
        if any(value <= 0 for value in shape_flags):
            parser.error("M, N, and K must be positive integers")
        cases = ((args.m, args.n, args.k),)
    else:
        cases = DEFAULT_CASES
    if args.driver:
        driver = args.driver if args.driver.is_absolute() else root / args.driver
        driver = driver.resolve()
    else:
        driver = build / "coarse_k_readiness_runtime_driver"
        if not args.skip_build:
            # Reconfigure even when CMakeCache.txt exists: this target may have
            # been added since the build tree was first generated.
            configure_command = [args.cmake, "-S", str(root / "die"), "-B",
                                 str(build), "-DCMAKE_BUILD_TYPE=Release"]
            result = run_to_files(configure_command, root,
                                  output / "configure.stdout.log",
                                  output / "configure.stderr.log")
            if result != 0:
                print(f"CMake configure failed; see {output / 'configure.*.log'}",
                      file=sys.stderr)
                return result
            build_command = [args.cmake, "--build", str(build), "--target",
                             "coarse_k_readiness_runtime_driver", "--parallel",
                             str(args.parallel)]
            result = run_to_files(build_command, root,
                                  output / "build.stdout.log",
                                  output / "build.stderr.log")
            if result != 0:
                print(f"driver build failed; see {output / 'build.*.log'}",
                      file=sys.stderr)
                return result
    if not driver.is_file():
        print(f"driver executable is missing: {driver}", file=sys.stderr)
        return 2

    oracle = root / "die/validation/grouped_gemm_oracle.py"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "driver": str(driver),
        "diagnostic_only": True,
        "cases": [],
    }
    failures = 0
    for m, n, k in cases:
        stem = f"m{m}_n{n}_k{k}"
        raw = output / f"{stem}.stdout.json"
        stderr = output / f"{stem}.stderr.log"
        command = [str(driver), "--m", str(m), "--n", str(n), "--k", str(k)]
        driver_status = run_to_files(command, root, raw, stderr)
        entry: dict[str, object] = {
            "shape": {"m": m, "n": n, "k": k},
            "driver_command": command,
            "driver_exit_code": driver_status,
            "raw_stdout": str(raw),
            "driver_stderr": str(stderr),
        }
        if driver_status != 0:
            failures += 1
            print(f"driver failed for {stem}; see {stderr}", file=sys.stderr)
        elif not args.skip_oracle:
            grouped = output / f"{stem}.grouped.json"
            oracle_stderr = output / f"{stem}.oracle.stderr.log"
            oracle_command = [sys.executable, str(oracle), "--raw", str(raw),
                              "--output", str(grouped)]
            with oracle_stderr.open("w", encoding="utf-8") as oracle_error:
                oracle_status = subprocess.run(
                    oracle_command, cwd=root, stdout=subprocess.DEVNULL,
                    stderr=oracle_error, check=False).returncode
            entry.update({"oracle_command": oracle_command,
                          "oracle_exit_code": oracle_status,
                          "grouped_report": str(grouped),
                          "oracle_stderr": str(oracle_stderr)})
            if oracle_status != 0:
                failures += 1
                print(f"grouped oracle returned {oracle_status} for {stem}; "
                      f"see {grouped} and {oracle_stderr}", file=sys.stderr)
        manifest["cases"].append(entry)  # type: ignore[union-attr]
        print(f"{stem}: raw={raw.name}" +
              (f" grouped={stem}.grouped.json" if not args.skip_oracle and
               driver_status == 0 else ""))

    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                              encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
