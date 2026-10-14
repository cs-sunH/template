#!/usr/bin/env python3
"""Re-run the preserved pre-fix coarse loader failure against frozen stage code."""
from __future__ import annotations
import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
PREP = HERE / "prepare_tiny_inputs.py"
spec = importlib.util.spec_from_file_location("tiny_prepare", PREP)
assert spec and spec.loader
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
variant = sys.argv[1]
repo_name = prep.SOURCE_VARIANTS[variant]
source = (prep.ROOT / repo_name).resolve()
stage = HERE / "stage" / variant
sh_test = stage / "sh_test_mesh"
workload = sh_test / "workload/llama2_7b_inference"
evidence = HERE / "evidence/coarse-loader-pre-fix" / repo_name
abi = prep.ROOT / "work/five-abi-validation-20261014" / f"build-abi-{repo_name}" / "lib"
die_cabi = abi / "libdie_model_cabi.so"
coarse_cabi = abi / "libcoarse_die_cabi.so"
if not die_cabi.is_file() or not coarse_cabi.is_file():
    raise FileNotFoundError(f"missing original ABI libraries under {abi}")
sys.path[:0] = [str(workload), str(sh_test), str(stage)]
import generate_face_trace as gft
import config_resolver as resolver

gft.SH_TEST_DIR = sh_test
import os
os.environ["DIE_MODEL_CABI_PATH"] = str(die_cabi)
os.environ["DIE_COARSE_CABI_PATH"] = str(coarse_cabi)

hardware, profile, capacity, template = prep._make_coarse_hw_inputs(
    source, sh_test, resolver=resolver
)
queue = workload / "face_tiny_requests.csv"
config_path = workload / "face_tiny_coarse_trace_config.csv"
prep._write_trace_config(
    workload / "trace_config.csv",
    config_path,
    queue=queue,
    model_name=f"face_tiny_{variant}_coarse" if variant == "face" else None,
    hardware_path=hardware.relative_to(sh_test).as_posix(),
    capacity_profile=profile,
    system_template=template,
)
try:
    prep._run_materializer(
        stage, config_path, die_cabi=die_cabi, coarse_cabi=coarse_cabi
    )
except Exception:
    traceback.print_exc()
    raise SystemExit(0)
raise SystemExit("pre-fix coarse loader unexpectedly succeeded")
