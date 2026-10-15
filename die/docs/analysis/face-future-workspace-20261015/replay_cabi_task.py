#!/usr/bin/env python3
"""Replay an exact isolated task batch through the actual coarse C ABI."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.die_coarse_abi_version.restype = ctypes.c_uint32
    if lib.die_coarse_abi_version() != 2:
        raise RuntimeError('expected ABI revision 2')
    lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.die_coarse_create_model_v2.restype = ctypes.c_int
    lib.die_coarse_estimate_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    lib.die_coarse_estimate_batch_v2.restype = ctypes.c_void_p
    lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
    lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
    lib.die_coarse_last_error_v2.restype = ctypes.c_char_p
    handle = ctypes.c_uint64()
    status = lib.die_coarse_create_model_v2(args.hardware.read_bytes(), ctypes.byref(handle))
    if status:
        raise RuntimeError(lib.die_coarse_last_error_v2().decode())
    try:
        pointer = lib.die_coarse_estimate_batch_v2(handle.value, args.task.read_bytes())
        if not pointer:
            raise RuntimeError(lib.die_coarse_last_error_v2().decode())
        try:
            response = ctypes.string_at(pointer)
        finally:
            lib.die_coarse_free_result_v2(pointer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(response)
        result = json.loads(response)
        print(json.dumps({'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
            'hardware_sha256': hashlib.sha256(args.hardware.read_bytes()).hexdigest(),
            'task_sha256': hashlib.sha256(args.task.read_bytes()).hexdigest(),
            'errors': result.get('errors')}, sort_keys=True))
        return int(any(error is not None for error in result.get('errors', [])))
    finally:
        if lib.die_coarse_destroy_model_v2(handle.value):
            raise RuntimeError('model cleanup failed')


if __name__ == '__main__':
    raise SystemExit(main())
