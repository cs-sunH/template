#!/usr/bin/env python3
"""Replay real Runtime middle-pair clock probes; no cycle skipping is applied."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--build-dir', type=Path, default=Path('work/build-clock'))
    parser.add_argument('--output-dir', type=Path, default=Path('work/runtime-clock-replay'))
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    root = args.repo_root.resolve()
    build = (root / args.build_dir).resolve()
    output = (root / args.output_dir).resolve()
    output.relative_to(root / 'work')
    output.mkdir(parents=True, exist_ok=True)
    driver = root / 'die/tests/k_readiness_runtime_driver.cc'
    source = driver.read_text()
    replacements = {
        '#include "compiled_gemm_oracle_driver.cc"': '#include "' + str(root / 'die/tests/compiled_gemm_oracle_driver.cc') + '"',
        '#include "../src/GemmOSReadiness.hh"': '#include "' + str(root / 'die/src/GemmOSReadiness.hh') + '"',
        'int main(int argc, char** argv)': 'int prepared_readiness_driver_entry(int argc, char** argv)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('diagnostic helper source changed: ' + old)
        source = source.replace(old, new)
    prepared = output / 'prepared_readiness_driver.cc'
    prepared.write_text(source)
    probe_source = Path(__file__).with_name('middle_pair_probe.cc')
    probe = output / 'middle_pair_probe.cc'
    probe.write_bytes(probe_source.read_bytes())
    library = build / 'libcoarse_die.a'
    binary = output / 'middle_pair_probe'
    command = [args.compiler, '-std=c++17', '-O2',
               '-DCOARSE_DIE_SOURCE_DIR="' + str(root / 'die') + '"',
               '-I', str(root / 'die/include'),
               '-I', str(root / 'astra-sim-wscllm/extern/helper'),
               str(probe), str(library), '-o', str(binary)]
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / 'result.json').open('w') as log, (output / 'stderr.log').open('w') as error:
        subprocess.run([str(binary)], cwd=root, stdout=log, stderr=error, check=True)
    paths = [driver, prepared, probe, library, binary,
             root / 'die/src/Execution.cc', root / 'die/include/coarse_die/Execution.hh']
    identity = {'scope': 'actual uncompressed isolated-pair clock study, not production lifecycle or efficiency certification',
                'compile_command': command,
                'source_and_binary_sha256': {str(p): digest(p) for p in paths}}
    (output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(output / 'result.json')


if __name__ == '__main__':
    main()
