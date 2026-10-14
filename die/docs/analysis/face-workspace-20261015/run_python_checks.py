#!/usr/bin/env python3
"""Run exact staged FACE Python sources with real protoc and both supplied ABIs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=['astra-sim-face', 'astra-sim-face-LRU'])
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--legacy-library', required=True, type=Path)
    parser.add_argument('--coarse-library', required=True, type=Path)
    parser.add_argument('modules', nargs='+')
    args = parser.parse_args()
    root = (args.repo_root or Path(subprocess.check_output([
        'git', '-C', str(Path(__file__).resolve().parent),
        'rev-parse', '--show-toplevel'], text=True).strip())).resolve()
    source = root / args.variant
    work = (args.work_dir or root / 'work/face-workspace-python' / args.variant).resolve()
    # The deletion is confined to this explicit intermediate directory.
    if not work.is_relative_to(root / "work") or work == root / "work":
        raise SystemExit('work-dir must be an isolated intermediate directory')
    stage = work / 'stage'
    if stage.exists():
        shutil.rmtree(stage)
    work.mkdir(parents=True, exist_ok=True)
    supplied = {}
    for name, library in [('legacy', args.legacy_library), ('coarse', args.coarse_library)]:
        if not library.is_absolute() or not library.is_file():
            raise SystemExit(name + ' library must be an existing absolute file')
        supplied[name] = library.resolve()
    shutil.copytree(source / 'sh_test_mesh', stage / 'sh_test_mesh',
                    ignore=shutil.ignore_patterns('generated', 'runs', '__pycache__', '*.pyc', '.pytest_cache'))
    relative_proto = Path('extern/graph_frontend/chakra/schema/protobuf/et_def.proto')
    (stage / relative_proto.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / relative_proto, stage / relative_proto)
    subprocess.run(['protoc', '--proto_path=' + str(source),
                    '--python_out=' + str(stage), str(relative_proto)], cwd=source, check=True)
    shutil.copytree(source / 'extern/die_model/coarse/configs',
                    stage / 'extern/die_model/coarse/configs')
    workload = Path('sh_test_mesh/workload/llama2_7b_inference')
    hashes = {}
    for path in sorted((source / workload).rglob('*.py')):
        relative = path.relative_to(source)
        if path.read_bytes() != (stage / relative).read_bytes():
            raise RuntimeError('staged source differs: ' + str(relative))
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(map(str, [stage, stage / 'sh_test_mesh', stage / workload]))
    environment['DIE_MODEL_CABI_PATH'] = str(supplied['legacy'])
    environment['DIE_COARSE_CABI_PATH'] = str(supplied['coarse'])
    environment['FACE_WORKSPACE_EVIDENCE_DIR'] = str(work / 'emitter-evidence')
    command = [sys.executable, '-m', 'unittest', '-v', *args.modules]
    (work / 'identity-and-command.json').write_text(json.dumps({
        'scope': 'isolated exact-source Python tests; real generated protobuf and actual two ABIs',
        'variant': args.variant,
        'source_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': hashes,
        'schema_sha256': hashlib.sha256((source / relative_proto).read_bytes()).hexdigest(),
        'library_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in supplied.values()},
        'command': command, 'cwd': str(stage / workload),
        'environment': {name: environment[name] for name in ['PYTHONPATH', 'DIE_MODEL_CABI_PATH',
              'DIE_COARSE_CABI_PATH', 'FACE_WORKSPACE_EVIDENCE_DIR']},
    }, indent=2) + '\n')
    # Ordinary module/root discovery sees only real protoc output. No marker,
    # generator, router, or model function is replaced.
    result = subprocess.run(command, cwd=stage / workload, env=environment)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
