#!/usr/bin/env python3
"""Save completed current-Runtime SDK phases as verified text-only archives."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[5]
WORK = ROOT / 'work/sdk-numeric-current-20261017'
REPOS = ('astra-sim-face', 'astra-sim-face-LRU', 'astra-sim-wscllm',
         'astra-sim-wscllm-LRU', 'astra-sim-joint')
CANONICAL = '23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('abi-python', 'native'))
    args = parser.parse_args()
    source = WORK / (args.phase + '-results')
    destination = ROOT / 'die/docs/evidence/numeric-continue-20261017/sdk-current' / args.phase
    rows = json.loads((source / 'summary.json').read_text())
    assert len(rows) == (33 if args.phase == 'abi-python' else 16), 'unfinished phase set'
    assert all(r['status'] in ('passed', 'failed', 'not_run_dependency_failed') for r in rows)
    snapshot = json.loads((WORK / 'manifest.json').read_text())
    stage = WORK / 'stage-root'
    assert all(not p.is_symlink() for p in stage.rglob('*'))
    assert all(sha(stage / rel) == want for rel, want in snapshot['frozen_files_sha256'].items())
    destination.mkdir(parents=True, exist_ok=False)
    public = snapshot['public_release']
    production_deltas = {rel: {'frozen_sha256': want, 'production_sha256': sha(ROOT / 'die' / rel)}
                         for rel, want in public['files'].items() if sha(ROOT / 'die' / rel) != want}
    formal = {}
    for repo in REPOS:
        coarse = ROOT / repo / 'extern/die_model/coarse'
        release = json.loads((coarse / 'RELEASE.json').read_text())
        files = {rel: {'expected': want, 'actual': sha(coarse / rel)} for rel, want in release['files'].items()}
        formal[repo] = {'source_digest': release['source_digest'], 'files': files,
                       'canonical24': release['source_digest'] == CANONICAL and len(files) == 24,
                       'all_files_match': all(v['expected'] == v['actual'] for v in files.values())}
    assert all(v['canonical24'] and v['all_files_match'] for v in formal.values())
    write(destination / 'formal-vendor-check.json', formal)

    text_inputs = [(p, str(Path('results') / p.relative_to(source)))
                   for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.json', '.log')]
    text_inputs += [(WORK / name, 'snapshot/' + name) for name in
                    ('manifest.json', 'protobuf-generation.json', 'build-map.json')]
    text_inputs += [(p, 'snapshot/protobuf-generation/' + p.name)
                   for p in sorted((WORK / 'protobuf-generation').glob('*.log'))]
    text_inputs.append((WORK / 'aggregate/CMakeLists.txt', 'snapshot/aggregate/CMakeLists.txt'))
    build_kind = 'build-abi' if args.phase == 'abi-python' else 'build-native'
    libraries = {}
    for repo in REPOS:
        build = WORK / build_kind / repo
        libraries[repo] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size, 'path': str(p)}
                           for p in sorted((build / 'lib').glob('*.so'))}
        for relative in ('CMakeCache.txt', 'CTestTestfile.cmake',
                         'Testing/Temporary/LastTest.log', 'Testing/Temporary/LastTestsFailed.log'):
            p = build / relative
            if p.is_file():
                text_inputs.append((p, str(Path('build-text') / repo / relative)))
        text_inputs += [(p, str(Path('build-text') / repo / p.relative_to(build)))
                        for p in sorted(build.rglob('link.txt'))]
    # CMakeCache is saved as text identity; compiler caches and all binaries are excluded.
    entries = []
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for p, rel in text_inputs:
            data = p.read_bytes()
            assert not data.startswith((b'\x7fELF', b'!<arch>'))
            entry = tarfile.TarInfo(rel)
            entry.size, entry.mtime, entry.mode = len(data), 0, 0o644
            archive.addfile(entry, io.BytesIO(data))
            entries.append({'path': rel, 'original_path': str(p), 'sha256': sha(p), 'bytes': len(data)})
    blob = gzip.compress(stream.getvalue(), mtime=0)
    archive_path = destination / 'results-text.tar.gz'
    archive_path.write_bytes(blob)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as archive:
        for entry in entries:
            data = archive.extractfile(entry['path']).read()
            assert hashlib.sha256(data).hexdigest() == entry['sha256'] and len(data) == entry['bytes']
    public_library = ROOT / 'work/build-numeric-final-20261017/lib/libcoarse_die_cabi.so'
    public_sha = sha(public_library) if public_library.is_file() else None
    suites = {}
    for repo in REPOS:
        phase = repo + ('-python-consumers' if args.phase == 'abi-python' else '-native-consumers')
        row = next(r for r in rows if r['label'] == phase)
        logfile = source / phase / 'run.log'
        if not logfile.is_file():
            suites[repo] = {'status': row['status'], 'tests': None}
            continue
        log = logfile.read_text()
        if args.phase == 'abi-python':
            tests = re.findall(r'Ran (\d+) tests?', log)
            skipped = re.findall(r'skipped=(\d+)', log)
            failures = re.findall(r'failures=(\d+)', log)
            errors = re.findall(r'errors=(\d+)', log)
            total = int(tests[-1]) if tests else None
            skip_count = int(skipped[-1]) if skipped else 0
            failed = (int(failures[-1]) if failures else 0) + (int(errors[-1]) if errors else 0)
            suites[repo] = {'status': row['status'], 'tests': total,
                'passed': total - failed - skip_count if total is not None else None,
                'failed': failed, 'skipped': skip_count,
                'skip_lines': [line for line in log.splitlines() if ' ... skipped ' in line]}
        else:
            result_lines = [line for line in log.splitlines() if re.search(r'^\s*\d+/\d+\s+Test\b', line)]
            totals = re.findall(r'(\d+) tests? failed out of (\d+)', log)
            failed, total = map(int, totals[-1]) if totals else (None, None)
            skipped = [line for line in result_lines if re.search(r'\*\*\*Skipped|\(Skipped\)|\*\*\*Not Run', line, re.I)]
            suites[repo] = {'status': row['status'], 'tests': total,
                'passed': sum(bool(re.search(r'\bPassed\b', line)) for line in result_lines),
                'failed': failed, 'skipped': len(skipped), 'result_lines': result_lines,
                'legacy_runner_zero_skips_check': row.get('checks', {}).get('zero_skips'),
                'r4_body_only_guard_failure_observed': 'prediction_not_available' in log}
    tool_info = {}
    for name in ('cmake', 'ctest', 'g++', 'protoc', 'python'):
        path = shutil.which(name)
        proc = subprocess.run([path, '--version'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tool_info[name] = {'path': path, 'sha256': sha(path), 'version': proc.stdout, 'return_code': proc.returncode}
    write(destination / 'toolchain.json', tool_info)
    summary = {'scope': 'actual fresh frozen current numerical Runtime SDK; unpublished, not full-business acceptance',
        'phase': args.phase, 'source_head_at_freeze': snapshot['source_head'],
        'source_digest': public['source_digest'], 'public_files_sha256': public['files'],
        'frozen_files_checked': len(snapshot['frozen_files_sha256']), 'no_source_symlinks_or_old_build_reuse': True,
        'production_deltas_at_export': production_deltas, 'phase_count': len(rows),
        'phase_status_counts': {status: sum(r['status'] == status for r in rows)
                                for status in sorted({r['status'] for r in rows})},
        'phases': [{'label': r['label'], 'status': r['status'], 'return_code': r.get('return_code')} for r in rows],
        'suites': suites, 'fresh_libraries': libraries,
        'current_public_library': {'path': str(public_library), 'sha256': public_sha},
        'all_five_coarse_equal_current_public': all(libraries[r].get('libcoarse_die_cabi.so', {}).get('sha256') == public_sha for r in REPOS),
        'configurations': [{'repo': r, 'status': next(x['status'] for x in rows if x['label'] == r + '-config')}
                           for r in REPOS] if args.phase == 'abi-python' else 'not rerun in native phase',
        'build_jobs': 2, 'repositories_executed_sequentially': True,
        'formal_sdk': 'five canonical24 actual file hashes unchanged; publisher not run',
        'not_run': ['full five-repository business E2E and CPU<=10% comparison', 'hardware calibration and target raw acquisition',
                    'SDK publication'], 'engineering_status': 'NOT ACCEPTED',
        'archive_sha256': sha(archive_path), 'archive_entries': len(entries), 'archive_roundtrip_verified': True,
        'exported_unix': time.time(), 'archiver_sha256': sha(__file__)}
    write(destination / 'checkpoint-summary.json', summary)
    write(destination / 'manifest.json', {'scope': 'text archive and metadata; no ELF/object/library/compiler cache',
        'archive_entries': entries, 'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                             for p in sorted(destination.iterdir()) if p.is_file()},
        'archiver_sha256': sha(__file__)})
    print(json.dumps({'phase': args.phase, 'statuses': summary['phase_status_counts'],
                      'suites': suites, 'all_five_equal_public': summary['all_five_coarse_equal_current_public'],
                      'archive_entries': len(entries), 'output': str(destination)}, indent=2))

if __name__ == '__main__':
    main()
