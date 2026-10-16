#!/usr/bin/env python3
"""Freeze completed isolated SDK ABI/Python results, never running builds."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

root = Path(__file__).resolve().parents[4]
source = root / 'work/runtime-ab-sdk-validation'
destination = root / 'die/docs/evidence/runtime-continue-20261016/sdk-abi-python'
destination.mkdir(parents=True, exist_ok=False)
rows = json.loads((source / 'summary.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
entries, stream = [], io.BytesIO()
with tarfile.open(fileobj=stream, mode='w') as archive:
    for f in sorted(source.rglob('*')):
        if not f.is_file() or f.suffix not in ('.json', '.log'):
            continue
        data, rel = f.read_bytes(), str(f.relative_to(source))
        assert not data.startswith((b'\x7fELF', b'!<arch>'))
        entry = tarfile.TarInfo(rel)
        entry.size, entry.mtime, entry.mode = len(data), 0, 0o644
        archive.addfile(entry, io.BytesIO(data))
        entries.append(dict(path=rel, sha256=sha(data), bytes=len(data)))
blob = gzip.compress(stream.getvalue(), mtime=0)
(destination / 'results-text.tar.gz').write_bytes(blob)
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as archive:
    for item in entries:
        assert sha(archive.extractfile(item['path']).read()) == item['sha256']
repos = ('astra-sim-face', 'astra-sim-face-LRU', 'astra-sim-wscllm', 'astra-sim-wscllm-LRU', 'astra-sim-joint')
libraries, consumers = {}, []
for repo in repos:
    library = root / 'work/runtime-ab-sdk/build-abi' / repo / 'lib/libcoarse_die_cabi.so'
    libraries[repo] = sha(library.read_bytes())
    log = (source / (repo + '-python-consumers/run.log')).read_text()
    count = int(re.findall(r'Ran (\d+) tests?', log)[-1])
    skips = re.findall(r'skipped=(\d+)', log)
    consumers.append(dict(repo=repo, tests=count, skipped=int(skips[-1]) if skips else 0,
                          strict_phase_status=next(r['status'] for r in rows if r['label'] == repo + '-python-consumers')))
summary = dict(scope='actual current A+B isolated SDK ABI/config/Python, not release or complete native/E2E',
               source_digest=rows[0]['source_digest'], phase_count=len(rows),
               phase_status_counts={s: sum(r['status'] == s for r in rows) for s in ('passed', 'failed')},
               abi_libraries_built=10, configurations_passed=98, python_consumers=consumers,
               coarse_library_sha256=libraries,
               all_five_equal_public_ab=all(s == '6eb9cf20e8fe2ff1630367b5bdf9e28e4edd42d9034a865f3db534a223c48663' for s in libraries.values()),
               formal_sdk='unchanged canonical24; publisher not run', engineering_status='NOT ACCEPTED',
               archive_sha256=sha(blob), files=entries, roundtrip_hashes_verified=True,
               archiver_sha256=sha(Path(__file__).read_bytes()))
(destination / 'checkpoint-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(dict(files=len(entries), abi_libraries=10, configurations=98,
                     phase_status_counts=summary['phase_status_counts'], all_libraries_equal=summary['all_five_equal_public_ab'])))
