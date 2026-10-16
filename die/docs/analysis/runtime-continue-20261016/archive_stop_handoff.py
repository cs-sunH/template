#!/usr/bin/env python3
"""Save existing stop-time research/native text only; never build or test."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[4]
out = root / 'die/docs/evidence/runtime-continue-20261016/stop-handoff-20261017'
out.mkdir(parents=True, exist_ok=False)
sha = lambda data: hashlib.sha256(data).hexdigest()
extensions = {'.json', '.log', '.md', '.cc', '.hh', '.h', '.py', '.patch',
              '.txt', '.csv', '.tsv', '.sh', '.cmake', '.yaml', '.yml', '.toml', '.proto'}
files = []
for name in ('runtime-ab-native-joint', 'runtime-ab-native-four',
             'runtime-positive-time', 'runtime-positive-time-v2', 'norm-runtime-current'):
    base = root / 'work' / name
    for f in sorted(base.rglob('*')):
        if not f.is_file() or f.is_symlink() or f.suffix not in extensions:
            continue
        if name == 'norm-runtime-current' and any(p in ('checkpoint-1', 'checkpoint-2') for p in f.relative_to(base).parts):
            continue  # Already independently archived with original manifests.
        files.append(f)
for name in ('manifest.json', 'build-map.json', 'protobuf-generation.json'):
    files.append(root / 'work/runtime-ab-sdk' / name)
entries, stream = [], io.BytesIO()
with tarfile.open(fileobj=stream, mode='w') as archive:
    for f in sorted(files):
        data = f.read_bytes()
        assert not data.startswith((b'\x7fELF', b'!<arch>')), str(f)
        data.decode('utf-8')
        rel = str(f.relative_to(root))
        entry = tarfile.TarInfo(rel)
        entry.size, entry.mtime, entry.mode = len(data), 0, 0o644
        archive.addfile(entry, io.BytesIO(data))
        entries.append(dict(path=rel, bytes=len(data), sha256=sha(data)))
blob = gzip.compress(stream.getvalue(), mtime=0)
(out / 'existing-text.tar.gz').write_bytes(blob)
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as archive:
    for row in entries:
        assert sha(archive.extractfile(row['path']).read()) == row['sha256']
        assert sha((root / row['path']).read_bytes()) == row['sha256']
summary = dict(scope='Stopped existing work; no new build/test; research not adopted',
               production_source_sha256=sha((root / 'die/src/Execution.cc').read_bytes()),
               archive_sha256=sha(blob), archive_bytes=len(blob), files=entries,
               all_source_hashes_verified=True, roundtrip_verified=True,
               archiver_sha256=sha(Path(__file__).read_bytes()))
(out / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(dict(files=len(entries), bytes=len(blob), archive_sha256=sha(blob))))
