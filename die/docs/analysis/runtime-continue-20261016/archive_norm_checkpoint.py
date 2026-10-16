#!/usr/bin/env python3
"""Archive a frozen Norm checkpoint without modifying ongoing experiments."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[4]
source = root / 'work/norm-runtime-current/checkpoint-1'
destination = root / 'die/docs/evidence/runtime-continue-20261016/norm-capacity1'
destination.mkdir(parents=True, exist_ok=False)
manifest_path = source / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
files = [manifest_path]
for row in manifest['files']:
    f = source / row['saved_path']
    data = f.read_bytes()
    assert sha(data) == row['saved_sha256'] == row['original_sha256'], str(f)
    assert not data.startswith((b'\x7fELF', b'!<arch>')), str(f)
    files.append(f)
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w') as archive:
    for f in sorted(files):
        data = f.read_bytes()
        entry = tarfile.TarInfo(str(f.relative_to(source)))
        entry.size, entry.mtime, entry.mode = len(data), 0, 0o644
        archive.addfile(entry, io.BytesIO(data))
blob = gzip.compress(buffer.getvalue(), mtime=0)
archive_path = destination / 'checkpoint-1-text.tar.gz'
archive_path.write_bytes(blob)
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as archive:
    for f in files:
        assert archive.extractfile(str(f.relative_to(source))).read() == f.read_bytes()
identity = dict(scope=manifest['scope'], file_count=len(manifest['files']),
                checkpoint_manifest_sha256=sha(manifest_path.read_bytes()),
                archive_sha256=sha(blob), archive_bytes=len(blob),
                archiver_sha256=sha(Path(__file__).read_bytes()),
                all_files_hash_verified=True, roundtrip_byte_equal=True,
                production_runtime_source='unchanged A+B; this candidate is not adopted')
(destination / 'archive-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
print(json.dumps(identity))
