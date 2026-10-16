#!/usr/bin/env python3
"""Preserve current text evidence; reject binary/build outputs and overwrites."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[4]
source = root / 'work/runtime-continue'
destination = root / 'die/docs/evidence/runtime-continue-20261016/public-ab'
destination.mkdir(parents=True, exist_ok=False)
sha = lambda b: hashlib.sha256(b).hexdigest()
entries = []
for f in sorted(source.rglob('*')):
    if not f.is_file() or f.suffix not in ('.json', '.log', '.gz', '.py'):
        continue
    if f.name in ('api-comparison.json',):
        continue  # Superseded by the preserved and actually executed final tool.
    data = f.read_bytes()
    if data.startswith(b'\x7fELF') or data.startswith(b'!<arch>'):
        raise RuntimeError('unexpected binary ' + str(f))
    rel = f.relative_to(source)
    target = destination / rel
    if len(data) > 262144 and f.suffix != '.gz':
        target = target.with_name(target.name + '.gz')
        saved = gzip.compress(data, mtime=0)
    else:
        saved = data
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(saved)
    entries.append(dict(original=str(f.relative_to(root)), saved=str(target.relative_to(destination)),
                        original_sha256=sha(data), saved_sha256=sha(saved), original_bytes=len(data)))
manifest = dict(scope='fresh A+B public build/regressions/API/matrix and failed budgets; no SDK publication',
                archive_script_sha256=sha(Path(__file__).read_bytes()), entries=entries)
(destination / 'artifact-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for f in (root / 'work/runtime-ab-sdk/manifest.json', root / 'work/runtime-ab-sdk/protobuf-generation.json'):
    shutil.copy2(f, destination / ('sdk-prepared-' + f.name))
print(json.dumps(dict(files=len(entries), destination=str(destination))))
