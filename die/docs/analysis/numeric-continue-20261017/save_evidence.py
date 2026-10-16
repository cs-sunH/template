#!/usr/bin/env python3
"""Freeze only source/text observations, with verified tar roundtrip hashes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
paths = set()
for name in ('work/runtime-resume-20261017', 'work/numeric-review-20261017'):
    paths.update(p for p in (root / name).rglob('*') if p.is_file() and
                 (p.suffix in ('.cc', '.hh', '.h', '.py', '.json', '.log', '.md', '.patch') or
                  p.name.endswith('.json.gz') or p.name == 'CMakeLists.txt') and
                 '__pycache__' not in p.parts)
build = root / 'work/build-numeric-final-20261017'
paths.update(p for p in (build / 'numerical').rglob('*.json') if p.is_file())
paths.update(p for p in (build / 'Testing').rglob('*.log') if p.is_file())
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = []
archive = out / 'text-source-and-results.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for path in sorted(paths):
        data = path.read_bytes()
        assert b'\0' not in data or path.name.endswith('.json.gz'), 'not a text artifact: ' + str(path)
        name = str(path.relative_to(root))
        item = tarfile.TarInfo(name)
        item.size, item.mtime, item.mode = len(data), 0, 0o644
        tar.addfile(item, io.BytesIO(data))
        manifest.append({'path': name, 'bytes': len(data), 'sha256': sha(data)})
with tarfile.open(archive) as tar:
    assert tar.getnames() == [item['path'] for item in manifest]
    for item in manifest:
        data = tar.extractfile(item['path']).read()
        assert len(data) == item['bytes'] and sha(data) == item['sha256']
report = {'scope': 'source/text only, no ELF/object/library/cache; evidence not acceptance',
          'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
          'production_execution_sha256': sha((root / 'die/src/Execution.cc').read_bytes()),
          'archive_sha256': sha(archive.read_bytes()), 'archive_bytes': archive.stat().st_size,
          'file_count': len(manifest), 'files': manifest}
(out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('file_count', 'archive_bytes', 'archive_sha256')}))
