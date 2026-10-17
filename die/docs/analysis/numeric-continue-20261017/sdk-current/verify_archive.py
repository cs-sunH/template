#!/usr/bin/env python3
"""Read-only archive verification plus explicit canonical command hashes."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import tarfile

ROOT = Path(__file__).resolve().parents[5]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', choices=('abi-python', 'native'), required=True)
args = parser.parse_args()
base = ROOT / 'die/docs/evidence/numeric-continue-20261017/sdk-current'
folder = base / args.phase
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = json.loads((folder / 'manifest.json').read_text())
for rel, meta in manifest['files'].items():
    data = (folder / rel).read_bytes()
    assert sha(data) == meta['sha256'] and len(data) == meta['bytes']
commands = {}
with tarfile.open(fileobj=io.BytesIO(gzip.decompress((folder / 'results-text.tar.gz').read_bytes()))) as archive:
    for row in manifest['archive_entries']:
        data = archive.extractfile(row['path']).read()
        assert sha(data) == row['sha256'] and len(data) == row['bytes']
        if row['path'].startswith('results/') and row['path'].endswith('/identity.json'):
            identity = json.loads(data)
            payload = {key: identity[key] for key in ('command', 'cwd', 'environment')}
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
            commands[identity['label']] = {'command_cwd_environment_sha256': sha(encoded),
                                           'identity_sha256': sha(data), 'status': identity['status'],
                                           'recorded_log_sha256': identity.get('log_sha256')}
assert len(commands) == (33 if args.phase == 'abi-python' else 16)
target = base / ('verification-' + args.phase + '.json')
assert not target.exists()
target.write_text(json.dumps({'phase': args.phase, 'archive_manifest_sha256': sha((folder / 'manifest.json').read_bytes()),
    'files_verified': len(manifest['files']), 'text_entries_verified': len(manifest['archive_entries']),
    'command_hash_encoding': 'UTF-8 json.dumps({command,cwd,environment}, sort_keys=True, separators=(comma,colon), ensure_ascii=False)',
    'commands': commands, 'verifier_sha256': sha(Path(__file__).read_bytes())}, indent=2) + '\n')
print(json.dumps({'phase': args.phase, 'commands_verified': len(commands), 'text_entries_verified': len(manifest['archive_entries'])}))
