from __future__ import annotations
import hashlib, json, os, shutil
from pathlib import Path

repo = Path('/workspace/template')
work = repo / 'work/integration-input-20261015'
snapshot = work / 'frozen-89'
stage = work / 'stage-root'
variants = ('astra-sim-wscllm', 'astra-sim-wscllm-LRU', 'astra-sim-face',
            'astra-sim-face-LRU', 'astra-sim-joint')

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def release_manifest(source: Path) -> dict:
    files = [source / 'CMakeLists.txt']
    for name in ('include', 'src', 'configs'):
        files += sorted(p for p in (source / name).rglob('*') if p.is_file())
    entries = {str(path.relative_to(source)): sha(path) for path in files}
    identity = hashlib.sha256(json.dumps(entries, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return {'schema': 'coarse_die_release_v2', 'source_digest': identity,
            'files': entries}

if stage.exists():
    raise SystemExit(f'refusing to replace existing stage: {stage}')
stage.mkdir(parents=True)
(stage / '.git').symlink_to(repo / '.git', target_is_directory=True)
manifest = release_manifest(snapshot)
source_map = {}
for variant in variants:
    original = repo / variant
    destination = stage / variant
    destination.mkdir()
    for child in original.iterdir():
        if child.name == 'extern':
            continue
        (destination / child.name).symlink_to(child.resolve(),
            target_is_directory=child.is_dir())
    original_extern = original / 'extern'
    staged_extern = destination / 'extern'
    staged_extern.mkdir()
    for child in original_extern.iterdir():
        if child.name == 'die_model':
            legacy = staged_extern / 'die_model'
            shutil.copytree(child, legacy, symlinks=True)
            coarse = legacy / 'coarse'
            shutil.rmtree(coarse)
            shutil.copytree(snapshot, coarse)
            (coarse / 'RELEASE.json').write_text(
                json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            source_map[variant] = {
                'original_variant': str(original),
                'staged_variant': str(destination),
                'legacy_tree_copied_from': str(child),
                'coarse_snapshot': str(coarse),
                'release': manifest,
            }
        else:
            (staged_extern / child.name).symlink_to(child.resolve(),
                target_is_directory=child.is_dir())
(work / 'stage-manifest.json').write_text(json.dumps({
    'source_commit': '89c2499c6e61de450291961d800594a8a83eeca7',
    'source_snapshot_tar_sha256': sha(work / 'die-89c2499.tar'),
    'release': manifest,
    'variants': source_map,
}, indent=2) + '\n', encoding='utf-8')
(work / 'build-map.json').write_text(json.dumps({
    variant: str(work / 'build' / variant) for variant in variants
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({'stage_root': str(stage), 'release_source_digest': manifest['source_digest'],
                  'release_files': len(manifest['files']), 'variants': list(variants)}, indent=2))
