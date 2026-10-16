"""Read-only identity/artifact checks and atomic records for SDK verification.

These helpers do not build, publish, or alter a frozen source snapshot.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shlex


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_root(command):
    argv = command['command']
    for flag in ('--build', '--test-dir', '-B'):
        if flag in argv:
            return Path(argv[argv.index(flag) + 1]).resolve()
    return None


def cache_validation(command):
    """Bind a completed configure to its still-live contemporary build tree."""
    argv = command['command']
    build = build_root(command)
    cache = build / 'CMakeCache.txt' if build else None
    if cache is None or not cache.is_file():
        return {'valid': False, 'reason': 'configured CMakeCache.txt missing'}
    values = {}
    for line in cache.read_text().splitlines():
        if line.startswith(('#', '//')) or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.split(':', 1)[0]] = value
    expected = {}
    if '-S' in argv:
        expected['CMAKE_HOME_DIRECTORY'] = str(Path(argv[argv.index('-S')+1]).resolve())
    if '-G' in argv:
        expected['CMAKE_GENERATOR'] = argv[argv.index('-G')+1]
    for arg in argv:
        if arg.startswith('-D') and '=' in arg:
            key, value = arg[2:].split('=', 1)
            expected[key.split(':', 1)[0]] = value
    mismatches = {key: {'expected': value, 'actual': values.get(key)}
                  for key, value in expected.items() if values.get(key) != value}
    return {'valid': not mismatches, 'path': str(cache), 'sha256': sha(cache),
            'expected_configuration': expected, 'mismatches': mismatches}


def command_targets(command, commands):
    argv = command['command']
    if '--target' in argv:
        return argv[argv.index('--target') + 1:]
    if '--test-dir' in argv and '-R' in argv:
        pattern = argv[argv.index('-R') + 1]
        if pattern.startswith('^(') and pattern.endswith(')$'):
            return pattern[2:-2].split('|')
    if command.get('new_build'):
        future = next((item for item in commands
                       if item['label'] == command['label'].removesuffix('-configure')+'-build'), None)
        if future:
            return command_targets(future, commands)
    return []


def artifact_snapshot(command, commands):
    """Resolve real CMake link outputs and CTest executable paths without running."""
    build = build_root(command)
    result = {'build_root': str(build) if build else None, 'targets': {}}
    targets = command_targets(command, commands)
    if not build:
        return result
    ctest_paths = {}
    for file in build.rglob('CTestTestfile.cmake'):
        for line in file.read_text().splitlines():
            found = re.match(r'add_test\((?:\[=\[)?([^\s\]]+)(?:\]=\])?\s+"([^"]+)"', line)
            if found:
                ctest_paths.setdefault(found.group(1), set()).add(Path(found.group(2)).resolve())
    for target in targets:
        paths = set(ctest_paths.get(target, ()))
        link_sources = []
        for file in build.glob('**/CMakeFiles/'+target+'.dir/link.txt'):
            link_sources.append({'path': str(file), 'sha256': sha(file)})
            try:
                tokens = shlex.split(file.read_text())
            except ValueError:
                continue
            if '-o' in tokens:
                value = Path(tokens[tokens.index('-o')+1])
                paths.add((file.parents[2] / value).resolve() if not value.is_absolute() else value.resolve())
        files = {}
        for path in sorted(paths):
            files[str(path)] = {'exists': path.is_file(),
                                'sha256': sha(path) if path.is_file() else None,
                                'bytes': path.stat().st_size if path.is_file() else None}
        result['targets'][target] = {'files': files, 'link_metadata': link_sources,
                                    'resolution': 'resolved' if paths else 'not_yet_generated'}
    return result


def tool_input_snapshot(command):
    argv = command['command']
    paths = []
    for value in argv[1:]:
        candidate = Path(value)
        if candidate.is_absolute() and candidate.is_file() and candidate.suffix in ('.py', '.json'):
            paths.append(candidate)
    if '-S' in argv:
        path = Path(argv[argv.index('-S')+1])/'CMakeLists.txt'
        if path.is_file():
            paths.append(path)
    return {str(path): sha(path) for path in sorted(set(paths))}


def retention(old_root, command, digest, manifest_sha):
    """Validate an old terminal record; mismatches never become retained passes."""
    folder = Path(old_root) / command['label']
    identity = folder / 'identity.json'
    log = folder / 'run.log'
    base = {'source_phase_root': str(folder), 'identity_present': identity.is_file(),
            'started_present': (folder/'started.json').is_file(), 'log_present': log.is_file()}
    if not identity.is_file():
        return {**base, 'retain': False, 'reason': 'incomplete_no_final_identity',
                'prior_log_sha256': sha(log) if log.is_file() else None,
                'prior_started_sha256': sha(folder/'started.json') if (folder/'started.json').is_file() else None}
    try:
        old = json.loads(identity.read_text())
    except (ValueError, OSError) as error:
        return {**base, 'retain': False, 'reason': 'invalid_identity_json', 'error': str(error)}
    reasons = []
    for key in ('label', 'command', 'cwd', 'environment', 'expected', 'requires', 'new_build'):
        if old.get(key) != command.get(key):
            reasons.append(key+'_mismatch')
    if old.get('status') not in ('passed', 'failed') or not isinstance(old.get('return_code'), int):
        reasons.append('not_terminal_completed_status')
    if old.get('source_digest') != digest:
        reasons.append('source_digest_mismatch')
    if old.get('snapshot_manifest_sha256') != manifest_sha:
        reasons.append('snapshot_manifest_sha256_mismatch')
    if not isinstance(old.get('runner_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', old['runner_sha256']):
        reasons.append('original_runner_sha256_missing')
    if not log.is_file() or old.get('log_sha256') != sha(log):
        reasons.append('log_sha256_mismatch')
    if command.get('new_build') and old.get('status') == 'passed':
        cache = cache_validation(command)
        if not cache['valid']:
            reasons.append('live_configured_build_mismatch')
        base['cache_validation'] = cache
    for path, expected in old.get('libraries', {}).items():
        if not Path(path).is_file() or sha(path) != expected:
            reasons.append('recorded_library_changed:'+path)
    if 'tool_inputs_sha256' in old and old['tool_inputs_sha256'] != tool_input_snapshot(command):
        reasons.append('tool_inputs_sha256_mismatch')
    for target, data in old.get('artifacts_after', {}).get('targets', {}).items():
        for path, meta in data.get('files', {}).items():
            if meta.get('exists') and (not Path(path).is_file() or sha(path) != meta.get('sha256')):
                reasons.append('recorded_target_changed:'+target+':'+path)
    return {**base, 'retain': not reasons, 'reason': 'matched' if not reasons else 'identity_mismatch',
            'mismatches': reasons, 'original_status': old.get('status'),
            'original_identity_sha256': sha(identity), 'original_runner_sha256': old.get('runner_sha256'),
            'original_log_sha256': old.get('log_sha256'), 'old_record': old}


def check_frozen_snapshot(stage, snapshot, manifest):
    bad = [rel for rel, expected in manifest['frozen_files_sha256'].items()
           if not (stage/rel).is_file() or sha(stage/rel) != expected]
    expected = manifest.get('helper_source_sha256', {}).get('aggregate/CMakeLists.txt')
    if expected and sha(snapshot/'aggregate/CMakeLists.txt') != expected:
        bad.append('../aggregate/CMakeLists.txt')
    if bad:
        raise RuntimeError('frozen snapshot source changed: '+repr(bad[:12]))
