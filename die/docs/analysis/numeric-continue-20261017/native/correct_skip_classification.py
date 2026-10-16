#!/usr/bin/env python3
"""Preserve the first export; distinguish CTest skips from business Skipped."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'die/docs/evidence/numeric-continue-20261017/native'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
initial = OUT / 'checkpoint-summary.json'
summary = json.loads(initial.read_text())
summary['initial_export_summary_sha256'] = sha(initial)
summary['skip_classification_correction'] = {
    'reason': 'Original strict runner matches any word Skipped, including successful assertions of the business terminal status Skipped. Actual CTest skip records are classified separately.',
    'original_runner_logs_and_checks_unchanged': True,
    'overall_failed_status_unchanged': 'all three lifecycle processes failed at the real R4 complete-prediction guard',
    'initial_export_manifest': 'manifest.json',
    'initial_exporter_source': 'die/docs/analysis/numeric-continue-20261017/native/export_evidence_before_skip_classification.py'}
for repo, case in summary['native'].items():
    phase = OUT / 'results' / (repo + '-native-consumers')
    log = (phase / 'run.log').read_text()
    identity = json.loads((phase / 'identity.json').read_text())
    lines = [line for line in log.splitlines() if re.search(r'^\s*\d+/\d+\s+Test\b', line)]
    assert len(lines) == case['tests'] == 4
    passed = [line for line in lines if re.search(r'\bPassed\b', line)]
    failed = [line for line in lines if 'Subprocess aborted' in line]
    skipped = [line for line in lines if re.search(r'\*\*\*Skipped|\(Skipped\)|\*\*\*Not Run', line, re.I)]
    assert len(passed) == 3 and len(failed) == 1 and not skipped
    case.update(skipped=False, skipped_test_count=0,
                legacy_runner_skip_word_flag=bool(re.search(r'\bskipped\b|Not Run', log, re.I)),
                legacy_runner_zero_skips_check=identity['checks']['zero_skips'],
                ctest_result_lines=lines)
target = OUT / 'checkpoint-summary-corrected.json'
assert not target.exists()
target.write_text(json.dumps(summary, indent=2) + '\n')
files = {str(p.relative_to(OUT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
         for p in sorted(OUT.rglob('*')) if p.is_file()}
manifest = OUT / 'corrected-manifest.json'
assert not manifest.exists()
manifest.write_text(json.dumps({'scope': 'successor text manifest; initial export remains byte-identical',
    'files': files, 'corrector_sha256': sha(__file__),
    'initial_exporter_source_sha256': sha(Path(__file__).with_name('export_evidence_before_skip_classification.py'))}, indent=2) + '\n')
print(json.dumps({'native': summary['native'], 'files': len(files) + 1, 'corrected_summary': str(target)}, indent=2))
