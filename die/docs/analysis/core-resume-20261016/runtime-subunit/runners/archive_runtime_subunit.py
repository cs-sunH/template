#!/usr/bin/env python3
"""Preserve source/text evidence only; no production edits or ELF/a/o/cache."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess

root=Path.cwd();lab=root/'work/runtime-lab'
analysis=root/'die/docs/analysis/core-resume-20261016/runtime-subunit'
evidence=root/'die/docs/evidence/core-resume-20261016/runtime-subunit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
def copy(src,dst,compress=False):
    dst.parent.mkdir(parents=True,exist_ok=True)
    data=src.read_bytes()
    assert not data.startswith((b'\x7fELF',b'!<arch>')),'binary archive forbidden: '+str(src)
    assert src.suffix not in ['.a','.o','.so','.pyc'],'compiled input forbidden: '+str(src)
    if compress:
        dst=dst.with_name(dst.name+'.gz');dst.write_bytes(gzip.compress(data,mtime=0))
        assert gzip.decompress(dst.read_bytes())==data
    else:shutil.copy2(src,dst)
    records.append({'source':str(src.relative_to(root)),'archive':str(dst.relative_to(root)),
                    'source_sha256':hashlib.sha256(data).hexdigest(),'archive_sha256':sha(dst),
                    'source_bytes':len(data),'archive_bytes':dst.stat().st_size,'compression':'gzip' if compress else None})

assert sha(lab/'candidate/Execution.cc')=='66f34d699d79b3c6dcffc5af04002799f3b1bfa4d4dcb569165825fc6a3f655a'
assert sha(lab/'baseline/Execution.cc')==sha(root/'die/src/Execution.cc')
for folder in [analysis,evidence]:folder.mkdir(parents=True,exist_ok=True)
copy(lab/'README.md',analysis/'README.md')
for name in ['subunit-curved-credit.patch','baseline/Execution.cc','candidate/Execution.cc']:
    copy(lab/name,analysis/name)
for name in ['scaled_credit_snapshot_guard_test.cc','coupled_credit_driver.cc','asymmetric_credit_driver.cc']:
    copy(lab/name,analysis/'tests'/name)
for p in lab.glob('*.py'):copy(p,analysis/'runners'/p.name)
for name in ['manifest.json','focused-identity.json','public-relinked-identity.json','coupled-identity.json',
             'baseline-coupled-identity.json','asymmetric-identity.json','coupled-results.json',
             'baseline-coupled-results.json','asymmetric-results.json','unit-invariance.json',
             'baseline-comparison.json','layered-assessment.json']:
    copy(lab/name,evidence/'current'/name)
for p in lab.glob('*.log'):
    copy(p,evidence/'current'/p.name,compress=p.stat().st_size>512*1024)
for folder in ['coupled-attempt-1','layered-attempt-1']:
    for p in (lab/folder).iterdir():
        if p.suffix in ['.cc','.py']:copy(p,analysis/folder/p.name)
        else:copy(p,evidence/folder/p.name,compress=p.stat().st_size>512*1024)
baseline=root/'work/norm-audit/current-baseline'
for name in ['runtime_credit_slack_test.cc','synchronous_three_activity_candidate.cc']:
    copy(baseline/name,analysis/'tests'/name)
for name in ['identity.json','sync-identity.json','synchronous-contract.json','README.md']:
    copy(baseline/name,evidence/'baseline'/name)
for p in baseline.glob('*.log'):copy(p,evidence/'baseline'/p.name)
for name in ['review.md','sub-epsilon-review.md','manifest-verification.json']:
    copy(root/'work/norm-audit'/name,analysis/'historical-review'/name)
copy(lab/'public-build/CTestTestfile.cmake',evidence/'public-relink/CTestTestfile.cmake')
copy(lab/'public-build/Testing/Temporary/LastTest.log',evidence/'public-relink/LastTest.log')

runner_status={
 'baseline_compile_run_wrapper':{'status':'not_preserved','description':'Python heredoc wrapper not saved; exact build/run argument arrays, source/library identities, return codes and logs retained in baseline identity.json'},
 'synchronous_compile_run_wrapper':{'status':'not_preserved','description':'Python heredoc wrapper not saved; full actual arguments and identities in sync-identity.json'},
 'initial_public_relink_runner':{'status':'not_preserved','executed_sha256':'dfc9ec918ac5d6dcc522ab25599b59c680714b2493d45f23ab643a41a7a63b21','description':'Executed version was changed after missing-ctest error; not reconstructed. Corrected archived run_public_relinked.py is future-only relative to that completed run.'},
 'public_ctest_resume_and_extra_wrapper':{'status':'not_preserved','description':'Actual Python heredoc wrapper not saved; exact subprocess commands and return codes in public-relinked-identity.json'},
 'focused_runner':{'status':'preserved','path':'runners/run_focused.py','qualification':'Current source copy preserved; no runner SHA recorded at historical execution start'},
 'coupled_attempt_1':{'status':'preserved','path':'coupled-attempt-1/run_coupled.py','qualification':'Executed SHA bound in that attempt identity'},
 'current_coupled_baseline_asymmetric':{'status':'preserved','paths':['runners/run_coupled.py','runners/run_baseline_coupled.py','runners/run_asymmetric.py'],'qualification':'Each actual runner SHA recorded by its execution identity'},
 'layered_attempt_1':{'status':'preserved','path':'layered-attempt-1/assess_coupled.py'},
 'current_layered_and_summaries':{'status':'preserved','paths':['runners/assess_coupled.py','runners/summarize_coupled.py','runners/compare_coupled_baseline.py'],'qualification':'Each output records executed runner SHA; no candidate build or budget performed'},
}
(analysis/'runner-source-status.json').write_text(json.dumps(runner_status,indent=2)+'\n')
(analysis/'SCOPE.md').write_text('''# Archived scratch experiment, not adopted

The README summarizes actual results. Evidence is the sibling docs/evidence/
core-resume-20261016/runtime-subunit tree. Original identities retain absolute
work paths and source hashes; artifact-manifest.json maps them to these copies.
Full large driver stdout files use lossless gzip, with original and compressed
SHA256 recorded. No ELF, library, object, cache or previous account artifact is
included. Production code/config/tests and formal vendors were not changed.

Current fixture drivers are under tests/, oracles and runners under runners/.
asymmetric_credit_driver.cc includes coupled_credit_driver.cc from the same
tests directory only to reuse input/observation helpers. Fraction expectations
are independent explicit formulas. For the original runner layout, restore
these files under work/runtime-lab, original tiny/sync inputs under
work/norm-audit/current-baseline, and use a genuinely fresh current-source public
build at work/build-core-release. Do not treat missing libraries as permission
to reuse old ELF. The checked source/headers/configs live in the repository;
their exact hashes are retained in each identity. No restoration was performed
by this archival script.

The source in coupled-attempt-1 and layered-attempt-1 is retained verbatim for
the original failed/mistaken test interpretation. It is not the final oracle.
Evidence/baseline/README.md is an older preparation note that said synchronous
was unrun. Its actual later sync-identity.json and full logs show build0/run0;
the actual result is pass/non-reproducer, as the current README states.

runner-source-status.json explicitly identifies unsaved executed wrappers and
the corrected future public-relink runner. No missing old source was invented.
The missing-ctest launch127 was not a test execution. The resumed51/51 used the
candidate C ABI and candidate static library, verified by member/ELF/loader
identities, but is a relink regression using this session's unchanged fresh
objects, not a final clean CMake build. Source-only/Python tests are identified
in public-relinked-identity.json and are not claimed to execute Runtime.

Combined candidate exact matrix:78/84 passed,6 failed; do not rewrite to84/84.
The six failed split imports remain not_run for cancel/completion. Separate
synthetic relative-error diagnostics meet10% by a very large margin on observed
samples, without waiving exact assertions, snapshot errors or correctness.
Original machine-defined snapshot/terminal ledger checks pass with no new
guard. Three public projected sums are non-additive after separate binary64
rounding; original mistaken assessment retained before correcting that method.

Open risks: arbitrary unit invariance is not proved; extreme subnormal/large
units not_run; relative guards may still erase tiny physical remainder. The
candidate leaves positive-event/time epsilon/progress thresholds unchanged,
does not address capacity1 Norm or saved-allocation snapshot reconstruction,
and has no final clean source build/API comparison/performance budget. It is
not target hardware calibration, SDK publication or engineering acceptance.
''')
manifest={'scope':'source and complete text-only scratch evidence; not production adoption',
          'head_at_archive':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'archive_runner_sha256':sha(Path(__file__)),
          'production_execution_sha256':sha(root/'die/src/Execution.cc'),
          'candidate_execution_sha256':sha(lab/'candidate/Execution.cc'),
          'baseline_fresh_library_sha256':sha(root/'work/build-core-release/libcoarse_die.a'),
          'candidate_pic_library_sha256':sha(lab/'public-build/libcoarse_die.a'),
          'candidate_cabi_sha256':sha(lab/'public-build/lib/libcoarse_die_cabi.so'),
          'copied_artifact_count':len(records),'artifacts':records,
          'not_archived':['ELF','static/shared libraries','objects','cache','unsaved old wrapper sources'],
          'original_exact_candidate_cases':{'passed':78,'failed':6,'total':84},
          'baseline_main_matrix':{'passed':32,'failed':16,'total':48}}
(evidence/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(evidence/'README.md').write_text('''# Runtime subunit evidence — not adopted

Read ../../../analysis/core-resume-20261016/runtime-subunit/README.md and SCOPE.md.
artifact-manifest.json maps original source/log paths and SHA256 to every copied
file. Full driver stdout is gzip-compressed without loss; decompress before
feeding the exact oracles. Original source/log hashes still match the identities.
current/ contains actual candidate public relink51/51, focused11/11, main42/48,
asymmetric36/36, baseline32/48, exact invariance and layered diagnostics.
baseline/ contains actual current-Core tiny red and sync pass/non-reproducer.
The two attempt-1 directories preserve prior failed/mistaken analysis unchanged.
No compiled artifact or cache is archived. No production code was changed.
''')
# Verify payloads after copying, including decompression, and reject binaries.
for item in records:
    p=root/item['archive'];assert sha(p)==item['archive_sha256']
    data=gzip.decompress(p.read_bytes()) if item['compression'] else p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==item['source_sha256']
print(json.dumps({'copied_artifacts':len(records),'all_payload_sha256_verified':True,
                  'source_bytes':sum(r['source_bytes'] for r in records),
                  'archive_bytes':sum(r['archive_bytes'] for r in records)}))
