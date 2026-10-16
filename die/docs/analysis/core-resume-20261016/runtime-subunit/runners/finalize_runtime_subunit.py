#!/usr/bin/env python3
"""Finalize archive metadata after the archival runner's stdout has closed."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

root=Path.cwd();lab=root/'work/runtime-lab'
analysis=root/'die/docs/analysis/core-resume-20261016/runtime-subunit'
evidence=root/'die/docs/evidence/core-resume-20261016/runtime-subunit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=lab/'manifest-before-coupled.json'
if not old.exists():shutil.copy2(lab/'manifest.json',old)
status=json.loads((lab/'manifest.json').read_text())
status.update({'status':'partial_scratch_regressions_not_adopted_existing_snapshot_failures_preserved',
               'readme_sha256':sha(lab/'README.md'),
               'candidate_main_exact_cases':{'passed':42,'failed':6,'total':48},
               'candidate_asymmetric_exact_cases':{'passed':36,'failed':0,'total':36},
               'baseline_main_exact_cases':{'passed':32,'failed':16,'total':48},
               'original_wrong_expectation_attempt':{'passed':39,'failed':9,'total':48,'status':'fixture_oracle_assumption_error_preserved'},
               'layered_assessment_sha256':sha(lab/'layered-assessment.json'),
               'unit_invariance_sha256':sha(lab/'unit-invariance.json'),
               'coupled_identity_sha256':sha(lab/'coupled-identity.json'),
               'asymmetric_identity_sha256':sha(lab/'asymmetric-identity.json'),
               'baseline_coupled_identity_sha256':sha(lab/'baseline-coupled-identity.json'),
               'final_clean_build_api_budget':'not_run','extreme_units':'not_run'})
(lab/'manifest.json').write_text(json.dumps(status,indent=2)+'\n')
manifest=json.loads((evidence/'artifact-manifest.json').read_text())
def replace(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    key=str(dst.relative_to(root));manifest['artifacts']=[r for r in manifest['artifacts'] if r['archive']!=key]
    manifest['artifacts'].append({'source':str(src.relative_to(root)),'archive':key,
        'source_sha256':sha(src),'archive_sha256':sha(dst),'source_bytes':src.stat().st_size,
        'archive_bytes':dst.stat().st_size,'compression':None})
replace(lab/'manifest.json',evidence/'current/manifest.json')
replace(old,evidence/'current/manifest-before-coupled.json')
replace(lab/'archive.stdout.log',evidence/'current/archive.stdout.log')
replace(lab/'archive.stderr.log',evidence/'current/archive.stderr.log')
replace(Path(__file__),analysis/'runners/finalize_runtime_subunit.py')
manifest['copied_artifact_count']=len(manifest['artifacts'])
manifest['archive_metadata_note']='Archive stdout/stderr copied after original archiver completed; current manifest updated without rewriting original per-run identities. Finalizer source preserved.'
(evidence/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in manifest['artifacts']:
    dst=root/row['archive'];assert sha(dst)==row['archive_sha256']
    data=gzip.decompress(dst.read_bytes()) if row['compression'] else dst.read_bytes()
    assert hashlib.sha256(data).hexdigest()==row['source_sha256']
    assert not data.startswith((b'\x7fELF',b'!<arch>'))
payload=[]
for directory in [analysis,evidence]:
    for p in directory.rglob('*'):
        if p.is_file() and p.name!='SHA256SUMS':payload.append((str(p.relative_to(root)),sha(p)))
(evidence/'SHA256SUMS').write_text(''.join(digest+'  '+name+'\n' for name,digest in sorted(payload)))
verification={'status':'verified','copied_payloads':len(manifest['artifacts']),
              'all_source_and_archive_payload_hashes_match':True,'checksummed_files':len(payload),
              'no_elf_static_archives_objects_cache':True,
              'production_execution_matches_baseline':sha(root/'die/src/Execution.cc')==sha(lab/'baseline/Execution.cc'),
              'candidate_execution_still_matches_public_relink':sha(lab/'candidate/Execution.cc')=='66f34d699d79b3c6dcffc5af04002799f3b1bfa4d4dcb569165825fc6a3f655a',
              'finalizer_sha256':sha(Path(__file__))}
(evidence/'archive-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification))
