import collections,hashlib,json,pathlib,shutil
root=pathlib.Path('/workspace/template');e=root/'die/docs/evidence/elementwise-drain-20261016/baseline';current=e.parent/'extra';rows=[]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(report,run,semantics):
 c=report['comparisons'][run][semantics]['comparison']
 return {'status':c['status'],'completion':c['completion'],'first_output_readiness':c['first_output_readiness'],'busy_original_definition':c['busy_original_definition']}
for path in sorted((e/'extra').glob('*.report.json')):
 name=path.name.removesuffix('.report.json');a=json.loads(path.read_text());other=current/path.name;b=json.loads(other.read_text())
 base={run:{sem:metrics(a,run,sem) for sem in a['comparisons'][run]} for run in a['comparisons']};phase3={run:{sem:metrics(b,run,sem) for sem in b['comparisons'][run]} for run in b['comparisons']}
 refs_equal=all(a['comparisons'][run][sem]['reference']==b['comparisons'][run][sem]['reference'] for run in a['comparisons'] for sem in a['comparisons'][run]);assert refs_equal
 row={'case':name,'baseline_ledger':a['ledger']['status'],'phase3_ledger':b['ledger']['status'],'baseline_report_sha256':h(path),'phase3_report_path':str(other.relative_to(root)),'phase3_report_sha256':h(other),'references_exactly_equal':refs_equal,'baseline':base,'phase3':phase3}
 rows.append(row)
 print(name,'solo latency error %',*[round(r['solo']['invocation_barrier']['completion'][0]['relative_error']*100,6) for r in [base,phase3]],'solo compute busy error %',*[round(r['solo']['invocation_barrier']['busy_original_definition']['compute_busy_s']['relative_error']*100,6) for r in [base,phase3]])
summary={'scope':'nine new offline single-Core synthetic cases only; frozen a0d baseline versus previously recorded phase3; no hardware calibration','baseline_commit':'a0d327aeeb4a966faab56b426ed88d61e5607478','phase3_source_identity_path':'die/docs/evidence/elementwise-drain-20261016/extra-cases-identity.json','phase3_source_identity_sha256':h(e.parent/'extra-cases-identity.json'),'conclusion':'Balanced511/512 solo completion >20% failures already exist at a0d, with exactly the same actual/reference times as phase3. Phase3 fixes compute-busy overstatement for these cases but does not fix their completion error.','case_count':len(rows),'all_baseline_ledgers_pass':all(r['baseline_ledger']=='passed' for r in rows),'all_references_equal':all(r['references_exactly_equal'] for r in rows),'cases':rows}
(e/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(root/'work/elementwise-drain-baseline/run_baseline.py',e/'run_baseline.py');shutil.copy2(pathlib.Path(__file__),e/'summarize.py')
(e/'files.sha256.json').write_text(json.dumps({str(x.relative_to(e)):h(x) for x in sorted(e.rglob('*')) if x.is_file() and x.name!='files.sha256.json'},indent=2)+'\n')
