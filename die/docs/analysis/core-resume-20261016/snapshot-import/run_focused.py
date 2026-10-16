import hashlib,json,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2];lab=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for name in ['service_curve_runtime_test','runtime_clock_translation_test','runtime_owner_symmetry_test','runtime_semantics_test','runtime_counters_test']:
 obj=root/'work/build-core-release/CMakeFiles'/('coarse_'+name+'.dir/tests')/(name+'.cc.o');exe=lab/name;cmd=['c++','-O3','-DNDEBUG',str(obj),str(lab/'libcoarse_die.a'),'-o',str(exe)]
 t=time.monotonic();build=subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(lab/(name+'-link.log')).write_text(build.stdout)
 if build.returncode:raise RuntimeError(build.stdout)
 run=subprocess.run([str(exe)],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(lab/(name+'.log')).write_text(run.stdout)
 row={'test':name,'link_command':cmd,'old_object':str(obj.relative_to(root)),'old_object_sha256':sha(obj),'binary_sha256':sha(exe),'exit':run.returncode,'elapsed_including_link_s':time.monotonic()-t};results.append(row);print(json.dumps(row),flush=True)
 if run.returncode:break
(lab/'focused-results.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
if any(r['exit'] for r in results):raise SystemExit(1)
