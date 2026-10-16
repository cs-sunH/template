from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--child',default='');a=p.parse_args()
root=Path(__file__).resolve().parents[2]
if a.child:
 repo=root/a.child;sys.path.insert(0,str(repo/'sh_test_mesh'))
 from config_resolver import derive_coarse_die_hardware,load_hardware_config
 descriptor=json.loads((root/'die/configs/example_hardware.json').read_text())
 summary=derive_coarse_die_hardware(descriptor);derived=summary['derived']
 assert derived['core_resource_table']['id']=='core32_v1'
 assert derived['core']['N_core']==88 and derived['core']['N_vec']==2 and derived['core']['N_pe']==40
 assert 'core_type' not in derived['core']
 import tempfile
 with tempfile.TemporaryDirectory() as tmp:
  authored=json.loads((repo/'sh_test_mesh/hardware/face_case5_config_c.json').read_text())
  authored.pop('die');authored['coarse-die']=descriptor
  capacity=int(derived['hardware_summary']['hbm_capacity_bytes'])
  authored['local-hbm']['capacity-profiles']={'explicit-fixture':{'bytes':capacity,'label':'explicit DOCX fixture','note':'test input, not selected research hardware'}}
  path=Path(tmp)/'typed_fixture.json';path.write_text(json.dumps(authored))
  hardware=load_hardware_config(path,'explicit-fixture')
  assert hardware.local_hbm_capacity_bytes==capacity
  assert hardware.local_hbm_bandwidth_gbps==sum(derived['hardware_summary']['hbm_row_peak_Bps'])/1e9
  assert hardware.die_profile=='coarse_die_v2' and hardware.peak_perf_tflops is None
 rejected=[]
 for value in ('small','large','core',None,1):
  bad=json.loads(json.dumps(descriptor));bad['architecture']['core_type']=value
  try:derive_coarse_die_hardware(bad)
  except ValueError as error:
   assert 'architecture.core_type' in str(error);rejected.append(value)
  else:raise AssertionError('obsolete Core selector accepted')
 print(json.dumps({'repo':a.child,'status':'passed','scope':'actual Python typed hardware ABI; not compiled five-repository C++ application or timing accuracy','coarse_library_path':str(a.library.resolve()),'coarse_library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'summary':summary,'obsolete_selector_values_rejected':rejected},ensure_ascii=False))
 sys.exit(0)
out=root/'work/core-integration-audit/results'/a.label;out.mkdir(parents=True,exist_ok=False)
rows=[]
for repo in ('astra-sim-face','astra-sim-face-LRU','astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-joint'):
 env=os.environ.copy();env['DIE_COARSE_CABI_PATH']=str(a.library.resolve());env['PYTHONDONTWRITEBYTECODE']='1'
 command=[sys.executable,'-B',str(Path(__file__).resolve()),'--library',str(a.library.resolve()),'--label',a.label,'--child',repo]
 started=time.time();result=subprocess.run(command,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (out/(repo+'.stdout.json')).write_text(result.stdout);(out/(repo+'.stderr.log')).write_text(result.stderr)
 rows.append({'repo':repo,'command':command,'return_code':result.returncode,'elapsed_seconds':time.time()-started,'resolver_sha256':hashlib.sha256((root/repo/'sh_test_mesh/config_resolver.py').read_bytes()).hexdigest()})
 print(json.dumps({k:rows[-1][k] for k in ['repo','return_code']}),flush=True)
 if result.returncode:print(result.stderr)
(out/'identity.json').write_text(json.dumps({'scope':'provisional parent public ELF, five actual typed Python resolvers; official vendor unpublished','library_path':str(a.library.resolve()),'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'descriptor_path':str(root/'die/configs/example_hardware.json'),'descriptor_sha256':hashlib.sha256((root/'die/configs/example_hardware.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows},indent=2)+'\n')
raise SystemExit(any(r['return_code'] for r in rows))
