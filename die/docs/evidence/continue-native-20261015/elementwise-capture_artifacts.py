import datetime,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/workspace/template');w=root/'work/native-dependency-check/elementwise-preview';e=root/'die/docs/evidence/continue-native-20261015';stage=json.loads((w/'stage-identity.json').read_text());label=sys.argv[1];build=w/'build';sdk=pathlib.Path(stage['stage_root'])/'astra-sim-joint/extern/die_model/coarse'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[sdk/'CMakeLists.txt']
for n in ['src','include','configs']:files+=list((sdk/n).rglob('*'))
manifest={str(p.relative_to(sdk)):sha(p) for p in sorted(files) if p.is_file()}
digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert digest==stage['sdk_release']['source_digest']
outputs=['lib/libcoarse_die_cabi.so','lib/libdie_model_cabi.so','lib/libjoint_projection.so','AstraSim/extern/die_model/coarse/libcoarse_die.a','AstraSim/extern/die_model/libdie_model.a','lib/libjoint_pipeline_core.a','bin/NativeCoarseDieDomainTest','bin/NativeCoarseProjectionTest','bin/NativeProjectionCApiTest','bin/JointPipelineWindowTest','bin/AstraSim_Analytical_Congestion_Aware_Online','lib/libjoint_ledger_cabi.so']
identities={}
for name in outputs:
 p=build/name
 if p.is_file():
  identities[name]={'path':str(p),'sha256':sha(p),'size':p.stat().st_size}
  if '.so' in name or name.startswith('bin/'):
   q=subprocess.run(['readelf','-n',str(p)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);identities[name]['readelf_notes']=q.stdout
   q=subprocess.run(['ldd',str(p)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);identities[name]['ldd']=q.stdout
commands={}
for p in sorted(build.rglob('*')):
 if p.name not in ['flags.make','link.txt']:continue
 if any(x+'.dir' in p.parts for x in ['coarse_die','coarse_die_cabi','die_model','die_model_cabi','joint_pipeline_core','joint_projection','NativeCoarseDieDomainTest','NativeCoarseProjectionTest','NativeProjectionCApiTest','JointPipelineWindowTest','AstraSim_Analytical_Congestion_Aware_Online','joint_ledger_cabi']):commands[str(p.relative_to(build))]={'sha256':sha(p),'contents':p.read_text()}
meta={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision_scope':'before-credit-endpoint-guard','joint_source_commit':stage['joint_source_commit'],'sdk_source_digest':digest,'sdk_source_file_sha256':manifest,'outputs':identities,'compiler_link_commands':commands,'state':'isolated preview; latest main worktree Mapping differs and is not certified by this build'}
(e/('elementwise-'+label+'-artifacts.json')).write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'label':label,'digest':digest,'output_count':len(identities)},indent=2))
