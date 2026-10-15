from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys
parser=argparse.ArgumentParser(description="Run real config checks with explicit five-repository build paths")
parser.add_argument('--repo-root',type=Path,required=True)
parser.add_argument('--builds',type=Path,required=True,help='JSON mapping each active variant name to its build directory')
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
root=args.repo_root.resolve();work=args.output_dir.resolve();work.mkdir(parents=True,exist_ok=True)
builds={variant:Path(folder).resolve() for variant,folder in json.loads(args.builds.read_text()).items()}
expected={'astra-sim-wscllm','astra-sim-wscllm-LRU','astra-sim-face','astra-sim-face-LRU','astra-sim-joint'}
if set(builds)!=expected:raise ValueError('build map must cover exactly the five active repositories')
results=[]
for variant,build in builds.items():
 environment=os.environ.copy()
 libraries={'DIE_MODEL_CABI_PATH':build/'lib/libdie_model_cabi.so',
            'DIE_COARSE_CABI_PATH':build/'lib/libcoarse_die_cabi.so'}
 for name,path in libraries.items():
  if not path.is_file():raise FileNotFoundError(path)
  environment[name]=str(path)
 command=[sys.executable,'-m','unittest','discover','-s',str(root/variant/'sh_test_mesh/tests'),'-p','test_config_resolver*.py','-v']
 result=subprocess.run(command,cwd=root,env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (work/(variant+'-config-tests.log')).write_text(result.stdout)
 count=re.search(r'Ran (\d+) tests',result.stdout)
 row={'variant':variant,'command':command,'exit_code':result.returncode,'tests':int(count.group(1)) if count else None,
      'libraries':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in libraries.items()},
      'release':json.loads((root/variant/'extern/die_model/coarse/RELEASE.json').read_text())}
 results.append(row);print(json.dumps({k:row[k] for k in ('variant','tests','exit_code')}),flush=True)
(work/'five-abi-config-results.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(row['exit_code']==0 for row in results)
assert len({row['libraries']['DIE_COARSE_CABI_PATH']['sha256'] for row in results})==1
