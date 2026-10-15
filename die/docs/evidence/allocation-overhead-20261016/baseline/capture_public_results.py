#!/usr/bin/env python3
"""Actual public C ABI output capture, excluding host timing measurements."""
import argparse,copy,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary
cfg=json.loads((root/'die/configs/example_hardware.json').read_text());tasks=json.loads((root/'die/configs/example_tasks.json').read_text())['specs']
repeated=copy.deepcopy(tasks[0]);repeated['layer_instances']=3;tasks.append(repeated)
lib=CoarseDieLibrary(a.library);model=lib.create_model(cfg)
compiled=None
try:
 compiled=lib.compile_batch(model,tasks)
 data={'hardware':lib.hardware_summary(model),'tasks':tasks,'compile':compiled,'estimate':lib.estimate_batch(model,tasks)}
finally:
 if compiled:
  for row in compiled['results']:
   if row is not None:lib.release_plan(row['plan_handle'])
 lib.destroy_model(model)
pathlib.Path(a.output).write_text(json.dumps({'library_sha256':hashlib.sha256(lib.path.read_bytes()).hexdigest(),'results':data},indent=2)+'\n')
