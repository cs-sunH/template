#!/usr/bin/env python3
"""Capture current-model public outputs, including cached/streamed Norm."""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary
out=pathlib.Path(a.output)
if out.exists():p.error('fresh output required')
lib=CoarseDieLibrary(a.library)
config=json.loads((root/'die/configs/example_hardware_explicit.json').read_text())
tasks=json.loads((root/'die/configs/example_tasks.json').read_text())['specs']
tasks += [{'op_kind':'norm','tensor_shapes':{'m':m,'hidden':h},'storage_dtype':'bf16','layer_instances':r}
          for m,h,r in [(8,4096,1),(128,19569,1),(5,257,3)]]
tasks += [{'op_kind':'residual','tensor_shapes':{'m':1,'hidden':257},'layer_instances':3,'storage_dtype':'bf16'}]
rows=[]
for task in tasks:
    model=lib.create_model(config)
    try:rows.append({'task':task,'hardware':lib.hardware_summary(model),'estimate':lib.estimate_batch(model,[task])})
    finally:lib.destroy_model(model)
out.write_text(json.dumps({'library':str(lib.path),'library_sha256':hashlib.sha256(lib.path.read_bytes()).hexdigest(),
    'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'rows':rows},indent=2)+'\n')
print(json.dumps({'cases':len(rows),'library_sha256':hashlib.sha256(lib.path.read_bytes()).hexdigest()}))
