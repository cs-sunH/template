#!/usr/bin/env python3
"""Capture untouched operators through a fresh process / one model per task."""
import argparse
import copy
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary
out=pathlib.Path(a.output)
if out.exists():p.error('fresh output required')
library=CoarseDieLibrary(a.library)
config=json.loads((root/'die/configs/example_hardware_explicit.json').read_text())
tasks=json.loads((root/'die/configs/example_tasks.json').read_text())['specs']
tasks=[task for task in tasks if task.get('op_kind')!='norm']
tasks.append({'op_kind':'residual','tensor_shapes':{'m':1,'hidden':257},'storage_dtype':'bf16','layer_instances':3})
rows=[]
for task in tasks:
    model=library.create_model(config)
    try:rows.append({'task':task,'hardware':library.hardware_summary(model),'estimate':library.estimate_batch(model,[task])})
    finally:library.destroy_model(model)
out.write_text(json.dumps({'library':str(library.path),'library_sha256':hashlib.sha256(library.path.read_bytes()).hexdigest(),
    'probe_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'rows':rows},indent=2)+'\n')
print(json.dumps({'cases':len(rows),'library_sha256':hashlib.sha256(library.path.read_bytes()).hexdigest()}))
