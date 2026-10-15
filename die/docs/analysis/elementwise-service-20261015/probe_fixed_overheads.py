#!/usr/bin/env python3
"""Read-only parameter response probe through the real public C ABI."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--library',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary

config=json.loads((root/'die/configs/example_hardware_explicit.json').read_text())
lib=CoarseDieLibrary(args.library)
tasks=[{'op_kind':op,'tensor_shapes':{'m':1,'hidden':257},'storage_dtype':'bf16'}
       for op in ('residual','norm')]
rows=[]
for launch,setup in ((0,0),(1000000,0),(0,1000000)):
    cfg=copy.deepcopy(config)
    cfg['performance_profile']['launch_ns_per_macro']=launch
    cfg['performance_profile']['memory_setup_ns_per_macro']=setup
    model=lib.create_model(cfg)
    try:
        rows.append({'launch_ns_per_macro':launch,'memory_setup_ns_per_macro':setup,
                     'hardware_summary':lib.hardware_summary(model),
                     'estimate':lib.estimate_batch(model,tasks)})
    finally:
        lib.destroy_model(model)
Path(args.output).write_text(json.dumps({'scope':'parameter response, not calibration',
    'library':str(lib.path),'library_sha256':hashlib.sha256(lib.path.read_bytes()).hexdigest(),
    'tasks':tasks,'rows':rows},indent=2)+'\n')
for row in rows:
    print(row['launch_ns_per_macro'],row['memory_setup_ns_per_macro'],
          [x.get('estimated_latency_ns') for x in row['estimate']['results']])
