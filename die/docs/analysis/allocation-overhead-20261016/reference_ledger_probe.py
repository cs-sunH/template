#!/usr/bin/env python3
"""XLSX residual workload ledger through real C ABI; no latency calibration."""
import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--hardware',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'die/validation'))
from cabi import CoarseDieLibrary
config_path=pathlib.Path(a.hardware).resolve();config=json.loads(config_path.read_text())
task={'op_kind':'residual','tensor_shapes':{'m':1024,'hidden':4096},'storage_dtype':{'activations':'bf16','output':'bf16'},'execution_profile':'fp32_vec_matrix_assumed_v1','layer_instances':1,'weight_passes':1}
lib=CoarseDieLibrary(a.library);model=lib.create_model(config)
try:response=lib.estimate_batch(model,[task])
finally:lib.destroy_model(model)
expected={'elements':1024*4096,'source_vec_ops':1024*4096,'vec_equivalent_ops':1024*4096,'hbm_read_bytes_total':2*2*1024*4096,'hbm_write_bytes_total':2*1024*4096}
row=response['results'][0];checks={k:row[k]==v for k,v in expected.items() if k!='elements'}
data={'scope':'XLSX shape/precision workload accounting on explicitly supplied example hardware only','latency_comparability':'not_available_hardware_and_execution_contract_differ','independent_target_calibration':'not_available','workbook':'TX8_单die建模_精简版.xlsx','workbook_sha256':hashlib.sha256((root/'TX8_单die建模_精简版.xlsx').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.path.read_bytes()).hexdigest(),'hardware_path':str(config_path),'hardware_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'hardware':config,'task':task,'expected_from_shape':expected,'checks':checks,'response':response}
pathlib.Path(a.output).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'checks':checks,'payload_bytes':expected['hbm_read_bytes_total']+expected['hbm_write_bytes_total'],'latency_comparability':data['latency_comparability']}));assert all(checks.values())
