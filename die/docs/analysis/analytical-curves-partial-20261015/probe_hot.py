import ctypes,json,time,hashlib
from pathlib import Path
p=Path('work/build-die/lib/libcoarse_die_cabi.so')
l=ctypes.CDLL(str(p.resolve()))
l.die_coarse_create_model_v2.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_uint64)]
l.die_coarse_estimate_cost_batch_v2.argtypes=[ctypes.c_uint64,ctypes.c_char_p]
l.die_coarse_estimate_cost_batch_v2.restype=ctypes.c_void_p
l.die_coarse_free_result_v2.argtypes=[ctypes.c_void_p]
l.die_coarse_destroy_model_v2.argtypes=[ctypes.c_uint64]
h=ctypes.c_uint64()
assert l.die_coarse_create_model_v2(Path('die/configs/example_hardware.json').read_bytes(),ctypes.byref(h))==0
s=json.load(open('die/configs/example_tasks.json'))['specs'][1]
request=json.dumps({'specs':[s]},separators=(',',':')).encode()
raw=l.die_coarse_estimate_cost_batch_v2(h,request)
warm=json.loads(ctypes.string_at(raw)); l.die_coarse_free_result_v2(raw)
assert warm['errors']==[None],warm
samples=[]
for _ in range(100):
    start=time.perf_counter_ns();raw=l.die_coarse_estimate_cost_batch_v2(h,request);stop=time.perf_counter_ns()
    assert raw
    samples.append(stop-start);l.die_coarse_free_result_v2(raw)
l.die_coarse_destroy_model_v2(h)
sorted_times=sorted(samples)
report={'scope':'interim diagnostic only, Python ctypes single hot cost call incl ctypes overhead; no final performance acceptance', 'library_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'samples_ns':samples,'p50_ns':sorted_times[49],'p95_ns':sorted_times[94]}
Path('work/analytical-curves/hot-norm-interim.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='samples_ns'})
