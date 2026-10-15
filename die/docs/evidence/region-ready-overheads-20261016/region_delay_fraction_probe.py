#!/usr/bin/env python3
"""Offline exact Fraction oracle for fixed-width C++ region delay arithmetic."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--driver',type=Path,required=True)
parser.add_argument('--output-prefix',type=Path,required=True)
args=parser.parse_args()
seed=2026101507
rng=random.Random(seed)
maximum=(1<<64)-1
records=[]

def bits(value):
    return struct.unpack('>Q',struct.pack('>d',value))[0]

def value(raw):
    return struct.unpack('>d',struct.pack('>Q',raw))[0]

def append(a,b,n):
    records.append((bits(a),bits(b),n))

counts=(0,1,2,3,7,1<<63,maximum)
for _ in range(8000):
    a=rng.randrange(0x7ff0000000000000)
    b=rng.randrange(0x7ff0000000000000)
    records.append((a,b,rng.choice(counts) if rng.randrange(2) else rng.getrandbits(64)))
for _ in range(8000):
    # Large count and ordinary fractional coefficients exercise all low bits
    # of the 117-bit products, unlike mostly-overflowing broad exponent draws.
    a=((rng.randrange(-120,2)+1023)<<52)|rng.getrandbits(52)
    b=((rng.randrange(-120,2)+1023)<<52)|rng.getrandbits(52)
    records.append((a,b,rng.getrandbits(64)))
for exponent in range(-1074,1):
    for n in (1,3,(1<<63)-1,maximum):
        append(math.nextafter(1.0,0.0),math.ldexp(1.0,exponent),n)
for base in (1,2,3,255,256,1<<31,1<<53,1<<63):
    center=float(base)
    below=math.nextafter(center,0.0)
    above=math.nextafter(center,math.inf)
    gap=center-below
    for a in (below,center,above):
        for b in (0.0,math.ulp(0.0),gap/4,gap/2,gap,2*gap,
                  math.nextafter(gap,0.0),math.nextafter(gap,math.inf)):
            for n in sorted(set((0,1,2,3,maximum,maximum//base))):
                append(a,b,n)
for bad in (-1.0,-math.ulp(0.0),math.inf,-math.inf,math.nan):
    for n in (0,1,maximum):
        append(bad,0.0,n)
        append(0.0,bad,n)
append(-0.0,0.0,maximum)

def expected(a_bits,b_bits,n):
    a,b=value(a_bits),value(b_bits)
    if not math.isfinite(a) or not math.isfinite(b) or a<0 or b<0:
        return 'invalid_input'
    exact=(Fraction.from_float(a)+Fraction.from_float(b))*n
    floor=exact.numerator//exact.denominator
    ceil=-(-exact.numerator//exact.denominator)
    if floor>maximum or ceil>maximum:
        return 'overflow'
    return f'{floor} {ceil}'

prefix=args.output_prefix
prefix.parent.mkdir(parents=True,exist_ok=True)
inputs=''.join(f'{a:016x} {b:016x} {n}\n' for a,b,n in records)
input_path=Path(str(prefix)+'.input.txt')
output_path=Path(str(prefix)+'.output.txt')
error_path=Path(str(prefix)+'.stderr.log')
input_path.write_text(inputs)
run=subprocess.run([str(args.driver.resolve()),'--fraction-driver'],input=inputs,
                   text=True,capture_output=True,check=False)
output_path.write_text(run.stdout)
error_path.write_text(run.stderr)
actual=run.stdout.splitlines()
mismatches=[]
classes={'finite_result':0,'overflow':0,'invalid_input':0}
for i,record in enumerate(records):
    truth=expected(*record)
    classes[truth if truth in classes else 'finite_result']+=1
    got=actual[i] if i<len(actual) else '<missing>'
    if got!=truth:
        mismatches.append({'index':i,'a_bits_hex':f'{record[0]:016x}',
                           'b_bits_hex':f'{record[1]:016x}','count':record[2],
                           'expected':truth,'actual':got})
paths=[args.driver,Path('die/src/RegionDelay.hh'),Path('die/tests/region_delay_math_test.cc'),
       Path(__file__),input_path,output_path,error_path]
summary={'status':'passed' if not mismatches and run.returncode==0 and len(actual)==len(records) else 'failed',
         'scope':'exact binary64 arithmetic only, not timing calibration',
         'reference':'Python Fraction.from_float independently evaluates (a+b)*count',
         'seed':seed,'cases':len(records),'classes':classes,'driver_returncode':run.returncode,
         'driver_output_lines':len(actual),'mismatch_count':len(mismatches),
         'first_mismatches':mismatches[:20],
         'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
Path(str(prefix)+'.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if summary['status']=='passed' else 1)
