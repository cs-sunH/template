#!/usr/bin/env python3
"""Produce a work-only Execution copy; assert exact anchors, no solver rewrite."""
import argparse
import pathlib
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[4]
out=pathlib.Path(a.output)
if out.exists():p.error('fresh output required')
source=(root/'die/src/Execution.cc').read_text()
source='#include "'+str(pathlib.Path(__file__).resolve().with_name('trace.hh'))+'"\n'+source
for name,key,args in [('update_capacity','Update','std::size_t resource, double rate,\n                              std::uint64_t now_ns'),
                      ('advance_to','Advance','std::uint64_t absolute_ns'),
                      ('integrate','Integrate','long double delta_ns'),('settle','Settle',''),('allocate','Allocate','')]:
    anchor=f'void Runtime::{name}({args}) {{'
    assert source.count(anchor)==1,anchor
    source=source.replace(anchor,anchor+'\n  update_trace::Scope trace_scope(update_trace::'+key+');')
anchor='  last_arbitration_={};'
assert source.count(anchor)==1
source=source.replace(anchor,'  update_trace::Timeline trace_timeline;\n'+anchor)
for anchor,key in [('  if (refs.empty()) return;','Collect'),
                   ('  const auto publish = [](Ref& ref, double rate, double alpha) {','Symmetry'),
                   ('  const bool nonnegative=std::all_of(base.begin(),base.end(),[](const auto& row) {','Rows')]:
    assert source.count(anchor)==1,anchor
    source=source.replace(anchor,'  trace_timeline.end(update_trace::'+key+');\n'+anchor)
# Solver cost is Allocate minus the disjoint prefix sections; inclusive calls
# remain separately reported. No extra clock reads in per-activity loops.
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(source)
