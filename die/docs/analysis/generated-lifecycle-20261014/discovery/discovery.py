"""Offline symbolic return discovery, generated from geometry rather than traces.

Synthetic one-Core FP32 OS, exact three-domain grouped allocator, bounded
2-input/2-output queues. Not production, not a granularity certificate.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from fractions import Fraction as F
from copy import deepcopy
from typing import Any
import json,argparse

@dataclass(frozen=True)
class Aff:
    a:F=F(0)
    b:F=F(0)
    def __add__(self,v):
        v=v if isinstance(v,Aff) else Aff(b=F(v));return Aff(self.a+v.a,self.b+v.b)
    __radd__=__add__
    def __neg__(self):return Aff(-self.a,-self.b)
    def __sub__(self,v):return self+-v if isinstance(v,Aff) else self+(-F(v))
    def __rsub__(self,v):return -self+v
    def __mul__(self,q):return Aff(self.a*F(q),self.b*F(q))
    __rmul__=__mul__
    def __truediv__(self,q):return self*(1/F(q))
    def at(self,x):return self.a*x+self.b
    def sub(self,x):return x*self.a+self.b
    def json(self):return {'slope':str(self.a),'intercept':str(self.b)}

@dataclass(frozen=True)
class Geometry:
    M:int=65;N:int=65;K:int=4097
    tm:int=16;tn:int=64;tk:int=64
    compute_rate:F=F(158216000)
    capacities:tuple= (F(1),F(50000000),F(50000000))
    def __post_init__(self):
        if any(v<=0 for v in (self.M,self.N,self.K,self.tm,self.tn,self.tk,self.compute_rate,*self.capacities)):raise ValueError('nonpositive geometry/rate')
    @property
    def mt(self):return (self.M+self.tm-1)//self.tm
    @property
    def nt(self):return (self.N+self.tn-1)//self.tn
    @property
    def kt(self):return (self.K+self.tk-1)//self.tk
    @property
    def outputs(self):return self.mt*self.nt
    @property
    def waves(self):return self.outputs*self.kt
    def facts(self,w):
        if not 0<=w<self.waves:raise ValueError('wave out of range')
        q,kidx=divmod(w,self.kt);mi,ni=divmod(q,self.nt)
        m=min(self.tm,self.M-mi*self.tm);n=min(self.tn,self.N-ni*self.tn);k=min(self.tk,self.K-kidx*self.tk)
        final=kidx==self.kt-1
        read=4*k*(m+n);write=4*m*n if final else 0;ops=2*m*n*k
        local=read+4*m*n+(4*m*n if kidx else 0)
        return (m,n,k,final,kidx==0,read,write,ops,local)
    def work(self,stage,w):
        m,n,k,final,first,read,write,ops,local=self.facts(w)
        if stage==0:return (F(0),F(read,2),F(read,2)),F(read),F(0)
        if stage==1:
            compute=F(ops)/self.compute_rate
            return (compute,F(0),F(0)),F(local),compute
        if not final:raise ValueError('store before final K')
        return (F(0),F(write,2),F(write,2)),F(write),F(0)

@dataclass
class Owner:
    started:bool=False
    next_load:int=0
    retired:int=0
    active:dict=field(default_factory=dict) # stage -> (wave, Aff progress)
    loaded:list=field(default_factory=list)
    computed:list=field(default_factory=list)
    finish:Aff|None=None

LEDGERS=('elapsed','vec_ops','local_bytes','hbm0','hbm1','compute_service',
         'compute_busy','hbm0_busy','hbm1_busy','hbm_busy','hbm_shared')

def rates_for(g,active):
    works={};qs={};private={};owner_q={}
    for ident in active:
        o,s,w=ident;works[ident],qs[ident],private[ident]=g.work(s,w)
        owner_q[o]=owner_q.get(o,F(0))+qs[ident]
    rates={i:F(0) for i in active};free=set(active)
    for _ in range(len(active)):
        if not free:break
        limits={}
        for r in range(3):
            used=sum(works[i][r]*rates[i] for i in active)
            slope=sum(works[i][r]/owner_q[i[0]] for i in free)
            if slope:limits[('r',r)]=(g.capacities[r]-used)/slope
        for i in free:
            if private[i]:limits[('p',i)]=(1/private[i]-rates[i])*owner_q[i[0]]
        if not limits:raise ValueError('unbounded rate')
        step=min(limits.values())
        if step<0:raise ValueError('over capacity')
        frozen=set()
        for i in free:rates[i]+=step/owner_q[i[0]]
        for (kind,k),v in limits.items():
            if v==step:
                frozen.update(i for i in free if works[i][k]>0) if kind=='r' else frozen.add(k)
        if not frozen:raise ValueError('no LP limiter')
        free-=frozen
    if free:raise ValueError('LP tier bound exhausted')
    return rates

class Discovery:
    def __init__(self,g,offsets=(F(0),F(1,20)),cap=256,enable_skip=True):
        if len(offsets)>2 or not offsets or offsets[0]!=0 or any(x<0 for x in offsets):raise ValueError('requires 1/2 owners and first arrival zero')
        self.g=g;self.sample=offsets[-1];self.arrivals=[Aff()]+([Aff(F(1),F(0))] if len(offsets)==2 else [])
        self.owners=[Owner() for _ in offsets];self.now=Aff();self.ledger={n:Aff() for n in LEDGERS}
        self.guards=[];self.events=0;self.cap=cap;self.enable_skip=enable_skip
        self.seen={};self.descriptors=[];self.skipped=0;self.skipped_k=0;self.max_active=0
        self.max_in=0;self.max_out=0
        self.outer_seen={};self.outer_returns=[]
    def active(self):return {(o,s,w):p for o,owner in enumerate(self.owners) for s,(w,p) in owner.active.items()}
    def occupancy(self,o):
        inp=len(o.loaded)+int(0 in o.active)+int(1 in o.active)
        out=len(o.computed)+int(2 in o.active)+int(1 in o.active and self.g.facts(o.active[1][0])[3])
        return inp,out
    def settle(self):
        for i,o in enumerate(self.owners):
            if not o.started and self.arrivals[i].at(self.sample)==self.now.at(self.sample):o.started=True
        while True:
            changed=False
            for o in self.owners:
                if not o.started or o.finish is not None:continue
                inp,out=self.occupancy(o)
                self.max_in=max(self.max_in,inp);self.max_out=max(self.max_out,out)
                if inp>2 or out>2:raise ValueError('slot capacity')
                if 0 not in o.active and o.next_load<self.g.waves and inp<2:
                    o.active[0]=(o.next_load,Aff());o.next_load+=1;changed=True
                _,out=self.occupancy(o)
                # Faithful to offline design oracle: output backpressure gates compute.
                if 1 not in o.active and o.loaded and out<2:
                    o.active[1]=(o.loaded.pop(0),Aff());changed=True
                if 2 not in o.active and o.computed:
                    o.active[2]=(o.computed.pop(0),Aff());changed=True
                if o.retired==self.g.outputs and not o.active and not o.loaded and not o.computed:
                    o.finish=self.now;changed=True
            if not changed:break
    def key(self,exact_phase=False):
        key=[];anchors=[]
        for index,o in enumerate(self.owners):
            if not o.started:
                key.append(('pending_arrival',index));anchors.append(0);continue
            if o.finish is not None:return None
            references=[w for w,p in o.active.values()]+o.loaded+o.computed
            base=min(references,default=o.next_load)
            anchors.append(base)
            def tag(w):return (w-base,self.g.facts(w),w%self.g.kt) if exact_phase else (w-base,self.g.facts(w))
            future=(self.g.facts(o.next_load),o.next_load%self.g.kt if exact_phase else None) if o.next_load<self.g.waves else None
            key.append((tuple((s,*tag(w)) for s,(w,p) in sorted(o.active.items())),tuple(tag(w) for w in o.loaded),tuple(tag(w) for w in o.computed),o.next_load-base,o.retired-base//self.g.kt,future))
        return tuple(key),anchors
    def record_step(self):
        if self.events>=self.cap:raise ValueError('symbolic event discovery cap reached (global; not reset per output)')
        active=self.active();rates=rates_for(self.g,active);self.max_active=max(self.max_active,len(active))
        deadlines={i:(1-p)/rates[i] for i,p in active.items() if rates[i]>0}
        for i,o in enumerate(self.owners):
            if not o.started:deadlines[('arrival',i)]=self.arrivals[i]-self.now
        if not deadlines:raise ValueError('unfinished deadlock')
        delta=min(deadlines.values(),key=lambda a:a.at(self.sample));selected=[i for i,a in deadlines.items() if a.at(self.sample)==delta.at(self.sample)]
        if delta.at(self.sample)<0:raise ValueError('negative deadline')
        self.guards.append((delta,'ge' if delta.at(self.sample)==0 else 'gt'))
        for ident,a in deadlines.items():
            margin=a-delta
            if ident in selected:
                if margin!=Aff():self.guards.append((margin,'eq'))
            else:self.guards.append((margin,'gt'))
        self.ledger['elapsed']+=delta
        busy=[set() for _ in range(3)]
        for i,p in active.items():
            o,s,w=i;rate=rates[i];work,_,_=self.g.work(s,w)
            increment=delta*rate;new=p+increment
            self.guards += [(new,'ge'),(1-new,'ge')]
            self.owners[o].active[s]=(w,new)
            facts=self.g.facts(w)
            for r in range(3):
                if rate>0 and work[r]>0:busy[r].add(o)
            if s==1:
                self.ledger['vec_ops']+=increment*facts[7];self.ledger['local_bytes']+=increment*facts[8]
                self.ledger['compute_service']+=increment*work[0]
            else:
                self.ledger['hbm0']+=increment*work[1];self.ledger['hbm1']+=increment*work[2]
        if busy[0]:self.ledger['compute_busy']+=delta
        if busy[1]:self.ledger['hbm0_busy']+=delta
        if busy[2]:self.ledger['hbm1_busy']+=delta
        if busy[1] or busy[2]:self.ledger['hbm_busy']+=delta
        if len(busy[1])>=2 or len(busy[2])>=2:self.ledger['hbm_shared']+=delta
        self.now+=delta
        # Complete all exact ties before arrival/start; L,C,S stable order.
        for s in range(3):
            for i in selected:
                if i[0]=='arrival' or i[1]!=s:continue
                o,_,w=i;del self.owners[o].active[s]
                if s==0:self.owners[o].loaded.append(w)
                elif s==1:
                    if self.g.facts(w)[3]:self.owners[o].computed.append(w)
                else:self.owners[o].retired+=1
        self.events+=1;self.settle()
    def snapshot(self,anchors):
        return {'owners':deepcopy(self.owners),'anchors':anchors,'guard_start':len(self.guards),'events':self.events,'ledger':dict(self.ledger),'now':self.now}
    @staticmethod
    def reparam(a,pivot):return Aff(a.a/pivot.a,a.b-a.a*pivot.b/pivot.a)
    @staticmethod
    def domain(guards):
        lo=None;hi=None;lo_strict=False;hi_strict=False
        for a,kind in guards:
            if kind=='eq':
                if a.a==0:
                    if a.b:raise ValueError('empty equality guard')
                else:
                    x=-a.b/a.a
                    if lo is None or x>lo:lo=x;lo_strict=False
                    if hi is None or x<hi:hi=x;hi_strict=False
                continue
            if not a.a:
                if a.b<0 or (kind=='gt' and a.b==0):raise ValueError('empty constant guard')
                continue
            x=-a.b/a.a;strict=kind=='gt'
            if a.a>0:
                if lo is None or x>lo:lo=x;lo_strict=strict
                elif x==lo:lo_strict|=strict
            else:
                if hi is None or x<hi:hi=x;hi_strict=strict
                elif x==hi:hi_strict|=strict
        if lo is not None and hi is not None and (lo>hi or (lo==hi and (lo_strict or hi_strict))):raise ValueError('empty guard domain')
        return lo,hi,lo_strict,hi_strict
    def skip_return(self,old,anchors):
        # Match active coordinates by normalized stage/owner identity, not traces.
        start=[(i,s,p) for i,o in enumerate(old['owners']) for s,(w,p) in sorted(o.active.items())]
        pivot=next((p for i,s,p in start if p.a),None)
        if pivot is None:
            # Exact state return is a zero-dimensional identity manifold.
            if any(p!=self.owners[i].active[s][1] for i,s,p in start):return False
            pivot=Aff(F(1),F(0));model=Aff(F(1),F(0));point=self.sample
        else:
            i,s,_=next(t for t in start if t[2]==pivot)
            model=self.reparam(self.owners[i].active[s][1],pivot)
            point=pivot.at(self.sample)
        relations={(i,s):self.reparam(p,pivot) for i,s,p in start}
        for i,s,p in start:
            end=self.reparam(self.owners[i].active[s][1],pivot)
            if end!=relations[(i,s)].sub(model):return False
        guards=[(self.reparam(a,pivot),k) for a,k in self.guards[old['guard_start']:]]
        guards += [(p,'ge') for p in relations.values()]+[(1-p,'ge') for p in relations.values()]
        try:domain=self.domain(guards)
        except ValueError:return False
        # Prove the full contraction orbit or identity belongs to the strict
        # domain. Generic expanding maps/guard crossings fail closed here.
        if model.a==1 and model.b==0:orbit=(point,point)
        elif 0<=model.a<1:
            fixed=model.b/(1-model.a);orbit=(min(point,fixed),max(point,fixed))
        else:return False
        for a,kind in guards:
            vals=[a.at(v) for v in orbit]
            if (kind=='eq' and any(vals)) or (kind=='ge' and min(vals)<0) or (kind=='gt' and min(vals)<=0):return False
        shifts=[anchors[i]-old['anchors'][i] for i in range(len(anchors))]
        if any(d<=0 for i,d in enumerate(shifts) if self.owners[i].started):return False
        # Only analytically bounded middle K translations initially. Outer
        # output return is deliberately not accepted by this compiler yet.
        count=None
        for i,d in enumerate(shifts):
            if not self.owners[i].started:continue
            if d>=self.g.kt:return False
            o=self.owners[i]
            references=[w for w,p in o.active.values()]+o.loaded+o.computed
            if o.next_load<self.g.waves:references.append(o.next_load)
            # Every referenced wave is in one output; reject store/K-final.
            if any(self.g.facts(w)[3] or self.g.facts(w)[4] for w in references):return False
            kmax=max(w%self.g.kt for w in references)
            n=(self.g.kt-2-kmax)//d
            count=n if count is None else min(count,n)
        if not count:return False
        # For this initial lowerer only exact middle identity is powered;
        # nonidentity maps are still emitted for research, never evaluated by
        # huge unbounded Fraction powers or approximate phase equality.
        if model!=Aff(F(1),F(0)):return False
        delta_ledger={n:self.reparam(self.ledger[n]-old['ledger'][n],pivot) for n in LEDGERS}
        global_pivot=pivot # point at old start, unchanged under exact return
        delta_time=delta_ledger['elapsed'].sub(global_pivot)
        # Keep the ordinary event integrator in charge of future arrivals.
        # The integer count is a policy cell in offset space, recorded as
        # an exact guard, not an interpolation across offset regions.
        for i,o in enumerate(self.owners):
            if not o.started:
                remaining=(self.arrivals[i]-self.now).at(self.sample)
                duration=delta_time.at(self.sample)
                if duration<=0:return False
                count=min(count,int(remaining//duration))
        if not count:return False
        for i,o in enumerate(self.owners):
            if not o.started:self.guards.append((self.arrivals[i]-self.now-delta_time*count,'ge'))
        for name,a in delta_ledger.items():self.ledger[name]+=a.sub(global_pivot)*count
        self.now=self.ledger['elapsed']
        for i,o in enumerate(self.owners):
            if not o.started:continue
            shift=shifts[i]*count
            o.active={s:(w+shift,p) for s,(w,p) in o.active.items()}
            o.loaded=[w+shift for w in o.loaded];o.computed=[w+shift for w in o.computed];o.next_load+=shift
        self.descriptors.append({'kind':'symbolically_proved_K_middle_identity','step_waves_by_owner':shifts,'repeat_count':count,'template_events':self.events-old['events'],'map':model.json(),'domain':[str(v) if v is not None else None for v in domain[:2]],'domain_endpoint_strict':list(domain[2:]),'orbit':[str(x) for x in orbit],'ledger':{n:a.json() for n,a in delta_ledger.items()},'proof':'exact affine identities + all deadline/progress guards over entire invariant orbit; K-final/first lookahead excluded by arithmetic bound'})
        self.skipped+=count;self.skipped_k+=sum(shifts)*count
        self.seen.clear();return True
    def prove_outer(self,old,anchors):
        """Synthesize a return cell without consuming any archived trace.

        Discovery is bounded, and coefficients remain offline rationals.
        This function does not execute/power the outer return in production.
        """
        shifts=[a-b for a,b in zip(anchors,old['anchors'])]
        if not all(d>0 and d%self.g.kt==0 for d in shifts):return
        start=[(i,s,p) for i,o in enumerate(old['owners']) for s,(w,p) in sorted(o.active.items())]
        pivot=next((p for i,s,p in start if p.a),None)
        if pivot is None:return
        i,s,_=next(t for t in start if t[2]==pivot)
        model=self.reparam(self.owners[i].active[s][1],pivot)
        relations={(i,s):self.reparam(p,pivot) for i,s,p in start}
        if any(self.reparam(self.owners[i].active[s][1],pivot)!=relations[i,s].sub(model) for i,s,p in start):return
        guards=[(self.reparam(a,pivot),k) for a,k in self.guards[old['guard_start']:]]
        guards += [(p,'ge') for p in relations.values()]+[(1-p,'ge') for p in relations.values()]
        try:domain=self.domain(guards)
        except ValueError:return
        point=pivot.at(self.sample)
        if not 0<=model.a<1:return
        fixed=model.b/(1-model.a);orbit=min(point,fixed),max(point,fixed)
        for a,kind in guards:
            vals=[a.at(v) for v in orbit]
            if (kind=='eq' and any(vals)) or (kind=='ge' and min(vals)<0) or (kind=='gt' and min(vals)<=0):return
        delta={n:self.reparam(self.ledger[n]-old['ledger'][n],pivot) for n in LEDGERS}
        def lifecycle(owner,index,initial):
            return {'next_load':owner.next_load,'retired':owner.retired,
                    'active':{str(s):{'wave':w,'progress':relations[index,s].json() if initial else self.reparam(p,pivot).json()} for s,(w,p) in sorted(owner.active.items())},
                    'loaded':list(owner.loaded),'computed':list(owner.computed)}
        record={'kind':'generated_guarded_output_return_not_lowered','coordinate':{'owner':i,'stage':s},'shift_waves':shifts,'retired_outputs':[o.retired-p.retired for o,p in zip(self.owners,old['owners'])],'map':model.json(),'initial_state':str(point),'fixed_point':str(fixed),'invariant_orbit':[str(v) for v in orbit],'domain':[str(v) if v is not None else None for v in domain[:2]],'domain_endpoint_strict':list(domain[2:]),'representative_steps':self.events-old['events'],'guard_count':len(guards),'relations':{f'{i}:{s}':p.json() for (i,s),p in relations.items()},'ledger':{n:a.json() for n,a in delta.items()},'proof':'exact symbolic initial-offset propagation, manifold closure, every pulled-back event/progress guard valid over full contraction orbit; geometry category lifetime and runtime lowering still required'}
        record['initial_lifecycle']=[lifecycle(o,i,True) for i,o in enumerate(old['owners'])]
        record['end_lifecycle']=[lifecycle(o,i,False) for i,o in enumerate(self.owners)]
        record['pulled_back_guards']=[{'affine':a.json(),'kind':kind} for a,kind in guards]
        # This first outer certificate is restricted to an all-full M/N
        # rectangle. K first/final/tail classes are retained by phase key.
        limits=[]
        if self.g.N%self.g.tn:limits.append((self.g.nt-1)*self.g.kt)
        if self.g.M%self.g.tm:limits.append((self.g.mt-1)*self.g.nt*self.g.kt)
        first_nonfull=min(limits,default=self.g.waves)
        high=[max([o.next_load]+[w for w,p in o.active.values()]+o.loaded+o.computed) for o in self.owners]
        if all(w<first_nonfull for w in high):
            record['all_full_MN_bound']={'first_nonfull_wave':first_nonfull,'maximum_reference_wave_at_end':high,'additional_complete_returns':min((first_nonfull-1-w)//d for w,d in zip(high,shifts))}
        if len(self.outer_returns)<32 and record['map'] not in [r['map'] for r in self.outer_returns]:self.outer_returns.append(record)
    def run(self):
        self.settle()
        while any(o.finish is None for o in self.owners):
            if self.enable_skip:
                if all(o.started for o in self.owners):
                    outer=self.key(exact_phase=True)
                    if outer:
                        key,anchors=outer
                        old=self.outer_seen.get(key)
                        if old:self.prove_outer(old,anchors)
                        self.outer_seen[key]=self.snapshot(anchors)
                item=self.key()
                if item:
                    key,anchors=item
                    old=self.seen.get(key)
                    if old and self.skip_return(old,anchors):continue
                    self.seen[key]=self.snapshot(anchors)
            self.record_step()
        n=len(self.owners);g=self.g
        expected_read=4*g.K*(g.M*g.nt+g.N*g.mt)*n
        expected_write=4*g.M*g.N*n
        assert self.ledger['vec_ops'].at(self.sample)==2*g.M*g.N*g.K*n
        assert self.ledger['hbm0'].at(self.sample)==F(expected_read+expected_write,2)
        assert self.ledger['hbm1'].at(self.sample)==F(expected_read+expected_write,2)
        return {'status':'passed','scope':'synthetic exact 1-Core strictFP32 OS; no production certificate; only guarded middle K identity skip, outputs are still expanded up to global event cap','geometry':{'M':g.M,'N':g.N,'K':g.K,'tm':g.tm,'tn':g.tn,'tk':g.tk},'offset':str(self.sample),'owners':n,'events':self.events,'global_event_cap':self.cap,'skipped_cycles':self.skipped,'skipped_wave_count':self.skipped_k,'descriptors':self.descriptors,'peak_active':self.max_active,'peak_input_slots':self.max_in,'peak_output_slots':self.max_out,'finish_s':[str(o.finish.at(self.sample)) for o in self.owners],'ledger':{k:str(v.at(self.sample)) for k,v in self.ledger.items()}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--k',type=int,default=4097);p.add_argument('--m',type=int,default=65);p.add_argument('--n',type=int,default=65);p.add_argument('--offset',default='1/20');p.add_argument('--solo',action='store_true');p.add_argument('--no-skip',action='store_true');p.add_argument('--cap',type=int,default=256);p.add_argument('--output');a=p.parse_args()
    try:result=Discovery(Geometry(M=a.m,N=a.n,K=a.k),offsets=(F(0),) if a.solo else (F(0),F(a.offset)),cap=a.cap,enable_skip=not a.no_skip).run()
    except (ValueError,AssertionError) as e:result={'status':'failed_closed','reason':str(e),'production':'not_implemented'}
    text=json.dumps(result,indent=2)+'\n'
    if a.output:open(a.output,'w').write(text)
    else:print(text,end='')
    raise SystemExit(result['status']!='passed')
