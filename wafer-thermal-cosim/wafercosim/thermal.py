"""Sparse conservative RC network with explicit advecting liquid state.

Backward Euler: (C/dt + L) T_next = C*T/dt + P + inlet.
The nonsymmetric coolant advection operator transports enthalpy downstream.
No neighbor-power correction is added: lateral conductances own coupling.
"""
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import splu

@dataclass
class Block:
    name: str
    die: int
    kind: str
    x: float
    y: float
    width: float
    height: float

class LiquidRC:
    name = 'rc_liquid'
    def __init__(self, config):
        self.config = config
        h, t = config['hardware'], config['thermal']
        self.nx, self.ny = h['nx'], h['ny']
        self.ndie = self.nx*self.ny
        q = h['compute_groups_per_side']
        self.q, self.blocks, self.die_blocks = q, [], {}
        side = h['compute_side_m']
        pitchx, pitchy = h['module_width_m'], h['module_height_m']
        for d in range(self.ndie):
            x, y = (d%self.nx)*pitchx, (d//self.nx)*pitchy
            ids = []
            for iy in range(q):
                for ix in range(q):
                    ids.append(len(self.blocks))
                    self.blocks.append(Block(f'd{d}_c{iy*q+ix}', d, 'compute',
                        x+ix*side/q, y+iy*side/q, side/q, side/q))
            self.die_blocks[d] = {'compute': ids}
            for kind, bx, by, w, hh in [('hbm', x, y+side+0.00015, 0.01626, 0.00492),
                                        ('router', x+side+0.00015, y, 0.001, side)]:
                self.die_blocks[d][kind] = [len(self.blocks)]
                self.blocks.append(Block(f'd{d}_{kind}', d, kind, bx, by, w, hh))
        self.nsolid = len(self.blocks)
        n = self.nsolid+self.ndie
        self.C = np.zeros(n)
        self.L = lil_matrix((n,n), dtype=float)
        self.boundary = np.zeros(n)
        self.T = np.full(n, t['initial_k'], dtype=float)
        self.mcp = t['coolant_mass_flow_kg_s_per_row']*t['coolant_cp_j_kgk']
        flows=t.get('coolant_mass_flow_kg_s_by_row',[t['coolant_mass_flow_kg_s_per_row']]*self.ny)
        if len(flows)!=self.ny or any(not np.isfinite(f) or f<=0 for f in flows):
            raise ValueError('one positive finite coolant mass flow per row required')
        self.mcp_rows=np.asarray(flows,dtype=float)*t['coolant_cp_j_kgk']
        htc=t.get('htc_w_m2k_by_die',[t['htc_w_m2k']]*self.ndie)
        if len(htc)!=self.ndie or any(not np.isfinite(v) or v<=0 for v in htc):
            raise ValueError('one positive finite HTC per die required')
        self.htc_by_die=np.asarray(htc,dtype=float)
        self.cache = OrderedDict()
        self.energy_residual_j = 0.0
        self.removed_j = 0.0
        self.input_j = 0.0
        def connect(i,j,g):
            self.L[i,i] += g; self.L[j,j] += g
            self.L[i,j] -= g; self.L[j,i] -= g
        for d, kinds in self.die_blocks.items():
            cool = self.nsolid+d
            self.C[cool] = (t['coolant_segment_volume_m3']*t['coolant_density_kg_m3']*
                           t['coolant_cp_j_kgk'])
            for i in kinds['compute']:
                area = self.blocks[i].width*self.blocks[i].height
                self.C[i] = area*t['silicon_thickness_m']*t['silicon_volumetric_heat_capacity_j_m3k']
                g = area/(1/self.htc_by_die[d]+t['tim_areal_resistance_m2k_w'])
                connect(i, cool, g)
            for kind in ['hbm','router']:
                i = kinds[kind][0]
                self.C[i] = t[f'{kind}_capacity_j_k_per_die']
                connect(i, cool, t[f'{kind}_cooling_g_w_k'])
                # Effective package conductance; explicit assumption, not direct silicon contact.
                edge=kinds['compute'][-q:] if kind=='hbm' else [kinds['compute'][row*q+q-1] for row in range(q)]
                for j in edge:connect(i,j,t['substrate_die_lateral_g_w_k']/q)
            for iy in range(q):
                for ix in range(q):
                    i = kinds['compute'][iy*q+ix]
                    for xx, yy in [(ix+1,iy),(ix,iy+1)]:
                        if xx<q and yy<q:
                            connect(i,kinds['compute'][yy*q+xx],
                                    t['silicon_k_w_mk']*t['silicon_thickness_m'])
            for dd in [d+1 if d%self.nx+1<self.nx else -1,
                       d+self.nx if d//self.nx+1<self.ny else -1]:
                if dd>=0:
                    for k in range(q):
                        i=kinds['compute'][k*q+q-1] if dd//self.nx==d//self.nx else kinds['compute'][(q-1)*q+k]
                        j=self.die_blocks[dd]['compute'][k*q] if dd//self.nx==d//self.nx else self.die_blocks[dd]['compute'][k]
                        connect(i,j,t['substrate_die_lateral_g_w_k']/q)
            mcp=self.mcp_rows[d//self.nx]
            self.L[cool,cool] += mcp
            if d%self.nx == 0:
                self.boundary[cool] = mcp*t['inlet_k']
            else:
                self.L[cool,cool-1] -= mcp
        self.L = self.L.tocsc()
        self.initial_T = self.T.copy()

    def distribute(self, die, kind, value, array):
        ids = self.die_blocks[die][kind]
        array[ids] += value/len(ids)

    def step(self, power, dt):
        if dt<=0: raise ValueError('thermal dt must be positive')
        power=np.asarray(power,dtype=float)
        if power.shape!=(self.nsolid,) or np.any(power<0) or not np.all(np.isfinite(power)):
            raise ValueError('nonnegative finite W required for every solid block')
        key=round(float(dt),12)
        if key not in self.cache:
            self.cache[key]=splu(self.L+diags(self.C/dt))
            if len(self.cache)>16: self.cache.popitem(last=False)
        old=self.T.copy()
        rhs=self.C/dt*old+self.boundary
        rhs[:self.nsolid] += power
        self.T=self.cache[key].solve(rhs)
        outlet=self.T[self.nsolid+self.nx-1::self.nx]
        removed=float(np.dot(self.mcp_rows,outlet-self.config['thermal']['inlet_k'])*dt)
        input_j=float(power.sum()*dt)
        stored=float(np.dot(self.C,self.T-old))
        self.energy_residual_j+=stored+removed-input_j
        self.removed_j+=removed; self.input_j+=input_j
        return self.T[:self.nsolid].copy()

    def die_max(self):
        return np.array([max(self.T[i] for ids in self.die_blocks[d].values() for i in ids)
                         for d in range(self.ndie)])

    def export_floorplan(self,path):
        with open(path,'w') as f:
            for b in self.blocks:
                f.write(f'{b.name} {b.width:.12g} {b.height:.12g} {b.x:.12g} {b.y:.12g}\n')
