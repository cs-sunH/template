"""Sparse conservative RC network with explicit advecting liquid state.

Backward Euler: (C/dt + L) T_next = C*T/dt + P + inlet.
The nonsymmetric coolant advection operator transports enthalpy downstream.
No neighbor-power correction is added: lateral conductances own coupling.

The solver keeps a complete physical state (solid *and* coolant temperature plus
the cumulative energy ledger) so a candidate can be forked from one committed
snapshot without disturbing it.  Copying only the returned solid temperatures
would be wrong: the coolant state feeds later results.
"""
from dataclasses import dataclass
from collections import OrderedDict
import math
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import splu
from .config import thermal_geometry

# dt cache: the LU factorisation is a performance cache only, never physical
# state.  Keys are rounded to this many decimals; a dt outside the accepted band
# or beyond the entry cap is solved without being cached rather than being
# silently merged with a different time step.
_DT_ROUND_DECIMALS = 12
_DT_CACHE_MAX_ENTRIES = 16
_DT_CACHE_MAX_RELATIVE_SPREAD = 1e-6


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
    supports_snapshot = True
    def __init__(self, config):
        self.config = config
        geometry, t = thermal_geometry(config), config['thermal']
        self.nx, self.ny = geometry['die_columns'], geometry['die_rows']
        self.ndie = self.nx*self.ny
        q = geometry['compute_groups_per_side']
        self.q, self.blocks, self.die_blocks = q, [], {}
        side = geometry['compute_side_m']
        pitchx, pitchy = geometry['module_width_m'], geometry['module_height_m']
        for d in range(self.ndie):
            x, y = (d%self.nx)*pitchx, (d//self.nx)*pitchy
            ids = []
            for iy in range(q):
                for ix in range(q):
                    ids.append(len(self.blocks))
                    self.blocks.append(Block(f'd{d}_c{iy*q+ix}', d, 'compute',
                        x+ix*side/q, y+iy*side/q, side/q, side/q))
            self.die_blocks[d] = {'compute': ids}
            for kind in ('hbm', 'router'):
                rectangle = geometry[kind]
                self.die_blocks[d][kind] = [len(self.blocks)]
                self.blocks.append(Block(f'd{d}_{kind}', d, kind,
                    x+rectangle['x_m'], y+rectangle['y_m'],
                    rectangle['width_m'], rectangle['height_m']))
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
        # Committed-step counter; candidate forks restore it, so it is the
        # authoritative "which physical state" version handed to the joint policy.
        self._state_version = 0

    def distribute(self, die, kind, value, array):
        ids = self.die_blocks[die][kind]
        array[ids] += value/len(ids)

    def step(self, power, dt):
        """Advance one thermal step with a validated, finite positive dt.

        Rejects non-positive/non-finite dt, wrong-shaped power and negative or
        non-finite watts, so an unvalidated window can never be integrated into
        the committed state.
        """
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise ValueError('thermal dt must be a finite positive number')
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('thermal dt must be a finite positive number')
        power = np.asarray(power, dtype=float)
        if power.shape != (self.nsolid,) or not np.all(np.isfinite(power)) or np.any(power < 0):
            raise ValueError('finite nonnegative W required for every solid block')
        solver = self._solver(dt)
        old = self.T.copy()
        rhs = self.C / dt * old + self.boundary
        rhs[:self.nsolid] += power
        self.T = solver.solve(rhs)
        if not np.all(np.isfinite(self.T)):
            raise ValueError('thermal step produced non-finite temperature')
        outlet = self.T[self.nsolid + self.nx - 1::self.nx]
        removed = float(np.dot(self.mcp_rows, outlet - self.config['thermal']['inlet_k']) * dt)
        input_j = float(power.sum() * dt)
        stored = float(np.dot(self.C, self.T - old))
        self.energy_residual_j += stored + removed - input_j
        self.removed_j += removed
        self.input_j += input_j
        self._state_version += 1
        return self.T[:self.nsolid].copy()

    def _solver(self, dt):
        """Cache the LU factorisation of one dt, bounded and spread-checked.

        Entries whose stored dt differs from the requested dt by more than the
        accepted relative spread are evicted instead of being reused, so two
        distinct segmentations can never silently share a stale factorisation.
        """
        key = round(dt, _DT_ROUND_DECIMALS)
        if key <= 0.0:
            key = dt
        cached = self.cache.get(key)
        if cached is not None:
            if abs(key - dt) <= _DT_CACHE_MAX_RELATIVE_SPREAD * dt:
                self.cache.move_to_end(key)
                return cached
            del self.cache[key]
        solver = splu(self.L + diags(self.C / dt))
        self.cache[key] = solver
        while len(self.cache) > _DT_CACHE_MAX_ENTRIES:
            self.cache.popitem(last=False)
        return solver

    def die_max(self):
        return np.array([max(self.T[i] for ids in self.die_blocks[d].values() for i in ids)
                         for d in range(self.ndie)])

    # --------------------------------------------------------- lifecycle --

    def state_version(self) -> int:
        """Monotone counter of committed physical steps (candidate changes never bump it)."""
        return self._state_version

    def snapshot(self) -> dict:
        """Complete restorable state: solid and coolant temperature + ledger.

        The full temperature vector already carries the coolant segments; they
        are also exposed separately for feedback consumers.
        """
        return {
            'kind': self.name,
            'temperature_k': self.T.copy(),
            'removed_j': self.removed_j,
            'input_j': self.input_j,
            'energy_residual_j': self.energy_residual_j,
            'state_version': self._state_version,
        }

    def restore(self, snapshot) -> None:
        """Reinstall a previously taken snapshot verbatim (fork rollback path)."""
        if not isinstance(snapshot, dict) or snapshot.get('kind') != self.name:
            raise ValueError('snapshot does not belong to this thermal backend')
        temperature = np.asarray(snapshot.get('temperature_k'), dtype=float)
        if temperature.shape != self.T.shape or not np.all(np.isfinite(temperature)):
            raise ValueError('snapshot temperature vector is invalid')
        self.T = temperature.copy()
        self.removed_j = float(snapshot['removed_j'])
        self.input_j = float(snapshot['input_j'])
        self.energy_residual_j = float(snapshot['energy_residual_j'])
        self._state_version = int(snapshot['state_version'])

    def reset(self) -> None:
        """Return to the initial temperature with a zeroed energy ledger."""
        self.T = self.initial_T.copy()
        self.removed_j = 0.0
        self.input_j = 0.0
        self.energy_residual_j = 0.0
        self._state_version = 0

    def close(self) -> None:
        """Release the dt cache (no native handle for the pure-Python backend)."""
        self.cache.clear()

    def feedback(self) -> dict:
        """Immutable thermal readout for the joint policy."""
        return {
            'block_k': {b.name: float(self.T[i]) for i, b in enumerate(self.blocks)},
            'coolant_k': {'row%d' % (d // self.nx): float(self.T[self.nsolid + d])
                          for d in range(self.ndie)},
            'die_block_max_k': {d: float(v) for d, v in enumerate(self.die_max())},
            'energy_residual_j': self.energy_residual_j,
            'input_j': self.input_j,
            'removed_j': self.removed_j,
            'state_version': self._state_version,
        }

    def export_floorplan(self,path):
        with open(path,'w') as f:
            for b in self.blocks:
                f.write(f'{b.name} {b.width:.12g} {b.height:.12g} {b.x:.12g} {b.y:.12g}\n')
