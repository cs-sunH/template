"""True persistent native HotSpot block-RC backend, not the liquid backend."""
import subprocess, selectors
from pathlib import Path
import numpy as np
from .thermal import LiquidRC

class HotSpotThermal(LiquidRC):
    name='hotspot_native_block'
    def __init__(self,config,bridge,hotspot_config,workdir):
        super().__init__(config)
        path=Path(workdir).resolve();path.mkdir(parents=True,exist_ok=True)
        self.export_floorplan(path/'native.flp')
        if not hotspot_config:
            size=max(max(b.x+b.width for b in self.blocks),max(b.y+b.height for b in self.blocks))
            hotspot_config=path/'native.config'
            hotspot_config.write_text(f'-model_type block\n-ambient {config["thermal"]["inlet_k"]}\n-init_temp {config["thermal"]["initial_k"]}\n-block_omit_lateral 0\n-r_convec 0.02\n-s_spreader {size*1.1}\n-s_sink {size*1.3}\n')
        self.log=open(path/'hotspot.stderr.log','w')
        self.process=subprocess.Popen([str(Path(bridge).resolve()),'--flp',str(path/'native.flp'),
                                      '--config',str(Path(hotspot_config).resolve())],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
        header=self._readline()
        if not header.startswith('# time_s') or len(header.split(','))!=self.nsolid+1:
            self.close();raise RuntimeError('HotSpot protocol mismatch; see stderr log')
        self.energy_residual_j=None

    def _readline(self):
        if not self.selector.select(timeout=30):
            self.process.kill();raise TimeoutError('native HotSpot step exceeded 30 s; use coarse backend or inspect log')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('HotSpot exited; see hotspot.stderr.log')
        return line.strip()

    def step(self,power,dt):
        if dt<=0 or np.any(np.asarray(power)<0):raise ValueError('invalid power/dt')
        self.process.stdin.write(','.join(map(str,[dt,*power]))+'\n');self.process.stdin.flush()
        row=np.array([float(x) for x in self._readline().split(',')])
        if len(row)!=self.nsolid+1 or not np.all(np.isfinite(row)):raise RuntimeError('HotSpot nonfinite state')
        self.T[:self.nsolid]=row[1:]
        return self.T[:self.nsolid].copy()

    def close(self):
        if getattr(self,'process',None):
            if self.process.poll() is None:
                self.process.stdin.close()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process.stdout.close();self.selector.close();self.log.close()
