import os
import subprocess
import time
import json
from typing import Dict, Optional

from loguru import logger


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HotspotManager():
    """
        Class to manage hotspot simulation
    """

    def __init__(self) -> None:
        self.hotspot_path = os.path.join(PROJECT_ROOT, "thirdparty/hotspot/hotspot")
        assert os.path.exists(self.hotspot_path), f"Hotspot executable bin {self.hotspot_path} does not exist."

    def run(self, directory: str, verbose: bool = True, force_rerun: bool = False, clean_up: bool = False, num_grid: Optional[int] = None) -> Dict[str, float]:
        """
            Run the hotspot simulation
        """
        if os.path.exists(os.path.join(directory, 'results.json')) and not force_rerun:
            logger.warning(f"Results already exist in {directory}. Skipping simulation.")
            with open(os.path.join(directory, 'results.json'), 'r') as f:
                results = json.load(f)
            return results

        logger.debug(f"Running hotspot simulation in directory: {directory}")

        cmd = [
            self.hotspot_path,
            "-c", os.path.join(directory, "hotspot.config"),
            "-p", os.path.join(directory, "hotspot.ptrace"),
            "-grid_layer_file", os.path.join(directory, "hotspot.lcf"),
            "-model_type", "grid",
            "-detailed_3D", "on",
            "-grid_steady_file", os.path.join(directory, "hotspot.grid.steady"),
            "-steady_file", os.path.join(directory, "hotspot.steady"),
        ]
        if num_grid is not None:
            cmd.append("-grid_rows")
            cmd.append(str(num_grid))
            cmd.append("-grid_cols")
            cmd.append(str(num_grid))

        logger.debug(f"Running command: {' '.join(cmd)}")

        start_time = time.time()
        p = subprocess.Popen(
            cmd,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE
        )
        p.wait()
        end_time = time.time()

        if not os.path.exists(os.path.join(directory, "hotspot.grid.steady")):
            raise RuntimeError("Hotspot simulation failed. Check the logs for more details.")
        
        results = {
            'max_temperature': self.parse_max_temperature(directory),
            'run_time': end_time - start_time,
        }
        if clean_up:
            subprocess.run(['rm', '-rf', directory])
        else:
            with open(os.path.join(directory, 'results.json'), 'w') as f:
                json.dump(results, f)

        return results
        
    def parse_max_temperature(self, directory: str) -> float:
        """
            Parse the maximum temperature of steady states
        """
        logger.debug(f"Parsing max temperature from directory: {directory}")

        grid_steady_file = os.path.join(directory, "hotspot.grid.steady")
        if not os.path.exists(grid_steady_file):
            raise RuntimeError(f"Grid steady file {grid_steady_file} does not exist.")

        with open(grid_steady_file, "r") as f:
            lines = f.readlines()

        # Find the line with the maximum temperature
        max_temp = 0
        for line in lines:
            if "Layer" in line:
                continue
            if line.strip() == "":
                continue
            try:
                temp = float(line.split()[1])
            except Exception as e:
                logger.warning(f"Error parsing temperature from line: {line}")
                temp = 0  # ignore this line
            if temp > max_temp:
                max_temp = temp

        logger.debug(f"Max temperature: {max_temp}")
        return max_temp
