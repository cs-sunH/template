# Instructions for ATLAS Artifact Evaluation

## 1. Getting Started

Follow the installation instructions in the project [README](../README.md).
The artifact evaluation is CPU-only and does not require a GPU. The validation
can run on any CPU machine. The configuration below is the reference machine
used for our experiments and reported timings; when using a different CPU
machine, each task's runtime will vary with the machine's performance.

> - OS: Ubuntu 24.04.4 LTS, Linux 6.8.0
> - CPU: 2 x Intel Xeon Platinum 8558P, 48 physical cores per socket
> - CPU threads: 192 logical CPUs total
> - Memory: 2.0 TiB

The DRAM scripts default to 80% of available logical CPUs. The chip scripts
default to one architecture worker because each architecture case performs its
own multiprocessing tiling exploration; increasing the outer worker count can
oversubscribe the host.

Before running the experiments, verify both native tools exist:

```bash
test -x simulator/build/bin/test_dram
test -x pyta/thirdparty/hotspot/hotspot
```

## 2. Run All Experiments

Run the all-in-one script from the repository root. It performs simulation,
CSV collection, and plotting for all four experiment families sequentially:

```bash
bash tests/run_all_experiments.sh
```

The script locates the repository root automatically and uses `python` from the
active environment. Set `PYTHON=/path/to/python` when an explicit interpreter
is needed. It checks the two required native executables, creates `results/`,
keeps Python output unbuffered, displays progress in the terminal, and records
each experiment under `results/<experiment>.log`.

Each Python driver prints its wall-clock `[TIMER]` line without requiring an
external `time` executable. The chip commands pass `--no-cache`, and the Edge
DRAM command omits `--use-cache`; therefore, the script performs clean
simulation runs even if an output directory already exists. The wrapper also
prints the total sequential runtime after all four experiment families finish.

## 3. Reference Runtime

The following clean-run times were measured in this repository on the
reference machine described above. Runtime varies with system load and CPU
frequency scaling.

| Experiment | Reference runtime |
| --- | ---: |
| Cloud DRAM DSE | 1,826.45 s (30.4 min) |
| Edge DRAM DSE | 4,395.66 s (73.3 min) |
| Cloud chip DSE | 26,881.95 s (7.47 h) |
| Edge chip DSE | 11,667.73 s (3.24 h) |
| Sequential total | 44,771.79 s (12.44 h) |

These values include simulation, result collection, and the plotting performed
by `--action all`. Plot-only reruns are much shorter. Note that
`pip install -e simulator` builds the Python extension but not necessarily the
`test_dram` executable; run the full CMake build from the root README before
starting DRAM DSE.

## 4. Run Individual Experiment Families

The `--action` option accepts `run`, `collect`, `plot`, or `all`. Use `run` to
generate raw results, then rerun `collect` or `plot` without repeating the
simulation.

### 4.1 Cloud DRAM DSE (Figures 10 and 12)

```bash
# Figure 10a: fig10a_channel_interleaving.{png,pdf}
python tests/dram_dse_cloud.py --experiment interleave --action all

# Figure 10b: fig10b_io_organization.{png,pdf}
python tests/dram_dse_cloud.py --experiment io --action all

# Figure 10c: fig10c_logical_row_size.{png,pdf}
python tests/dram_dse_cloud.py --experiment row --action all

# Figure 10d and Figure 12: fig10d_full_space_models.{png,pdf} and
# fig12_full_space_average.{png,pdf}
python tests/dram_dse_cloud.py --experiment full --action all
```

Results are under `results/dram_dse_cloud/output_results/`; collected CSVs and
figures are in its `summary/` and `figures/` directories.

### 4.2 Edge DRAM DSE (Figure 20)

```bash
# Figure 20a: fig20a_channel_interleaving.{png,pdf}
python tests/dram_dse_edge.py --experiment interleave --action all

# Figure 20b: fig20b_io_organization.{png,pdf}
python tests/dram_dse_edge.py --experiment io --action all

# Figure 20c: fig20c_logical_row_size.{png,pdf}
python tests/dram_dse_edge.py --experiment row --action all

# Figure 20d: fig20d_full_space_heatmap.{png,pdf}
python tests/dram_dse_edge.py --experiment full --action all
```

Results are under `results/dram_dse_edge/output_results/`, with collected CSVs
and figures under `summary/` and `figures/`.

### 4.3 Cloud Chip DSE (Figures 13-19)

```bash
# Figure 13a-d: fig13a_bandwidth_compute.{png,pdf},
# fig13b_temperature_ratio.{png,pdf}, fig13c_avg_speedup.{png,pdf}, and
# fig13d_case_speedup.{png,pdf}.
python tests/chip_dse_cloud.py --experiment bandwidth --action all

# Figure 14a-c: fig14a_sram_area_compute.{png,pdf},
# fig14b_avg_speedup.{png,pdf}, and fig14c_case_speedup.{png,pdf}.
python tests/chip_dse_cloud.py --experiment sram --action all

# Figure 15a-c: fig15a_matrix_vector_compute.{png,pdf},
# fig15b_avg_speedup.{png,pdf}, fig15c_top_case_speedup.{png,pdf}, and
# fig15c_bottom_fc_attention.{png,pdf}.
python tests/chip_dse_cloud.py --experiment matrix_vector --action all

# Figure 16a-c: fig16a_noc_area_compute.{png,pdf},
# fig16b_avg_speedup.{png,pdf}, fig16c_top_case_speedup.{png,pdf}, and
# fig16c_bottom_comm_compute_ratio.{png,pdf}.
python tests/chip_dse_cloud.py --experiment noc --action all

# Figures 17-19: fig17_atlas_ppa_thermal.{png,pdf},
# fig18_stratum_ppa_thermal.{png,pdf}, and
# fig19_baseline_comparison.{png,pdf}.
python tests/chip_dse_cloud.py --experiment baseline --action all
```

Each experiment is stored under `results/chip_dse_cloud/<experiment>/`.
Per-architecture thermal results, test-case performance files, summaries, and
figures remain grouped below that experiment.

### 4.4 Edge Chip DSE (Figure 21)

```bash
# Figure 21a-c: fig21a_matrix_vector_compute.{png,pdf},
# fig21b_avg_speedup_efficiency.{png,pdf}, and
# fig21c_case_speedup_efficiency.{png,pdf}.
python tests/chip_dse_edge.py --experiment matrix_vector --action all
```

Results are under
`results/chip_dse_edge/01_matrix_vector_allocation/`.

## 5. Validate Results

Due to privacy requirements associated with our industry collaborator, we
anonymize architecture parameters and simplify selected simulator microarchitecture 
implementation details. Consequently, reproduced numerical results may differ 
slightly from those reported in the paper. These differences do not affect the 
reproduced hardware-parameter trade-off trends demonstrated in the paper.
For comparison, we provide pre-run reference figures under [`docs/ref_figs/`](ref_figs/).

A successful all-in-one run should produce these high-level artifacts:

- Cloud DRAM: five figures in
  `results/dram_dse_cloud/output_results/figures/` for Figures 10a-d and 12.
- Edge DRAM: four figures in
  `results/dram_dse_edge/output_results/figures/` for Figures 20a-d.
- Cloud chip: experiment summaries and Figures 13-19 under
  `results/chip_dse_cloud/01_*` through `05_*`.
- Edge chip: architecture/case summaries and three Figure 21 plots under
  `results/chip_dse_edge/01_matrix_vector_allocation/`.

### 5.1 Cloud DRAM Figures

- **Figure 10a:** `fig10a_channel_interleaving.{png,pdf}`, generated by
  `--experiment interleave`; compares DRAM channel-interleaving positions.
  Compare with the reference [PNG](ref_figs/fig10a_channel_interleaving.png).
- **Figure 10b:** `fig10b_io_organization.{png,pdf}`, generated by
  `--experiment io`; compares DRAM channel organizations. Compare with the
  reference [PNG](ref_figs/fig10b_io_organization.png).
- **Figure 10c:** `fig10c_logical_row_size.{png,pdf}`, generated by
  `--experiment row`; compares logical row sizes. Compare with the reference
  [PNG](ref_figs/fig10c_logical_row_size.png).
- **Figure 10d:** `fig10d_full_space_models.{png,pdf}`, generated by
  `--experiment full`; shows the OPT, LLaMA, Mixtral, and Qwen full-space
  heatmaps in one row. Compare with the reference
  [PNG](ref_figs/fig10d_full_space_models.png).
- **Figure 12:** `fig12_full_space_average.{png,pdf}`, generated by
  `--experiment full`; shows the full-space average heatmap separately. Compare
  with the reference [PNG](ref_figs/fig12_full_space_average.png).

### 5.2 Edge DRAM Figures

- **Figure 20a:** `fig20a_channel_interleaving.{png,pdf}`, generated by
  `--experiment interleave`; compares DRAM channel-interleaving positions.
  Compare with the reference [PNG](ref_figs/fig20a_channel_interleaving.png).
- **Figure 20b:** `fig20b_io_organization.{png,pdf}`, generated by
  `--experiment io`; compares DRAM channel organizations. Compare with the
  reference [PNG](ref_figs/fig20b_io_organization.png).
- **Figure 20c:** `fig20c_logical_row_size.{png,pdf}`, generated by
  `--experiment row`; compares logical row sizes. Compare with the reference
  [PNG](ref_figs/fig20c_logical_row_size.png).
- **Figure 20d:** `fig20d_full_space_heatmap.{png,pdf}`, generated by
  `--experiment full`; shows the OPT, LLaMA, PaLM, and average full-space
  heatmaps in one row. Compare with the reference
  [PNG](ref_figs/fig20d_full_space_heatmap.png).

### 5.3 Cloud Chip Figures

- **Figure 13a:** `fig13a_bandwidth_compute.{png,pdf}`; DRAM bandwidth and
  matrix compute across channel counts. Reference:
  [PNG](ref_figs/fig13a_bandwidth_compute.png).
- **Figure 13b:** `fig13b_temperature_ratio.{png,pdf}`; temperature and
  achieved-compute ratio across channel counts. Reference:
  [PNG](ref_figs/fig13b_temperature_ratio.png).
- **Figure 13c:** `fig13c_avg_speedup.{png,pdf}`; average speedup across
  channel counts. Reference: [PNG](ref_figs/fig13c_avg_speedup.png).
- **Figure 13d:** `fig13d_case_speedup.{png,pdf}`; dense and MoE case speedups
  across channel counts. All Figure 13 files are generated by
  `--experiment bandwidth`. Reference:
  [PNG](ref_figs/fig13d_case_speedup.png).
- **Figure 14a:** `fig14a_sram_area_compute.{png,pdf}`; SRAM area and matrix
  compute. Reference: [PNG](ref_figs/fig14a_sram_area_compute.png).
- **Figure 14b:** `fig14b_avg_speedup.{png,pdf}`; average SRAM-allocation
  speedup. Reference: [PNG](ref_figs/fig14b_avg_speedup.png).
- **Figure 14c:** `fig14c_case_speedup.{png,pdf}`; per-case SRAM-allocation
  speedup. All Figure 14 files are generated by `--experiment sram`.
  Reference: [PNG](ref_figs/fig14c_case_speedup.png).
- **Figure 15a:** `fig15a_matrix_vector_compute.{png,pdf}`; matrix and vector
  compute allocation. Reference:
  [PNG](ref_figs/fig15a_matrix_vector_compute.png).
- **Figure 15b:** `fig15b_avg_speedup.{png,pdf}`; average matrix/vector
  allocation speedup. Reference: [PNG](ref_figs/fig15b_avg_speedup.png).
- **Figure 15c:** `fig15c_top_case_speedup.{png,pdf}` and
  `fig15c_bottom_fc_attention.{png,pdf}`; per-case and operator analysis.
  All Figure 15 files are generated by `--experiment matrix_vector`.
  References: top [PNG](ref_figs/fig15c_top_case_speedup.png) and bottom
  [PNG](ref_figs/fig15c_bottom_fc_attention.png).
- **Figure 16a:** `fig16a_noc_area_compute.{png,pdf}`; NoC area and matrix
  compute. Reference: [PNG](ref_figs/fig16a_noc_area_compute.png).
- **Figure 16b:** `fig16b_avg_speedup.{png,pdf}`; average NoC-allocation
  speedup. Reference: [PNG](ref_figs/fig16b_avg_speedup.png).
- **Figure 16c:** `fig16c_top_case_speedup.{png,pdf}` and
  `fig16c_bottom_comm_compute_ratio.{png,pdf}`; per-case and communication
  analysis. All Figure 16 files are generated by `--experiment noc`.
  References: top [PNG](ref_figs/fig16c_top_case_speedup.png) and bottom
  [PNG](ref_figs/fig16c_bottom_comm_compute_ratio.png).
- **Figure 17:** `fig17_atlas_ppa_thermal.{png,pdf}`; ATLAS area, power, and
  thermal results, generated by `--experiment baseline`. Reference:
  [PNG](ref_figs/fig17_atlas_ppa_thermal.png).
- **Figure 18:** `fig18_stratum_ppa_thermal.{png,pdf}`; Stratum area, power,
  and thermal results, generated by `--experiment baseline`. Reference:
  [PNG](ref_figs/fig18_stratum_ppa_thermal.png).
- **Figure 19:** `fig19_baseline_comparison.{png,pdf}`; GPU, Stratum, and
  ATLAS performance and energy-efficiency comparison, generated by
  `--experiment baseline`. Reference:
  [PNG](ref_figs/fig19_baseline_comparison.png).

### 5.4 Edge Chip Figures

- **Figure 21a:** `fig21a_matrix_vector_compute.{png,pdf}`; matrix and vector
  compute allocation. Reference:
  [PNG](ref_figs/fig21a_matrix_vector_compute.png).
- **Figure 21b:** `fig21b_avg_speedup_efficiency.{png,pdf}`; average speedup
  and energy efficiency. Reference:
  [PNG](ref_figs/fig21b_avg_speedup_efficiency.png).
- **Figure 21c:** `fig21c_case_speedup_efficiency.{png,pdf}`; per-case speedup
  and energy efficiency. All Figure 21 files are generated by
  `--experiment matrix_vector`. Reference:
  [PNG](ref_figs/fig21c_case_speedup_efficiency.png).
