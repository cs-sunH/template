# ReThermal paper basis for the thermal co-simulator

## Scope

This note extracts the thermal parameters and modeling claims in *ReThermal: Co-Design of Thermal-Aware Static and Dynamic Scheduling for LLM Training on Liquid-Cooled Wafer-Scale Chips* (HPCA 2026, DOI `10.1109/HPCA68181.2026.11408476`). Page references below are PDF page numbers in the supplied 15-page file. The paper is a useful source for a representative wafer-scale configuration, a behavioral power model, and qualitative liquid-cooling trends. It does **not** specify a layer-resolved transient thermal model. Its learned temperature surrogate is based on steady-state ANSYS samples, so the reported prediction error does not establish transient inference accuracy.

## Parameters and claims reported by the paper

| Item | Paper value or model | Evidence |
|---|---|---|
| Evaluation topology | 5 × 4 dies (20 total), with PP=5 pipeline stages mapped to rows and TP=4 tensor partitions across columns | PDF p.10, §V-A, Table II context; Fig. 2 on p.3 shows the PP/TP layout |
| Die footprint | 25 × 30 mm | PDF p.10, Table II |
| Compute settings | 0.85 V, 2 GHz, effective capacitance 3.5 aF; peak throughput 700 TFLOPS; TDP 800 W | PDF p.10, Table II. The table's row grouping does not make the 800 W / 700 TFLOPS granularity sufficiently clear for safe per-die power derivation; treat them as reported configuration figures, not calibrated per-die limits. |
| Coolant settings | Water, inlet temperature 300 K, “Flow Rate” 2 m/s | PDF p.10, Table II. The reported unit is velocity; volumetric or mass flow is not given. |
| Cooling direction and scale | Coolant warms and heat-transfer effectiveness decreases downstream; paths often exceed 100 mm | PDF p.2, Fig. 1 and introduction; PDF p.3, §II-B and Fig. 2. Fig. 2 labels inlet-side dies cold and outlet-side dies hot. |
| Heat-transfer coefficient | No calibrated WSC HTC values are tabulated. Figs. 1–2 plot an illustrative HTC axis reaching 50,000 W/(m²·K), with lower HTC downstream. | PDF pp.2–3, Figs. 1–2. Read the trend and axis scale only; the plotted curves are not a reproducible HTC profile. |
| Compute power | `Pdyn = α Ceff Vdd² f Nops`; `Pleak = λ Pdyn`; `Pcomp = α Ceff Vdd² f Nops (1 + λ)` | PDF p.7, Eqs. (1)–(3). The paper says α is profiled on ResNet-50 and VCD-validated with <3% error; λ is temperature-dependent and regressed from several 7 nm chips. It does not provide α, λ(T), Nops calibration, or a per-operation interpretation sufficient to reproduce watts. |
| Inter-die communication power | `Pcomm = Ebit Rdata γxtalk / ηdrv`; rate is derived from task traffic and scheduling interval | PDF p.7, Eq. (4) and adjacent text. Table II gives `Ebit=0.3 pJ/bit`, `γxtalk=1.2×`, and `ηdrv=80%` (PDF p.10). |
| Behavioral coupling heuristic | `Ptotal(i,j) = Plocal(i,j)(1+ε) + ε Σneighbor Plocal`; `ε=0.01`, to be calibrated with ANSYS | PDF p.7, Eqs. (5)–(6) and adjacent text. This is a power-map heuristic, not a physical lateral thermal conductance. Do not add it on top of explicit RC heat transfer without a separate calibration, or it can double-count heat. |
| Learned thermal output | Per die: mean temperature, peak temperature, standard deviation, and Z-score; 4×M×N features | PDF pp.7–8, §IV-C and Eq. (7) context |
| Surrogate training | 25 ANSYS-labeled power/temperature cases expanded to >200 with RBF interpolation under thermal-equilibrium constraints; 5×4 DNN input is 20 powers plus 40 position values, output is 80 thermal features | PDF p.8, §IV-C and Fig. 8. The 10-layer encoder-decoder uses ReLU and batch normalization; Adam, batch size 32, 2,000 epochs. |
| Reported surrogate fit | Validation MAE below 2.0 °C; evaluation reports correlation 0.987 and average deviation 1.9 °C against ANSYS | PDF p.8, §IV-C; PDF p.11, Fig. 12 and §V-D. The paper does not specify a transient test set or enough independent held-out data detail to transfer this error bound to inference traces. |
| Reported simulator speed | 120 ms per ReThermal iteration including power and thermal estimation, versus about 210 ms for McPAT; the ANSYS comparison reports 13,846× speedup per iteration | PDF p.11, §V-C–D and Fig. 11–12. These are the paper's evaluation results, not a cost guarantee for this co-simulator. |
| Static scheduling | Stage ratios `αi ∈ [0.5,1]`; starts from balanced full load, adjusts stage compute ratios, swaps tasks between hot/cool regions, then uses simulated annealing | PDF p.9, §IV-D and Algorithm 1 |
| Runtime adjustment | Die-local compute ratios increase below thermal capacity and decrease under thermal saturation; updates use bounded, smoothed changes and temperature-aware hysteresis. Sampling/adjustment occurs between PP phases and is stated to be <5% of a stage duration. | PDF p.10, §IV-D(2). No numerical temperature thresholds, controller period, hysteresis width, or ratio increment is reported. |
| DVFS | The evaluated controller does not specify a DVFS policy. Table II fixes the evaluation at 0.85 V and 2 GHz; frequency/voltage are mentioned as possible temperature-estimation signals. | PDF pp.9–10, Fig. 9 and §IV-D(2), Table II. |

## What the paper does not specify

The paper does not provide a WSC layer stack, material list for the evaluated package, layer thicknesses, thermal conductivity, density or heat capacity, die-to-coolant/contact resistance, channel cross-section, channel count/branching, hydraulic diameter, mass/volume flow, pump curve, ambient boundary, numerical HTC-vs-position function, or calibrated lateral thermal resistance. The CPU cooling stack drawn in Fig. 1 (TIM, integrated heat spreader, baseplate and cooler) is a generic air-cooling illustration, not the WSC package stack used in the experiments. The WSC liquid-cooling figures show computing dies and coolant schematically, without package materials or channel geometry.

There is also no thermal integration time step or measured thermal time constant. The DNN maps a power distribution directly to equilibrium temperature features; it has no state for silicon/package heat capacity or coolant residence time. The LLM experiments concern training benchmarks. No result compares full prefill against KV reuse, and the paper does not calibrate power for HBM/DRAM/SRAM traffic, KV placement, cache movement, or inference collectives. Eq. (4) covers task-to-task inter-die communication only.

## Recommended compact RC plus flowing-coolant model

Use the paper's 5×4 grid, 25×30 mm dies, 300 K inlet, and 2 m/s nominal coolant velocity as the initial scenario. Represent each die with one effective silicon/junction temperature state and each channel segment under a die with one mixed-coolant state. Route five die segments in series per assumed TP lane, with four parallel lanes; make the lane-to-die ordering an explicit configuration because the paper's figure orientation is illustrative.

For die `i`, use:

```text
Cdie dTdie_i/dt = Pi - q_i - sum_j Glat_ij (Tdie_i - Tdie_j)
q_i = (Tdie_i - Tfluid_i) / (Rpkg + 1/(hi Aeff))
Cfluid dTfluid_i/dt = mdot cp_water (Tfluid_upstream - Tfluid_i) + q_i
Tfluid_inlet = 300 K
```

Here `Pi` is calibrated die power from the workload trace, `Glat` is physical RC coupling, and the fluid equation explicitly carries coolant enthalpy downstream. Use the fluid energy balance to conserve heat; do not translate the paper's `ε=0.01` directly into `Glat`.

| Starting value | Recommended default | Status and use |
|---|---:|---|
| Grid, footprint, inlet, velocity | 5×4; 25×30 mm; 300 K; 2 m/s | Reported in PDF p.10, Table II / PDF p.3, Fig. 2 |
| Die top area | `7.5e-4 m²` | Derived from the reported footprint; effective wetted area may differ. |
| Silicon thickness / properties | 0.5 mm; `ρ=2330 kg/m³`, `cp=700 J/(kg·K)` | Explicit engineering assumptions, not in the paper. Gives `Cdie≈0.61 J/K` for a 25×30 mm slab. Sweep thickness 0.3–0.8 mm and heat capacity ±25%. |
| HTC profile | `h=40,000 W/(m²·K)` at inlet to `15,000 W/(m²·K)` at outlet, linearly by channel segment | Assumed initial profile guided only by the qualitative direction and plotted scale in PDF pp.2–3, Figs. 1–2. Sweep 10,000–50,000 W/(m²·K); do not describe this as a paper measurement. |
| Package/contact resistance | `Rpkg=0.02 K/W` per die | Placeholder assumption for the unreported stack/contact path. Sweep 0.01–0.10 K/W and calibrate to measured or independent thermal-solver results. |
| Physical lateral conductance | `Glat=0.05 W/K` per neighboring-die edge | Placeholder assumption, independent of ε. Sweep 0.01–0.20 W/K; consider a shared spreader node if calibration shows stronger lateral coupling. |
| Water properties | `ρwater=997 kg/m³`, `cp=4180 J/(kg·K)` near 300 K | Standard engineering defaults, not in the paper; update if coolant composition/temperature changes. |
| Channel flow area | `Aflow=2 mm²` per lane, `mdot=ρ v Aflow≈0.004 kg/s` per lane | Scenario assumption because only velocity is reported. With 800 W interpreted as total and split evenly over four lanes, the 200 W/lane coolant rise is about 12 K. Treat this only as a baseline scenario; the table's TDP granularity is unclear. Sweep `Aflow` 0.5–4 mm² (or, preferably, measured `mdot`). |
| Fluid segment state | One segment per die, `L=25 mm`; `Cfluid=ρ cp Aflow L≈0.21 J/K` | Derived from the assumed channel area and one die-row pitch. At 2 m/s the segment residence time is 12.5 ms. The 5-segment path is 125 mm, consistent with the paper's qualitative >100 mm scale. |
| Thermal integration | Begin at `dt=1 ms` for explicit updates; verify by halving to 0.5 ms | Engineering choice: about one tenth of the assumed 12.5 ms coolant residence time. Use event boundaries for workload power changes. If using a different integrator, establish stability and convergence from its own timestep study. |
| Controller timing | Update mapping/compute ratios at prefill/decode or collective phase boundaries; impose bounded changes and hysteresis | Paper-inspired policy. It gives no numeric update period or thresholds. Do not model this as DVFS unless a separate DVFS policy is supplied. |

These defaults make a transparent, sensitivity-testable starting model; they are not a reconstruction of the proprietary ANSYS setup. For power, use workload-derived per-die `Pcomp + Pcomm` where coefficients are calibrated. Since α, λ(T), Nops scaling, communication volume, and per-die TDP are not fully specified, do not infer precise wattage from the table alone. Keep compute, inter-die communication, memory/KV traffic, and cooling-flow assumptions as separate power/energy terms so the full-prefill versus KV-reuse comparison can change each term independently.

## Validation required before interpreting inference temperatures

1. Check conservation at every step: die electrical power equals change in stored solid/fluid energy plus heat leaving the modeled boundaries, within the numerical solver tolerance.
2. Verify the single-die RC step response against its analytic first-order solution and verify a channel-only enthalpy rise against `ΔT=Q/(mdot cp)`.
3. Halve the integration step; require peak temperature and temperature-gradient changes below a chosen project tolerance (for example, 0.2 °C) before scaling to hour-long traces.
4. Compare steady-state and transient step/phase-change cases against independent ANSYS or hardware measurements, including inlet/outlet temperatures and hotspot location. A fit to one steady-state gradient is calibration, not validation.
5. Sweep HTC, channel flow area/mass flow, package resistance, die heat capacity, and lateral conductance. Report how the four requested workload/architecture objectives move under these thermal ranges.
6. Validate power traces separately for full prefill, KV reuse, cache transfer, and inter-die collectives. ReThermal's <2 °C / 1.9 °C results apply to its steady-state training/evaluation setup only and must not be claimed as expected inference accuracy.
