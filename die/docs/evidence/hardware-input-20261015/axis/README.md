# Analytical OS service curve: six-shape axis check

Each row compares the actual selected OS Plan and Runtime with the existing grouped discrete K-wave oracle.
The fixture is single-Core 1x1, FP32, with synthetic finite resource capacities. It is model-to-model evidence; it contains no silicon measurements or target calibration.
Description counts and available framework rows are read from each actual serialized Plan; consistency is checked by summing each epoch's activity segment counts, with empty curves and empty epochs contributing one description.
Acceptance limits are copied per case from each actual oracle report; the original standalone completion, concurrent completion, HBM busy, and shared HBM busy criteria are used unchanged.

| Shape M×N×K | Plan E / descriptions | Status | Solo finish rel. / abs. | Solo HBM busy rel. / abs. | Sync finish rel. / abs. | Sync HBM busy rel. / abs. | Sync shared busy rel. / abs. / ref. duration | Offset finish rel. / abs. | Offset HBM busy rel. / abs. | Offset shared busy rel. / abs. / ref. duration | Events / allocation updates (solo; sync; offset) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16×64×64 | 3 / 3 | passed | 8.63e-07 / 9.27e-10s | 0 / 0s | 3.97e-07 / 8.53e-10s | 0 / 0s | 0 / 0s / 0.000492s | 8.63e-07 / 9.27e-10s | 0 / 0s | 0 / 0s / 0s | 6/4; 9/5; 12/8 |
| 16×64×4096 | 3 / 4 | failed | 5.72e-09 / 3.04e-10s | 2.93 / 0.0385s | 5.72e-09 / 6.09e-10s | 2.93 / 0.0769s | 2.93 / 0.0769s / 0.0263s | 0.0118 / 0.000663s | 2.98 / 0.0783s | 7.25 / 0.00256s / 0.000354s | 8/6; 11/7; 16/12 |
| 80×128×4096 | 4 / 7 | failed | 8.38e-11 / 4.44e-11s | 3.02 / 0.397s | 8.38e-11 / 8.89e-11s | 3.02 / 0.795s | 3.02 / 0.795s / 0.263s | 0.00642 / 0.00648s | 3.03 / 0.798s | 2.68e+03 / 0.948s / 0.000354s | 13/10; 17/11; 26/20 |
| 65×65×4097 | 4 / 7 | failed | 0.0692 / 0.0163s | 1.91 / 0.143s | 0.0692 / 0.0325s | 1.91 / 0.285s | 1.91 / 0.285s / 0.149s | 0.0659 / 0.0281s | 1.93 / 0.288s | 15 / 0.322s / 0.0215s | 13/10; 17/11; 28/22 |
| 256×64×64 | 4 / 6 | failed | 6.12e-08 / 8.26e-10s | 2.22 / 0.00874s | 2.42e-08 / 6.52e-10s | 2.22 / 0.0175s | 2.22 / 0.0175s / 0.00786s | 6.12e-08 / 8.26e-10s | 2.22 / 0.0175s | 0 / 0s / 0s | 11/8; 15/9; 22/16 |
| 16×64×6400 | 3 / 4 | failed | 7.98e-09 / 6.63e-10s | 2.97 / 0.0609s | 1.96e-09 / 3.26e-10s | 2.97 / 0.122s | 2.97 / 0.122s / 0.041s | 0.00574 / 0.000665s | 3 / 0.123s | 176 / 0.0622s / 0.000354s | 8/6; 11/7; 16/12 |

## Reading the result

Completion errors and HBM busy errors are reported separately. A smooth cumulative service curve can keep completion close to the discrete oracle while extending the interval classified as HBM busy; the table preserves the original busy integral and exposes the absolute error and reference duration.

The tool only covers these six single-Core FP32 OS points. It does not certify other shapes, dataflows, core grids, hardware, or minimum description counts. A failed precision status remains failed.

Historical initial-model 80×128×4096 result is excluded from this batch: `null`.

Machine summary: `/workspace/template/work/analytical-curves/axis/summary.json`. Per-case actual raw JSON and oracle reports are beside this file.
