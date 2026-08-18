# ASTRA-sim
[ASTRA-sim](https://astra-sim.github.io/) is a distributed AI system simulator. It models the end-to-end software and hardware stack of modern AI systems - encompassing workload scheduling, collective communication algorithms, and hardware architectures (compute/memory/network). Through a suite of APIs, it enables plug-and-play of external open/proprietary components for modeling different parts of the AI system. This provides end-to-end multi-fidelity simulation capabilities for aiding in design and deployment of next-generation distributed AI systems. 


### Overview and Documentation
Here is a concise visual summary of ASTRA-sim, showing its layers and APIs:
![alt text](https://github.com/astra-sim/astra-sim/blob/master/docs/images/astrasim_overview_codesign.png)

For a comprehensive understanding of the tool, and to gain insights into its capabilities, please visit our [website](https://astra-sim.github.io/).

For information on how to use ASTRA-sim, please visit our [Wiki](https://astra-sim.github.io/astra-sim-docs/index.html).

ASTRA-sim accepts MLCommons Chakra Execution Traces as workload-layer inputs. For details, please visit [Chakra Github](https://github.com/mlcommons/chakra).


### Releases and Contributions

ASTRA-sim is currently at **version 2.0.**
The previous version, ASTRA-sim 1.0, is available in the `ASTRA-sim-1.0` [branch](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0).

We encourage community contributions to ASTRA-sim via PRs.


## Contact Us
For any questions about using ASTRA-sim, you can email the ASTRA-sim User Mailing List: astrasim-users@googlegroups.com

To join the mailing list, please fill out the following form: https://forms.gle/18KVS99SG3k9CGXm6


We appreciate your interest and support in ASTRA-sim!

## 残留补清（2026-08-18，主 agent 执行，工作树未提交）

删①②主链后的验收工具类残留清理：build_analytical_aware.sh（目标已删脚本必坏）、
run_metric_microbench.sh + generate_metric_microbench.py（microbench=①族工具链）、
( [ sh_1.0 = sh_3.0 ] && echo "b3_canonical_compare.py（对照①离线产物）" )( [ sh_1.0 = wscllm ] && echo "tier_b_compare.py（②族 oracle）" )（tier_b_compare 已在主清理中删除）；README_COMMANDS 悬空引用行同步清理。保留定性：clean_history.sh（清 generated/results，③④ 同用）、run_metrics_postprocess 链（③④共用）、contracts/tier_b 报告/实录（历史记录载体）。剩余文字性提及均为注释或历史文档，无功能性依赖。
