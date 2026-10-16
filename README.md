# Architecture research and single-Die simulation

当前硬件依据：[架构建模_单Core版_严格复核修订.docx](架构建模_单Core版_严格复核修订.docx)。硬件只有一种 Core 设计，SRAM 与 NoC 位宽选择 32 行资源配置，边长为 `L0=a`；Die 外形及 Core 网格由用户给定。

仿真器入口：[die/README.md](die/README.md)。[仿真设计文档](die内仿真器设计文档_最粗粒度参数响应版.md)继续规定解析累计服务曲线、精度与速度合同；这些条款不由硬件 DOCX 替代。

真实开发进度、源码身份、验证范围与交接：[die/docs/HANDOFF.md](die/docs/HANDOFF.md)。旧设计与证据仅按其历史范围参考，不能视作当前硬件已验收。
