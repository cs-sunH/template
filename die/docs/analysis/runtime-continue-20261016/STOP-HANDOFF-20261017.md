# 立即停止交接（2026-10-17，Asia/Shanghai）

用户要求停止；没有启动新开发、实验或回归。进程检查时已无构建/测试进程。本页覆盖此前“继续开发”的说明；生产源码没有采用下面的研究候选。

- 有效生产：`codex/die-runtime-continue-20261016`，停止前已推送 `fe2569abe532c610da8901b98949ee24697625a2`。`Execution.cc` SHA256 `6fcc637e13c876d240d92d8677e8f9d2e323b325aea46176848063ffe93e710e`；头 `4ae56348aab0f39c6e67ecc28fe157cf6aaffd9478ac107af4b2f1d3003f0d6a`。本次仅保存已有研究、日志、脚本及交接文档，恢复应取该分支最终 HEAD，不只取上述前置提交。
- 当前 A+B 的 fresh Release 53/53、main 对照两个新测试 red、API/快照验证见 [public-ab](../../evidence/runtime-continue-20261016/public-ab/checkpoint-summary.json)。严格矩阵仍78/84，六个原精确舍入断言失败；两配置正常30样本性能均失败。不是硬件校准或最终验收。
- 五仓本轮冻结 digest `25c51345c5ddcedb45b21c8b4dc89805c1f9ea2166d49792ddaf639ccbf91013`：10 ABI 库、98配置通过；Python384pass/1原optional skip，严格runner保留该失败。旧历史22/4 native不作为本轮结果。
- 本轮 native：JOINT fresh build、10/10 C++、6/6 Python实际通过；FACE fresh build、3/4 C++通过，Lifecycle因真实R4 body-only guard失败。FACE-LRU configure成功，build日志停在100%编译阶段但无完成identity/summary，**不算构建通过**；其消费者、WS/WSLRU native本轮未完成/未运行。没有为交接补跑。原日志和started/identity/summary全部归档。
- Norm checkpoint1/2已推送，候选2因负4ULP快照与service_curve真实失败被拒绝。capacity1 Norm、内部数值/每角色可表示性、原busy、长行/R>1与速度仍未解决。
- 独立 time-only C1：六项既有测试和强subepsilon探针通过，但仅替换Execution对象的mixed研究。独立审查**实际复现提前ready服务错误**，不能采用。完整正小terminal/1ns后正尾探针有改善；未改变此拒绝结论。`time-review-ab-time-only-v2`指审查driver第二版，仍链接C1，**不是C2测试**；初版busy oracle错误、第二版最后duplicate activity id工具错误均原样保留。
- time-only C2：仅执行prepare生成源码/patch/identity，位于归档的 `work/runtime-positive-time-v2/`；**未编译、未测试、未采用**。生成脚本 `prepare_time_only_v2.py` 已保存。没有修改正式Execution/头/ABI。

## 恢复

1. fetch远端，读最新HANDOFF，从上述恢复分支最终HEAD另建开发分支，核对main共享指针；保留main其他修改。只使用唯一Core权威DOCX与当前粗粒度路线。
2. [停止归档](../../evidence/runtime-continue-20261016/stop-handoff-20261017/manifest.json)保存全部新增研究text及本轮native原始结果；`existing-text.tar.gz`条目以`work/`开头，在仓库根恢复到work，**不要把候选覆盖到生产源码**。先验证manifest逐文件hash。Norm旧checkpoint1/2另在`norm-capacity1/`，不用重复解包。
3. 优先审查C1真实ready反例与尚未编译C2；修正审查driver重复ID并保留原失败。若继续候选，先独立定向验证，再fresh全源/API/性能，不能把mixed结果当采用。按依赖续接剩余native；必要时全新构建，不能把部分日志/旧ELF当通过。

五份正式SDK保持canonical24/digest `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`，publisher未运行。A+B未进入main；main仅计划同步交接入口。工程 **NOT ACCEPTED**。远端保存、main生产集成、SDK发布与工程验收分别核实。
