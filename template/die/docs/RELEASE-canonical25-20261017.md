# 正式 SDK 发布声明：canonical25（2026-10-17，Wave 4）

- **发布身份**：canonical = **canonical25**；source_digest =
  `5de03f5f2921e39abad3718b82199f98bf1e7e6c7c478379f5031d86593e401c`（33 文件；`die/tools/sync_repositories.py --check` 五仓 passed，
  2026-10-17 实测）；die 侧发布提交 = 本提交（HEAD 见下方签署栏前的 git log）。
- **相对上一身份（repeat-v1 工作同步 85a7998e / 历史正式版 canonical24）的内容增量**：
  1. 生成式串行 repeat v1（plan v5、repeat_seam_modeled、五仓 R4 转绿——已随 4fd520c
     以 85a7998e 同步，本次重同步随下列新增一并冻结）；
  2. P2：busy 计数器 canonical dyadic 语义（u128 精确累加 + 快照精确 dyadic 字段，
     CONTRACTS.md C-3）；
  3. P11：快照导入 ready_ns 归纳下界收紧（issue+launch+round×seam）；
  4. 工具面：run_strict_matrix.py（P1 豁免复现入口）、benchmark.py 比率哨兵模式（P4）、
     busy_partition_invariance_test（随 COARSE_DIE_BUILD_TESTS 开启）。
- **E2E 证据（本 digest，五仓 aggregate 重建 BUILDTARGET=congestion_aware）**：
  - DieExecutionLifecycleTest **5/5 全绿**（face / face-LRU / wscllm / wscllm-LRU / joint）；
  - DieModelAdapterTest 全绿（face、joint 实跑；face-LRU/wscllm/wsclm-LRU 为各自
    aggregate 中通过的那个）；
  - WorkspaceAdmissionTest：face 绿；face-LRU/wscllm/wsclm-LRU 在本次临时 harness 中
    **Not Run**（目标未随显式列表构建——P14 已记录的 build-by-contract 类别，非测试
    失败；该三仓 consumer C++ 代码本轮零改动，绿记录结转 4fd520c）；
  - python：joint 声明联测 30+15+7 绿；wscllm/face 侧 python 结论结转（本轮零相关改动）。
- **发布状态**：**工程发布（engineering release）生效**。工程整体 **NOT ACCEPTED** 保持：
  P15-L2 稳态机绝对预算门未执行（无稳态机资源；P3 D 路线的单次更新口径重测义务同批），
  维护者签署栏开放。ACCEPTED 需 L2 测量 + ACCEPTANCE.md 签署。
- **合同快照**：CONTRACTS.md C-1/C-2/C-3；legacy ×R not_available 一字未动；guard 未放宽。

## 状态更新（2026-10-18）：验收签署生效

- 工程整体状态由 **NOT ACCEPTED** 更新为 **ACCEPTED-WITH-DISCLOSURES**：维护者签署栏已于
  2026-10-18 填写（见 die/docs/ACCEPTANCE.md）；L2-1 稳态机绝对门与 P4 双机一致性未执行、
  经维护者豁免结案（豁免≠通过，如实留档）；五仓 CPU≤10% 隐藏项经维护者裁决删除。
- 本发布声明的其余内容（发布身份/digest、内容增量、E2E 证据、合同快照）不变；本批为纯
  文书改动，未触碰 include/src/configs，source_digest 不受影响。
