# Codex Cloud Session 交接记录

> 这是不同 Plus 账号间的**共享状态摘要**，不是可恢复的原始 Session。只有相关代码已提交并进入共享集成基线后，才算可安全接棒。

## Current Status

SETUP_ONLY — 接力文档初始化完成，尚未对 `die/` 的实际功能开发进度作出判断。

## Integration Branch

`main`（当前默认分支；如改用专门的接力分支，请同步更新 PROJECT.md）

## Last Verified Base

待首次接棒任务核实：记录合并后的 commit SHA，并确认云端任务工作区包含该提交。

## Completed

- [x] 建立 die 子项目的接力规则与文档模板。
- [ ] 检查 die/ 代码结构和真实项目需求。
- [ ] 识别并运行项目已有测试。
- [ ] 根据实际项目确定下一阶段开发计划。

## Next Step

1. 读取 `die/AGENTS.md` 和 `die/docs/PROJECT.md`。
2. 检查仓库 `main` 最新提交、当前工作区的 HEAD、`die/` 文件结构与已有测试；确认当前任务基线未落后。
3. **先查明 die 项目的真实目标和现状**，补充 PROJECT.md 中标记“待确认/待核实”的项目资料。
4. 在确认真实需求后把可独立验证的下一阶段任务写入本节，再开展功能开发。不要根据这份模板臆造功能需求。

## Important Decisions

- 用 GitHub 已提交/已合并的代码 + 此交接文件衔接两个独立的 Codex Cloud Session；不假设可以跨账号恢复聊天历史。
- 优先通过新建 Cloud Task 读取最新集成分支，而不是继续可能已经过时的旧工作区。
- 按最小可验证阶段提交，尽量避免接近用量上限时仍存在大量未交付改动。
- 暂不创建或假定 `codex-relay` 分支；当前基线是仓库现有的 `main`。

## Files to Continue

- `die/AGENTS.md`
- `die/docs/PROJECT.md`
- `die/docs/HANDOFF.md`
- `die/` 内实际源码与测试（由下一任务核实后补充）

## Tests

- 本轮仅建立流程与文档，**未运行项目代码测试**。
- GitHub Actions 只检查交接文件是否齐全及关键章节是否非空，不代替功能测试。

## Blocking Issues

- 项目具体目标、运行方式和验收条件尚待根据仓库内容与用户需求确认。
- 未验证 Codex Cloud 账号在该仓库上的实际推送或创建 PR 权限。

## Handoff Notes

每次结束一轮任务时，必须把以下内容写进本文件：

- 已完成内容（含相关文件）及尚未完成内容。
- 下一步明确的操作和预期验收标准。
- 已核实的测试命令及其成功/失败结果；未运行要注明。
- 本轮的关键设计决策、失败尝试与已知阻塞。
- 当前工作区 HEAD / 交付提交 SHA、PR 地址及是否已合并（不能将未合并的 PR 写为已交接）。
- 如果代码未成功提交或 PR 未合并，明确标记 **NOT READY FOR HANDOFF**。
