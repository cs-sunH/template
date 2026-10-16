# 唯一Core迁移：完整交接归档

用户要求立即停止。本归档保存全部有效部分成果，未将未经一致验证的迁移激活到main。
代码/文档/配置/测试/原始证据完整在远端分支`codex/die-single-core-20261016`，
实现提交`96c4e9475792293814608c93b2dc383a6c1cb09e`，最后证据提交`b8ddbcfd62e03738b0c3996444355cbbe2e99b80`。
main保存全部差异的二进制Git补丁（含新增源码、gzip/tar证据），manifest记录hash。
已在原基线执行`git apply --check --binary`通过，无需依赖本机work文件或ELF。

优先恢复：

```sh
git fetch origin main codex/die-single-core-20261016
git switch -c codex/die-core-next b8ddbcfd62e03738b0c3996444355cbbe2e99b80
```

若独立分支不可用，可从main归档恢复到全新工作树，不在有改动目录强行应用：

```sh
git worktree add -b codex/die-core-recovered ../die-core-recovered 41f119135fd2d4cc87292f59b4b00adc918eece0
gzip -dc die/docs/analysis/single-core-handoff-20261016/complete-migration.patch.gz > /tmp/die-core-recovery.patch
git -C ../die-core-recovered apply --check --binary /tmp/die-core-recovery.patch
git -C ../die-core-recovered apply --binary /tmp/die-core-recovery.patch
```

准确测试范围、当前未运行项和下一步见[HANDOFF](../../HANDOFF.md)。
原长行/数值失败原型位于恢复后analysis/pre-core-migration-20261016，
五仓所有脚本/raw/identity/授权前源及最后中断状态位于恢复后的
`evidence/single-core-20261016/integration/integration-artifacts.tar.gz`。
停止后不再启动开发、实验或额外测试。
