# 交接保存：Norm 长行首版部分成果，未进入生产

用户于2026-10-16要求立即停止开发并交接。基线为已共享的 `5caede8ca81902baf15c7d3228db70f2087302fb`。所有新增/修改源码已完整保存为 `partial-source/`、`partial-existing-files.patch` 和 SHA manifest；生产CMake/Mapping/旧离线driver已只反向应用本任务patch，恢复到基线，未覆盖其他工作。**这不是已验收或已发布的长行实现；下一账号不得把部分源码的测试套给当前main生产模型。**

部分实现：R1且rows≤Core数时，每Core独占一行；第一遍输入以两份256元素source槽流式统计，16B统计状态在片上保留，完整统计kernel终结后才第二遍HBM读取输入并应用gamma/beta。第二遍采用已有固定首/中/尾方法、两份输入/输出槽、最多三活动；7epoch/12服务描述（launch计入后13），不逐tile/row/repeat执行。数学6/8 eq-op拆为2+4/3+5，仍工程SFU计数；每次转换、参数单独INT4打包、SRAM/LOCAL和第二遍HBM显式归账。多行/Core及R>1未实现，不能把16B状态借给多行免费保留。

实际已有验证：24种Norm/type/Core/row独立账本及实际Runtime完成、固定epoch/描述、统计阻塞/有限预取、cancel、snapshot恢复的单项测试最终 **1/1通过，0.04s**。初次预取断言错把累计读取当驻留输入，public next_event向上取整前已有少量统计消费，2053.12B累计读取并非2048B容量越界；保留两次失败，改为累计输入减实际3eqops/element统计消费核驻留，原物理credit容差未改。初次CMake编辑helper猜错foreach变量而assert，未编辑；之后真实注册/构建/运行，所有日志保留。

已启动的12任务/36场景oracle在交接时已完成，无进程失败。单Core/单行/BF16/R1离散统计+apply参考中，60完成样本和60首输出样本均≤10%，max相对0.354286%/1.440490%；**36联合场景全部failed**，108项原busy指标仅64达10%、44失败（最大相对169451.06倍）。raw/reference/差值全部保留，不改busy或门槛，不拟参数，不是硬件校准。完整50公共回归、新scope性能预算、fractional/更宽shape审查、五仓新模型集成均 **not_run**；未启动这些工作。

`die/docs/evidence/norm-long-row-20261016/handoff-summary.json`为简表，`curves-initial/`保留12份raw/report/stderr/log及suite，`*-identity.json`保存实际源/ELF/命令。归档后 **work/build-die-continue里的库和driver仍是部分长行版本**；未为交接另做构建/测试，不能将其当作main已测ELF。canonical模型最近有效CABI为e1347630…43a3c6，历史身份/49测试/预算在compiler-ranking检查点；本机只读基线副本在work/j32-update/baseline-binaries（work不跨账号，GitHub来源为Git5caede8）。

恢复方式：先同步最新main、确认无未知修改，在独立分支/worktree中校验manifest；以下只应用本归档，不能强制覆盖冲突。

```sh
git apply --check die/docs/analysis/norm-long-row-20261016/partial-existing-files.patch
git apply die/docs/analysis/norm-long-row-20261016/partial-existing-files.patch
```

随后按manifest `new_files_sha256`将 `partial-source/<原路径>`逐一复制回原路径（四文件），逐一核hash；已有目标若不同须先查明，不覆盖。恢复后CMake注册 `coarse_norm_long_row_test` 和 `coarse_norm_long_oracle_driver`，验证入口分别是相应CTest和 `run_cases.py --driver <新构建driver> --output <fresh目录>`。不直接运行canonical树中残留的旧work binary。先完成50公共回归及编译预算/普通模型完整对象兼容，再决定采用；原busy失败和多行/Core、R>1边界必须保留。未采用的J32遍历融合不是待恢复路线，见相邻j32-update归档。
