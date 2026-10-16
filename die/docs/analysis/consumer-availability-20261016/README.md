# 不完整预测的消费者边界

基线为已实核 main/HEAD `9bb8c0ac74ad3b78edb0004d7ebbc61f00942dda`，
工作分支 `codex/die-explicit-continue-20261015`。这是 HANDOFF 要求的
新 SDK 接入前置修复；公共编译/Runtime 继续允许运行 body 诊断，但真实
执行、workspace admission、依赖完成与成本评分不能把该数值当完整预测。
正式五仓 vendor 仍是 canonical24，未在主工作树运行 publisher。

## 最新补充：隔离五仓已完成（基于 main 4114533）

`preview-complete-final-summary.json` 为当前source-bound总表，原411的
`checkpoint-summary-final.json` 及所有原raw保持不变。五仓consumer和双ABI
均构建成功，配置18/17/20/15/13合计 **83/83，无skip**；五份coarse CABI
均逐byte等于公共47。32文件eb0472bf freeze始终未变，actual文件hash全对，
公共生产源/配置/头均与当前相同，CMake仅少CLI注册（SDK testsOFF）。
五个真实R3 compile/prepare/ready-scan probe均通过，含合法R1正例。
正式五份24文件仍逐个吻合23160c…7e9b，没有运行正式publisher。

新SDK原标准CTest四仓lifecycle在原样实际R4 FFN处failed；JOINT为5/6，
NativeProjection原共享HBM断言failed。该矩阵整体failed，不能因ABI/config
通过就称业务通过。两个WS fixture原来首个Runtime事件后立即取消；现在首
事件可以只结束launch等待，测试改为最多32事件等实际compute服务。与原
冻结生产objects链接的完整测试main中，取消/typed/两rank指标原断言全部过，
后续原R4仍以prediction_not_available abort（真实-6、外层250）。没有改变
生产模型、放宽断言或缩小实际R4任务。

JOINT原1MiB端点50ns到达，已在计算1000ns真实launch等待内结束；专测
sharing的parity和capacity fixture改为真实launch之后偏移50ns到达。
硬件/shape/字节保持，direct Runtime读取相同wire时刻，capacity同tick语义
保持；其它snapshot/cancel的早到达50ns用例不动。新SDK完整NativeProjection
main通过，原标准失败二进制/log仍保留。初次runner错误假定target在frontend
导致未编译/未运行，保留preflight transcript；随后从实际flags.make定位。

两个WS和JOINT修正后旧SDK重建并实测2+6项通过；未变FACE两仓各1项的
已绑定当前源证据仍有效，故当前旧SDK **10/10**。新的LastTest复制到新总表
旁，不覆盖历史日志。新版recorder区分隔离archive和外层Git HEAD，stage的
base_head=null，另列archive base9bb、freeze SHA、overlay核验；旧raw中
base_head是当时外层HEAD，不能用作stage版本身份。

`run_repeated_prepare_probe.py --kind lifecycle --fixture-source PATH` 链接
完整fixture main以核取消采样，`--kind native_projection` 核JOINT完整main；
默认仍是R3拒绝probe，均不改标准目标。汇总复现：

```sh
python3 die/docs/analysis/consumer-availability-20261016/collect_preview_results.py \
  --output die/docs/evidence/consumer-availability-20261016/FRESH-summary.json
```

这次只有测试和证据工具变化，公共47、9bb/412性能/精度证据按原范围复用，
不声称新测量。R>1真实窗口、busy/首输出、五仓真实负载/CPU和target raw
仍未通过/缺失，正式发布not_published，工程NOT ACCEPTED。继续公共模型。

## 源码边界

公共 `include/coarse_die/Prediction.hh` 提供一个无状态检查。仅检查 Plan
summary 顶层 `prediction_status`、`alone_estimate_status`，明确
`not_available` 时抛 `prediction_not_available`；已存在字段非 string
报 `invalid_prediction`。旧 Plan 缺字段仍兼容，不因 nested target/heldout
缺测而拒绝已有预测。原结构、硬件和物理验证不被替代。

五仓离线适配各保留与公共头逐 byte 相同的 `CoarseDiePrediction.hh`，使
原 canonical24 无需先发布新 SDK 即可编译。公共测试约束六份源一致；
相同 include guard 允许将来 SDK 同时提供公共头而不重复定义。

- 五仓 CoarseDieRuntime 在冷/热编译和直接 submit 时检查，发生在缓存
  插入/LRU mutation、Sys tick 采样和 Runtime submit 之前。
- 五仓 DieModelAdapter 在 prepare、手工 Prepared acquisition/validation
  检查，发生在 workspace 字节查询、ledger acquisition 或执行发布之前。
- JOINT native chain 检查全部未来 Plan，先于 advance；snapshot 与本地
  Runtime sample 导入也检查，避免从恢复路径绕过。NativeProjection 的
  直接编译入口同样检查。原 sh30/provider/live import guard 保留。
- FACE/FACE-LRU/JOINT Python full、FACE 两仓 cost/可选 full fallback
  检查返回行的顶层 status，拒绝发生在数字读取和 seconds 换算之前。
  不可用行返回 None，并在原位置记录 last_errors；合法/原生错误兄弟
  不变，malformed status 抛异常，已有 finally 释放结果 buffer。
  FACE scorer 对 None/异常原本就停止，不改候选、评分或 legacy fallback。
- 公共 estimate.py CLI 对不可用或 malformed status 行保持原 body，标记
  整体 partial_error 并非零退出。行 status_code=0 仍准确表示 ABI 调用成功，
  CLI 层 error.reason 标明 prediction_not_available/invalid_prediction_status。

不修改曲线、原子工作量、资源、公平、ABI、硬件输入或校准参数。没有把
R>1 改成 R1 来获得生产成功；body 数值仍可由公共 ABI/Runtime 读取。

## 已有直接证据与构建过程

- 公共 Release CTest 46/46，18.38s。包括实际 R3 编译标为不可用，消费者
  helper 拒绝而公共 Runtime 可 submit/cancel，legacy/缺测/字段类型及
  五仓头一致性。首次启动漏 activate、ctest 不在 PATH 为 not_run，原样保存。
- JOINT NativeCoarseDieDomainTest 修复前真实失败、修复后通过：合法首
  Plan 后接不可用 Plan 不能推进当前 blocker；snapshot 未来 Plan、本地
  body sample 恢复均拒绝，原输入和状态保持。见 cpp/native-*。
- Python focused 21/21，完整相关套件 FACE70/FACE-LRU63/JOINT15，合计
  148/148、无 skip。mock 和 staging 缺依赖/fixture 的初次失败全部保留，
  不能当生产语义失败；完整命令/源/库身份见 python/README.md。
- 增补真实 C ABI 对拍，不只 mock：同一当前共享库对 R3 residual 与未经
  修改的实际 FACE 发射 FFN（layer_count=2、weight_passes=2）返回有数字的
  not_available。Git9bb 消费者在 full/cost/optional fallback/estimate_one
  接受这些数字，真实 red failed；当前三消费者全部拒绝，合法 R1 和原生
  错误兄弟保持，真实 green passed。见 python/actual-native-{red,green}。
- 第一轮 FACE C++ regression 编译因新测试使用不存在的 lookup 失败；
  四仓改为实际 lookup_ptr，JOINT 原接口仍 lookup。FACE 随后实际完整
  lifecycle 通过。FACE-LRU 第二轮发现 Workload prepared seam 是 private；
  测试改为已有 public adapter issue seam，未为测试扩大生产 API。
  其它四仓保留已有 public Workload seam。该差异不等同于五仓都测到
  Workload 手工 bypass；真实新 SDK R3 ready-scan 另测。

最终旧 SDK 五仓 **10/10 CTest 通过**：四个 lifecycle 各1，JOINT 的 legacy
lifecycle、coarse workspace 与四个 native/window 合6。所有实际被测仓源
SHA 与当前文件一致、正式五份24文件均吻合原 manifest，六份 policy 头
逐 byte 一致。old-three 原总体 exit1保留，原因是已单独修复并重跑通过的
FACE-LRU private seam 编译；不能只看最后进程退出值推翻各项真实记录。
汇总/冻结 matrix 在 checkpoint-summary-final；完整成功断言在各 LastTest.log。

CLI 最终相同7项测试实际 Git9bb red failed、当前 green 7/7（原始6项 red
也保留）。含实际只有 R3、ABI无原生错误仍须 exit1，R1 exit0、mixed/native
错误/原body保持和 nested heldout不误拒绝。最终一致公共 Release **47/47
通过，53.59s**；当时有并行功能构建，不将墙钟作性能证书。46/47的公共静态
库与 CABI 全部 byte-identical，die/src 全部与Git9bb相同；不重复无影响的
模型精度或速度测量，沿9bb/412绑定证据及历史失败，不宣称消费者CPU已通过。

隔离预览32文件digest `eb0472bfa6e4370232d058fa6c1d6b4169132f206d86770449cce0e4d293e55a`
在 CLI 新测试注册前冻结，CMake少该Python测试注册，SDK tests显式OFF；
生产源/公共头与当前一致，FACE coarse CABI与公共47的SHA相同。
截至此检查点，仅 FACE 已完成：两ABI/完整consumer构建成功、配置 **18/18
通过、0skip**，标准 lifecycle 在原样R4 FFN处 **failed**。真实R3独立
prepare/ready-scan probe **passed**：冷/热R1正例、R3真实marker与公共body诊断、
两次compile/prepare拒绝不改cache/graph/tick/lease、真实ready-scan不发terminal。
不将这个拒绝测试冒充R4业务成功。其余四仓预览构建/测试当时仍未完成，
继续执行。FACE完整源/ELF/失败在 preview-face-checkpoint，probe/config另存。
最初配置发现路径错误，Ran0/exit5记录failed_or_incomplete；修正到真实tests
目录后18通过，不把0tests当通过。构建并行仅作功能验证，不作性能测量。

## 隔离发布与复现入口

`prepare_preview.py --tag FRESH_TAG` 从当前 Git HEAD archive 五仓及公共
SDK 源，显式叠加当前修改/新文件，拒绝不明删除和正式 vendor 修改；仅在
新建 stage-root 调用该 stage 自己的 sync_repositories。它核验主工作树
五份正式 manifest 前后不变，记录 archive/overlay/SDK 摘要与完整命令。
最初两个 freeze 未构建，测试 API 修正后使用新的 freeze，不覆盖旧记录。
这些是隔离验证，不能宣称正式五仓已发布或工程已通过。

```sh
source /workspace/cloud-setup/activate.sh
unset PROTOBUF_FROM_SOURCE
python3 die/docs/analysis/consumer-availability-20261016/prepare_preview.py --tag fresh-sdk-preview
python3 die/docs/analysis/consumer-availability-20261016/run_cpp_matrix.py \
  --mode preview --freeze work/consumer-availability/fresh-sdk-preview/stage-identity.json \
  --label fresh-preview-checks --repos astra-sim-face astra-sim-face-LRU \
  astra-sim-wscllm astra-sim-wscllm-LRU astra-sim-joint
```

`run_repeated_prepare_probe.py` 复用每仓真实 Sys/Graph fixture，另编 test
main，再与标准目标完全相同的生产/backend objects、archives、SDK 和
flags 链接。不会修改 CMake 或标准测试二进制，不另写延迟公式；保留每条
实际编译/链接/执行命令、所有链接输入 SHA。此入口要求真实 R3 编译器
标记 not_available，检查实际 prepare/ready scan 拒绝前后完整状态、合法
R1 冷/热缓存正例和公共 body diagnostics。它不能代替标准业务回归。

```sh
python3 die/docs/analysis/consumer-availability-20261016/run_repeated_prepare_probe.py \
  --root work/consumer-availability/fresh-sdk-preview/stage-root \
  --build work/consumer-availability/fresh-sdk-preview/build/astra-sim-face \
  --repo astra-sim-face --label fresh-face-repeat-probe
python3 die/docs/analysis/consumer-availability-20261016/probe_python_native.py \
  --oracle-root . --library work/build-die/lib/libcoarse_die_cabi.so \
  --output work/actual-python-native.json
```

所有 probe/matrix 要求新 label/输出，保留失败。work 内二进制和 archive
不入 Git，必须按记录重建。原始日志需显式 force-add，不能因 .gitignore
丢失。R>1 非服务窗口、原 busy/首输出精度、真实五仓业务和 CPU、独立
target raw 仍未验收，TX8/目标效率冻结，整体 NOT ACCEPTED。
