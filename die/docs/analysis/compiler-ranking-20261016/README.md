# GEMM 候选排序诊断延后生成

基线 Git `22c4106281bc44da49bf1ba44f5df6d3c43c9e4b`。只减少内部候选 JSON 存储；不减少候选、估值、结构/硬件/资源/credit/overflow/status 校验，不改变软件映射顺序、排序键、时间模型或校准参数。公开 enumerator 仍输出完整 sealed Plan。胜者用同一 GEMM builder 的原 tile/dataflow 重建完整摘要一次，不再估值或搜索；核对身份、launch/status 和九个排序键。所有 OS 曲线 audit 仍生成，不以缺元数据跳过描述预算检查。

`finalize_summary` 即使不存 JSON 也依原顺序执行所有 u128→u64、合并 HBM 加法和 EqWork 分数表示校验。完整路径的 JSON 类型/字段保持原合同；非 GEMM 仍走完整路径。`compiler_summary_test` 对两个 execution profile、两个 κ、四个 shape、两种 storage、R1/R3 的64任务共1092候选逐个比较完整物化 wire/hash，并核四个极值错误一致及篡改排序键原子拒绝。真实旧库/新库独立公开枚举的15,459,219字节输出全等；只保留一份确定性 gzip，SHA/命令/ELF 见 evidence。

最终一致 Release **49/49 通过，21.52s**；36候选的完整完成时间、仲裁路径、枢轴、切向轮次均与基线全等。11完整公开对象（包含缓存记账）与真实基线 CABI 逐字段相同。Norm 新 driver 32任务/96场景实际重跑，128份 raw/report/stderr/log 同已保留 v2逐byte一致；fresh suite、身份和内容引用另存。保持 Norm scoped completion96/first-output160 达10%、原 busy54 failed；不是硬件校准。

初版完整接口比较失败：routing profile 的临时字符串改为具名变量后，JSON字符串实际capacity从30变20，缓存正确记账因此少10B；预测/wire/hash未变。已恢复原 full-expression，初版 raw、解释、真实容量诊断全部保留。scratch诊断最初误用不存在的 Compiler.hh，未生成/运行二进制，换真实Plan.hh后实际old/new诊断成功；不冒充模型测试。Norm suite自身含命令路径、Python路径、ELF身份，不能与旧suite要求byte相同；仅128个实际结果文件做byte比较。

同一100样本计数诊断中，128³内部枚举分配7460→5038，完整编译16498→14447（少2051次、109514B请求），候选Runtime仍7293次，原逐候选估值保留。完整 thread CPU 中位1.470→1.340ms；计数器增开销、跨次环境可变，只作成本定位，不替代正常预算测量。最终正常30样本 **failed**：128³ cold P95 **1.508469ms**（初版1.564922ms，两次均保留），全部本次适用热/常规冷门槛通过；OS J1/J8/J32 **14.792/24.476/77.745μs**，J32超过50μs。初版J32 56.073μs和历史所有慢点保留，不选最快样本宣称全域通过。

正式五仓 SDK仍canonical24/digest `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`，五份逐文件hash实核相同，未执行publisher；新公共源五仓业务/CPU not_run。R>1真实非服务窗口、Norm长行两遍、原busy及其它模板精度依然未验收，target独立raw/heldout缺失，TX8/目标先验冻结，工程 **NOT ACCEPTED**。

复现入口（先激活环境；完整原命令在各`*-identity.json`）：

```sh
source /workspace/cloud-setup/activate.sh
unset PROTOBUF_FROM_SOURCE
cmake -S die -B work/build-die-continue -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die-continue -j 2
ctest --test-dir work/build-die-continue --output-on-failure
work/build-die-continue/coarse_service_curve_benchmark --samples 30 --output work/fresh-budget.json
python die/docs/analysis/compiler-ranking-20261016/collect_results.py --output work/fresh-summary.json
```

collector核当前生产/头/测试源、静态库/CABI/预算身份、实际完整对象、候选轨迹、保留raw内容及五份正式SDK；输出必须fresh。离开本源码版本后不能把collector旧源校验通过记录套给新模型。基线库在work只作本机便利，可按Git22c重建；重建ELF必须另记身份，不能假装旧ELF。`capture_candidates.cc`和`capture_retained.cc`可分别链接实际旧/新静态库，命令及ELF在 evidence。
