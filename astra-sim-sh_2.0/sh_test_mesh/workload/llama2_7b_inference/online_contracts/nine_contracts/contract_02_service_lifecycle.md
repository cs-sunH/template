# 合同② 服务生命周期合同

冻结时间: 2026-08-16（阶段 0 步骤 0-4）；审查通过是阶段 1 开工硬门槛。

## 口径
- 状态机：IDLE / ACTIVE / DRAINING / FINISHED。
- 空 EventQueue ≠ FINISHED：`finished()` 条件 = 输入已关闭（EOF/显式 close）
  && active_request_count==0 && 无 pending alarm/fence。
- EOF vs 显式 close vs 异常：三态 InputCloseReason{ExplicitClose, EndOfFile,
  Error}；异常 fail-closed abort（退出非 0，不静默降级）。
- 无请求启动 → 保持 IDLE（不退出、不忙等，阻塞在桥接/注入通道）；
  request-neutral：不提供 request 输入时不读任何预置队列。

## 裁决
- **最终结束权唯一归 ServiceCoordinator**：在线模式禁用 Workload::call 里
  `sim_notify_finished()/is_finished=true` 副作用（Workload.cc:672-684），
  但保留 `local_hbm_bandwidth_model->has_active_jobs()` 条件语义
  （:679-680——HBM 模型有 active job 时不得视为空闲，§4.3.2）。
- 在线模式构造链 = 执行模式工厂：Sys 创建时注入动态 GraphSource，不构造
  ETFeeder、不要求 .et 文件；静态构造路径原样。
  LocalHbmBandwidthModel 装配时机与参数不变（Workload.cc:57-60）。

## 验证方法
- IDLE fixture（步骤 1-2/1-10）：五态迁移日志齐全、注入 2 request 后
  ACTIVE、close 后 FINISHED 退出 0。
