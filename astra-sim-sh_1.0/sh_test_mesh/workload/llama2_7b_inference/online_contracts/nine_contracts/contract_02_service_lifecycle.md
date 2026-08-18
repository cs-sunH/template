# 合同② 服务生命周期合同(IDLE / ACTIVE / DRAINING / FINISHED)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径
- 状态机四态:`IDLE / ACTIVE / DRAINING / FINISHED`,由 `ServiceCoordinator` 维护;
  字段含 `input_open`、`accepted_request_count`、`completed_request_count`、
  `active_request_count`、pending alarm/fence 计数。
- IDLE:无请求且输入未关闭(无任何预置 request 来源);空 EventQueue 必须保持
  IDLE **不退出、不忙等**(等待外部注入通道)。
- IDLE→ACTIVE:外部 Producer 注入(request 或 close/EOF 命令)。
- →DRAINING:输入已关闭且仍有余留活动请求在排空。
- →FINISHED:输入已关闭 && active==0 && 无 pending alarm/fence。
- 空 EventQueue 的等待语义:EventQueue 空 ≠ FINISHED;在线主循环 EventQueue 空
  时调用 `svc.wait_for_work()` 阻塞(禁止忙等);收到注入或关闭信号后必须保证
  EventQueue 有 alarm 或 svc 进入 FINISHED。
- EOF vs 显式 close vs 异常退出:命令队列三态(submit/close/EOF/error);
  异常 → fail-closed abort(退出非 0)。
- **最终结束权唯一归 ServiceCoordinator**:在线模式必须禁用 `Workload::call`
  里 `sim_notify_finished()/is_finished=true` 的副作用(Workload.cc:625-634),
  否则阶段性空图把在线仿真永久结束、后续注入被丢弃。
- 线程合同:外部 Producer 只写线程安全有界 command queue;仿真线程统一取
  命令并排 alarm。

## 裁决
- 首版 request-neutral 默认:不提供 `--request-queue-csv` 时服务保持 IDLE,
  不读任何预置队列(正式入口 fail-closed 已由步骤 0-1 保证,
  `create_default_request_queue` 随机 stub 不进正式路径)。

## 验证方法
- IDLE fixture(步骤 1-2/1-10):无请求启动→IDLE;注入 2 request→ACTIVE;
  注入关闭→DRAINING→FINISHED 退出 0;全程记录五态迁移日志与时间。
- 阶段 2 B1:生命周期不变量成立。
