# 合同② 服务生命周期合同(IDLE / ACTIVE / DRAINING / FINISHED)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- 状态机四态:`IDLE / ACTIVE / DRAINING / FINISHED`,由 `ServiceCoordinator`
  维护;字段含 `input_open`、`accepted_request_count`、
  `completed_request_count`、`active_request_count`、pending alarm/fence 计数。
- 状态进入/退出条件(阶段 1 步骤 1-2 实现,本表冻结口径):
  - IDLE:无请求且输入未关闭(无任何预置 request 来源);空 EventQueue 必须
    保持 IDLE **不退出、不忙等**(等待外部注入通道)。
  - IDLE→ACTIVE:外部 Producer 注入(request 或 close/EOF 命令)。
  - ACTIVE:存在被接受/活动请求。
  - →DRAINING:输入已关闭(close/EOF)且仍有余留活动请求在排空。
  - →FINISHED:输入已关闭 && active==0 && 无 pending alarm/fence。
- **空 EventQueue 的等待语义**:EventQueue 空(静态 `finished()` 语义)≠
  FINISHED;在线主循环 `while(!svc.finished())` 中 EventQueue 空时调用
  `svc.wait_for_work()` 阻塞在桥接 FIFO/注入通道(禁止忙等);收到注入或
  关闭信号后必须保证 EventQueue 有 alarm 或 svc 进入 FINISHED,否则死循环。
- EOF vs 显式 close vs 异常退出:Producer 命令队列三态区分(submit/close/
  EOF/error);`mark_input_closed()` → DRAINING;异常/错误 → fail-closed abort
  (进程退出非 0,不做静默降级)。
- **最终结束权唯一归 ServiceCoordinator**:在线模式必须禁用
  `Workload::call` 里 `sim_notify_finished()/is_finished=true` 的副作用
  (Workload.cc:625-634【亲核】),否则阶段性空图会把在线仿真永久结束、
  后续注入被丢弃(见步骤 1-4 的执行模式构造工厂)。
- 线程合同:外部 Producer 只写线程安全有界 command queue(submit/close/EOF/
  error),仿真线程统一取命令并排 alarm;禁止外部线程直接调 EventQueue/
  schedule_arrival。

## 裁决

- 首版 request-neutral 默认:不提供 `--request-queue-csv` 时服务保持 IDLE,
  不读任何预置队列(正式 runner 不触发 `create_default_request_queue`
  stub——步骤 0-1 fail-closed 已保证)。
- 静态基线配置(trace_config.csv:12 永久指向 20 档 30s 物化)与服务默认
  配置分离。

## 验证方法

- IDLE fixture(阶段 1 步骤 1-2/1-10):无请求启动 → 停在 IDLE 不退出不忙等;
  注入 2 request → ACTIVE;注入关闭 → DRAINING → FINISHED 退出 0;全程记录
  五态迁移日志与时间。
- 阶段 2 B1:生命周期不变量(IDLE/ACTIVE 转换、输入关闭/排空、不提前决策)
  成立。
