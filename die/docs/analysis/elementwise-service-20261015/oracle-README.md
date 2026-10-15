# 逐元素固定服务曲线：离线独立误差验证入口

本入口仅覆盖一个 Core 的 synthetic residual、GELU、SwiGLU，比较真实公共 Mapping/C++ Runtime 与独立逐块离散参考。它不是 TX8 条件参考、工程数据校准或目标硬件实测。没有拟合参数，不给多 Core/路由或整个项目发通过证书。

`die/tests/elementwise_oracle_driver.cc` 读取显式硬件配置，保留硬件源常数后，**显式覆盖 synthetic 服务率和 Vec 峰值**。输出完整输入/派生硬件、13 域容量、生产 Plan wire/活动、solo/sync/5ms offset 的真实 Runtime 事件、首 cohort 输出就绪、终态和原始 counters。这些覆盖不属于生产硬件校准。构建由主代理统一加入现有 CMake；本入口自身不改 CMake。

```bash
python3 die/validation/elementwise_service_oracle.py --self-test
python3 die/validation/elementwise_service_oracle.py --driver work/build-die/elementwise_oracle_driver --output work/elementwise-oracle
python3 die/validation/elementwise_service_oracle.py --raw work/elementwise-oracle/residual-e256-i4-balanced-bf16-fp32.raw.json --output work/elementwise-oracle/recheck.json
```

完整 suite 有 17 个小场景，每个运行 solo、同刻两 owner、5ms offset 两 owner。覆盖每 invocation 单 cohort/多个 cohort、257 full/tail、重复执行、INT4 独立奇数打包，以及均衡/访存/计算/LOCAL 主导的 synthetic 响应。逐 Core 反例不在此单 Core 验证范围内。

Python 仅从 shape、dtype、固定算术 eq-op 和显式硬件率生成每块 L/C/S 基本工作，不读取生产 credit 曲线，不复制生产 fluid 稳态。256 元素块、各 invocation 独立尾块、有限输入/输出槽逐块枚举；输入槽从 load 开始保留到 compute 完成，输出槽从 compute 开始保留到 store 完成。每 owner 最多一个 L/C/S。公平分配使用与既有 grouped oracle 相同的 owner Q/alpha 定义，独立非负 progressive filling 同时约束全部 13 共享域和各活动 PRIVATE。路由域在该 synthetic fixture 非绑定；HBM 的显式地址假设是**每 invocation 从 row0 重新开始整数交替字节条带，输入 streams 在每 block 拼接**。完整 256 元素块具有偶数字节，奇数尾块位于 invocation 末端；累计行量是每 invocation 的 ceil(bytes/2)/floor(bytes/2) 再分别乘执行次数，不是全部 repeat 共用一个条带游标，也不是已知硬件实测地址。

每个测量同时对比两种不同的内部执行语义：

- `invocation_barrier`：下一 invocation 的 load 必须等待上一 invocation 全部输出 materialize。用于显示 bulk 聚合可能提前跨 invocation 依赖的问题。
- `independent_invocations_pipeline`：invocation 之间没有 materialization barrier，只受逐块 L→C→S 依赖与有限槽位限制。它是另一模型假设；与其接近不证明真实逐层依赖正确。

首块就绪、完成时间、HBM busy/shared busy 和 compute busy 分开报告。busy 仍是正服务时段的并集；shared 按不同 owner 计数，不按利用率加权。保留绝对误差、相对误差、参考共享时长、10% 目标和 20% 覆盖。20% 内只标记 `within_20_percent_unaccepted_without_recorded_attempts`；没有多次合理调整记录，不能自动当作已接受。>20% 标记 failed，零参考的非零差异也失败，不删除失败场景。

独立 oracle 自检包含可手算的单位 L/C/S：单块 3 秒、四个串行 invocation 12 秒、允许流水的四块 6 秒、首输出 2 秒、两个同步 owner 各 6 秒，以及非等 Q 的 owner 公平 floor。条带新增自检：1B 奇数 payload ×3 invocation 为 row0=3、row1=0；256 元素 INT4 ×3 的偶数 full payload 仍为 [192,192]；257 元素 INT4 full+tail ×3 为 [195,192]。自检通过仅验证该离散参考的这些局部性质，不是生产模型准确性通过。

路由修正前的实际运行：单进程 `g++ -O3 -std=c++17` 将 driver 与当时的 `work/build-die-continue/libcoarse_die.a` 链接为 `work/elementwise_oracle_driver`。当时 17/17 driver 成功、17/17 独立总量账本通过；原始 Plan/HW/Runtime、stderr 和逐项误差都保留。LOCAL 三算子 solo/sync/offset 的完成和 busy 达到 10%。报告生成命令正常退出只代表成功保存证据，不能当准确性通过。**路由修正前的 oracle 源码、driver 源码/ELF 与完整旧 suite 已保存在 `work/elementwise-oracle-initial/before-route/`，以下原始误差数字仍保留；新版本重跑见下文。**

已经捕获的有效失败包括：

- residual，256 元素、4 次 invocation、balanced、solo：生产 90ms；严格 invocation barrier 参考 120ms，完成时间误差 **25% failed**。无 barrier 的离散参考 80ms，误差 12.5%，尚未按有记录合理尝试的规则接受。sync 同样存在 25% barrier 误差。
- residual，1 元素、INT4、3 次 invocation：生产 solo 0.15625ms；严格 barrier 参考 0.234375ms，完成时间误差 **33.33% failed**；无 barrier 参考完成时间相同。说明独立存储打包守恒不等于逐 invocation 完成依赖正确。
- residual，257 元素、单 invocation、balanced：首输出 20ms 与离散参考一致；完成误差约 0.13%，但 compute busy 为 20.0390625ms，离散参考 10.0390625ms，误差约 **99.61% failed**。memory fixture 的 compute busy 误差更大，未隐藏，也不改原指标。

初始 fixture 将非绑定资源设为 `1e30`，三个 LOCAL 场景在 compiler standalone estimate 报 `invalid_hardware: standalone plan has no positive service rate`，完整失败保存在 `work/elementwise-oracle-initial/`。最终改为非绑定 `1e12`、LOCAL fixture HBM `1e9`，避免极端数值条件，三个场景成功。该修改仅改变离线 synthetic fixture，未改生产模型和验收门槛；初始失败不能冒充通过。最终源码和 ELF 身份由主代理整合到正式 evidence。

路由工作量及 credit 端点防护修正后，已重新链接当前公共库，完整重跑到 `work/elementwise-oracle-final/`：17/17 driver 成功（每项 solo/sync/offset），17/17 总量与**各 HBM 行 payload**独立账本通过，oracle 手算及条带自检 9/9 通过。GELU 的 1 元素 INT4 ×3 invocation 输入/输出行量均为 [3,0]；residual/SwiGLU 双输入为 [3,3]、输出为 [3,0]。每个 raw/report/stderr、suite stdout/stderr 和源码/库/driver ELF 的 SHA256 都由该目录 `source-identity.json` 绑定，记录了 suite 期间源码和库均未变化；旧失败与 before-route 目录保留。

新版本仍复现上列 25%/33.33% invocation barrier 完成时间失败及 99.61% compute busy 失败，未作参数拟合。首输出的 solo 误差最大 0.25ns；并发 offset 下误差最大 2.500001ms（8.33334%，SwiGLU 256 元素 ×4、第二 owner），因此只说明这组首输出比较达到 10%，**不能称并发首输出精确到 ns**。并发差异包含 macro 与离散活动集合、Q 归一化单位和预取时机的区别，不能用 solo 的首块验证替代并发验证。
