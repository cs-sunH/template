# K readiness 修正与紧凑状态基础（2026-10-15）

本轮从 main `d1151179` 恢复持续开发。旧 `k-readiness-20261015/raw/` 全部原样保留。新源码、原始输出与身份见 `../../evidence/k-readiness-corrected-20261015/`。

## 修正和真实验证

Python 离散参考只在 final-K Compute 预约输出槽；非 final 更新独立累加缓冲，不因两个 pending output 阻塞。Store terminal 才释放输出槽。参考语义升为 `final_k_output_reservation_v2`，grouped oracle identity 升 v3。诊断报告显式说明 compiled Plan 只提供 mapping/工作账本，实际运行的是手工封装的 per-K facts，不再给出 E5/minimum-E 标签。

独立 Python 和 C++ Runtime 小夹具均从空状态达到反例：三输出、每输出两 K，Load=HBM1 1B@1B/s、Compute=1s、Store=HBM0 100B@25B/s。q2/k0 在 5s、output occupancy=2 时运行；q2/k1 于 6s 等待，q0 Store 在 7s 终结释放槽后开始；三个 Store 在 7/11/15s 完成。初始把三个活动放同一 COMPUTE_GATE 的夹具未达到输出满，失败计数保留，不能用它作见证。

`k_readiness_runtime_driver.cc` 复用现有 Geometry/ResourceFacts，显式两输入/两输出槽驱动公共 Runtime；同时逐次校验新的六计数器 Cursor 的全部准入、终态、占用和部分状态恢复。M80/N128/K4096、M80/N128/K4097、M65/N65/K4097 都选 OS16×64×64；solo/sync/50ms-offset 九项 grouped 比较全部 passed，保持原 3%/5% 限制。最大完成误差约 2.367e-6，最大 shared-HBM 相对误差约 5.060e-4。三 shape 的真实 traces 都没有 output-full 非 final admission，反例见独立小夹具，不外推为这些 traces 的覆盖。

旧 K4096 raw 用修正参考重评时，原三场景 `comparisons` 原有数值与字段全等，仅新增 readiness revision 元数据。新比较的工作量严格整数检查和累计浮点序列化差值分别保留在完整报告，不能由延迟通过推断整数守恒。

定向 CTest 3/3 passed（Cursor、13项 Python、实际 Runtime 反例）。本次未重跑未修改且先前通过的 public30、五仓83及业务压力；五份新固定源均成功独立 CMake configure（tests OFF），不称为新的 Online 工程验收。公共 Runtime/Mapping/Plan/Compiler 与 TX8/目标先验未改；发布新增内部头文件和 CMake 测试目标，canonical24 digest `dd62e6787557ff2f764fc5be8f38865a2c15916c20f1a9fc95a28ebcd4fb7005`。仅活动 JOINT 参与同步，old 未改。

## 六计数器表示

`die/src/GemmOSReadiness.hh` 的 Descriptor 以现有 Geometry 的最多32个类构建真实单 Core facts，直接调用原 resource builder，不重写计数/带宽公式。它拒绝多 Core，并 checked-u64 收窄内部宽整数 wave count。Descriptor 不进入当前生产 Plan schema。

严格 row-major/K 顺序、每角色至多一活动，令 `(Ls,Lc,Cs,Cc,Ss,Sc)` 为各阶段 started/completed 数：

```text
Load ready FIFO       = [Cs, Lc)
Store ready FIFO      = [Ss, floor(Cc/Tk))
input occupancy       = Ls - Cc
output occupancy      = floor(Cs/Tk) - Sc
active stage counts   = Ls-Lc, Cs-Cc, Ss-Sc  (each 0 or 1)
```

因此无需 token 数组保存两个 FIFO。Compute 只取下一个已终态 Load；只有 final admission 增加 output occupancy。State restore 先验证单调依赖、角色差值和两槽上限，失败不改原状态。单 Core、原子拒绝、u64 overflow、K=1/full/tail 与巨大重复的固定类数已有测试。State 本身不绑定硬件/Plan 指纹，外层完整 wire 仍需做这些检查；Descriptor 必须比 Cursor 活得更久。

## 下一步及复杂度边界

当前 driver 是明确的 O(events) 诊断；有界的是描述与状态大小，不是运行工作。没有生产生命周期、signed-credit 曲线、周期压缩、取消/完整快照、multi-Core、WR、E8/E16 或 host 预算证书。整体工程仍 NOT ACCEPTED。

优先审查一个简单窄域：同一输出、同类非 final middle K、Store/pending output 为空、单 owner 且仅 Load(i+1)/Compute(i) 同时零进度启动。输入槽使较快活动结束后无法再启动第三活动；慢者结束后两阶段可同时从零重启，计数器平移一 K。全部13域、Q/alpha 和 PRIVATE 仍取公共 Runtime。只有证明整个跳跃 horizon 内 deadline 的整数 ns 舍入、LP 区域及 busy/served 累加安全后，才能按闭式跳过。一次重复状态见证不足以证明任意绝对时间的 long-double 舍入一致。第二 owner/端点、非零 credit、输出/尾类或外部事件边界均使此专项失效；这只是下一实现方向，没有新压缩通过结果。多 owner/输出行周期不由该窄域外推。

## 复现

在仓库根目录，使用当前环境实际 CMake：

```sh
python3 die/docs/analysis/k-readiness-20261015/run_corrected.py --cmake work/tool-env/bin/cmake --output-dir work/k-readiness-final
work/tool-env/bin/cmake --build work/build-k-readiness --target coarse_gemm_os_readiness_test -j 2
work/tool-env/bin/ctest --test-dir work/build-k-readiness -R 'coarse_(gemm_os_readiness_test|grouped_gemm_oracle_test|k_readiness_runtime_self_test)$' --output-on-failure
```

默认三组 shape；可指定 `--m/--n/--k`，raw/stdout、stderr、grouped 报告和 manifest 分开保存到 work/。原未修 probe 的 replay.py 仍是旧结果重放入口，不替代新诊断。
