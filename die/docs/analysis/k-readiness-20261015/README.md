# K 就绪诊断：停止时保存的部分成果（2026-10-15）

用户已要求立即交接。此目录保存当前最小诊断及只读设计调查，不包含新的生产生命周期实现；公共源码仍为 e73a378。生产 E5 原 3%/5% 仍 failed，工程 NOT ACCEPTED。

## 已执行与边界

`evidence/k-readiness-20261015/raw/probe.cc` 是实际执行源码，原始绝对 include 保留以绑定字节。它复用既有 E5 硬件/选中 OS mapping、GemmOSResourceFacts 的真实 13 域工作量、Q/private 与公共 Runtime；以同一 owner 的三个独立单阶段 Plan、完整 Read/Compute 终态和两输入/两输出槽做 O(events) 诊断。没有修改公共 Runtime、物理速率、TX8 参数或门槛。

实际只运行 M80/N128/K4096，OS16×64×64/单Core/strict FP32 Vec。solo、同步双 owner、50ms offset 的 grouped 对照均 passed；solo finish相对误差1.118012e-6，offset shared-HBM误差0.000324556。solo 1281次advance/1290阶段完成/3种fact类。详情、原始计数尾差和旧 equal-progress 对照（可能 failed，保留）在 evidence；不能把摘要中的 grouped passed 套到旧对照或生产 E5。

**新发现的共同缺陷，尚未修改/量化影响：** probe 和 `compiled_gemm_oracle.py:455-472` 对所有 Compute 检查 output occupancy<2；设计§6.2/§6.3/§8.5及 `die/tests/os_lifecycle_runtime_driver.cc:588-603` 仅 final-K Compute 预留输出槽。非final Compute 不应因两份待输出阻塞。现有比较可能共同保守，不能视为该依赖规则的独立验证。先修正并增加输出槽全满时非final仍可执行的独立反例，再复测原场景；不要仅修改一个实现以制造 pass。

**原始报告元数据限制：** 原 grouped checker 继承了 E5 driver 的“没有手工构造 Plan”、E5/minimum-E 标签，这些断言不适用于此诊断。实际手工封装的是已有 C++ facts 的单阶段 Plan；原 compiled E5 只提供 mapping/账本来源，未被该 probe 执行。raw 原样保存用于审计，`summary.json` 明确范围；不得将其作为生产验收证书。

K4097、M/N tail 未运行；没有多Core、WR、生产 wire/snapshot/cancel、有限周期压缩或效率通过证据。原 `k-readiness-proposal-original.md` 为设计草案，非已采纳生产合同；其中 signed-credit 生命周期方案仍需推导和评估，不能直接照搬。

## 从新 clone 复现已测诊断

只在 work/ 运行；不要向 evidence 目录生成文件。可先激活该环境实际安装的工具（本次 `/workspace/cloud-setup/activate.sh`），然后：

```sh
cmake -S die -B work/k-readiness-replay/build -DCMAKE_BUILD_TYPE=Release
cmake --build work/k-readiness-replay/build --target coarse_die -j 2
python die/docs/analysis/k-readiness-20261015/replay.py --repo-root . --build-dir work/k-readiness-replay/build --output-dir work/k-readiness-replay/run
```

replay.py 是交接时补的可移植入口：复制原 probe 到隔离目录并只替换两条绝对 include，使用指定静态库；不修改生产源或原始证据。原 measured 命令是直接 g++链接 `work/build-attention/libcoarse_die.a` 后运行 `probe --m 80 --n 128 --k 4096`，再调用 grouped_gemm_oracle.py --raw/--output。新入口仅语法/帮助检查，停止前未重新编译重放，不宣称已重跑。输出仍含上述共同门控缺陷和旧 checker 的标签限制。

## 下一步

先修参考/诊断的 final-K输出门并测试反例，再测 K tail/MN tail；复用已有 bounded geometry、per-K counter、13域 facts 和本目录 inventory。生产需固定数量 facts、紧凑 row-major cursor、有限证明周期/闭式跳过，在 arrivals/capacity/class/LP 区域边界分段；不得把 O(R×Tk) 诊断直接部署或引入重型 Fraction、扩大256pivot/1M事件预算。完成守恒/ready/busy/并发/导入取消及成本验证后才可称生产最小粒度通过。其他模块和优先级见 HANDOFF。
