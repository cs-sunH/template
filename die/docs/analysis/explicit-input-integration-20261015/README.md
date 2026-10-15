# 显式硬件输入的五仓隔离集成

设计先行 `35f4580`、实现 `89c2499c6e61de450291961d800594a8a83eeca7` 已分别合入main。本次从后者的git archive冻结29个发布文件，在work构造五个独立vendor预览；没有修改正式vendor或恢复旧JOINT。预览source digest为 `74426b866937a077600c76db2d8e94b19329a68e354475d372756f89c58242d6`。

五仓Release双ABI目标全部构建成功。实际配置检查83/83、0skip：WSCLLM 20、WSCLLM-LRU 15、FACE 18、FACE-LRU 17、活动JOINT 13。五个coarse ABI的SHA均为 `e2cfaf612642e4ebf2e07a235a68004eea9ad056a1eb76061105be31615510c3`；legacy ABI沿用各仓自己的源，JOINT与另四仓不同，分别记录，不能要求不同legacy源产生同一ELF。

真实FACE cost-only调用发现两仓相同缺陷：归一化的结果append后继续执行full-report的通用append，每个spec产生两行，调度器读取交错的原始行时缺total_seconds。各补一个continue，保留旧库fallback。实际新ABI探针重跑均通过；新增回归检查7个不同spec返回7项、顺序/秒转换/字段/完整估值一致，并强制检查fallback。FACE 133/133（原132+1）、FACE-LRU 152/152（原151+1），0skip。第一次探针失败及缺Python protobuf的环境失败保留；最终使用真实protoc3.21.12和Python runtime4.21.12，不提交安装包或构建产物。

证据在对应evidence目录，verification.json给出逐文件SHA；integration-summary.json记录每仓命令、退出码、源码和ELF身份。stage_five.py/build_variant.py是当次真实复现脚本，根路径和冻结SHA显式写出；换目录复现须调整路径，不能沿用原ELF身份。正式publisher尚未运行；这些结果限SDK/config/consumer，未覆盖五仓完整Online及固定图CPU预算。

同一检查点另保存仲裁分配的优化：预留小数组容器、避免无效Packing副本和Ratio行复制，最多3折点使用固定插入排序，已有可行floor与匹配上界时省略剩余求根但仍验证全行；Ratio还保留各变量的独立上界证据行，以向外舍入的可行包围盒省略确定不可能成为瓶颈的非负行。最后仍严格认证全部原binary64物理/PRIVATE/signed行，不改Q/alpha或容差。新增精确边界/证据行保留与多冗余行回归，独立有理数测试及全42项CTest通过（22.34s）。allocation-source-identity.json绑定此版本，早于后续向量实现。

中间allocation-reuse速度仍failed，最终含冗余行省略的版本尚未重新计时。profile-compile-before-row-prune是并行构建期间的CPU/墙钟定位工具，使用行省略前的静态链接库，不是速度证书。busy条件界见上一分析目录的busy-contract-scope.md；误差目标/20%规则、busy定义和全部速度门槛不变。TX8条件参考与目标先验未改，无独立目标raw，不称目标重校准。
