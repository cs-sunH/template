# 未执行的审查草案

exact_lp_oracle.py 与 member_change_runtime.cc 是此 Session 子代理准备的额外验证工具，**没有执行/编译**，不能作为 pass 证据。已通过的公共 owner symmetry/rollback 测试在 die/tests/，最终 25/25 结果另存 evidence。

静态审查支持：完整同构 owner 块的 LP 可交换，平均投影保持全局 work/Q 行和局部 PRIVATE/credit；同一 fair owner 跨 execution 时排除压缩。提交失败回滚以旧世界已经 advance/settle 为前提，保留真实已提供的服务并恢复新 allocation 前状态。注意极端 finite work/Q×multiplicity 可能 overflow 并 fail closed；尚无 uncompressed 可解的相同极端反例，不能宣称确认回归。

member_change_runtime.cc 旧延迟 fixture 的终止预期写成 **7s**，按当前三份各3字节、owner2晚1s加入的服务积分应为 **6s**，shared HBM=**4s**。接手者先独立核算并修正草案，再配置公共 tests/test_support.hh include 和库链接后运行；本 Session 没有修草案或取得该证书。部分脚本仍引用原工作目录，应先改为当前仓库/work 路径。不要直接把这个草案移入正式测试并解释为已验证。
