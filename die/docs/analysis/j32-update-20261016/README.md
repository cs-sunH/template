# J32 更新定位：融合遍历未采用

生产基线 `5caede8ca81902baf15c7d3228db70f2087302fb`。`make_instrumented.py` 仅在work生成当前Execution副本，加入按thread CPU计时的function scope和allocate固定段标记；没有逐活动计时，不改算法，不进入生产构建或正式SDK。`update_profile.cc` 复用真实compiled OS bulk选择和Runtime，每次只启用一个同刻capacity update的计数/时钟。时钟及global-new计数自身有开销，以下不是50μs预算证书。

100样本baseline中J32 update/allocate中位约42.123/40.475μs，collect（含分配值重置和活动收集）17.021μs、symmetry7.616μs、rows1.732μs，余部含ratio求解/发布/释放。advance中位0.180μs，integrate/settle调用均0；本fixture不是旧速率结算成本。不据此跳过其他时刻真实advance、事件或统计。

尝试把先重置全部rate/alpha、再收集活动的两次遍历合成一次，保留所有过去/未来/终态活动重置、Ref顺序和后续数值逻辑。真实实验版本公共 **49/49通过，24.73s**；11完整API对象与baseline相等；Norm32/96新driver实际重跑，128 raw/report/stderr/log同保留v2逐byte一致。实验正常30样本仍 **failed**：cold128³ P95 1.949150ms，本次OS J1/J8/J32 15.823/26.259/**95.682μs**，另multi-domain J8失败，全部原样保留。

实验插桩J32 collect中位17.206μs、allocate41.537μs；C++new次数/请求字节不减，普通diagnostic thread CPU也无下降证据。跨次环境可变，不能证明因果减速，但**没有支持采用的速度收益**。因此不采用；精确源码patch和全部测试/失败raw保留供审查，**不作为下一待办或待恢复方案**。

已只反向应用本实验自己的patch，未reset/清理外部改动。原生产重建后实际静态库与CABI均逐byte等于baseline已测ELF，CABI重新实核 `e134763079d36e30f1f9ad06785847421bde1f05c330e8896f7008b6cd43a3c6`。所有生产/头/测试源与baseline49身份一致，所以当前版本沿用compiler-ranking/final-public49及其预算（cold1.508469ms/J32 77.745μs failed），不能把实验binary62567b…的测试冒充当前版本。无新增硬件精度、五仓发布或业务/CPU结论，TX8/目标先验不动。

复现：`make_instrumented.py --output work/FRESH.cc`；使用trace-baseline-build身份中的真实C++命令替换fresh输出路径，再顺序跑`trace-baseline-run`命令，避免测量并发CPU任务。实验仅离线审查时在baseline worktree应用`fused-traversal-not-adopted.patch`并单独记身份；不向生产恢复。`summarize.py --output work/FRESH-summary.json`核还原后的当前源/两库、实验API/raw内容引用，完整证据在同名evidence。trace未使用的solver槽为0，solver余部由allocate减固定前缀段解释，不能当作独立求解测量。

下一方向回到公共模型真实依赖：R>1无Q非服务窗口/region barrier仍未解，不前置等待、不PRIVATE替身、不逐repeat；Norm长行两遍HBM可作为有限两kernel映射研究，必须先明确完整row统计→apply依赖、实际中间统计张量字节、每Core小tile驻留、参数缓存和有界曲线，不能恢复旧256元素假整行驻留。持续自主开发，工程NOT ACCEPTED。
