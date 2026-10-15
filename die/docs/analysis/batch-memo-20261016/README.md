# 两份已验证raw请求的有界LRU memo

基线main/HEAD已实核41254718c03433c0786aad9f5de9b259f3738fdb，继续在
codex/die-explicit-continue-20261015，仅改die/。生产仅Capi.cc有变化，公共模型
静态库byte-identical，不改曲线、候选、硬件、校准、13域/Q/alpha/PRIVATE或ABI。

## 问题、实现与资源界

此前单memo只记住最后一个成功raw batch。canonical/alias交替会反复JSON解析、
规范化、dump和查规范Plan；没有重新编译Plan，但已足以越过20μs热查询预算。
新增独立交替基准，计时仍只含一次真实API调用，JSON解析/验证和release在外；
所有固定门槛不变。旧生产真实P95：GEMM1/128 alias交替27.300/29.193μs，
attention32.148μs均failed，OS J32本次也failed。原单alias历史失败仍保留。

固定array两份unique_ptr memo，命中交换到MRU；第三份只替换LRU，必要时再
移除MRU。每memo实际raw/string容量、指针vector、对象和分配allowance全计账；
**Plan+memo总项数≤4096、总字节≤64MiB**，不为memo淘汰Plan。每份仍最多
64KiB raw/256行，空/失败兄弟/未留存Plan不建memo。新候选全部分配和node验证
完成后，先核plan-only空间；候选单独放不下则原memo也不动。所有cache结构
插入/回滚删除/淘汰前清空两份指针，普通LRU splice保持节点和指针有效。

fresh handle、每行错误隔离、规范Plan碰撞核对和响应合同不变。两种不同raw
同时保留时cache_accounted_bytes合法增加；命中及等长第三份替换保持双份footprint，
不是用漏计内存换速度。只读复核未发现指针、事务或计账阻塞。

## 验证

- 新交替memo回归在旧生产真实failed，最终通过：两份精确计账、20次交替不增长、
  第三份LRU替换/命中更新recency、失败兄弟不建memo、新Plan插入清空两份，
  原handle仍可submit/cancel。
- 在原真实64MiB Plan LRU边界补测合法64KiB padded请求：大memo单独放不下时
  不丢小memo、不淘汰Plan，不超过原边界。原4096输入/真实byte cap/evicted行
  lifetime与全部旧用例保留。
- 首轮44/45，唯一失败是旧legacy测试假定单memo替换后字节相等；原日志保留。
  更新为第二份真实正增量、A/B命中及等长第三份替换的双份稳定计账，保留
  64MiB边界/fresh handle/摘要相等。最终一致Release **45/45，18.71s**。
- 改前生产源与Git412逐文件核对一致；真实改前/改后8个task完整hardware/
  compile/estimate对象逐字段全等，含R3 not_available、hash、handle、summary、
  cost和缓存字节（该批次共用同一raw）。库SHA改变、公共模型静态库SHA完全相同。
  不重复无影响的42case精度oracle；沿用412模型证据，未冒称新精度重跑。

## 性能

相同新增基准、无并行构建30样本，最终本次所测全部通过：

| 指标 | 旧生产P95 | 新CABI P95 |
|---|---:|---:|
| GEMM1 canonical/alias交替 | 27.300μs | 2.534μs |
| GEMM128³ canonical/alias交替 | 29.193μs | 2.904μs |
| attention canonical/alias交替 | 32.148μs | 2.193μs |
| 所有5组交替 | 见完整raw | 2.193—3.295μs |
| GEMM128³常规冷编译 | 见完整raw | 1.903321ms |
| hot cost | 见完整raw | 4.517—8.173μs |
| OS J1/J8/J32更新 | 见完整raw | 15.213/24.296/39.779μs |

单alias新P95 2.263—3.656μs。GEMM/OS模型及allocator未改，不能把其本次
时延变化归因于memo；旧J32尾部和全部历史失败保存，不称稳定全负载证书。
五仓CPU not_run，正式vendor仍旧canonical24，未publisher。

整体NOT ACCEPTED：412长16/48完成全10%仅其范围，仍24busy failed、一项
首输出11.379%未接受，R>1窗口not_available、原17还有27联合failed。TX8/目标
先验冻结，缺独立target raw/heldout；没有硬件校准声称。下一步核查五仓消费者
是否误用not_available的body数值，并继续公共模型与隔离集成。

## 复现与证据

同名evidence保存baseline/after benchmark全样本、public-before/after完整对象、
source/ELF/CPU/命令、旧回归失败/44项初轮/最终45及全部构建日志。summarize.py
校验Git生产源、静态库一致性和public输出，不以一次更好样本改验收。

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/batch-memo-benchmark.json
python3 die/docs/analysis/allocation-overhead-20261016/capture_public_results.py --library work/build-die/lib/libcoarse_die_cabi.so --output work/public.json
```

此前单memo生产的基准使用同一新增benchmark源码，仅两份memo尚未实现。
最终稳定footprint比较改用所有请求预热后的canonical状态；旧single和新two均
可比，未跳过首次alias切换开销，也未从样本删除慢调用。交替增加一项严格20μs
指标，不替代原所有指标。当前持续自主开发。
