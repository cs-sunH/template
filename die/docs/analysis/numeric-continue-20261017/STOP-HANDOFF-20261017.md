# 2026-10-17 停止交接（Asia/Shanghai）

用户要求立即停止；不启动新开发、实验或非必要测试。生产源码保持已验证的
`c734ce1145915279bc26367cb8e1e5076629adfe`，Execution.cc SHA256
`54b8f6f37ec7fc7f45549e51b4c86b8e70888b0cc923f3286cd269721ea38aab`。
恢复分支为 `codex/die-numeric-continue-20261017`；最终文档提交不会改生产。

## 恢复与现有证据

先 fetch 全部 heads，核实恢复分支最终远端 HEAD，不仅看 main。保留 main
上的其他有效修改。当前分支已普通 merge main 共享指针，禁止 reset/force。
阅读本分支 HANDOFF 顶部、die/AGENTS.md、die/docs/PROJECT.md、根唯一Core
权威 DOCX 与粗粒度参数响应设计。只有一种 Core；不恢复旧Core类型或逐K路线。

- `evidence/numeric-continue-20261017/runtime-checkpoint/`：355项无ELF文本源码、
  全部失败/日志/预算，manifest 与 tar SHA/逐项解包实核。fresh final Release
  58/58；五新回归对 fresh A+B 实际 red。相同源码两配置各一次正常30样本
  速度预算当前全过，非稳定全业务或硬件校准。严格矩阵78/84，六原失败保留。
  严格API比较false：13个estimate busy微差，最大4.538e-15ns/一个double投影1ULP；
  其他字段/8条实际轨迹/28ABI符号相同。跨库snapshot通过。
- `analysis/numeric-continue-20261017/sdk-current/README.md` 及对应 evidence：
  当前新Runtime从 c734ce1 冻结，33文件digest
  `0fb24b47de6b9a3622508d59b60d86542ee2413a20d4a482add0f3cdd56ef7dc`，
  4493冻结文件。10 ABI/98配置通过；Python384pass/1原optional skip，
  strict33阶段32pass/1failed。不要套旧A+B的 native 结果；当前五仓fresh native均构建成功，C++实际22pass/4原R4guard fail/0skip，JOINT10/10 C++及6/6 Python通过。16阶段/358项原始text单独归档并实核。
- `analysis/numeric-continue-20261017/NORM-CURRENT.md` 与
  `evidence/numeric-continue-20261017/norm-current-runtime/`：实际当前Runtime
  完整7epoch Norm在5.130029297s无next停滞；11观察中6非法库存，terminal
  totals未达到。11快照与旧A+B逐byte相同。不是长行可用性通过。
- `norm-affine/`：1952固定输入的独立Fraction对照，首anchor仍4错误headroom符号，拒绝。
- `norm-affine-v2/`：独立FMA残量求值修正样本符号，但815区间仍未认证，
  实测微成本约764ns/dot，不能全局安装。22文件hash/6 gzip roundtrip实核。
- `norm-dyadic128/`：停止前的独立固定128位dyadic研究，以其README/manifest
  记录实际完成程度；21文件hash/4gzip roundtrip实核。1952项中1836支持、116非零subnormal progress明确拒绝；支持域exact sign/integer identity/directed interval断言通过，仍未经后续独立review/接入Runtime。不得将算术样本通过称Norm、snapshot、
  API或性能通过。

## 下一账号优先顺序

1. fetch/核实分支及上述源码SHA，按归档manifest恢复work，不复用旧ELF。
   当前源码如需继续验证使用fresh构建；当前58项与SDK冻结身份不因文档变化失效。
2. 审查 `norm-dyadic128/Dyadic128.hh` 的受限域、精确符号和有向区间证明后，
   才考虑隔离的最窄affine physical-view实验：仅两点直线credit、最多3角色，
   q/headroom分别求值；同一视图必须贯通boundary/LP/event/terminal/import。
   每次积分前核对所有角色的实际可表示progress及精确库存，不能clip负值。
   先dyadic容量1、完整7epoch、原service/split负4ULP/新守卫，失败保留。
   旧candidate2存在篡改snapshot与service_curve失败，不整体套用。
3. 保留当前五仓native真实R4 guard失败与optional skip，调查完整业务可用性；
   正式SDK发布/CPU≤10%/硬件校准均未验收。长行/R>1/busy精度继续按原合同，
   精度目标10%、有据条件20%，不改参考/容差/速度门槛。

## 状态边界

main只保存共享入口，未激活A+B/本次数值Runtime。正式五份SDK仍canonical24，
digest `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`，
publisher未运行。无最终验收，工程NOT ACCEPTED。研究无Runtime采用；不存在
必须用户决定的阻碍。本账号交接后停止。
