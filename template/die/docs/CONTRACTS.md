# coarse_die 数值合同与具名豁免记录（CONTRACTS）

本文件是数值精度合同与 (c) 类豁免的规范落点（执行基线 v3 §2-P1.3）。

## C-1 现行数值精度合同

- **引擎合同**：公开输出为 binary64；内部分配/求解使用 long double 并按
  **每运算 1-ULP 认证**折回 double（RatioAllocation 认证族；多资源耦合求解的
  链式运算按认证规则累积）。
- **严格矩阵 oracle 合同**：精确有理（Python Fraction）零容差断言
  （coupled/asymmetric_credit_oracle.py）。
- 两合同存在结构性不对齐：oracle 断言的"有理精确 rate"超出引擎合同的表示
  能力。该不对齐的处置程序见 C-2；**不是容差放宽**（可行性/守恒/台账检查
  保持零容差）。

## C-2 P1 六例具名 (c) 类豁免（条件触发式签署；实体已生效）

- **框架生效**：2026-10-17 共识决策①（用户转发确认）。
- **实体生效条件**：反事实重放证明全部差异条目为**单源线性传播**
  （机器判据 = 与 double-rate 重放逐位相等；busy 为有界层）。
- **证明结果（2026-10-17）**：`legacy-closeout-20261017/P1-REPLAY/
  replay_counterfactual.py` 对全部 12 个 same_owner_split 案例
  （含 6 个失败例与 6 个通过例对照）判定 **12/12 single_source_confirmed**。
  规则 R1-R7 逐条成立；原始计数与身份见
  `docs/evidence/legacy-closeout-20261017/strict-matrix/`。
- **豁免对象（具名）**：same_owner_split × owners{2,8} × unit_exponent{0,−10,−60}
  共 6 例。差异条目 = 多资源 LP 认证舍入产生的 rate 偏差（实测 ULP 偏差由
  replay 结果的 rate_ulp_deltas 自证：A 侧 Load/Store +8 ULP、竞争者 −2 ULP
  （J=8）；J=2 相应缩小；**不写入断言**）及其下游线性传播
  （progress/vec/hbm 台账/快照 credit/总量；busy 为有界传播层）。
- **判分语义**：严格矩阵保持双层判分——(1) 可行性/守恒/台账零容差（现行
  已全过，豁免不触碰）；(2) 对"有理精确 rate"及其精确传播值的断言标为
  **超出引擎 binary64+1-ULP 认证合同的参考上界**：6 例在该层失败仍可见，
  归入 certified-approximation 层，不算引擎缺陷。
- **作废条款（自动）**：此后任何重放运行出现非单源条目，或 6 例之外的
  新失败出现 → 本豁免自动作废，P1 自动转引擎修复立项（无需另行拍板）。
- **复现入口**：`python3 die/tools/run_strict_matrix.py --output-dir <dir>`
  （预期 verdict = "78-passed + 6 named (c)-exemption (single-source proven)"）。
- **引擎精确硬化**：本轮不做（热路径精确有理化与 P3 性能门冲突；列为其
  后可选增强，须独立立项与全门槛）。

## C-3 busy 计数器 canonical 语义（P2，2026-10-17 生效）

busy 系字段（compute/hbm/row × ns 与 _exact）= **事件窗口的精确 dyadic 和**：
累加在 128 位尾数上精确进行（对齐超限时按 nearest-even 舍入至第 128 位），
double 投影 = 对精确值的单次正确舍入，_exact = 投影的 18 位 decimal_measure，
快照携带规范 dyadic 字段（{hbm,compute,row}_busy_ns_dyadic；导入优先之，旧十进制
文本仍兼容）。该语义使 busy 输出对事件划分、求和顺序与构建历史不变；跨构建严格
比较不再出现 busy 尾差（2026-10-17 前的 13 处历史微差按"舍入依赖→精确"具名收档，
见 analysis/legacy-closeout-20261017/P2-BUSY/）。hbm_shared_ns/row_shared_ns 暂保留
long double 累加（无历史差异；若未来出现同类微差按本条同法处理）。
