# Barrett Isaac Gym / Isaac Sim 配置对齐与穿透伪成功分析

生成日期：2026-08-10

## 1. 结论

当前 MultiDex 45k Barrett OOD10 主实验的 Isaac Gym 结果为
477/640（74.53%），Isaac Sim 结果为 552/640（86.25% raw；
552/616，89.61% valid-only）。两者不能视为完全同协议结果，11.72 个百分点
的 raw 差值同时包含手资产、物体碰撞几何、闭合目标、驱动参数、初始化方式、
有效性判据和 PhysX 世代差异，不能归因成纯模拟器差异。

Isaac Sim 中观察到的“手指穿透物体后卡住”可能形成物理伪成功：当前 Sim
有效性检查不排除穿透，只要状态有限且物体最终净位移不超过 2 cm，数值卡死的
物体也会被记为成功。主 552/640 实验没有启用接触 separation 记录，因此目前
不能定量给出 552 个成功中穿透伪成功的数量。

## 2. 主实验中显式可见的差异

| 项目 | Isaac Gym | Isaac Sim | 影响 |
|---|---|---|---|
| PhysX 世代 | Isaac Gym Preview 4 旧版 GPU PhysX | Isaac Sim 6.0.1 新版 PhysX | 接触生成、摩擦求解与 joint drive 语义不能保证一致 |
| contact/rest offset | 显式 0.01/0.0 m | 未显式设置，使用 Sim/USD 默认值 | 影响提前接触、允许分离距离和去穿透 |
| 手 URDF | D(R,O) Barrett 资产 | GenDex `model.urdf` | 关节、碰撞体、质量惯量来源不同 |
| 手碰撞 cooking | Gym URDF importer | Sim URDF importer + Convex Decomposition | 视觉相近不代表实际碰撞壳一致 |
| URDF 预处理 | `collapse_fixed_joints=True` | 删除 mimic、补缺失 inertial、添加虚拟根、转 USD | 可能改变拓扑、质量和惯量 |
| 虚拟根 | base 固定，但 URDF 内六个虚拟 DOF 采用 1000/200 | 六个合成虚拟根采用 1e6/1e5 | Gym 手腕更顺应，Sim 近似刚性锁定 |
| 闭合 | D(R,O) 逐姿态决定方向，25% outer/15% inner | 固定 close-dir，0% outer/5% inner | 实际起始姿态和闭合行程不同 |
| 手指 drive | Kp/Kd=1000/200 | Kp/Kd=1000/50 | Sim 阻尼更低，更易过冲 |
| object collision | 预先生成的 D(R,O) COACD URDF | 原始 mesh 运行时 `convexDecomposition` | 最可疑的穿透和互锁来源之一 |
| object COM/inertia | `override_com/inertia=True` | 按 Sim 碰撞分解与 density 计算 | 碰撞壳不同会带来惯量差异 |
| armature | 继承 Gym 资产/默认设置，主结果未完整记录 | 显式 0.001 | 关节瞬态响应不同 |
| 自碰撞 | 主结果没有完整记录 | `allow_self_collision=False` | 不能确认过滤完全一致 |
| effort/velocity limit | 继承 D(R,O) URDF/Gym importer | 继承 GenDex URDF，合成根另写限制 | 高增益下饱和行为可能不同 |
| restitution/材料组合 | friction=3，其余多为 Gym 默认值 | restitution=0，combine mode 默认 | 摩擦值相同不代表材料完全相同 |
| CCD/最大去穿透速度 | 未显式核对 | 未显式核对 | 离散闭合时可能发生隧穿或深穿透 |
| 初始化 | 每物体 64 个独立 env 并行闭合 | 手空载 settle 3 步，物体从远处放回，逐候选运行 | 初始重叠、接触缓存和重置顺序不同 |
| 世界复用 | 每物体 64 env | 一个 world 顺序复用手和多个物体 | solver/contact cache 条件不同 |
| 穿透有效性 | 无专门穿透判据 | 仅检查 finite、坐标及位移小于 10 m | 穿透卡住仍被视为 valid |
| 成功判据 | final，并计算 strict；本次二者均 477 | 主汇总使用 final | 552 未经过穿透过滤 |

共同的显式设置包括：100 Hz、2 substeps、零重力、TGS 类求解配置、position
iterations=8、velocity iterations=0、手/物体摩擦=3、物体密度=500 kg/m³、
每方向 1 秒、外力为质量乘 0.5 m/s²、方向顺序 +X,+Y,+Z,-X,-Y,-Z，以及
2 cm final 位移阈值。共同项不足以抵消上表差异。

## 3. 穿透后卡住的可能机制

1. **碰撞代理不一致。** Sim 对原始 mesh 运行时凸分解，凸包可能覆盖视觉凹槽、
   在相邻 hull 间形成重叠/内部接缝，或比视觉表面更厚。手指可能与这些代理形成
   非真实互锁。
2. **位置目标阶跃。** 手从 outer 直接以完整 inner 目标闭合 100 步，不是连续插值。
   outer 已重叠或单步跨过碰撞面时，位置 drive 和去穿透约束会互相竞争。
3. **Sim 手腕过硬且手指阻尼更低。** 1e6/1e5 的根 drive 几乎不允许整手退让，
   Kd=50 又比 Gym 的 200 更容易产生闭合过冲。
4. **高摩擦维持错误接触。** 摩擦 3 不一定制造穿透，但会使已经嵌入的碰撞代理
   难以沿表面退出。
5. **有效性和成功判据未排除数值互锁。** 穿透卡住后，六方向外力无法使物体移动
   2 cm，反而满足当前 final 成功条件。

候选文件中的 mean penetration 是 FK 几何近似量，不等同于 PhysX 接触 separation，
不能用其中位数 0.11 mm 证明物理阶段没有严重穿透。

## 4. 能否完全对齐

协议层可以显著对齐：使用同一 URDF、同一预分解 convex collision、同一逐样本
outer/inner、相同虚拟根结构、Kp/Kd、armature、effort/velocity、contact/rest
offset、质量惯量、初始化顺序、批处理方式和成功/穿透判据。

数值层不能保证完全相同。Isaac Gym Preview 4 与 Isaac Sim 6.0.1 使用不同 PhysX
世代、资产 importer 和 GPU 执行栈；即使所有可见参数一致，也不应期待逐帧或逐样本
完全相同。因此后续报告应使用“显式协议对齐”，而不是“模拟器完全等价”。

## 5. 新增缺口：旧 Barrett 执行栈

现有历史目录已经复现过 YCB10 的旧候选 + 旧协议 Isaac Sim：3200 条中 final
成功 2557 条（79.91%）。但它使用历史保存候选，不是 MultiDex 45k 模型针对当前
OOD10 重新生成的候选，而且没有同一批候选的对齐 Isaac Gym 全量验证。

因此下一项独立实验定义为：

- checkpoint：MultiDex success-only 45k `best_val.pt`；
- 物体：当前 OOD10 十物体；
- 候选预算：每物体 64 contact sets、每组 32 FK particles、top1、FK 400 steps；
- 旧手资产：dex-urdf Barrett `bhand_model.urdf`；
- 旧闭合与旧原生手/GenDex协议（与 ShadowHand 475/640 Gym、594/640 Sim
  同口径）：60 Hz、2 substeps、200 闭合步、每方向 50 步、摩擦 10/10、
  density 10000、线性/角阻尼 10/100、手指 Kp/Kd 400/400、solver 4/0、
  GenDex 方向顺序，以及逐方向 2 cm 判据；
- 为满足本次对齐要求，Gym 不沿用旧 Shadow Gym 中未对齐的根参数，而是显式采用
  Sim 的虚拟根 Kp/Kd=1e6/1e5、armature=0.001、相同 prepared 原始 object mesh、
  相同 outer/inner 和零地面；
- Gym 与 Sim 必须消费同一批 640 条候选，并分别报告 raw、valid、strict、穿透诊断；
- Gym 在接口支持范围内对齐 Sim 显式参数，同时保留无法对齐的 PhysX/Importer
  差异说明。

该实验不得与 D(R,O) Barrett 477/640、552/640 混写为纯参数消融，因为候选 URDF
和碰撞资产也发生了变化。

## 6. 代码依据

- Gym 主验证器：`scripts/validate_contactdiff_ood10_dro_isaacgym.py`
- D(R,O) Gym 基类：`remote_assets/dro_grasp/validation/isaac_validator.py`
- D(R,O) 闭合控制：`remote_assets/dro_grasp/utils/controller.py`
- Sim prepared 转换：`scripts/prepare_gendex_ood10_matched_isaacsim.py`
- Sim 主验证器：`../CEDex-Grasp/scripts/validate_isaacsim_six_direction.py`
- Barrett 主实验入口：`scripts/run_multidex_ood10_barrett64x32_top1_steps400_4gpu.sh`
- 历史旧协议复现：`outputs/barrett_previous_high_repro_isaacsim601/summary/REPORT.md`
