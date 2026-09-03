# 基础实验配置冻结后的模型、数据、参数与结果台账

更新日期：2026-09-03（Asia/Shanghai）

## 1. 结论先行

从首个可复现基础闭环冻结到当前版本，基础协议经历了六代：

1. **v1（2026-08-13）**：MultiDex success-only 45k、64×32×400、旧
   feasible/GraspQP Top-1；Barrett O0/I5、ShadowHand O10/I20。
2. **v2（2026-08-14）**：模型和候选不变，两手统一使用 O10/I20。
3. **v3（2026-08-18）**：模型改为 MultiDex-filtered
   Barrett/ShadowHand balanced N=3/5 step-50k，保留全部32粒子后用
   EAWQ `fusion_old_full_residual` 选 Top-1。
4. **v4（2026-08-20，历史冻结）**：Barrett和ShadowHand均将
   `palm_distance_weight` 从200改为0；掌心距离计算、1 mm目标、12 mm feasible
   gate、O10/I20和其他冻结参数不变；EAWQ使用精确 `ranking-only` 简化计算路径。
5. **v5（2026-09-02，历史冻结）**：接触生成器替换为 partial-only、
   full-normalization 的 K=128 AR step-64k；每物体/手 diffusion contact sets 从64
   改为32。FK、exact ranking-only EAWQ、O10/I20和D(R,O) PhysX设置继承v4。
6. **v6（2026-09-03，当前唯一默认）**：模型改为完整/partial各50%混合训练的
   K=128 AR step-56k；推理继续直接使用完整2048点云。其他设置继承v5。

当前最重要的状态不是一个新的成功率，而是：

- v6 协议、full-cloud推理适配和专用运行入口已经冻结；
- v6 checkpoint 已在远端按step、model type、训练观测与SHA256校验，但尚未完成完整
  OOD-10 GPU PhysX运行；
- filtered-50k 的 40,960 粒子历史运行已经完成，可用于诊断和规则探索；
- EAWQ 在这批旧标签上将离线 Top-1 从 58.20% 重排到 63.59%；
- ShadowHand 掌心权重严格512-env common-filler复验中，`w=0` 为
  327/640（51.09%），相对 `w=200` 的289/640（45.16%）提高5.94个百分点；
- **该640次结果是 v4 定权证据，不是完整双手 v4 结果；冻结 v4 后尚未完成独立
  1,280 次 Barrett+ShadowHand GPU PhysX Top-1 复验**；
- 因而 63.59% 只能标记为探索性离线重排，不能作为当前基础实验正式结果。

## 2. 统计边界和时间依据

本报告把“基础实验配置确定以后”的起点定义为首个完整可审计基础路线落盘，即
2026-08-13 的 v1 OOD-10 闭环。2026-08-13 以前的 FK800、旧 GenDex 栈、
Gym/Sim 对齐实验只作为形成基础配置的前史，不纳入本台账主表。

时间优先采用报告、summary 或状态文件的本地落盘时间。若远端 GPU run-root 没有
同步开始/结束时间，则明确写成“时间未同步”，不根据聊天顺序虚构精确时刻。

实验结果分为四类：

- **完整 Top-1 闭环**：每手10物体×64候选，分母固定为640；
- **全粒子诊断**：每手10×64×32=20,480次，双手共40,960次；
- **配对消融**：复用同一候选，只改变闭合或执行参数；
- **可视化/数据审计**：不能作为模型生成成功率。

## 3. 使用过的五类模型

| 代号 | 模型与 checkpoint | 训练数据 | 主要训练参数 | 本报告中的用途 |
|---|---|---|---|---|
| M45 | `contact_diffusion_multidex_seen48_success_n235_4x4090/best_val.pt`，step 45,000，SHA256 `bceef2fd51f1b13dcd86887e8204e0b988619f9e32248a4e7ec0dbceb7437bad` | MultiDex success-only；GenDex seen-48；N=2/3/5；执行器标签包含 ezgripper、Barrett、Robotiq-3F、ShadowHand | 4×RTX 4090；50k训练上限；单卡 batch 128；梯度累积4；有效 batch 2048；lr 2e-4；seed 42；DDIM 50 | v1/v2、闭合消融、Panda N=2、历史全粒子基线 |
| M145 | 与 M45 同一训练族继续到 step 145,000，SHA256 `4b00f0d139b8c208a0844a6b92a5073ae76684b49d8d392628f028ee536fee13` | 与 M45 相同 | 候选和仿真预算与全粒子基线一致 | 检验继续训练是否改善 Top-1/Oracle |
| MF50 | MultiDex-filtered Barrett/ShadowHand balanced N=3/5，显式 step 50,000，SHA256 `0934a15218f0f35cfd978d207e5c374556c0f8afbfb597e0489795d15ae97cc5` | seen-48；仅 Barrett、ShadowHand；N=3/5；成功索引 allowlist；N=3/5 round-robin 平衡采样 | 4×RTX 4090；batch 128×累积4×4卡=2048；lr 2e-4；50k；seed 42；DDIM 50；PointNet + 6层 Transformer | 历史 v3/v4 模型；40,960粒子诊断；EAWQ与稳定性研究 |
| PAR64 | partial-only AR K=128，显式 step 64,000，SHA256 `1516d96091e91d19c9e177a9a5aeda34fece16ed37983b9385868d0a1d86cfb8` | MultiDex seen-48成功抓取K=128 allowlist；N=2/3/5；每次随机view-facing 50% crop；完整几何归一化 | 4×RTX 4090；batch 192×4卡=768；lr 8e-4；64k；seed 20260903；DDIM 50/点；AR逐点生成；由full-PC 64k warm start | 历史v5模型；完整OOD-10结果未运行 |
| MFP56 | mixed full/partial AR K=128，显式 step 56,000，SHA256 `05bd542bb0c10a74ed8dafd7586a58ceedb17097dd08eb344011bb91e98516b4` | 与PAR64同一K=128体系；完整点云50%、synthetic partial 50%；两者均用完整几何归一化；N=2/3/5 | 4×RTX 4090；batch 192×4卡=768；lr 2e-4；从上一mixed阶段step-8k model warm start后执行56k新更新；seed 20260905；DDIM 50/点 | 当前v6模型；完整OOD-10结果待运行 |

M45 训练跑到50k，但 45k 的 validation total 最低，因此历史主实验选择
`best_val.pt`。MF50 则按当前冻结定义显式选择 step-50k，不使用 `latest.pt` 或
`best_val.pt` 的可变指针。

## 4. 固定数据与执行参数

### 4.1 OOD-10

完整基础实验统一使用以下10个未见/OOD物体：

- ContactDB：apple、camera、cylinder_medium、door_knob、rubber_duck、
  water_bottle；
- YCB：baseball、pear、potted_meat_can、tomato_soup_can。

### 4.2 候选预算

v1–v4主线每物体/手为64 sets；v5/v6固定为32 sets。其余预算如下：

| 参数 | 值 |
|---|---:|
| 每物体、每手 diffusion contact sets | v5/v6为32；v1–v4为64 |
| 每 contact set FK particles | 32 |
| 每粒子 FK steps | 400 |
| Diffusion sampler | DDIM，50 steps |
| FK 后进入解析排序的粒子 | v1/v2为历史Top-1；v3–v6保留全部32 |
| 每 contact set 最终进入仿真的候选 | 1 |
| 每手 OOD-10 Top-1 trial | v5/v6为320；v1–v4为640 |
| 双手 OOD-10 Top-1 trial | 1,280 |
| 双手全粒子 trial | 40,960 |

FK 使用精简六类能量框架：contact、点云穿透 mean+CVaR、self-collision
mean+CVaR、approach、joint prior、palm unsigned distance。v4中两手的palm
distance梯度权重均为0，但仍计算掌心距离并保留12 mm筛选gate。FC/DFC均关闭，
学习率0.005。Barrett自碰撞修复后冻结26对，ShadowHand冻结210对。

EAWQ固定使用精确 `ranking-only` 路径：保留冻结融合公式实际使用的末端+掌心残差
与全手残差，两路QP仍各迭代80次；只跳过epsilon/凸包、unweighted QP、composite
及全手2/5 mm等未使用诊断，Top-K与完整诊断路径一致。

### 4.3 D(R,O)-aligned Isaac Gym

- Isaac Gym Preview 4，PhysX，100 Hz，substeps=2，TGS iterations=8/0；
- gravity=0，无地面；contact/rest offset=0.01/0 m；
- hand/object friction=3/3，object density=500 kg/m³；
- position drive，Kp/Kd=1000/200，100步闭合；
- 两手当前统一固定方向 O10/I20；虚拟根保持 FK 值；
- 连续施加 `+X,+Y,+Z,-X,-Y,-Z`，每方向1秒，中间不重置状态；
- `F = Isaac 导入质量 × 0.5 m/s²`；
- 主判据：闭合结束位置到六方向结束位置的最终位移 ≤2 cm；
- strict：每段均≤2 cm，只作诊断；invalid 始终留在总分母并计失败。

正式结果要求 GPU PhysX。本地 RTX 5080 上的 CPU PhysX compatibility 运行只用于
工程闭环、消融和可视化，不得标成 D(R,O) 论文 GPU baseline 复现。

## 5. 按时间排列的实验台账

### 5.1 2026-08-12 23:02 至 2026-08-13 02:52：v1 首个完整闭环

- 模型：M45；数据：OOD-10；双手各640个 Top-1；64×32×400。
- 本地 Isaac Gym CPU PhysX；D(R,O) 资产和六方向协议。
- Barrett O0/I5，ShadowHand O10/I20。

| 手 | final | strict | valid/invalid |
|---|---:|---:|---:|
| Barrett | 487/640（76.09%） | 486/640（75.94%） | 623/17 |
| ShadowHand（错误 root 映射） | 1/640（0.16%） | 1/640（0.16%） | 469/171 |

该结果完成了流水线验收，但 ShadowHand 后来确认存在 FK root 到仿真 root 的放置
变换错误，因此 1/640 是缺陷复现，不是模型能力结论。

来源：[`BASIC_EXPERIMENT_DRO_ALIGNED_LOCAL_OOD10_FINAL_ZH.md`](BASIC_EXPERIMENT_DRO_ALIGNED_LOCAL_OOD10_FINAL_ZH.md)。

### 5.2 2026-08-13 14:04：修复 ShadowHand root 映射后的完整重放

复用同一 M45、同一 OOD-10 contact sets、FK候选和物理参数，只修正
FK-to-simulation root transform，并录制全部视频。

| 手 | final | strict | valid/invalid | 视频 |
|---|---:|---:|---:|---:|
| Barrett | 487/640（76.09%） | 486/640（75.94%） | 623/17 | 640 |
| ShadowHand | 241/640（37.66%） | 247/640（38.59%） | 542/98 | 640 |

ShadowHand 从 0.16% 回升到 37.66%，说明早期“物体位于拇指或掌背”的根本原因是
手物初始放置变换，而不是闭合比例或 diffusion contact 本身。

来源：[`REPORT.md`](../outputs/basic_experiment_root_aligned_full_recordings/REPORT.md)。

### 5.3 2026-08-13 14:30–22:19：ShadowHand 闭合与掌根配对消融

全部实验复用 M45 的640个 ShadowHand Top-1，不重新采样 diffusion 或重做 FK。

| 实验 | 规模与唯一变量 | 主要结果 | 决策 |
|---|---|---|---|
| outer/inner 8组筛选 + Top-2全量 | 8×160；再比较 O10/I20 与 O10/I10 的640重放 | O10/I20 241/640；O10/I10 236/640，-0.78 pp，p=0.583 | 保留 O10/I20 |
| smooth closure | 同160候选；旧 step target 对比 20/100/20 cubic smooth | 40.00% → 27.50%；闭合漂移中位数 415.9→588.4 mm | 平滑插值不能单独解决弹飞 |
| outer/inner 扩展至25% | tune 8×160、validate 3×160、confirm 2×320 | confirm：O10/I20 122/320；O15/I25 106/320，-5.00 pp | 不扩大到25% |
| 固定 root retreat | 0–20 mm；tune/validate/confirm | confirm：0 mm 122/320；8 mm 117/320，-1.57 pp，p=0.679 | 不使用固定后退 |

这些结果共同否定了“只调一个统一闭合比例/后退距离即可消除 ShadowHand 高失败率”。

来源：[`FINAL_REPORT.md`](../outputs/shadowhand_closure_parameter_sweep_v1/FINAL_REPORT.md)、
[`PREVIEW_REPORT.md`](../outputs/shadowhand_closure_smooth_preview_v1/PREVIEW_REPORT.md)、
[`REPORT_ZH.md`](../outputs/shadowhand_closure_expanded25_v1/REPORT_ZH.md)、
[`REPORT_ZH.md`](../outputs/shadowhand_root_retreat_sweep_v1/REPORT_ZH.md)。

### 5.4 2026-08-14 12:53–12:59：Barrett O0/I5 与 O10/I20 配对实验

复用 M45、同一640个候选、同一资产与物理参数，只改变闭合端点。

| 闭合 | final | strict | valid |
|---|---:|---:|---:|
| O0/I5 fresh control | 486/640（75.94%） | 485/640 | 623/640 |
| O10/I20 | 494/640（77.19%） | 493/640 | 629/640 |

O10/I20 增加8次成功（+1.25 pp），但53次失败转成功、45次成功转失败，McNemar
`p=0.4797`，整体增益不显著且有物体依赖性。统一 O10/I20 是为了减少跨手协议
分支，不应宣称它已显著优于 O0/I5。

来源：[`REPORT.md`](../outputs/barrett_closure_o10_i20_ab/REPORT.md)。

### 5.5 2026-08-14 13:20：v2 统一 O10/I20 完整结果

模型仍为 M45；数据仍为 OOD-10；候选与 root 修复后的历史运行一致；两手统一
O10/I20。后端为本地 CPU PhysX compatibility。

| 手 | final | strict | valid/invalid | fallback |
|---|---:|---:|---:|---:|
| Barrett | 494/640（77.19%） | 493/640（77.03%） | 629/11 | 271 |
| ShadowHand | 240/640（37.50%） | 246/640（38.44%） | 542/98 | 386 |

这是历史 v2 的完整工程基线，但不是当前 v4，也不是 GPU PhysX 正式结果。

来源：[`REPORT.md`](../outputs/basic_experiment_o10i20_v2_full/REPORT.md)。

### 5.6 2026-08-14 15:36：Panda N=2 扩展验证

- 模型：M45（包含 N=2 训练）；OOD-10；64×32×400；O10/I20 v2；
- Panda 每物体64个 Top-1，共640次；本地 CPU PhysX compatibility。

结果为 final/strict 均 118/640（18.44%），valid 496、invalid 144，fallback 264。
该实验验证了 N=2 路径可运行，但 Panda 不属于当前 v4 的 Barrett/ShadowHand
balanced N=3/5 主比较。

来源：[`summary.json`](../outputs/basic_experiment_panda_n2_ood10_full64/summary.json)。

### 5.7 时间未完整同步：M45 的 GPU PhysX 全粒子历史基线

该远端 run-root 没有随本地结果同步精确开始/结束时间；其指标保存在 M145 对照报告中。

- 规模：双手×OOD-10×64 sets×32粒子=40,960；O10/I20；GPU PhysX；
- 模型：M45；同一投影 contact set 与 seed 重放；invalid计失败。

| 范围 | 原 rank-0 Top-1 | Oracle@32 | 粒子成功率 | invalid |
|---|---:|---:|---:|---:|
| 总体 | 57.19% | 99.77% | 50.27% | 12.71% |
| Barrett | 80.16% | 100.00% | 69.68% | 4.94% |
| ShadowHand | 34.22% | 99.53% | 30.87% | 20.47% |

这批结果首次清楚表明：绝大多数 contact set 内存在成功粒子，主要损失来自 Top-1
排序而不是“整个 diffusion set 无法抓取”。

### 5.8 2026-08-15 15:45：M145 全粒子 GPU PhysX

仅把 checkpoint 换为 M145，其余 OOD-10、64×32×400、O10/I20、GPU PhysX 与
历史全粒子基线保持一致。

| 范围 | Top-1 | Oracle@32 | 粒子成功率 | invalid |
|---|---:|---:|---:|---:|
| 总体 | 55.39% | 100.00% | 50.68% | 12.67% |
| Barrett | 76.25% | 100.00% | 69.39% | 5.11% |
| ShadowHand | 34.53% | 100.00% | 31.97% | 20.23% |

相对 M45：总体 Top-1 -1.80 pp，Oracle +0.23 pp，粒子成功率 +0.41 pp。配对检验
总体 Top-1 `p=0.312`，没有证据表明继续训练到145k带来稳定总体提升。1,280个
set 均至少有一个成功粒子，但 ShadowHand 有42个 set 仅≤4/32粒子成功。

来源：[`FINAL_REPORT_ZH.md`](../outputs/basic_experiment_step145k_all_particles_o10i20_v1_remote4090/FINAL_REPORT_ZH.md)、
[`STATISTICAL_ANALYSIS_ZH.md`](../outputs/basic_experiment_step145k_all_particles_o10i20_v1_remote4090/STATISTICAL_ANALYSIS_ZH.md)。

### 5.9 2026-08-17：MF50 全粒子 GPU PhysX 与排序空间

模型改为当前 MF50；OOD-10、双手、64×32×400、O10/I20、GPU PhysX，共
40,960次。原 `candidate_rank=0` 仍是旧优化排序，不是 v3/v4 EAWQ。

| 范围 | 原 rank-0 Top-1 | Oracle@32 | 粒子成功率 | invalid |
|---|---:|---:|---:|---:|
| 总体 | 58.20% | 99.92% | 50.67% | 12.89% |
| Barrett | 78.44% | 100.00% | 69.41% | 5.16% |
| ShadowHand | 37.97% | 99.84% | 31.93% | 20.62% |

只有1/1,280个 set 的32个粒子全部失败，来自 ShadowHand。ShadowHand 的 Top-4
Oracle 为79.84%，Top-8为94.53%，Top-32为99.84%；rank 0–3 单粒子成功率约
36.6%–38.0%，显示旧能量排序几乎没有形成明显梯度。

2026-08-17 15:57 还筛出10组“rank-0失败、rank-1/2/3成功”的 CPU/GPU一致可视化
案例；它们用于证明排序错失，不构成独立总体成功率。

来源：[`REPORT.md`](../outputs/shadowhand_rank_pattern_visualizations_step50k/REPORT.md)、
[`REPORT.md`](../outputs/top1_vs_top4_step50k_isaacgym_comparison/final/REPORT.md)。

### 5.10 2026-08-17 17:35–22:04：EAWQ 探索性离线重排

在 MF50 的同一40,960粒子及其成功标签上，先无标签计算解析指标，随后用标签评价
AUC与重排结果。指标计算不读标签，但最终规则是在比较这批标签后选定，因此属于
post-hoc 探索。

| 排序 | 总体 Top-1 | Barrett | ShadowHand | 组内 AUC |
|---|---:|---:|---:|---:|
| 原 candidate rank | 58.20% | 78.44% | 37.97% | 0.5725 |
| 末端+掌心 EAWQ-R | 61.41% | 81.56% | 41.25% | 0.6284 |
| EAWQ-ε | 62.11% | 81.41% | 42.81% | 0.5943 |
| **末端残差 + 全手残差等权 rank fusion** | **63.59%** | **83.12%** | **44.06%** | **0.6524** |

v3 冻结并由 v4 继承最后一行 `fusion_old_full_residual`：末端+掌心 EAWQ-R 与全手 EAWQ-R
分别做组内0–1名次，等权平均，低分优先。相对旧 rank-0，离线总体 +5.39 pp，
Barrett +4.69 pp，ShadowHand +6.09 pp。

这不是独立实验：总体206个 set 被挽救、137个丢失，公式选择本身看过同一批标签。
必须重新仿真冻结规则选出的1,280个 Top-1，才能得到 v3 正式结果。

来源：[`FINAL_ANALYSIS_ZH.md`](../outputs/basic_experiment_balanced_n35_step50k_all_particles_o10i20_v1_remote4090/execution_aware_wrench_auc/FINAL_ANALYSIS_ZH.md)、
[`RANK_FUSION_ZH.md`](../outputs/basic_experiment_balanced_n35_step50k_all_particles_o10i20_v1_remote4090/full_hand_contact_auc/RANK_FUSION_ZH.md)。

### 5.11 2026-08-17 21:17 至 2026-08-18 11:44：GPU PhysX 重复稳定性

对 MF50 的40,960个唯一姿态各重复10次；不扰动物理参数，只改变隔离环境 batch
排列，共409,600次 PhysX trial。

| 范围 | 重复成功率 | 始终成功 | 1–9/10翻转 | 始终失败 | 二值稳定率 |
|---|---:|---:|---:|---:|---:|
| 总体 | 50.55% | 12,260 | 16,442 | 12,258 | 59.86% |
| Barrett | 69.49% | 11,972 | 3,831 | 4,677 | 81.29% |
| ShadowHand | 31.61% | 288 | 12,611 | 7,581 | 38.42% |

ShadowHand 只有288/20,480（1.41%）姿态10/10稳定成功；69.22%的 Shadow set
至少含一个≥9/10成功粒子，但其中91.65%被 rank-0 漏选。该结果说明排序提升空间
真实存在，同时也说明单次二值成功标签噪声很大。

来源：[`final_report.md`](../outputs/physx_stability_r10_results/final_report.md)、
[`REPORT_ZH.md`](../outputs/physx_stability_r10_results/set_particle_analysis/REPORT_ZH.md)。

### 5.12 2026-08-18 13:16：filtered ShadowHand seen-48 数据审计

这不是 diffusion 推理实验，而是从 D(R,O) `MultiDex_filtered/shadowhand.pt` 中按
seen-48 物体抽取已有成功姿态并在本地 CPU PhysX 重放、录制。

- 请求48个物体；46个有可录制样本；39个至少有10条 filtered success；
- 共录制433个数据集姿态和46个有效视频；
- 本地 final 256/433（59.12%），strict 253/433（58.43%）。

它说明“数据集中标为成功”不等于在当前 CPU PhysX 协议中必然成功，但不能用来
评价 MF50 生成质量，因为这些姿态不是 MF50 diffusion 输出。

来源：[`manifest.json`](../outputs/multidex_filtered_shadowhand_seen48_videos_lowmem/manifest.json)。

### 5.13 2026-08-18 17:02：当前 v3 冻结

v3 将 MF50、64×32×400、EAWQ rank fusion、两手 O10/I20、D(R,O) 资产和 GPU
PhysX 六方向协议统一冻结。2026-08-18 19:09 已将显式 step-50k checkpoint、训练
config、train log 和 filtered 样本索引同步到本地，并验证 checkpoint 内部
`step=50000`、SHA256一致。

截至 v3 冻结当时：

- 当时 `outputs/` 或 `reports/` 中没有正式结果声明 v3 协议 ID；
- v3 的解析排序脚本与协议审计已通过；
- v3 独立1,280 Top-1 GPU PhysX验证尚未执行；
- 当前可引用的正式数字仍必须带上“历史 v1/v2、模型、排序和后端”限定。

权威定义：[`BASIC_EXPERIMENT_CONFIG_PROMPT.md`](../docs/BASIC_EXPERIMENT_CONFIG_PROMPT.md)、
[`basic_experiment_filtered50k_eawq_o10i20_protocol.yaml`](../configs/basic_experiment_filtered50k_eawq_o10i20_protocol.yaml)。

### 5.14 2026-08-19 至 2026-08-20：ShadowHand 掌心距离权重演进

本阶段模型、目标接触、seed、32粒子、400 FK steps、学习率0.005、EAWQ Top-1、
O10/I20和物理判据均保持不变，仅改变 `palm_distance_weight`。

**小样本初筛。** 在4物体、Barrett+ShadowHand、每物体/手16个 contact set上测试
`{200,100,50,20,10,2}`。`w=20`总体86/128（67.2%），高于 `w=200` 的
82/128（64.1%）；ShadowHand由26/64升至33/64（+10.9 pp），但小样本差异不显著。
同时 contact Chamfer 37.45→16.12 mm、掌心间隙1.61→5.22 mm、feasible
52.3%→18.0%，首次显示“接触匹配改善、静态几何 gate下降”的交换关系。

**中等权重完整 OOD-10。** 每个权重1,280次 PhysX：

| 权重 | ShadowHand final | 相对 w=200 |
|---:|---:|---:|
| 200 | 179/640（27.97%） | — |
| 20 | **247/640（38.59%）** | **+10.63 pp** |
| 30 | 227/640（35.47%） | +7.50 pp |
| 40 | 232/640（36.25%） | +8.28 pp |

Barrett在 `w=20` 下为533/640，对照534/640，基本持平；因此证据支持分手型定权，
不支持把低掌心权重无差别套到 Barrett。

**低权重标准64-env搜索。** `{0,0.001,0.01,0.1,1}` 均提高 ShadowHand，最高
为 `w=0.01` 的42.03%，其次 `w=1` 的41.72%，`w=0` 为38.91%，同批
`w=200` 为29.22%。但诊断确认 GPU PhysX 对 active env数量、`envs_per_row`与
filler状态敏感，因此这些数字只作为趋势证据。

**严格512-env common-filler复验。** 固定512 active env、`envs_per_row=23`；
slot 0–63为目标，64–511为六组完全相同的448个 filler。10个OOD物体共640个
有效统计目标：

| 权重 | final | strict | invalid |
|---:|---:|---:|---:|
| 200 | 289/640（45.16%） | 47.19% | 43 |
| 1 | 315/640（49.22%） | 51.41% | 41 |
| 0.1 | 310/640（48.44%） | 51.88% | 42 |
| 0.01 | 313/640（48.91%） | 52.19% | 41 |
| 0.001 | 326/640（50.94%） | 53.59% | 42 |
| **0** | **327/640（51.09%）** | **54.22%** | **37** |

`w=0` 相对200提高38/640，即+5.94个百分点。但相对 `w=0.001` 只多1次成功，
五重 Holm 校正后低权重相对200也未达到 `p<0.05`；因此不能声称0是唯一统计最优。

来源：[`REPORT_ZH.md`](palm_distance_weight_pilot_v3_4obj16set_20260819/REPORT_ZH.md)、
[`REPORT_ZH.md`](palm_distance_weight_full_ood10_w200_w20_w30_w40_20260820/REPORT_ZH.md)、
[`FINAL_REPORT_ZH.md`](palm_distance_weight_lowrange_full_ood10_20260820/FINAL_REPORT_ZH.md)、
[`summary_padded512_common_fillers.json`](palm_distance_weight_lowrange_full_ood10_20260820/summary_padded512_common_fillers.json)。

### 5.15 2026-08-20：v4 冻结

用户在完整证据与统计边界已知的前提下作出工程定权：Barrett和ShadowHand均使用
`palm_distance_weight=0.0`。v4 的两手主能量为：

```text
100 E_contact + 100 E_pc-penetration + 100 E_self
+ 2 E_approach + 0.001 E_q + 0 E_palm-distance
```

掌心距离模块、1 mm目标和12 mm gate仍保留，EAWQ feasible-first继续使用该gate；
FC/DFC不启用，其余权重、32粒子、400 steps、学习率0.005均不变。

选择0的收益是释放多指接触匹配空间：contact Chamfer约33.68→16.76 mm，严格
PhysX final +5.94 pp，invalid 43→37。明确代价为 ShadowHand静态 feasible
31.09%→7.97%、`max penetration≤7 mm` 31.09%→10.16%、envelope
73.91%→52.97%、palm gate 100%→74.69%，掌心间隙中位数1.64→7.63 mm。

因此 v4 的含义是从“强制掌心贴近的几何保守方案”切换为“以接触匹配和最终
PhysX成功率为主、掌心距离仅作候选筛选约束”。严格640次 ShadowHand结果支持这一
工程选择；Barrett采用0是统一精简能量的工程定义，目前没有独立掌心权重复验，
不能套用ShadowHand的收益。完整双手v4仍需独立跑完1,280个Top-1后才能形成正式结果。

权威定义：[`BASIC_EXPERIMENT_CONFIG_PROMPT.md`](../docs/BASIC_EXPERIMENT_CONFIG_PROMPT.md)、
[`basic_experiment_filtered50k_eawq_o10i20_palm0_v4_protocol.yaml`](../configs/basic_experiment_filtered50k_eawq_o10i20_palm0_v4_protocol.yaml)。

### 5.16 2026-09-02：v5 冻结

v5 在不覆盖v4历史真源的前提下，只改变两类基础设置：

1. 将 MF50 joint-set step-50k 接触生成器替换为 partial-only、
   full-normalization、K=128 的 AR step-64k checkpoint；
2. 每物体/手型 diffusion contact sets 从64改为32，因此解析候选由2048降为1024，
   OOD-10双手最终 PhysX Top-1由1280降为640。

该checkpoint训练时每次访问都构造随机view-facing 50% crop并重采样至`2048×3`，
不是full/partial各半。最初接入实现按训练分布在推理中复现这一流程；用户随后明确
冻结v5推理为完整点云条件，因此当前v5已删除随机观察方向、保留50%与partial重采样，
直接输入完整`2048×3`点云。完整点云center/max-radius归一化、AR逐点生成free XYZ、
模型外向完整表面投影仍保留。候选JSON记录checkpoint训练观测、实际推理观测和二者
是否匹配；v5中该值明确为false，partial seed不适用。

远端 checkpoint：
`/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_partialonly_normfull_warmfull64_gb768_64k_4x4090/model/checkpoints/step_00064000.pt`；
远端实测大小86,708,970 bytes，SHA256为
`1516d96091e91d19c9e177a9a5aeda34fece16ed37983b9385868d0a1d86cfb8`。

FK能量、两手`palm_distance_weight=0`、32粒子、400 steps、学习率0.005、exact
ranking-only EAWQ、Top-1、O10/I20以及D(R,O) GPU PhysX参数全部继承v4。

冻结时尚未生成v5完整候选或PhysX结果。因此台账中此前的v1–v4数值都不是v5结果；
另外，因contact-set预算同时减半且存在partial-only训练/full-cloud推理的输入分布
偏移，未来v5对v4的差异不能单独归因于模型架构。严格模型消融需要另建相同sets预算
并对齐推理观测分布的配对协议。

权威定义：[`BASIC_EXPERIMENT_CONFIG_PROMPT.md`](../docs/BASIC_EXPERIMENT_CONFIG_PROMPT.md)、
[`basic_experiment_partial_ar64k_eawq_o10i20_palm0_v5_protocol.yaml`](../configs/basic_experiment_partial_ar64k_eawq_o10i20_palm0_v5_protocol.yaml)、
[`LATEST_PARTIAL_AR_MODEL_V4_REPLACEMENT.md`](../docs/LATEST_PARTIAL_AR_MODEL_V4_REPLACEMENT.md)。

### 5.17 2026-09-03：v6 冻结

v6只替换v5的接触生成checkpoint。新模型训练观测为
`mixed_full_synthetic_partial`：完整点云与synthetic partial各占50%，partial分支仍
保留随机view-facing 50%区域并重采样到2048点；两分支统一使用完整几何归一化。
v6推理继承v5决定，直接输入完整`2048×3`点云，不执行partial构造。不同于v5，完整
点云是新模型训练时明确覆盖的分支，因此消除了partial-only训练/full-cloud推理冲突。

远端checkpoint：
`/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_mixedfullpartial_normfull_from8k_lr2e4_gb768_56k_4x4090/model/checkpoints/step_00056000.pt`。
2026-09-03经本地3333 SSH隧道实测：`step=56000`、
`model_type=autoregressive_contact_diffusion`、输入`2048×3`、N=`[2,3,5]`、
`v_prediction`、文件大小86,709,034 bytes，SHA256为
`05bd542bb0c10a74ed8dafd7586a58ceedb17097dd08eb344011bb91e98516b4`。

该文件的step语义是从上一mixed阶段step-8k有限模型状态warm start、重新初始化
optimizer/scheduler后完成56k次新更新；累计暴露设计为8k+56k，但checkpoint正式身份
仍是step-56k。

32 contact sets、每set 32粒子×400步、两手掌心权重0、exact ranking-only EAWQ、
O10/I20、D(R,O)资产与GPU PhysX参数均与v5完全一致。冻结时没有v6完整OOD-10
候选或PhysX结果，历史结果不得重标为v6。

同日已将checkpoint按上述SHA256保存为仓库内
`weights/v6/step_00056000.pt`，并完成真实权重本地集成冒烟：`contactdb_apple`、两手
各1个set、每set 32粒子×400 FK steps、exact ranking-only EAWQ Top-1、D(R,O)
参数下CPU PhysX六方向验证。两手trial均`status=complete`且invalid=0，最终均为0/1。
该缩减协议只验证“模型→接触→FK姿态→EAWQ→PhysX”代码闭环，不可作为v6成功率，
也不改变“完整OOD-10 GPU PhysX结果尚未产生”的状态。可复现入口为
[`run_basic_experiment_v6_local_smoke.sh`](../scripts/run_basic_experiment_v6_local_smoke.sh)。

权威定义：[`BASIC_EXPERIMENT_CONFIG_PROMPT.md`](../docs/BASIC_EXPERIMENT_CONFIG_PROMPT.md)、
[`basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml`](../configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml)。

## 6. 跨实验可比较结论

### 6.1 已有充分证据支持

1. **ShadowHand 初始放置/root 映射曾是一级错误。** 修复后同候选从1/640提升到
   241/640，是本阶段最大的单项变化。
2. **统一闭合比例不是主要性能开关。** ShadowHand 的 O10/I10、扩大到25%、平滑
   闭合、固定 root retreat 都未稳定优于 O10/I20；Barrett O10/I20 的+1.25 pp
   也不显著。
3. **继续把旧模型从45k训练到145k没有改善 Top-1。** Oracle和粒子成功率几乎
   不变，Top-1反而略降，差异不显著。
4. **当前主要瓶颈是粒子选择与标签稳定性。** 三个模型的 Oracle@32 都接近100%，
   但 ShadowHand Top-1仅约34%–38%；MF50 重复运行中61.58%的 Shadow姿态会翻转。
5. **执行感知接触比目标接触/旧 GraspQP 更有排序信息。** EAWQ融合的离线组内
   AUC 0.6524，明显高于旧 candidate rank 0.5725。
6. **ShadowHand 的掌心距离权重200过强。** 严格512-env common-filler中，降至0
   后 final +5.94 pp且接触 Chamfer约减半；但静态可行率显著下降，0与0.001也
   没有被证明存在统计差异。

### 6.2 目前不能声称

1. 不能声称当前v6成功率为63.59%；这是历史v4模型粒子上的规则选择离线值。
2. 不能把本地 CPU v2 的 Barrett 77.19% / ShadowHand 37.50% 与论文 GPU baseline
   直接比较。
3. 不能把 MF50 相对 M45/M145 的几个百分点差异纯归因于模型，因为 contact sets、
   训练数据过滤和平衡策略同时变化。
4. 不能把 filtered 数据集姿态的59.12%称为 diffusion 模型成功率。
5. 不能把单次 PhysX 二值标签当作稳定真实标签，特别是 ShadowHand。
6. 不能把327/640称为双手 v4 成功率；它只对应严格布局下的 ShadowHand掌心
   权重定权实验。

## 7. 当前正式基础实验还缺什么

要完成当前v6的第一份可引用完整结果，至少需要：

1. 在远端使用已校验的mixed full/partial AR `step_00056000.pt`；
2. 按冻结seed对OOD-10、Barrett/ShadowHand各生成32个contact sets，并记录
   checkpoint训练观测、实际full-cloud推理观测、full-normalization和free-XYZ投影
   provenance；
3. 每 set 运行32粒子×400步并保留全部粒子；
4. Barrett和ShadowHand均使用掌心权重0；二者均保留1 mm目标与12 mm gate；
5. 可行性门控后执行冻结的 `fusion_old_full_residual`，不读取成功标签；
6. 每 set 只送 EAWQ Top-1，共640个候选进入同一 GPU PhysX协议；
7. invalid计失败，同时报告 final、strict、per-object、fallback和重复稳定性；
8. 将 checkpoint/config/资产/候选/EAWQ/结果 hash 和实际开始结束时间写入 manifest；
9. 明确把该独立结果与离线63.59%和ShadowHand掌心定权640次分栏，不覆盖历史
   v1/v2/v3/v4文件。

建议至少对最终640个 Top-1 再做3次重复；如果算力有限，可先完整跑一次作为主结果，
再对 ShadowHand、阈值附近和结果翻转候选进行分层重复。

## 8. 主要证据入口

- v4历史协议说明：[`BASIC_EXPERIMENT_DRO_ALIGNED.md`](../docs/BASIC_EXPERIMENT_DRO_ALIGNED.md)
- 当前配置提示：[`BASIC_EXPERIMENT_CONFIG_PROMPT.md`](../docs/BASIC_EXPERIMENT_CONFIG_PROMPT.md)
- 当前机器配置：[`basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml`](../configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml)
- Barrett v4 FK配置：[`multigripper_fk_multidex_ood10_barrett_steps400_palm0_v4.yaml`](../configs/multigripper_fk_multidex_ood10_barrett_steps400_palm0_v4.yaml)
- ShadowHand v4 FK配置：[`multigripper_fk_multidex_ood10_shadow_dro_steps400_palm0_v4.yaml`](../configs/multigripper_fk_multidex_ood10_shadow_dro_steps400_palm0_v4.yaml)
- 掌心权重完整演进：[`FINAL_REPORT_ZH.md`](palm_distance_weight_lowrange_full_ood10_20260820/FINAL_REPORT_ZH.md)
- v1完整报告：[`BASIC_EXPERIMENT_DRO_ALIGNED_LOCAL_OOD10_FINAL_ZH.md`](BASIC_EXPERIMENT_DRO_ALIGNED_LOCAL_OOD10_FINAL_ZH.md)
- v2完整报告：[`REPORT.md`](../outputs/basic_experiment_o10i20_v2_full/REPORT.md)
- M145全粒子：[`FINAL_REPORT_ZH.md`](../outputs/basic_experiment_step145k_all_particles_o10i20_v1_remote4090/FINAL_REPORT_ZH.md)
- MF50 EAWQ：[`FINAL_ANALYSIS_ZH.md`](../outputs/basic_experiment_balanced_n35_step50k_all_particles_o10i20_v1_remote4090/execution_aware_wrench_auc/FINAL_ANALYSIS_ZH.md)
- EAWQ融合：[`RANK_FUSION_ZH.md`](../outputs/basic_experiment_balanced_n35_step50k_all_particles_o10i20_v1_remote4090/full_hand_contact_auc/RANK_FUSION_ZH.md)
- PhysX稳定性：[`final_report.md`](../outputs/physx_stability_r10_results/final_report.md)
