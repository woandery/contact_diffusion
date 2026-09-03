# 基础实验配置提示（当前唯一版本：v6）

后续提到“基础实验”“基础配置”或“默认实验”时，除非显式声明做历史消融，均指：

- 协议：`contactdiff-mixed-full-partial-ar-k128-step56k-fk32x32x400-eawq-rankfusion-dro-gym-o10i20-palm0-v6`；
- 模型：K=128、full-normalization、完整/partial点云混合训练的自回归
  ContactDiffusion，显式step-56,000 checkpoint；远端路径为
  `/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_mixedfullpartial_normfull_from8k_lr2e4_gb768_56k_4x4090/model/checkpoints/step_00056000.pt`，
  SHA256为
  `05bd542bb0c10a74ed8dafd7586a58ceedb17097dd08eb344011bb91e98516b4`；
- 仓库内权重：`weights/v6/step_00056000.pt`；runner优先使用该文件，缺失时才回退到
  远端绝对路径；该文件作为唯一显式例外纳入Git，其他checkpoint仍由`.gitignore`
  排除；
- 训练观测：`mixed_full_synthetic_partial`，完整点云与synthetic partial各占50%；
  partial分支保留随机view-facing 50%区域并重采样为`2048×3`；两分支都使用完整
  物体点云进行canonical normalization；
- v6推理观测：继承v5决定，直接输入完整`2048×3`物体点云，不生成随机观察方向、
  不裁掉50%、不做partial重采样、不追加observation flag；完整点云条件属于该模型
  训练时明确覆盖的50%分支，因此不存在v5的partial-only/full-inference分布冲突；
- 输出适配：AR原生生成free XYZ，`model.sample(project_to_surface=False)`；反归一化
  后在模型外投影到完整2048点表面，保存raw XYZ、投影点和投影距离；
- 采样与FK：每物体/手型32个diffusion contact sets，每组32个FK particles，
  每粒子400步，学习率0.005；Barrett每组3个接触，ShadowHand每组5个接触；
- seed：模型/FK seed为
  `20260808 + 1000003*global_hand_index + 1009*global_object_index + sample_index`；
  v6不构造synthetic partial，因此没有partial seed；
- Barrett、ShadowHand FK均使用`palm_distance_weight=0.0`；掌心距离仍计算，
  `palm_target_distance_m=0.001`、`selection_max_palm_distance_m=0.012`和12 mm
  feasible gate保持不变；
- 主能量：`100 E_contact + 100 E_pc-penetration + 100 E_self +
  2 E_approach + 0.001 E_q + 0 E_palm-distance`；FC/DFC均未启用；
- 排序：保留全部32粒子，先做可行性门控，再用最新版EAWQ
  `fusion_old_full_residual`；末端+掌心残差和全手残差分别转换为组内0–1平均名次，
  等权平均，低分优先，原candidate rank仅作最终tie-break；
- EAWQ计算：固定使用精确`ranking-only`简化路径；保留两路80步QP残差，跳过不进入
  冻结融合公式的诊断量，与完整诊断模式产生相同Top-K；
- Top-1：每组只将EAWQ Top-1送入Isaac Gym；每物体/手型32次，OOD-10双手共
  640次；成功标签不参与EAWQ；
- 执行：Barrett、ShadowHand均使用固定关节方向O10/I20；
- 仿真：D(R,O)同源资产及Isaac Gym参数，连续`+X,+Y,+Z,-X,-Y,-Z`，每方向1秒且
  方向切换不重置状态；主判据为闭合结束位置到最终位置不超过2 cm；
- 正式后端：GPU PhysX；CPU PhysX只作本地smoke，不混合统计；invalid计失败。

## 相对v5的变更

v6只替换接触生成模型：从partial-only AR step-64k改为完整/partial各50%混合训练的
AR step-56k。完整点云推理、32×32×400、FK能量、exact ranking-only EAWQ、Top-1、
O10/I20、D(R,O)资产和GPU PhysX判据全部继承v5。

该56k训练阶段从上一轮mixed模型的step-8k有限状态进行model warm start，重新初始化
优化器和scheduler后执行56k次新更新；checkpoint内嵌`step=56000`，不能将文件名改写
为step-64k。可以说明累计训练暴露意图为8k+56k，但正式checkpoint身份仍是step-56k。

当前只完成v6协议与执行链建设，尚未产生完整OOD-10 GPU PhysX结果；不得引用v5、
partial/full分布评价或其他历史结果作为v6物理成功率。

2026-09-03已用仓库内真实权重完成本地端到端集成冒烟：对`contactdb_apple`分别为
Barrett和ShadowHand生成1个contact set，每set执行32粒子×400步FK、exact
ranking-only EAWQ Top-1及CPU PhysX六方向验证。两次PhysX trial均正常完成、invalid=0，
但恰好均失败（0/1）；该结果只证明代码链路和协议适配可执行，不构成成功率估计，也不
与正式GPU PhysX结果混合。可复现入口为
`scripts/run_basic_experiment_v6_local_smoke.sh`，对应缩减范围协议为
`configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_smoke1_protocol.yaml`。

权威机器可读配置：
`configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml`。

默认远端运行入口：
`scripts/run_basic_experiment_v6_mixed_full_partial_ar56k_ood10_8gpu.sh`。

历史口径：v5为partial-only AR64k、完整点云推理、32×32×400；v4为
MultiDex-filtered step-50k、64×32×400、两手掌心权重0；v3为同一filtered-50k +
EAWQ但掌心权重200；v2为MultiDex 45k + feasible/GraspQP Top-1；v1包含旧Barrett
O0/I5。它们只用于历史对照或消融，不得再称为当前基础实验。
