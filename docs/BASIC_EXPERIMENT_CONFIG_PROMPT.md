# 基础实验配置提示（当前唯一版本：v7）

更新日期：2026-09-12。默认实验为v7；v6及更早版本用于历史对照。

- 协议：`contactdiff-fetchbench-real60-synth20-full20-ar-k128-step32k-fk32x32x400-eawq-rankfusion-dro-gym-o10i20-palm0-v7`。
- 模型：K=128体系AR ContactDiffusion，FetchBench伪真实相机partial混合训练，
  best_val内嵌step=32000。仓库权重：`weights/v7/best_val.pt`。
- SHA256：`c55badbd2e1ce7bc9cda9b58832e68003b4b757ee02eedee47e01e697c7ec491`；
  文件大小86,705,982 bytes。best_val与同目录显式32k文件模型参数已核对相同。
- 远端来源：`/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_real60_synth20_full20_from_mixed56k_lr5e5_gb768_32k_4x4090/model/checkpoints/best_val.pt`。
- 训练观测：60% FetchBench仿真相机partial + 20% synthetic partial + 20% full；
  非真机相机采集。仍采用原MultiDex接触标签与K=128 allowlist。
- 训练阶段：mixed56k best模型warm start，4×4090、batch192/卡、有效batch768、
  lr5e-5、32k新更新。step32000是本阶段步数。
- **推理继承v6：完整2048×3点云**，不生成随机观察方向、不裁掉50%、不重采样
  synthetic partial、不追加观测flag。三路混合训练的20% full分支支持该输入。
- 输入使用完整物体点云中心与最大半径归一化。AR顺序生成free-XYZ contacts，
  每点DDIM50步，模型内project_to_surface=False；反归一化后模型外最近邻投影到
  完整2048点表面，保存raw XYZ、投影目标与距离。
- 每物体每手32个contact sets；每set32个FK粒子、400 steps、lr0.005。
  Barrett每set3个接触，ShadowHand每set5个接触。
- seed=`20260808 + 1000003*global_hand_index + 1009*global_object_index + sample_index`；
  full推理不需要partial seed。
- 两手FK能量：
  `100 E_contact + 100 E_pc-penetration + 100 E_self + 2 E_approach + 0.001 E_q + 0 E_palm-distance`。
  FC/DFC关闭，palm_distance_weight=0；距离仍计算，目标1mm、feasible gate12mm保留。
- 保留32粒子，feasible-first的exact ranking-only EAWQ
  `fusion_old_full_residual`：末端+掌心残差和全手残差分别归一化为组内平均名次，
  等权融合，低分优先，并列用原candidate rank；无可行候选时按相同规则fallback。
  两路QP各80步，成功标签不参与排序。
- 每set只仿真EAWQ Top-1。每物体每手32次，OOD-10两手总640次。
- FK/PhysX使用v6同源D(R,O)资产与既有root/joint映射；两手固定关节方向O10/I20。
- PhysX100Hz、2 substeps、闭合100步；连续+X,+Y,+Z,-X,-Y,-Z、
  每方向1秒、0.5m/s²；方向切换不重置位置或速度。
- final判据：闭合结束位置到六方向结束位置的位移≤2cm；同时报告strict，invalid计失败。
  摩擦3/3、物体密度500、关节及virtual root stiffness1000/damping200、
  position iterations8/velocity0、contact offset0.01/rest0、无重力/地面。
- 正式后端GPU PhysX；本地CPU PhysX冒烟独立标识。v7完整640次正式结果尚未生成。

## 入口与历史

本地验收（2026-09-12）：apple两手各1-set、32×400、exact EAWQ、CPU PhysX，
两手final/strict均1/1，invalid=0；此为链路冒烟，不作为正式成功率估计。

- [v7全流程说明](BASIC_EXPERIMENT_V7_WORKFLOW.md)
- [机器配置](../configs/basic_experiment_fetchbench_ar32k_eawq_o10i20_palm0_v7_protocol.yaml)
- [正式运行入口](../scripts/run_basic_experiment_v7_fetchbench_ar32k_ood10_8gpu.sh)
- [本地冒烟入口](../scripts/run_basic_experiment_v7_local_smoke.sh)
- [权重与协议审计](../scripts/audit_basic_experiment_v7_protocol.py)
- [演进台账](../reports/BASIC_EXPERIMENT_POST_FREEZE_LEDGER_ZH.md)

v7相对v6仅替换模型权重和训练来源。v6为50% full/50% synthetic-partial AR56k；
v5为partial-only AR64k；v5–v7均为full输入、32×32×400。v4为filtered50k、
64×32×400、两手掌心权重0。FetchBench的相机输入、环境碰撞二阶段优化、
其他闭合或抬升判据均不自动纳入v7基础路线。
