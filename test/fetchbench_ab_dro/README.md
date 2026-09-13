# FetchBench A/B + D(R,O) 历史验证实验（test）

本目录保存 2026-09-10 三场景实验的生成、环境筛选、坐标系修正与 CPU PhysX 验证流程。
来源为实际算力节点，2026-09-13 取回。它是**独立的历史实验包，不是基础实验 v7 的替代实现**。
主仓库 `models/`、`utils/`、`configs/`、`scripts/` 不被覆盖；不包含点云、候选、视频、日志或新权重。

## 1. 配置与比较边界

|场景|FetchBench 场景 / 工厂|task index|目标 index|合法单视角数|
|---|---|---:|---:|---:|
|架子 CerealBox|RigidObjLayerShelf_4 / LayerShelfSceneFactory_32|20|1|79|
|篮子 Shampoo|RigidObjDrawerShelf_6 / DrawerShelfSceneFactory_28|0|1|45|
|抽屉 Shampoo|RigidObjDrawerShelf_6 / DrawerShelfSceneFactory_28|10|1|48|

每场景原始候选视角 111 个，半径 1/1.5/2 m，极点去重；按可见性筛选后的原始视角列表必须保存。
不是将所有视角融合后再生成，每个合法视角独立生成两只手的候选。

|参数|A / B|DRO|
|---|---|---|
|条件输入|A：当前视角 partial 2048 点；B：SAM3D 伪完整 2048 点|当前视角 partial 512 点|
|候选预算|4 contact sets × 4 FK 粒子；每 set 保留 1，共 4 个|64 个原始候选|
|接触生成 / FK|DDIM 50；FK 200，学习率 0.0075|DRO 后处理优化 64 步|
|环境处理|ENV 200，学习率 0.003，w=10|外侧预抓取手点云到 partial scene 最近邻距离，5 mm 筛选|
|执行|CPU PhysX；无 GUI、无视频|相同验证框架；CPU PhysX|

**A/B 历史实验更换了输入和 checkpoint 两个因素，不能称为固定权重的纯输入消融。**
A 权重与主仓库 `weights/v7/best_val.pt` 的 SHA256 一致；B 使用 fullpc 模型 step 64000，
DRO 使用 `model_3robots_partial.pth`。三个权重的路径、大小、完整 SHA256 见
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。不允许为了省事把 B 权重替换为 A 权重再声称复现。

另一个边界：A/B 的 Barrett 为 GenDex `barrett_adagrasp`，DRO 的 Barrett 来自 DRO 手模型资产。
因此 AB4 与 DRO64 不是预算或手模型完全匹配的生成器消融。

## 2. 数据流与物理口径

1. FetchBench RGB-D + segmentation → 单视角目标 partial 与不含目标/机器人的 partial scene。
2. SAM3D 使用 OpenCV 相机约定重建，并融合 25% 可见点 + 75% 重建点形成 8192 点几何代理。
3. A 条件来自 partial；B 条件来自 SAM3D。两者的中心/归一化、接触点表面投影、法向、
   FK 初始化、物体接触/穿透/抓取质量计算均使用 SAM3D 几何代理，而不是原始物体完整 mesh。
4. FK 根据对应手 URDF 计算手表面采样点与运动学；物体穿透通过点云表面/法向近似计算，
   不是 mesh 精确体积求交。可见环境约束使用当前视角 partial scene；ENV 阶段还保留 object_pc 约束。
5. ENV 使用 5 mm 环境间隙、闭合路径 4 个采样状态、约束归一化、CVaR 与最佳可行状态恢复，
   再按 contact set 选择，详见冻结 shell 的完整命令行。
6. 排除根姿态或关节中的 NaN/Inf，并保存原始 record index、来源 SHA256 与分母审计。
   不重新生成替代失败样本，不把未执行样本冒充仿真失败。
7. Isaac Gym：`q_outer → q_inner → 抬升 25 cm`。outer=0.10、inner=0.20；随机种子 20260808。
   主指标是 **`final_object_lift_m >= 0.10`**，不是过程中的最大抬升。
   原始严格 `success`（含闭合前位移 ≤2 cm 等门限）仍单独保存，不被高度指标覆盖。

仿真本身仍需 FetchBench 场景/目标碰撞资产；“不使用真实完整 mesh 生成/优化”不代表物理引擎不加载 mesh。
历史视角可见性/覆盖率分析也读取目标真实 mesh 采样作为评估参照；这一部分不是接触生成或穿透优化的输入。
YAML 内 `isaac:` 是旧通用验证字段，不能单独当作本次有效物理参数；本次实际参数来自
`FetchPtdDRORender*.yaml` 继承链和 `validate_extension3_finite_ab.py` / DRO worker 的 Hydra 覆盖。

注意有效覆盖：FK 掌距能量权重为 0，生成 shell 显式传入 `--disable-palm-selection-gate`；
YAML 虽保留 `selection_max_palm_distance_m: 0.012`，本实验该筛选并未启用。这里仅记录历史事实，不修改参数。

## 3. 坐标系修正：不能省略

原始相机元数据标记 `fetchbench_world`，但历史文件名仍是 `*_robot_base.npy`。
必须以元数据为准，而非文件名：

`T_base_world = inverse(T_world_base)`，`p_base = R_base_world @ p_world + t_base_world`。

`materialize_fetchbench_world_to_robot_base.py` 同步变换目标、场景和 SAM3D 点云，保存变换矩阵与
round-trip 误差，要求 <1e-6 m；已是 `robot_base` 的输入会拒绝再次转换。
FK 局部坐标以 SAM3D 物体表面中心定义，再经 candidate transform 回到 robot_base。
执行器仅在进入场景时做一次 base→world。不能再次变换已经是 world 的姿态。

## 4. 文件地图与可信范围

- `snapshot/ContactDiffusion/`：历史推理/FK/环境细化/筛选/Isaac 适配及其项目内依赖。
- `snapshot/FetchBench-CORL2024/`：相机采集、SAM3D、坐标修正、调度、DRO、仿真任务和场景配置。
- `configs/config_{barrett,shadow}.yaml`：实际使用的 4×4、FK200、0.0075 配置。
- `configs/historical_manifest.json`：实际任务、运行环境路径与预算；`status: running` 是历史清单原值，
  **不是当前进度**。不要将历史绝对路径误当作通用安装路径。
- `SOURCE_MANIFEST.json`：53 个源码快照文件 SHA256（包含相机 reference111 辅助脚本）；另有 12 个生成时冻结 hash，审计逐项对照。
  没有生成时 hash 的文件仅能证明 9 月 13 日抓取时的内容，不宣称全部代码均有运行前冻结证据。
- `run_workflow.py`：新增分阶段入口，默认只展示计划；显式 `--execute` 才运行。
- `summarize_results.py`：修正后的统计入口，A/B 读 **ab_finite_validation/summary.json**，DRO 逐条读执行结果。
- `tests/`：离线回归，不要求 Isaac Gym 或模型权重。
- `reports/HISTORICAL_RESULTS.md`：最终聚合计数，仅作来源对照；不是本次重新跑出的成功率。

历史 `run_extension3_ab4x4_dro64_2gpu.py`、`validate_fetchbench_extension_ab_cpu_physx.sh`、
`summarize_extension3_ab4x4_dro64.py` **仅供追溯，不作为新入口**：旧 AB 路径要求每批 4 个有限姿态，
不适用于本次 25 个非有限姿态的情况。新入口直接调用 finite validator，DRO 要求 AB 验证已完成。

原始 manifest 准备脚本未在当前远端找到（`SOURCE_MANIFEST.missing` 保留记录）；本包提供已经冻结的
三场景清单，不提供或宣称“一键从零重新选出同样的 172 个视角”。需保留既有合法视角与原始 RGB-D 数据。

## 5. 本地审计（不启动实验）

在本目录运行，Python 3.10+，测试另需 NumPy、PyYAML：

```bash
python audit_package.py
python -m unittest discover -s tests -v
python export_overlay.py --destination /tmp/fetchbench_ab_dro_overlay_new
```

导出目标必须不存在。导出的是覆盖层源码，**不是完整可运行仓库**；不会自动修改任何已有 checkout。
`notices/FetchBench-LICENSE` 保留 FetchBench 的 BSD-3-Clause 许可。Isaac Gym、SAM3D、DRO、
GenDex 的第三方代码、权重、URDF/mesh 按各自许可单独配置，不在本提交中重新分发。

## 6. 有资产的独立实验工作区如何运行

先准备独立工作区，不要覆盖用于正式 v7 的目录：

```text
PROJECT/
  ContactDiffusion/       完整基础仓库 + 本包 ContactDiffusion 覆盖层
  FetchBench-CORL2024/    完整 FetchBench + 本包 FetchBench 覆盖层
  GenDexGrasp/            Barrett URDF/mesh
  dro_grasp_reproduction/ DRO-Grasp 与 pydeps_py310
  sam-3d-objects/         重建仓库与完整 checkpoints
  miniconda3/envs/        contactdiff (3.10), sam3d-objects (3.11), fetchbench (3.8)
```

外部必需项还包括：FetchBench `Task/benchmark_eval` 任务数组、benchmark_objects、场景资产、Isaac Gym
安装和 gymtorch；`InfiniGym/assets/contactdiff_hands/{barrett,shadowhand}` 与对应物理 URDF；
配置中的 tip-offset calibration 等资产；DINO 缓存、兼容驱动/运行库。不能用新生成的手资产替代后仍声称原样复现。
本包不安装 CUDA、驱动、Python 环境，也不自动下载这些资产。

将覆盖层合入**隔离的实验副本**，调整 `historical_manifest.json` 的 `project_root`、runtime 路径、
观测路径、任务路径；保存为新运行目录 `RUN/manifest.json`。从 `configs/` 复制两份 YAML 并在 manifest 指向它们。
保留协议值，配置中的相对资产布局也需满足。`--execute` 会检查部署代码 hash、权重 hash 和冻结 YAML；
若更改算法/手参数，请创建新协议，不要绕过检查。

分阶段运行（将路径替换为已配置的实际路径；以下不含 `--execute`，只检查并展示计划）：

```bash
python run_workflow.py --run /path/to/new_run --runtime-source /path/to/source_runtime --phase sam3d
python run_workflow.py --run /path/to/new_run --runtime-source /path/to/source_runtime --phase ab-generate
python run_workflow.py --run /path/to/new_run --runtime-source /path/to/source_runtime --phase ab-validate
```

确认路径后，每条分别加 `--execute`。SAM3D 使用两卡；AB 每卡 3 个生成进程；有限姿态验证 8 个 CPU 进程，
每进程 PhysX 4 线程；不要同时启动多份入口。物理不是 GPU PhysX，也不会录视频。
AB 完成并汇报后，才执行 `--phase dro --execute`；DRO 每卡 2 个生成进程、2 个 CPU 验证进程。
环境变量/命令行完整计划可先重定向保存审阅。新运行必须新目录，不要修改正在生成的 manifest 或复用旧候选缓存。

最后统计（输出目录必须不存在；无须 GPU）：

```bash
python summarize_results.py --run /path/to/new_run --output /path/to/new_report
```

AB 计划分母 2752，DRO 计划分母 22016。每组同时输出计划、NaN、环境筛除、执行、缺失、
高度成功、严格成功与两种分母的成功率。未完成组不输出最终率；A/B 和 DRO 预算差异始终注明。

## 7. 本次同步验收范围

离线 hash、Python 语法、shell 语法、配置/统计/坐标转换与入口预算测试通过，结果见
`reports/SYNC_ACCEPTANCE.md`。**此次只整理并同步代码，没有重新跑 SAM3D、抓取生成或 Isaac Gym**。
物理复现仍需在配置好上述外部资产的节点上做两手各 1 个案例冒烟后再运行全量。
