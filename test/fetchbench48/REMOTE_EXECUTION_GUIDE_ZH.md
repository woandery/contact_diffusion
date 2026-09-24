# 远端平台 A/B/DRO 48 任务实验：代码、目录与执行过程

本文件记录 2026-09-23 开始的四批实验。**一批为 48 个“场景＋目标物”任务，四批共 192 个不同目标资产**；每个任务最多使用 8 个选定的单相机视角。A、B、DRO 均测试 Barrett 和 ShadowHand。下面的路径是实际算力节点上的路径，不是仓库克隆后的通用路径。

## 1. 远端目录与运行环境

节点：`dex-grasp--62e259e82782-ugjr3pf32f`，4 × RTX 4090。根目录：

```text
P=/inspire/qb-ilm2/project/zhanghanbo/public/mck
F=$P/FetchBench-CORL2024
C=$P/ContactDiffusion
PY_CONTACT=$P/miniconda3/envs/contactdiff/bin/python
PY_FETCH=$P/miniconda3/envs/fetchbench/bin/python
PY_SAM3D=$P/miniconda3/envs/sam3d-objects/bin/python
DRO=$P/dro_grasp_reproduction/DRO-Grasp
SAM3D=$P/sam-3d-objects
```

远端 FetchBench 的 `scripts/` 放调度、相机、坐标变换、DRO、排序与验证适配代码；ContactDiffusion 的 `scripts/`、`models/`、`utils/` 放接触生成与 FK 实现；`F/InfiniGym/` 放 Isaac Gym 任务和物理验证。模型和场景资产不在本代码包中。各批 `manifest.json` 的 `runtime` 字段给出当时使用的 A/B/DRO 权重、E3 配置、SAM3D 配置与 Python 绝对路径。

## 2. 冻结任务与实际启动

实验前，本地用 `F/scripts/prepare_fetchbench_four48_20260923.py` 联合选出四批任务；它调用 `prepare_fetchbench_newassets48_20260922.py` 生成各批任务清单、`DESIGN_LOCK.json`、模型与资产清单。固定种子为 `2026092301`；每批四类环境各 12 个任务，同一批每场景最多两个任务，不按抓取结果重新抽样。

远端系列目录：

```text
$C/outputs/fetchbench_four48_series_20260923/
  JOINT_ALLOCATION.json
  SERIES.json
  launch.json
  driver.log
  status.json
  SERIES_RESULTS.json
  SERIES_REPORT.md
```

四个单批目录为 `$C/outputs/fetchbench_v2_four48_batch{1,2,3,4}_ab_dro64_20260923/`。本仓库 `test/fetchbench48/frozen/` 保存准备阶段的四批设计文件；实际相机选择产生于运行时，保存在各批目录的 `VIEW_SELECTION_LOCK.json`。

原始主程序是 `F/scripts/run_fetchbench_four48_series_20260923.py`，调用 `F/scripts/run_fetchbench_four48_cohort_20260923.py`。远端 `launch.json` 记录的启动命令为：

```bash
cd /inspire/qb-ilm2/project/zhanghanbo/public/mck/FetchBench-CORL2024
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff/bin/python -u \
  scripts/run_fetchbench_four48_series_20260923.py
```

执行前可加 `--preflight-only`，检查四张可见 GPU、设计锁、外部资产、运行源代码/权重和 Vulkan 配置。主程序先预检四批，再按批次顺序启动；每批写入 `driver.log`，完成异常审计与报告后才进入下一批。同一系列使用 `series.lock` 防止并发重复运行。上述命令是历史执行记录，不应在现有运行目录再次启动。

## 3. 一批 48 任务的数据与代码链

| 阶段 | 主要代码 | 远端输入 → 输出 |
| --- | --- | --- |
| 相机捕获 | `F/InfiniGym/isaacgymenvs/capture_extension_111.py`，由 `run_fk_top1_extension6_20260915.py:capture` 调用 | `F/Task/benchmark_eval/<scene>/task_config.npz` 与场景资产 → `observations/<case>/visibility/rgbd_views/` |
| 锁定视角 | `F/scripts/fetchbench_confirmatory_rules_v1.py:select_views`、`run_fetchbench_autolabel_pilot_v2.py:locked_views` | 每任务 111 个候选机位中满足可见性条件的视角 → 最多 8 个，低/中/高可见度目标配额 2/4/2 |
| SAM3D 重建 | `F/scripts/sam3d_resident_batch.py`、`reconstruct_fetchbench_sam3d.py` | 同一视角 RGB-D/分割 → `observations/<case>/sam3d/<view>/`；重建失败的视角不进入 A/B FK |
| 坐标与输入 | `F/scripts/materialize_fetchbench_world_to_robot_base.py` | 相机的 FetchBench 世界坐标数据 → robot-base 数据；分别写入 `dro_inputs/<case>/` 和 `generation/<case>/corrected_inputs/` |
| A/B 抓取生成 | `run_fetchbench_autolabel_pilot_v2.py` → `run_fk_top1_round3_20260916.py` → `supplement_extension6_20260916.py:fk_case`；ContactDiffusion 的 `infer_contactdiffusion_camera_partial_v5_six.py` / `infer_contactdiffusion_explicit_condition_v6.py`、`infer_local_contactdiffusion_grasp.py` | A 以 `camera_partial_sam3d_centered.npy` 作为模型条件；B 以 `sam3d_fused_centered.npy` 作为模型条件。两者用 SAM3D 物体几何做投影、法向、FK 初始化与物体几何计算，场景 partial 点云做环境计算 → `fk/<case>/cases/J0/<view>/<A或B>/<hand>/` |
| A/B 物理验证与 Top1 | `run_shampoo_joint_fk_staged.py`、`validate_shampoo_particle_order.py`，再由 `run_fk_top1_extension6_20260915.py:score_case` 使用冻结排序器 | 每组 4 粒子按固定顺序分别用 CPU PhysX 执行；已冻结的排序器从四粒子中选一个 → `top1/fk/<case>/unified_top1.json` |
| DRO 生成、筛选、验证 | `supplement_extension6_20260916.py:dro_generate/dro_valid`，`dro_generate_grasps.py`，`filter_dro_environment_candidates.py`，`validate_dro_cached_only.py` / `InfiniGym/isaacgymenvs/validate_dro_lift.py` | 同一视角 `target_partial_robot_base.npy` 和 `scene_partial_robot_base.npy` → 64 个候选、环境筛选、CPU PhysX 执行；输出在 `dro/<case>/views/<view>/simulation/<scene_factory>/task_<index>/<hand>/` |
| 统计与审计 | `run_fetchbench_four48_cohort_20260923.py:report`，`run_fetchbench_four48_series_20260923.py:audit_report` | `comparison_inputs.json`、Top1、DRO 的 `lift_validation/summary.json` → `EXECUTION_STATISTICS.json`、`SERIES_COHORT_AUDIT.json` |

这里的 `<case>` 是各批 `manifest.json` 中的 `id`，`<view>` 是锁定的相机视角 ID。每任务视角集合由该任务的可见性决定，**不是所有任务共用固定的 8 台相机**。DRO 对已锁定的合法 partial 视角运行；A/B 需要相应视角的 SAM3D 重建有效。

## 4. 固定配置和成功率口径

- A：partial 点云条件，2048 点，独立 A 权重；B：SAM3D 伪完整点云条件，2048 点，独立 B 权重。两者都采用 E3 预处理、4 个 contact sets × 4 粒子、DDIM 50、FK 200 步、FK 学习率 0.0075、独立 ENV 0 步、32 次固定接触预算。
- Barrett 与 ShadowHand 都使用同一冻结 Top1 排序流程。排序器为每批目录的 `ranker_unified_frozen.joblib`；先独立仿真全部可用粒子，再按预先训练好的特征排序。新 48 任务的成功标签没有用于重新训练排序器。
- DRO 每视角每只手生成 64 候选，采样 512 个目标 partial 点、优化 64 步；用同视角场景 partial 点云做环境筛选，间隙参数 0.005 m。数值求解失败只按预先规定的种子序列重试，不按抓取结果补抽。
- 两条链路的物理执行均为 headless CPU PhysX，不录视频；`q_outer → q_inner → 抬升 0.25 m`。运行参数还包含运行时环境接触拒绝和闭合前位移 0.02 m 设置。**主成功定义只取有限的最终物体净抬升 ≥ 0.10 m**，不能用最大瞬时高度代替。
- `EXECUTION_STATISTICS.json` 中 A/B 分母为实际选出并执行的 Top1，DRO 分母为环境筛选后实际执行的候选；筛除、重建失败、接触预算耗尽和执行错误另行披露。两个方法的候选数不同，这个成功率不是等预算比较。

四卡分工：4 路相机捕获、4 路 SAM3D、A/B 或 DRO 生成最多每卡 2 路（总 8 路），CPU PhysX 验证最多 12 路。实际并行度也受当前任务阶段和数据可用性限制。

## 5. 本次运行的续跑记录

最初的顺序入口完成第 1 批；第 2 批的执行统计已生成，但异常审计发现一条初始化环境能量断言，原系列因此停止。该条异常没有被改写为算法失败或静默删除。用户随后要求继续第 3、4 批；远端使用独立脚本：

```text
$C/outputs/fetchbench_four48_series_20260923/resume_batch34_20260924.py
```

脚本在本包的 `snapshot/ContactDiffusion/outputs/fetchbench_four48_series_20260923/` 也有一份副本。它保留第 2 批异常标记，重新预检第 3、4 批，依次运行并审计，记录 `RESUME34_AUTHORIZATION.json`、`launch_resume34.json`、`driver_resume34.log`、`RESUME34_RESULTS.json`。继续运行不代表第 2 批异常已修复。协作者应按各批审计状态解释结果。

## 6. 查找结果

```text
$C/outputs/fetchbench_four48_series_20260923/status.json
$C/outputs/fetchbench_four48_series_20260923/driver.log
$C/outputs/fetchbench_four48_series_20260923/driver_resume34.log
$C/outputs/fetchbench_v2_four48_batch1_ab_dro64_20260923/EXECUTION_STATISTICS.json
$C/outputs/fetchbench_v2_four48_batch2_ab_dro64_20260923/EXECUTION_STATISTICS.json
$C/outputs/fetchbench_v2_four48_batch3_ab_dro64_20260923/EXECUTION_STATISTICS.json
$C/outputs/fetchbench_v2_four48_batch4_ab_dro64_20260923/EXECUTION_STATISTICS.json
```

某批仍在执行时，其统计文件可能尚未产生。查看 `pipeline_status.json`、`coverage.json`、`events.json` 与分阶段日志可区分尚未执行、重建失败、接触预算耗尽和真实仿真失败。完整原始点云、权重、Isaac Gym 安装与视频未包含在 Git 代码包中。
