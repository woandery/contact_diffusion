# 48 任务复现协议

每批 48 个新目标：桌面、架/柜、抽屉、篮子各 12 个；每目标最多选 8 个合法单视角。四批顺序执行，上一批完成异常审计和报告后进入下一批。自动几何类别待人工复核；不同批次允许场景实例复用。

| 方法 | 接触条件输入 | 候选与优化 | 最终选择 |
| --- | --- | --- | --- |
| A | 相机 partial 点云 | 4 接触集 × 4 粒子，DDIM50，FK200/学习率 0.0075，ENV0 | 冻结排序器 Top1 |
| B | SAM3D 伪完整点云 | 同 A，但使用独立冻结权重 | 同一排序器 Top1 |
| DRO | 单视角 partial 点云，采样 512 点 | 每视角每手 64 候选，优化 64 步 | 环境筛选后执行 |

A/B 沿用 E3 预处理、32 次接触预算和原有物理参数。SAM3D 用于物体中心、点投影、法向、FK 初始化和物体几何计算；场景 partial 点云用于环境计算。机器人基座/世界坐标转换必须与冻结链路一致。没有独立 ENV200 阶段。

Barrett、ShadowHand 均使用 CPU PhysX；执行序列为 `q_outer → q_inner → 抬升 25 cm`。唯一主成功标签是最终有限净抬升 `final_object_lift_m >= 0.10`。A/B 分母为实际可执行 Top1，DRO 分母为环境筛选后实际执行；筛掉的 DRO 候选和未生成的 A/B 组另行披露。候选预算不同，执行成功率不是等算力比较。

原节点目录布局：

```text
/inspire/qb-ilm2/project/zhanghanbo/public/mck/
  ContactDiffusion/
  FetchBench-CORL2024/
  GenDexGrasp/
  dro_grasp_reproduction/DRO-Grasp/
  sam-3d-objects/
  miniconda3/envs/{contactdiff,sam3d-objects,fetchbench}/
```

`manifest.json` 含原节点绝对路径；`ASSET_FILE_INVENTORY.json` 列出外部资产大小。四批排序器随包提供。A/B/DRO 生成权重、场景资产、Isaac Gym、历史运行库和接触生成配置须另行安装/共享，并按冻结 manifest 与上游哈希核对。

源代码快照包含从原算力节点读取并核对的 51 个上游文件，以及带 20260914 日期后缀的四个历史脚本。它不是 FetchBench 的完整安装。前序链条还依赖历史运行目录中的配置与运行库；把代码复制到新节点后仍需完成这些依赖。原始绝对路径和设计锁不可直接重写后宣称等价复现。

在具有上述原始布局与完整外部依赖的节点运行：

```bash
python FetchBench-CORL2024/scripts/run_fetchbench_four48_series_20260923.py --preflight-only
python FetchBench-CORL2024/scripts/run_fetchbench_four48_series_20260923.py
```

入口检查四张可见 GPU、设计锁、资产大小、源代码与模型哈希以及 Vulkan ICD。准备器拒绝覆盖已有运行目录。代码快照供核对与协作开发，不能覆盖正在运行的节点。

结果文件是每批运行目录的 `EXECUTION_STATISTICS.json`、`SERIES_COHORT_AUDIT.json` 和 `SERIES_COHORT_REPORT.md`。异常分为接触预算耗尽、空场景裁剪和数值/代码错误。跨批统计应考虑场景实例复用，不能把所有视角当独立样本。
