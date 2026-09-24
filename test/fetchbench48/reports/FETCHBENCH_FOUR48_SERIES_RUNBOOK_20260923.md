# 四批48任务顺序实验

## 冻结设计

共192个不同的新目标资产，与20260920和20260922两批48目标资产均无交集。联合MILP在生成结果之前一次性分配，种子2026092301；每批四环境各12任务，每场景每批最多2任务。四批涉及63个场景实例，允许跨批复用场景，因此统计不能将192任务视作192独立场景。剩余目标主要是复杂体，不是几何平衡设计。

本地冻结清单：`outputs/fetchbench_four48_series_20260923/JOINT_ALLOCATION.json`及`SERIES.json`。各批任务、模型、资产清单与哈希锁位于：

- `outputs/fetchbench_v2_four48_batch1_ab_dro64_20260923`
- `outputs/fetchbench_v2_four48_batch2_ab_dro64_20260923`
- `outputs/fetchbench_v2_four48_batch3_ab_dro64_20260923`
- `outputs/fetchbench_v2_four48_batch4_ab_dro64_20260923`

远端对应目录在`/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/`下。不读取历史抓取结果选任务，不根据成功率补抽或提前停止。

## 算法与统计

沿用上一轮Barrett、ShadowHand的A/B/DRO算法与物理协议，不使用Panda专用耦合或求解参数。

- 每任务111候选机位，按原2/4/2可见度规则最多锁定8个单视角；不是所有任务使用相同相机位置。
- A partial、B SAM3D使用原两套冻结权重，各4接触集×4粒子，DDIM50，FK200、lr0.0075、ENV0；E3、32次采样预算、冻结Top1排序不变。
- DRO每视角每手64候选，原环境筛选与数值重试不变。
- CPU PhysX、无GUI、无视频，q_outer→q_inner→抬升25cm；有限最终净抬升≥10cm。
- A/B按实际选中并执行的Top1统计；DRO按环境筛选后实际执行统计。不把未执行项静默作为已执行失败，也不将执行分母成功率误称为端到端成功率。

## 资源与顺序

当前节点`dex-grasp--62e259e82782-ugjr3pf32f`：4张4090，CPU配额55核，内存配额400GiB。四卡各一路采集/SAM3D；抓取生成每卡2路，总8路；CPU PhysX验证最多12路。保留原4卡core及历史脚本，仅增加独立系列入口。

严格顺序：第1批全部阶段终止→异常审计→成功率报告落盘→第2批→第3批→第4批。不会同时启动多个批次，不创建持续轮询或定时监视任务。执行器等待它自己启动的子进程是实验执行流程，不是外部监控服务。

已知终止性不可用（固定接触预算耗尽、局部环境裁剪为空）单独披露，不额外抽样；未知执行/数值/基础设施错误以及DRO未完成会停止系列，不能在漏报的情况下推进下一批。

每批生成`EXECUTION_REPORT.md`、`EXECUTION_STATISTICS.json`及额外的`SERIES_COHORT_REPORT.md`、`SERIES_COHORT_AUDIT.json`。系列目录更新`SERIES_REPORT.md`、`SERIES_RESULTS.json`、`status.json`。

## 启动和恢复

在远端FetchBench项目中运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 /inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff/bin/python -u scripts/run_fetchbench_four48_series_20260923.py --preflight-only
```

预检通过后，去掉`--preflight-only`即执行系列。已有系列进程时不要重复启动；文件锁阻止重复主流程。恢复时复用已完成批次的报告/哈希及已完成子任务，不能覆盖冻结设计。代码变化会触发运行锁检查，不应盲目删除锁绕过。

## 时间估计

基于前轮四卡约10.5小时和双卡约18.7小时，预估本系列40–60小时，建议节点至少72小时。这是启动前估计，复杂场景、筛选保留数、I/O与错误处理均会改变时间。阶段依赖和CPU验证意味着GPU利用率不会全程100%；不为满载而更改科学协议或并行四个批次。
