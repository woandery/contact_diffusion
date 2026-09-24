# FetchBench 48 任务 A/B/DRO 实验代码

这里保存 2026-09-23 四批、每批 48 个任务的实验入口、依赖脚本快照、冻结任务设计和排序模型。早期 `test/fetchbench_ab_dro` 是三场景 ENV200 实验；本实验使用 FK200、ENV0。

在 ContactDiffusion 根目录执行静态审计：

```bash
python test/fetchbench48/audit_package.py
python test/fetchbench48/audit_upstream.py
```

- `frozen/JOINT_ALLOCATION.json` 与 `frozen/SERIES.json`：共同抽样与批次顺序。
- `frozen/batch1` 至 `batch4`：每批任务、协议、资产清单、排序器与设计锁。相机视角在捕获后选定，记录于运行目录的 `VIEW_SELECTION_LOCK.json`。
- `snapshot/`：FetchBench 与 ContactDiffusion 的相关脚本快照；`SOURCE_SHA256.json` 记录全部快照哈希；`frozen/upstream_fk_manifest.json` 记录上游 51 个文件的运行时锁定哈希。
- [复现说明](REPRODUCTION.md)：外部依赖与成功率口径。
- [远端执行说明](REMOTE_EXECUTION_GUIDE_ZH.md)：实际代码、节点目录、启动和续跑记录。

这份代码包不包含 Isaac Gym、SAM3D/DRO/GenDex 安装、场景与物体资产、生成模型权重、相机观测或仿真结果。冻结文件保留原节点的绝对路径；换目录需重新准备并生成新的设计锁。
