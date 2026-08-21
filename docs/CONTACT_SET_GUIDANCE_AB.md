# Contact set 引导作用 A/B 实验

该实验回答一个单独且先于方法大消融的问题：在相同 FK 预算下，ContactDiffusion
生成的稀疏接触集合，是否比不读取成功标签的随机物体表面点更能引导出可执行抓取。

## 与当前基础路线的关系

唯一基础协议仍是
`contactdiff-multidex-filtered50k-fk64x32x400-eawq-rankfusion-dro-gym-o10i20-palm0-v4`。
本实验是它的 **all-32 诊断扩展**，不是新的部署协议，也不替代 v4 的 EAWQ Top-1
正式结果。以下项目与 v4 完全一致：

- MF50 step 50,000 checkpoint 及 SHA256；
- DDIM50，每物体64组 contact set，每组32个 FK 粒子，优化400步；
- Barrett、ShadowHand 的 palm0 FK 配置及六项能量权重；
- O10/I20 闭合、D(R,O) 资产、六方向、final/strict 判据；
- GPU PhysX 每批512个 active env、`envs_per_row=23`，每个物体—手型—变体4批。

不同之处只有 contact target：

- A `diffusion`：投影到物体表面的 ContactDiffusion 输出；
- B `matched_random`：从完整 XYZ 表面点云随机取互异点，并用与物体尺度归一化的
  点间距离、到物体中心的径向距离和接触质心半径做几何匹配。匹配器不匹配质心
  方向/表面语义区域，也不读取 FK 或 PhysX 成功标签。

主比较采用 `--initialization-contact-source diffusion`。因此 A、B 每个配对子共享
diffusion 输出、sample seed 和完全相同的32粒子 FK 初始状态，只有优化目标不同。
启动器会先完成每个 A 文件，再把其中保存的 diffusion contacts 直接传给对应 B，
不会依赖两次独立 CUDA 采样碰巧逐位相同。
这是较保守的 target-only 检验：若 A 仍优于 B，可以把差异归因于 diffusion contact
在固定 FK 能量中的引导信息，而不是更好的随机初始化。

“删除 contact Chamfer 项”不作为主对照，因为它同时改变了优化问题与约束强度，不能
回答哪一组 contact 更有引导性。它可在主 A/B 完成后作为独立的 `no-contact-energy`
机制消融。

## 2×/4×H100 运行

在算力平台拉取代码、准备 checkpoint 和 Isaac Gym 环境后：

```bash
cd /path/to/ContactDiffusion
export CONTACTDIFF_PYTHON=/path/to/contactdiff/python
export CONTACTDIFF_ISAAC_RUNNER="$PWD/scripts/run_remote_isaacgym_python.sh"
export CONTACTDIFF_GPU_IDS=0,1
mkdir -p outputs/contact_set_guidance_ab_palm0_v4_h100
nohup bash scripts/run_contact_set_guidance_ab_4h100.sh \
  > outputs/contact_set_guidance_ab_palm0_v4_h100/launcher.log 2>&1 &
```

generation 默认每张 H100 使用三个进程；两卡为6个、四卡为12个。若显存或主存压力
过大，可设置 `CONTACTDIFF_GUIDANCE_AB_GENERATION_WORKERS=2`。GPU PhysX 阶段始终
每张卡一个 worker。重复执行同一命令会
利用 candidate `--resume` 并跳过完整的512-trial PhysX batch。

只从指定阶段恢复：

```bash
CONTACTDIFF_GUIDANCE_AB_START_STAGE=gym \
  bash scripts/run_contact_set_guidance_ab_4h100.sh
```

允许的阶段为 `generate`、`prepare`、`gym`、`report`。如果使用非默认输出目录，所有
恢复调用都必须继续设置同一个 `CONTACTDIFF_GUIDANCE_AB_RUN_ROOT`。

进度与结果：

```bash
tail -f outputs/contact_set_guidance_ab_palm0_v4_h100/supervisor/status
tail -f outputs/contact_set_guidance_ab_palm0_v4_h100/supervisor/pairing_audit.log
watch -n 5 nvidia-smi
```

最终配对结果位于：

- `reports/pairing_audit.json`：进入 PhysX 前的 seed、源 contact、初始化状态审计；
- `reports/diffusion_summary.json` 与 `matched_random_summary.json`：各变体 all-particle
  成功率、Oracle@32 和 rank-prefix 指标；
- `reports/paired_summary.json` 与 `PAIRED_SUMMARY.md`：按原始粒子配对的成功率差、
  contact-set Oracle@32 配对 McNemar 精确检验和 object-cluster bootstrap 95% CI。

主结论优先看 invalid-inclusive 的 all-particle final/strict success；Oracle@32说明
contact set 是否至少包含一个可执行粒子。统计独立单位应按 object × contact-set
处理，不能把同一 contact set 内32个高度相关粒子当作32个独立 contact set。
