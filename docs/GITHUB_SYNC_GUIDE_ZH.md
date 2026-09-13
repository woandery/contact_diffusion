# GitHub 同步与协作说明

更新日期：2026-09-13。本说明记录基础实验 v7 的 GitHub 交付位置、同步范围及协作方法。

## 1. 当前发布位置

- 仓库：[woandery/contact_diffusion](https://github.com/woandery/contact_diffusion)。
- 当前基础实验分支：[codex/basic-experiment-v7](https://github.com/woandery/contact_diffusion/tree/codex/basic-experiment-v7)。
- v7 基础路线冻结提交：`4d5e75d2b0a2acdafa86ac488ccfaa524994f357`。
- 历史 v6 分支：`codex/basic-experiment-v6`，仅供追溯，不是当前 v7 入口。
- 本地工作区：`/home/zhb1/mck/dexgrasp/ContactDiffusion`。

2026-09-13 已核对上述 v7 冻结提交与 GitHub 分支一致。本说明作为后续文档提交发布，
因此分支最新提交可能晚于冻结提交；冻结提交标识的是实验实现版本，不要求永远等于分支 HEAD。
v7 发布在独立分支，未合并或覆盖 `main`。

## 2. 已同步与未打包的内容

已同步：

- [v7 完整流程](BASIC_EXPERIMENT_V7_WORKFLOW.md)和[基础配置提示文件](BASIC_EXPERIMENT_CONFIG_PROMPT.md)。
- [配置演进台账](../reports/BASIC_EXPERIMENT_POST_FREEZE_LEDGER_ZH.md)。
- [正式 v7 协议](../configs/basic_experiment_fetchbench_ar32k_eawq_o10i20_palm0_v7_protocol.yaml)、独立冒烟协议和模型训练来源配置。
- 接触生成、FK、exact ranking-only EAWQ、Isaac Gym 适配及运行入口。
- [模型权重](../weights/v7/best_val.pt)和[权重来源说明](../weights/v7/README.md)。
- 协议审计、回归测试和[本地冒烟验收报告](../reports/BASIC_EXPERIMENT_V7_LOCAL_ACCEPTANCE_20260912.md)。

权重是 FetchBench real60/synth20/full20 混合观测训练模型，内嵌 step=32000，
文件大小为 86,705,982 字节，随普通 Git 文件保存。SHA256：

```text
c55badbd2e1ce7bc9cda9b58832e68003b4b757ee02eedee47e01e697c7ec491
```

完整训练数据、外部 D(R,O)/GenDex 资产、Isaac Gym 安装环境，以及本地生成的候选、
视频和全部日志，不属于此次 GitHub 交付包。克隆代码后仍需按流程说明配置资产路径与运行环境。
本地工作区还存在其他实验的未提交修改，未混入 v7 冻结提交。

GitHub 同步也不等于向算力平台部署代码。本地端口 3333 的 SSH 连接用于访问算力平台；
从该平台取得权重、向该平台部署代码、向 GitHub 推送提交，是三个独立操作。
本说明不表示算力平台代码已经更新。

## 3. 协作者首次获取

在准备存放项目的目录执行；目标目录 `contact_diffusion` 应不存在：

```bash
git clone --branch codex/basic-experiment-v7 --single-branch https://github.com/woandery/contact_diffusion.git
cd contact_diffusion
git log -1 --oneline
sha256sum weights/v7/best_val.pt
```

确认权重 hash 与上文一致，然后阅读 `docs/BASIC_EXPERIMENT_V7_WORKFLOW.md`。
依赖安装完成后，可执行配置与权重审计：

```bash
python scripts/audit_basic_experiment_v7_protocol.py --output /tmp/v7_audit.json
```

协议审计通过不代表 GPU PhysX 成功率已经验证。冻结时仅完成 Apple 两手各 1 次 CPU PhysX
集成冒烟，均成功；正式 OOD-10 的 640 次 GPU PhysX 不应被当作已有实验结果。

## 4. 已有克隆如何更新

先检查工作区，以下切换和拉取命令只应在当前工作区干净时执行：

```bash
git status --short
git fetch origin
git switch codex/basic-experiment-v7
git pull --ff-only origin codex/basic-experiment-v7
```

如果本地尚无 v7 分支，将 `git switch` 那行替换为：

```bash
git switch --track origin/codex/basic-experiment-v7
```

如果有未提交修改，先将其保存在自己的工作分支，或另建一个干净克隆；不要强制覆盖。
`--ff-only` 拒绝执行时，表示历史需要检查，不要通过强制推送或硬重置处理。

## 5. 后续修改与同步规则

协作者建议从最新 v7 新建个人功能分支，例如 `codex/v7-doc-update`，完成改动、测试后推送，
再创建以 `codex/basic-experiment-v7` 为目标分支的 Pull Request。没有仓库写权限时，
可由仓库所有者授予协作权限，或通过自己的 fork 提交 Pull Request。

每次提交只暂存此次任务涉及的文件。例如只修改本说明时：

```bash
git diff -- docs/GITHUB_SYNC_GUIDE_ZH.md
git add docs/GITHUB_SYNC_GUIDE_ZH.md
git diff --cached --name-only
git diff --cached --check
git diff --cached
git commit -m "Document GitHub synchronization workflow"
git push -u origin HEAD
```

提交前必须检查整个暂存区；如果存在其他任务已暂存的内容，先协调处理，不要一起提交。
避免在混合实验工作区使用 `git add .`、`git add -A`、强制推送或破坏性重置。
修改正式实验参数时，应同时更新协议、提示文件、台账及测试；不要悄悄改变已冻结 v7 的含义。

维护者在 v7 分支同步完成后，可用以下命令核对本地和远端提交 ID：

```bash
git rev-parse HEAD
git ls-remote --heads origin codex/basic-experiment-v7
```

两者提交 ID 一致，才说明当前 v7 提交已同步。请向协作者分享明确的 v7 分支链接，
不要仅依赖仓库首页的默认分支，也不要误发历史 v6 链接。
