# v7 本地集成验收（2026-09-12）

新权重为FetchBench real60/synth20/full20混合训练AR模型的best_val.pt，
内嵌step32000。SHA256：
`c55badbd2e1ce7bc9cda9b58832e68003b4b757ee02eedee47e01e697c7ec491`。

运行入口：`scripts/run_basic_experiment_v7_local_smoke.sh`。
输出根目录：`outputs/v7_local_smoke_20260912`。
协议为独立v7 smoke1：apple两手各1-set，32粒子、400步、DDIM50、full2048输入、
投影到完整表面、exact EAWQ Top-1、O10/I20、D(R,O)参数CPU PhysX。

| 手型 | PhysX状态 | final | strict | invalid | 最终位移 | 原候选rank |
|---|---|---:|---:|---:|---:|---:|
| Barrett | complete | 1/1 | 1/1 | 0 | 0.676171mm | 0 |
| ShadowHand | complete | 1/1 | 1/1 | 0 | 0.096458mm | 4 |

两手候选均记录full_object_pc和training_supports_inference_observation=true。
EAWQ在64个解析粒子上计算两个残差并选出2个Top-1；生成与排序阶段不读取
PhysX标签。闭合阶段物体分别移动约18.661mm和17.935mm，按继承的D(R,O)
判据，final从闭合结束开始测量。

协议审计核对checkpoint hash、内嵌训练模式与步数，以及v6的generation、
fk_optimization、ranking、hands、isaac_gym、execution六段逐项一致。
回归覆盖三路混合full输入支持、无full分支、非法概率拒绝和v5/v6兼容性。

这两次试验验证代码集成与参数传递。样本量不足以判断v7优于v6，
也不能替代完整OOD-10的640次GPU PhysX基准结果。
仓库保存本报告；原始候选、日志、prepared与结果JSON在上述本地输出目录。
