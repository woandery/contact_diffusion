# MultiDex ContactDiffusion D(R,O) Gym/Sim 总结

## 实验范围

Barrett 与 ShadowHand 在本报告中分别总结；不将两种执行器的成功率作为优劣排名。每种执行器内部比较同一批 640 条候选在 D(R,O) Isaac Gym 和 Isaac Sim 中的表现。

共同预算：OOD10、每物体 64 个 contact sets、每组 32 个 FK particles、保留 top1、400 个 FK 优化步、50 个 diffusion steps。

## 总体结果

| 执行器 | Gym D(R,O) | Gym strict | Sim raw | Sim valid-only | Sim 无效 | Sim raw - Gym |
|---|---:|---:|---:|---:|---:|---:|
| Barrett | 477/640 (74.53%) | 477/640 (74.53%) | 552/640 (86.25%) | 552/616 (89.61%) | 24 | +11.72 pp |
| ShadowHand | 3/640 (0.47%) | 4/640 (0.62%) | 33/640 (5.16%) | 33/509 (6.48%) | 131 | +4.69 pp |

## Barrett：Gym 与 Sim

| OOD 物体 | Gym D(R,O) | Gym strict | Sim raw | Sim valid-only | 无效 | Sim raw - Gym |
|---|---:|---:|---:|---:|---:|---:|
| contactdb_apple | 51/64 (79.69%) | 51/64 (79.69%) | 61/64 (95.31%) | 61/64 (95.31%) | 0 | +15.62 pp |
| contactdb_camera | 49/64 (76.56%) | 49/64 (76.56%) | 53/64 (82.81%) | 53/62 (85.48%) | 2 | +6.25 pp |
| contactdb_cylinder_medium | 53/64 (82.81%) | 53/64 (82.81%) | 59/64 (92.19%) | 59/63 (93.65%) | 1 | +9.38 pp |
| contactdb_door_knob | 40/64 (62.50%) | 40/64 (62.50%) | 62/64 (96.88%) | 62/64 (96.88%) | 0 | +34.38 pp |
| contactdb_rubber_duck | 54/64 (84.38%) | 54/64 (84.38%) | 60/64 (93.75%) | 60/64 (93.75%) | 0 | +9.38 pp |
| contactdb_water_bottle | 49/64 (76.56%) | 49/64 (76.56%) | 57/64 (89.06%) | 57/64 (89.06%) | 0 | +12.50 pp |
| ycb_005_tomato_soup_can | 32/64 (50.00%) | 32/64 (50.00%) | 42/64 (65.62%) | 42/50 (84.00%) | 14 | +15.62 pp |
| ycb_010_potted_meat_can | 41/64 (64.06%) | 41/64 (64.06%) | 47/64 (73.44%) | 47/57 (82.46%) | 7 | +9.38 pp |
| ycb_016_pear | 55/64 (85.94%) | 55/64 (85.94%) | 58/64 (90.62%) | 58/64 (90.62%) | 0 | +4.69 pp |
| ycb_055_baseball | 53/64 (82.81%) | 53/64 (82.81%) | 53/64 (82.81%) | 53/64 (82.81%) | 0 | +0.00 pp |

候选：checkpoint step 45000，640 records，32 particles，400 FK steps。

诊断统计：候选 contact chamfer 中位数 35.07 mm，assigned contact error 中位数 18.09 mm，mean penetration 中位数 0.11 mm；Gym 最终位移中位数 0.000 m，P90 5.219 m，最大方向段位移中位数 0.000 m。

## ShadowHand：Gym 与 Sim

| OOD 物体 | Gym D(R,O) | Gym strict | Sim raw | Sim valid-only | 无效 | Sim raw - Gym |
|---|---:|---:|---:|---:|---:|---:|
| contactdb_apple | 0/64 (0.00%) | 0/64 (0.00%) | 0/64 (0.00%) | 0/61 (0.00%) | 3 | +0.00 pp |
| contactdb_camera | 0/64 (0.00%) | 0/64 (0.00%) | 9/64 (14.06%) | 9/48 (18.75%) | 16 | +14.06 pp |
| contactdb_cylinder_medium | 0/64 (0.00%) | 0/64 (0.00%) | 0/64 (0.00%) | 0/44 (0.00%) | 20 | +0.00 pp |
| contactdb_door_knob | 1/64 (1.56%) | 1/64 (1.56%) | 3/64 (4.69%) | 3/42 (7.14%) | 22 | +3.12 pp |
| contactdb_rubber_duck | 0/64 (0.00%) | 0/64 (0.00%) | 3/64 (4.69%) | 3/54 (5.56%) | 10 | +4.69 pp |
| contactdb_water_bottle | 1/64 (1.56%) | 1/64 (1.56%) | 0/64 (0.00%) | 0/53 (0.00%) | 11 | -1.56 pp |
| ycb_005_tomato_soup_can | 0/64 (0.00%) | 1/64 (1.56%) | 7/64 (10.94%) | 7/47 (14.89%) | 17 | +10.94 pp |
| ycb_010_potted_meat_can | 1/64 (1.56%) | 1/64 (1.56%) | 11/64 (17.19%) | 11/42 (26.19%) | 22 | +15.62 pp |
| ycb_016_pear | 0/64 (0.00%) | 0/64 (0.00%) | 0/64 (0.00%) | 0/56 (0.00%) | 8 | +0.00 pp |
| ycb_055_baseball | 0/64 (0.00%) | 0/64 (0.00%) | 0/64 (0.00%) | 0/62 (0.00%) | 2 | +0.00 pp |

候选：checkpoint step 45000，640 records，32 particles，400 FK steps。

诊断统计：候选 contact chamfer 中位数 26.36 mm，assigned contact error 中位数 15.86 mm，mean penetration 中位数 0.39 mm；Gym 最终位移中位数 2.746 m，P90 15.012 m，最大方向段位移中位数 0.784 m。

ShadowHand 的低成功率不能仅由 Sim invalid 解释：排除 131 条无效试验后 Sim 仍为 6.48%。候选的几何损失有限，但 Gym 位移达到米级，说明主要失败发生在物理闭合/扰动阶段；后续应优先核查 D(R,O) ShadowHand controller、根坐标与碰撞初始状态，而不是只继续降低 contact chamfer。

## 口径说明

- Gym D(R,O)：依次施加六方向扰动后，最终位移不超过 0.02 m。
- Gym strict：要求每一个方向段的位移都不超过 0.02 m。
- Sim raw：无效仿真计入总试验数并按失败处理。
- Sim valid-only：排除验证器标记为 invalid 的试验，仅用于诊断仿真有效性影响。
- Gym 与 Sim 使用相同候选，但物理引擎实现、资产导入和控制细节仍不同；百分比差值描述 simulator gap，不直接归因于模型。
- 控制轨迹并非完全相同：D(R,O) Gym controller 使用 25% outward / 15% inward 调整；当前 Isaac Sim preparation 对 Barrett 使用 0% / 5%，对 ShadowHand 使用 10% / 20%。因此本报告是完整执行栈差异，而不是纯 PhysX 后端差异。
