# GOATBench 统一工作流实施状态

**日期**: 2026-06-28
**设计文档**: `docs/plans/2026-06-28-goatbench-unified-workflow-design.md`
**实现计划**: `docs/plans/2026-06-28-goatbench-unified-workflow-implementation.md`

---

## 总览

| 阶段 | 状态 | 完成度 |
|---|---|---|
| 代码实现（Task 1-10） | ✅ 完整 | 100% |
| Code Review 修复（C1-C3, I1/I4） | ✅ 完整 | 100% |
| 集成测试（Task 11-13） | 🚧 进行中 | 33% |

**commits**: 16 个（10 feat + 6 fix）

---

## 代码实现（Task 1-10）✅

| Task | 内容 | Commit | 状态 |
|---|---|---|---|
| 1 | 扩展 TwoTierState（9 新字段） | `0c9597c` | ✅ |
| 2 | 扩展 Resources（goal_type/goal_metadata） | `f7c1cf9` | ✅ |
| 3 | 新增 note_node（任务分类） | `0baed24` | ✅ |
| 4 | 新增 check_arrival_node（欧氏距离终止） | `85400f5` | ✅ |
| 5 | executor_node 追加目标位置观测 | `8fa6eb2` | ✅ |
| 6 | submit_node 分流（GOATBench/AEQA） | `f9e86c5` | ✅ |
| 7 | 新增 after_check_arrival 边 | `51d04c4` | ✅ |
| 8 | 改 graph.py 拓扑（12 nodes） | `585d152` | ✅ |
| 9 | 扩展 entrypoint（GOATBench 参数 + 线程） | `b9bba97` | ✅ |
| 10 | 改 runner 用 langgraph entrypoint | `9ea7a43` | ✅ |

### 图拓扑（最终）

```
START → note → init → build_context → planner → critic → loop_guard
                                                          ↓ submit_answer → submit → END
                                                          ↓ else → executor → check_arrival
                                                                              ↓ within → submit → END
                                                                              ↓ not within → memory_update → build_context
```

12 nodes，4 conditional edges（after_critic / after_guard / after_check_arrival / after_memory + after_submit）。

### State 扩展字段

- 任务分类：`task_type` / `task_plan` / `is_terminal_task`
- 跨 subtask：`subtask_index` / `subtask_total` / `cross_subtask_notes`
- GOATBench proximity：`observed_goal_positions`（last-writer-wins）/ `within_target` / `agent_target_distance`

### Resources 扩展字段

- `goal_type`（Optional[str]）
- `goal_metadata`（Optional[dict]，仅含 `goal_description` 文本，不含真值坐标）

---

## Code Review 修复 ✅

### Critical（3 项，全部修复）

| ID | 问题 | 修复 | Commit |
|---|---|---|---|
| C1 | `goal_description` = `str(subtask_goal)` 泄漏真值 `view_points` 坐标 | 改为从 `subtask_metadata["class"]` / `subtask_goal[0]["lang_desc"]` 提取人类可读描述 | `cf5a0d5` |
| C2 | object-type matcher 双向子串匹配过宽（`"a"` 匹配任意） | 改为严格等值（case-insensitive equality） | `de6c6f6` |
| C3 | `observed_goal_positions` 用 `operator.add` 累积历史位置，早期误匹配永久拉低 min dist | 改为 last-writer-wins（drop reducer） | `b793851` |

### Important（2 项，全部修复）

| ID | 问题 | 修复 | Commit |
|---|---|---|---|
| I1 | graph.py docstring 仍写 8-node 拓扑 | 更新为 12-node 拓扑 | `6135784` |
| I4 | note_node 冗余返回 `subtask_index`/`subtask_total`（entrypoint 已设） | 移除 note_node 的这两个字段 | `6135784` |

---

## 集成测试 🚧

### Task 11: AEQA 回归验证 🚧 进行中

**目标**: 确认 note/check_arrival 插入后 AEQA 行为不变（变量隔离）

**服务器验证**:
- ✅ 代码同步到 server 2（8.157.94.238:57249）
- ✅ import 链通过（3dmem env + .pth bridge）
- ✅ 图编译通过（12 nodes: `__start__`, note, init, build_context, planner, critic, loop_guard, executor, check_arrival, memory_update, stall_recovery, submit）
- ✅ GPU 状态确认（残留进程清理后 15.5GB free）
- 🚧 2-episode 回归运行中

**回归中发现的 bug 及修复**:

| Bug | 原因 | 修复 | Commit |
|---|---|---|---|
| `call_vlm` 404 | fallback 用 `OPENAI_BASE_URL`（`.../v1`，缺 `/chat/completions`） | fallback 改用 `QWEN_PLANNER_BASE_URL`（完整 URL） | `fda90b1` |
| `EvidenceNotebook is not JSON serializable` | `episode_payload = **result` 展开 `_notebook` 等非序列化字段 | 过滤 `_-prefixed` 字段 | `1eb66e5` |
| `ndarray is not JSON serializable` | `final_pts` 是 np.ndarray | entrypoint 返回时 `ndarray.tolist()` | `9121afb` |

**待验证**:
- AEQA 2-episode 跑通无异常
- 对比 baseline 结果（LLM Match / Acc / SPL 与 baseline_langgraph_aeqa 一致）

### Task 12: GOATBench 冒烟测试 ⏳ 待执行

**目标**: 跑 1 个 GOATBench episode，验证全链路

**验证点**:
- note_node 输出 `task_type="object_nav"` 等，`is_terminal_task=True`
- executor 追加 `observed_goal_positions`
- check_arrival 计算距离，within_target 在 < 1m 时为 True
- submit_node 走 GOATBench 分支，输出 `reached_goal`/`not_found`
- 跨 subtask: subtask 2 的 `cross_subtask_notes` 含 subtask 1 摘要

### Task 13: 全量 GOATBench 评估 ⏳ 待执行

**目标**: 全量跑 GOATBench，评分，对比 legacy baseline，关机

---

## 变量隔离保证

AEQA runner（`run_two_tier_aeqa_evaluation.py`）不传 GOATBench 参数 → entrypoint 走 AEQA 分支：

| 守卫 | AEQA 路径 | GOATBench 路径 |
|---|---|---|
| `note_node` | `goal_type=None` → `is_terminal_task=False` | `goal_type` truthy → `is_terminal_task=True` |
| `check_arrival_node` | `is_terminal_task=False` → 返回 `{}`（直通） | 计算欧氏距离 |
| `after_check_arrival` | `is_terminal_task=False` → 恒 `memory_update` | `within` → `submit` / else → `memory_update` |
| `executor_node` | `goal_type=None` → 跳过目标提取 | 从 `scene.objects` 提取匹配目标 |
| `submit_node` | `is_terminal_task=False` → 现有 AEQA 逻辑 | 输出 `reached_goal`/`not_found` |

---

## 目标位置来源约束

目标物体真实位置**只能来自观测反投影**，不能来自真值：

- executor_node 每步从 `scene.objects[oid]["bbox"].center` 提取匹配目标位置
- 匹配逻辑：object-type → 严格等值；description/image → CLIP 相似度
- `goal_metadata` 仅含 `goal_description` 文本，不含 `view_points` 等真值
- 真值 `viewpoints` 仅用于事后评分（`calc_agent_subtask_distance`），不进 workflow runtime
- 复用现有反投影代码（`scene_goatbench.py:596`），不新增

---

## 跨 subtask 记忆

GOATBench 允许跨 subtask 记忆。runner 线程以下资源跨 subtask：

| 资源 | 线程方式 |
|---|---|
| `scene` | 每 episode 创建一次，所有 subtask 共享 |
| `tsdf_planner` | 每 episode 创建一次，所有 subtask 共享 |
| `notebook` | entrypoint 返回 `_notebook`，runner 回收传给下一 subtask |
| `scene_graph` | entrypoint 返回 `_scene_graph`，runner 回收传给下一 subtask |
| `cross_subtask_notes` | submit_node 追加摘要（reached/missed + dist），下一 subtask 的 note_node 读取 |
| `pose` (pts/angle) | entrypoint 返回 `final_pts`/`final_angle`，runner 回收传给下一 subtask |

---

## 下一步

1. **等 AEQA 回归完成**（当前在跑，修复 final_pts 序列化后重跑）
2. **GOATBench 冒烟测试**（Task 12）
3. **全量 GOATBench 评估 + 评分**（Task 13）
4. **关机**
