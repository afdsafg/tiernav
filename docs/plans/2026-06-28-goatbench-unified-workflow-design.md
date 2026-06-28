# GOATBench 统一工作流设计文档

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 GOATBench 评估接入 LangGraph Two-Tier 工作流，以 AEQA 工作流为基础扩展，新增 note 节点做任务分类，新增 check_arrival 节点做几何终止判定，支持跨 subtask 记忆。

**Architecture:** GOATBench 是完整工作流，AEQA 是其子集。在现有 `src/two_tier_graph/` 8 节点图前插入 `note`（任务分类），在 `executor` 与 `memory_update` 之间插入 `check_arrival`（欧氏距离 < 1m 终止判定）。AEQA 路径行为不变（note 输出 question 类型，check_arrival 直通）。GOATBench runner 线程 scene/tsdf/notebook/scene_graph 跨 subtask，目标位置从观测反投影（复用 `scene.objects[oid]["bbox"].center`），不读真值。

**Tech Stack:** LangGraph StateGraph, habitat-sim, YOLO/SAM/CLIP, Qwen3-vl-flash API

---

## 背景与动机

GOATBench 与 AEQA 根本相似：无预建图进入场景探索完成任务。AEQA = 探索后回答问题；GOATBench = 探索后导航至目标物体 1m 内。AEQA 是 GOATBench 的初始阶段/子集。GOATBench 允许跨 subtask 记忆。

当前问题：`run_goatbench_two_tier_evaluation.py` 调用 legacy `run_episode_two_tier`（手写循环），未接入 LangGraph runtime；无任务分类机制。

## 完成条件差异

- **AEQA**: planner 主动 `submit_answer` 即终态，输出答案文本
- **GOATBench**: 终态 = 智能体位置距目标物体欧氏距离 < 1m（几何判定，不需答案文本）；planner 主动 submit = 认输/预算耗尽

## 目标位置来源约束

目标物体真实位置**只能来自观测反投影**（YOLO/GroundingDINO 识别 bbox 中心 → 世界坐标），**不能来自真值**。现有代码已实现：
- `scene_goatbench.py:596`: `np.linalg.norm(gobs["bbox"][idx].center[[0, 2]] - pts[[0, 2]])`
- `scene.objects[oid]["bbox"].center` 即反投影世界坐标
- `agent_tools.py:250,479` 同类用法

复用现有反投影，不新增代码。

## 状态扩展（`state.py`）

```python
class TwoTierState(TypedDict):
    # ... 现有字段 ...

    # ── 任务分类（note_node 设置一次）──
    task_type: str                    # "question" | "object_nav" | "description_nav" | "image_nav"
    task_plan: str                    # note 生成的自然语言计划
    is_terminal_task: bool            # False=回答(AEQA), True=导航(GOATBench)

    # ── 跨 subtask（GOATBench；AEQA 未用）──
    subtask_index: int
    subtask_total: int
    cross_subtask_notes: Annotated[list, operator.add]

    # ── GOATBench proximity（check_arrival 写入）──
    observed_goal_positions: Annotated[list, operator.add]  # executor 每步 append 匹配目标
    within_target: bool
    agent_target_distance: float
```

## 资源扩展（`resources.py`）

`Resources` 已有 `scene`/`tsdf_planner`/`notebook`/`scene_graph` 字段。新增：

```python
@dataclass
class Resources:
    # ... 现有字段 ...
    goal_type: Optional[str] = None          # "object"|"description"|"image"|None
    goal_metadata: Optional[dict] = None     # {"goal_description": str, ...}（不含真值位置）
```

## note 节点（新）

```python
def note_node(state, resources) -> dict:
    goal_type = resources.goal_type
    prior_notes = state.get("cross_subtask_notes", [])
    if goal_type is None:
        return {"task_type": "question", "task_plan": f"探索场景，收集证据，回答：{state['question']}",
                "is_terminal_task": False, "subtask_index": 0, "subtask_total": 1}
    type_map = {"object": "object_nav", "description": "description_nav", "image": "image_nav"}
    task_type = type_map.get(goal_type, "object_nav")
    prior_summary = "\n前序 subtask 已发现: " + "; ".join(prior_notes[-3:]) if prior_notes else ""
    return {"task_type": task_type,
            "task_plan": f"导航至 {goal_type} 目标: {state['question']}{prior_summary}",
            "is_terminal_task": True,
            "subtask_index": state.get("subtask_index", 0),
            "subtask_total": state.get("subtask_total", 1)}
```

Phase-1 不调 LLM —— 确定性分类器。未来 lever：LLM 拆解（Phase C/D）。

## check_arrival 节点（新）

```python
def check_arrival_node(state, resources) -> dict:
    if not state["is_terminal_task"]:
        return {}  # AEQA 直通
    observed = state.get("observed_goal_positions", [])
    if not observed:
        return {"within_target": False, "agent_target_distance": float("inf")}
    pts = state["pose"]["pts"]
    min_dist = min(float(np.linalg.norm(pts[[0,2]] - g[[0,2]])) for g in observed)
    return {"within_target": min_dist < 1.0, "agent_target_distance": min_dist}
```

## executor 节点改动

executor_node 每步执行后，从现有观测结果提取与 goal 描述匹配的物体位置，append 到 `observed_goal_positions`：
- `goal_type="object"` → class_name 字符串匹配
- `goal_type="description"/"image"` → CLIP 相似度（scene 已有 clip_model）
- 匹配物体 → `scene.objects[oid]["bbox"].center`（现成反投影世界坐标）

## submit 节点分流

```python
def submit_node(state, config) -> dict:
    if state.get("is_terminal_task"):
        within = state.get("within_target", False)
        return {"answer": "reached_goal", "success": within, "terminal": True,
                "cross_subtask_notes": [f"subtask {state['subtask_index']}: {'到达' if within else '未到达'} {state['task_type']} (dist={state.get('agent_target_distance',-1):.2f}m)"]}
    # else: 现有 AEQA 回答路径不变
```

## 图拓扑（`graph.py`）

```
START → note → init → build_context → planner → critic → loop_guard
                                                          ↓ submit_answer → submit → END
                                                          ↓ else → executor → check_arrival
                                                                              ↓ within → submit → END
                                                                              ↓ not within → memory_update → build_context
```

改动：
1. `START → init` 改为 `START → note → init`
2. `executor → memory_update` 改为 `executor → check_arrival`
3. 新增条件边 `after_check_arrival`: `within → submit`, `not_within → memory_update`

## entrypoint 分流

`run_episode_two_tier_langgraph` 新增参数：`scene`/`tsdf_planner`/`notebook`/`scene_graph`/`goal_type`/`goal_metadata`/`subtask_index`/`subtask_total`/`cross_subtask_notes`

- `scene is None` → AEQA 路径（现有行为，自建）
- `scene is not None` → GOATBench 路径（跳过构建，用注入的；notebook/scene_graph 若 None 则新建，否则复用）

返回值新增：`_notebook`/`_scene_graph`/`final_pts`/`final_angle`/`cross_subtask_notes`

## runner 改动

`run_goatbench_two_tier_evaluation.py`:
- 改用 `run_episode_two_tier_langgraph`（替代 legacy `run_episode_two_tier`）
- 每 episode 创建一次 scene + tsdf_planner
- 每 subtask 调用 entrypoint，线程 scene/tsdf/notebook/scene_graph/cross_subtask_notes
- `goal_metadata` 含 goal 描述文本（不含真值位置）；viewpoints 仅用于事后评分
- 最终评分仍用 `calc_agent_subtask_distance`（测地）+ snapshot match

## 变量隔离保证

AEQA runner（`run_two_tier_aeqa_evaluation.py`）不传新参数 → entrypoint 走 AEQA 分支 → note_node 输出 `task_type="question"` → check_arrival 直通 → 现有行为不变。
