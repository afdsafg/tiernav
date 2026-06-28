# GOATBench 统一工作流实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `src/two_tier_graph/` 上扩展，让 GOATBench 评估接入 LangGraph Two-Tier 工作流，AEQA 行为不变。

**Architecture:** 插入 note（任务分类）+ check_arrival（欧氏距离终止）两个节点，扩展 State/Resources/entrypoint，改 runner 线程跨 subtask 资源。详见 `docs/plans/2026-06-28-goatbench-unified-workflow-design.md`。

**Tech Stack:** LangGraph StateGraph, habitat-sim, YOLO/SAM/CLIP, Qwen3-vl-flash

---

## Task 1: 扩展 TwoTierState

**Files:**
- Modify: `src/two_tier_graph/state.py:44-123`

**Step 1: 编辑 state.py**

在 `TwoTierState` 末尾（`failure_type` 之后）追加：

```python
    # ── 任务分类（note_node 设置一次；AEQA 路径默认 question）──
    task_type: str                    # "question" | "object_nav" | "description_nav" | "image_nav"
    task_plan: str                    # note 生成的自然语言计划
    is_terminal_task: bool            # False=回答(AEQA), True=导航(GOATBench)

    # ── 跨 subtask（GOATBench；AEQA 未用）──
    subtask_index: int
    subtask_total: int
    cross_subtask_notes: Annotated[list, operator.add]

    # ── GOATBench proximity（check_arrival / executor 写入）──
    observed_goal_positions: Annotated[list, operator.add]  # executor 每步 append 匹配目标世界坐标
    within_target: bool
    agent_target_distance: float
```

**Step 2: 验证语法**

Run: `python -c "from src.two_tier_graph.state import TwoTierState; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add src/two_tier_graph/state.py
git commit -m "feat(state): add GOATBench task-classification + proximity fields to TwoTierState"
```

---

## Task 2: 扩展 Resources

**Files:**
- Modify: `src/two_tier_graph/resources.py:16-43`

**Step 1: 编辑 resources.py**

在 `Resources` dataclass 末尾（`output_dir` 之后）追加：

```python
    # ── GOATBench 任务上下文（AEQA 路径为 None）──
    goal_type: Optional[str] = None          # "object"|"description"|"image"|None
    goal_metadata: Optional[dict] = None     # {"goal_description": str, ...}（不含真值位置）
```

注意：`scene`/`tsdf_planner`/`notebook`/`scene_graph` 字段已存在（行 21-29），无需新增。

**Step 2: 验证语法**

Run: `python -c "from src.two_tier_graph.resources import Resources; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add src/two_tier_graph/resources.py
git commit -m "feat(resources): add goal_type/goal_metadata to Resources for GOATBench"
```

---

## Task 3: 新增 note_node

**Files:**
- Modify: `src/two_tier_graph/nodes.py`（在 `init_node` 之前插入）

**Step 1: 编辑 nodes.py**

在 `init_node` 定义之前（约行 276 前）插入：

```python
def note_node(state: TwoTierState, config) -> dict:
    """Node 0: 任务分类与拆解。

    AEQA 路径（goal_type=None）：输出 task_type="question"，is_terminal_task=False。
    GOATBench 路径：按 goal_type 分类，结合 cross_subtask_notes 生成计划。

    Phase-1 确定性分类器，不调 LLM。未来 lever：LLM 拆解（Phase C/D）。

    Reads: question, cross_subtask_notes (从 resources 读 goal_type)。
    Writes: task_type, task_plan, is_terminal_task, subtask_index, subtask_total。
    """
    res: Resources = config["configurable"]["resources"]
    goal_type = res.goal_type
    prior_notes = state.get("cross_subtask_notes", [])

    if goal_type is None:
        # AEQA 路径
        return {
            "task_type": "question",
            "task_plan": f"Explore scene, gather evidence, answer: {state['question']}",
            "is_terminal_task": False,
            "subtask_index": 0,
            "subtask_total": 1,
        }

    # GOATBench 路径
    type_map = {
        "object": "object_nav",
        "description": "description_nav",
        "image": "image_nav",
    }
    task_type = type_map.get(goal_type, "object_nav")

    prior_summary = ""
    if prior_notes:
        prior_summary = "\nPrior subtasks found: " + "; ".join(prior_notes[-3:])

    return {
        "task_type": task_type,
        "task_plan": f"Navigate to {goal_type} target: {state['question']}{prior_summary}",
        "is_terminal_task": True,
        "subtask_index": state.get("subtask_index", 0),
        "subtask_total": state.get("subtask_total", 1),
    }
```

**Step 2: 验证语法**

Run: `python -c "from src.two_tier_graph.nodes import note_node; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add src/two_tier_graph/nodes.py
git commit -m "feat(nodes): add note_node for task classification (AEQA vs GOATBench)"
```

---

## Task 4: 新增 check_arrival_node

**Files:**
- Modify: `src/two_tier_graph/nodes.py`（在 `executor_node` 之后插入）

**Step 1: 编辑 nodes.py**

在 `executor_node` 定义之后（约行 710 后）插入：

```python
def check_arrival_node(state: TwoTierState, config) -> dict:
    """Node 5b: GOATBench 几何终止判定。

    AEQA 路径（is_terminal_task=False）：返回空 dict，直通 memory_update。
    GOATBench 路径：用 observed_goal_positions（观测反投影，非真值）计算
    欧氏距离（xz 平面），< 1m 则 within_target=True。

    目标位置来源：executor_node 每步从 scene.objects[oid]["bbox"].center
    提取匹配目标位置 append 到 observed_goal_positions。

    Reads: is_terminal_task, observed_goal_positions, pose。
    Writes: within_target, agent_target_distance。
    """
    import numpy as np

    if not state.get("is_terminal_task", False):
        return {}  # AEQA 直通

    observed = state.get("observed_goal_positions", [])
    if not observed:
        return {"within_target": False, "agent_target_distance": float("inf")}

    pts = state["pose"]["pts"]
    if pts is None:
        return {"within_target": False, "agent_target_distance": float("inf")}

    # 欧氏距离，xz 平面（忽略 y 轴高度差）
    min_dist = min(
        float(np.linalg.norm(np.asarray(pts)[[0, 2]] - np.asarray(g)[[0, 2]]))
        for g in observed
    )

    return {
        "within_target": min_dist < 1.0,
        "agent_target_distance": min_dist,
    }
```

**Step 2: 验证语法**

Run: `python -c "from src.two_tier_graph.nodes import check_arrival_node; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add src/two_tier_graph/nodes.py
git commit -m "feat(nodes): add check_arrival_node for GOATBench geometric termination"
```

---

## Task 5: executor_node 追加目标位置观测

**Files:**
- Modify: `src/two_tier_graph/nodes.py:684-709`（executor_node）

**Step 1: 读现有 executor_node + scene.objects 结构**

Run: `grep -n "bbox.*center" src/scene_goatbench.py src/agent_tools.py src/agent_executor.py | head -20`

确认 `scene.objects[oid]["bbox"].center` 返回 np.ndarray [x,y,z]。

**Step 2: 编辑 executor_node**

在 `executor_node` 返回 dict 之前，插入目标位置提取逻辑：

```python
    # ── GOATBench: 从观测提取匹配目标位置（反投影，非真值）──
    observed_goal_positions = []
    if res.goal_type is not None:
        goal_desc = (res.goal_metadata or {}).get("goal_description", "")
        scene = res.scene
        if scene is not None and hasattr(scene, "objects") and goal_desc:
            import numpy as np
            try:
                from src.agent_tools import clip_text_similarity
                clip_model = res.models.get("clip_model")
                for oid, obj in scene.objects.items():
                    if not isinstance(obj, dict) or "bbox" not in obj or obj["bbox"] is None:
                        continue
                    if not hasattr(obj["bbox"], "center"):
                        continue
                    class_name = obj.get("class_name", "")
                    matched = False
                    if res.goal_type == "object":
                        # 字符串匹配（容错：小写包含）
                        matched = goal_desc.lower() in class_name.lower() or class_name.lower() in goal_desc.lower()
                    elif clip_model is not None:
                        # CLIP 相似度（description/image）
                        sim = clip_text_similarity(clip_model, class_name, goal_desc)
                        matched = sim > 0.25  # 阈值，实现时调参
                    if matched:
                        observed_goal_positions.append(obj["bbox"].center.copy())
            except Exception as e:
                logger.warning(f"goal position extraction failed: {e}")
```

在返回 dict 中追加：

```python
    return {
        "last_evidence": evidence,
        "pose": {"pts": res.executor._pts, "angle": float(res.executor._angle)},
        "steps_taken": steps_taken,
        "action_history": [action.action_type],
        "observed_goal_positions": observed_goal_positions,  # append via reducer
    }
```

注意：`clip_text_similarity` 若不存在，实现时改用 `res.models["clip_model"]` 直接算余弦相似度。具体函数名实现时 grep 确认。

**Step 3: 验证语法**

Run: `python -c "from src.two_tier_graph.nodes import executor_node; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add src/two_tier_graph/nodes.py
git commit -m "feat(nodes): executor extracts observed goal positions for GOATBench proximity check"
```

---

## Task 6: submit_node 分流

**Files:**
- Modify: `src/two_tier_graph/nodes.py:933-1020`（submit_node）

**Step 1: 编辑 submit_node**

在 `submit_node` 函数开头（`res = config[...]` 之前）插入 GOATBench 分流：

```python
def submit_node(state: TwoTierState, config) -> dict:
    # ── GOATBench 分流：导航任务终态由 proximity 判定 ──
    if state.get("is_terminal_task", False):
        within = state.get("within_target", False)
        dist = state.get("agent_target_distance", -1.0)
        res_sub: Resources = config["configurable"]["resources"]
        from src.agent_tools import silent_perception_step
        steps_taken = int(getattr(silent_perception_step, "_step_counter", 0))
        logger.info(f"GOATBench subtask {state.get('subtask_index', 0)} terminal: within={within} dist={dist:.2f}m")
        note_summary = f"subtask {state.get('subtask_index', 0)}: {'reached' if within else 'missed'} {state.get('task_type', '')} (dist={dist:.2f}m)"
        if res_sub.run_logger is not None:
            if res_sub.scene_graph is not None:
                try:
                    res_sub.run_logger.save_graph(res_sub.question_id, res_sub.scene_graph.to_dict())
                except Exception:
                    pass
            res_sub.run_logger.finalize_episode(
                episode_id=res_sub.question_id, success=within,
                answer="reached_goal" if within else "not_found",
                num_steps=int(steps_taken),
            )
        return {
            "answer": "reached_goal" if within else "not_found",
            "success": within,
            "steps_taken": steps_taken,
            "rounds_used": state["rounds_used"],
            "terminal": True,
            "failure_type": "" if within else "target_not_reached",
            "cross_subtask_notes": [note_summary],
        }

    # ── AEQA 路径（现有逻辑不变）──
    res: Resources = config["configurable"]["resources"]
    action: PlannerAction = state["current_action"]
    # ... 现有代码不变 ...
```

**Step 2: 验证语法**

Run: `python -c "from src.two_tier_graph.nodes import submit_node; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add src/two_tier_graph/nodes.py
git commit -m "feat(nodes): submit_node branches on is_terminal_task for GOATBench"
```

---

## Task 7: 新增 after_check_arrival 边

**Files:**
- Modify: `src/two_tier_graph/edges.py`

**Step 1: 编辑 edges.py**

在文件末尾追加：

```python
def after_check_arrival(state: TwoTierState) -> str:
    """Conditional edge after check_arrival_node.

    GOATBench: within_target=True → "submit" (成功终止)
               within_target=False → "memory_update" (继续探索)
    AEQA: is_terminal_task=False → 恒 "memory_update"（check_arrival 返回空 dict）
    """
    if state.get("is_terminal_task", False) and state.get("within_target", False):
        return "submit"
    return "memory_update"
```

**Step 2: 验证语法**

Run: `python -c "from src.two_tier_graph.edges import after_check_arrival; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add src/two_tier_graph/edges.py
git commit -m "feat(edges): add after_check_arrival conditional edge"
```

---

## Task 8: 改 graph.py 拓扑

**Files:**
- Modify: `src/two_tier_graph/graph.py:30-131`

**Step 1: 编辑 imports**

```python
from .edges import after_check_arrival, after_critic, after_guard, after_memory, after_submit
from .nodes import (
    build_context_node,
    check_arrival_node,
    critic_node,
    executor_node,
    init_node,
    loop_guard_node,
    memory_update_node,
    note_node,
    planner_node,
    stall_recovery_node,
    submit_node,
)
```

**Step 2: 编辑 build_two_tier_graph**

在节点列表中添加（`init` 之前）：

```python
    g.add_node("note", note_node)
    g.add_node("check_arrival", check_arrival_node)
```

静态边改动：

```python
    # START → note → init（原 START → init）
    g.add_edge(START, "note")
    g.add_edge("note", "init")

    # executor → check_arrival → memory_update（原 executor → memory_update）
    g.add_edge("executor", "check_arrival")
    # 删除原 g.add_edge("executor", "memory_update")
```

新增条件边（在 after_memory 边之后）：

```python
    # ── Conditional edge: after_check_arrival ──
    # GOATBench: within_target → submit (终止); 否则 → memory_update (继续)
    # AEQA: is_terminal_task=False → 恒 memory_update
    g.add_conditional_edges(
        "check_arrival",
        after_check_arrival,
        {
            "submit": "submit",
            "memory_update": "memory_update",
        },
    )
```

**Step 3: 验证图编译**

Run: `python -c "from src.two_tier_graph.graph import build_two_tier_graph; g = build_two_tier_graph(); print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add src/two_tier_graph/graph.py
git commit -m "feat(graph): insert note + check_arrival nodes, wire GOATBench termination path"
```

---

## Task 9: 扩展 entrypoint

**Files:**
- Modify: `src/two_tier_graph/entrypoint.py`

**Step 1: 编辑函数签名**

`run_episode_two_tier_langgraph` 新增参数（在现有参数之后）：

```python
def run_episode_two_tier_langgraph(
    scene_id: str,
    question: str,
    question_id: str,
    cfg,
    detection_model,
    sam_predictor,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    output_dir: str = "/root/MyAgent/results/hmge",
    max_planner_rounds: int = 10,
    max_total_steps: int = 50,
    start_pts: Optional[np.ndarray] = None,
    start_angle: float = 0.0,
    run_logger=None,
    method_config: Optional[dict] = None,
    # ── GOATBench 新增（AEQA 路径不传，走默认值）──
    scene=None,
    tsdf_planner=None,
    notebook=None,
    scene_graph=None,
    goal_type: Optional[str] = None,
    goal_metadata: Optional[dict] = None,
    subtask_index: int = 0,
    subtask_total: int = 1,
    cross_subtask_notes: Optional[list] = None,
) -> Dict:
```

**Step 2: 编辑初始化分流**

在 scene 构建段（约 `scene = None` / `try:` 处），改为分流：

```python
    # ── Build Resources (mirrors :1160-1226) ──
    scene_built = scene  # None for AEQA, pre-built for GOATBench
    try:
        # ... 现有 silent_perception_step / cfg 处理 ...

        if scene_built is None:
            # AEQA 路径：自建 scene（现有逻辑）
            from src.scene_aeqa import Scene
            scene_built = Scene(...)
            # ... 现有 start_pts / tsdf_planner 构建 ...
        else:
            # GOATBench 路径：用注入的 scene + tsdf_planner
            scene = scene_built
            # notebook / scene_graph: None 则新建，否则复用
            if notebook is not None:
                # 复用跨 subtask
                pass
            # tsdf_planner 已传入
```

具体实现时读现有 entrypoint 的 scene 构建块，把 `scene = Scene(...)` 包在 `if scene_built is None:` 里。

**Step 3: 编辑 Resources 构造**

在 `Resources(...)` 构造时填入新字段：

```python
        resources = Resources(
            scene=scene_built,
            tsdf_planner=tsdf_planner_built,
            # ... 现有字段 ...
            goal_type=goal_type,
            goal_metadata=goal_metadata,
        )
```

**Step 4: 编辑 initial_state**

在 `initial_state: TwoTierState` dict 中追加：

```python
            "task_type": "",  # note_node 会设置
            "task_plan": "",
            "is_terminal_task": False,
            "subtask_index": subtask_index,
            "subtask_total": subtask_total,
            "cross_subtask_notes": cross_subtask_notes or [],
            "observed_goal_positions": [],
            "within_target": False,
            "agent_target_distance": float("inf"),
```

**Step 5: 编辑返回值**

在终态 state → result dict 映射处，追加：

```python
        result["_notebook"] = resources.notebook
        result["_scene_graph"] = resources.scene_graph
        result["final_pts"] = final_state.get("pose", {}).get("pts")
        result["final_angle"] = final_state.get("pose", {}).get("angle", 0.0)
        result["cross_subtask_notes"] = final_state.get("cross_subtask_notes", cross_subtask_notes or [])
```

**Step 6: 验证语法**

Run: `python -c "from src.two_tier_graph.entrypoint import run_episode_two_tier_langgraph; print('ok')"`
Expected: `ok`

**Step 7: Commit**

```bash
git add src/two_tier_graph/entrypoint.py
git commit -m "feat(entrypoint): add GOATBench params + resource threading + return fields"
```

---

## Task 10: 改 runner 用 langgraph entrypoint

**Files:**
- Modify: `run_goatbench_two_tier_evaluation.py`

**Step 1: 改 import**

```python
# 替换
from src.agent_workflow import run_episode_two_tier
# 为
from src.two_tier_graph.entrypoint import run_episode_two_tier_langgraph
```

**Step 2: 改 subtask 循环**

在 `run_goatbench_episode` 函数内，改调用：

```python
    # 跨 subtask 线程变量（episode 级）
    threaded_notebook = None
    threaded_scene_graph = None
    cross_subtask_notes = []

    for subtask_idx, (goal_type, subtask_goal) in enumerate(
        zip(all_subtask_goal_types, all_subtask_goals)
    ):
        subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
        # ... 现有 init_subtask / goal_question 构建 ...

        goal_metadata = {
            "goal_description": subtask_goal if isinstance(subtask_goal, str) else str(subtask_goal),
            "goal_type": goal_type,
        }

        result = run_episode_two_tier_langgraph(
            scene_id=scene_id,
            question=goal_question,
            question_id=subtask_id,
            cfg=cfg,
            detection_model=detection_model,
            sam_predictor=sam_predictor,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
            output_dir=cfg.output_dir,
            max_planner_rounds=cfg.get("max_planner_rounds", 10),
            max_total_steps=cfg.get("max_total_steps", 50),
            start_pts=pts,
            start_angle=angle,
            run_logger=logger,
            method_config={"use_notebook": True, "use_scene_graph": True, "use_active_query": True, "use_rejected_tracking": True},
            scene=scene,                 # 线程
            tsdf_planner=tsdf_planner,   # 线程
            notebook=threaded_notebook,  # 线程（首 subtask None）
            scene_graph=threaded_scene_graph,  # 线程
            goal_type=goal_type,
            goal_metadata=goal_metadata,
            subtask_index=subtask_idx,
            subtask_total=len(all_subtask_goals),
            cross_subtask_notes=cross_subtask_notes,
        )

        # 回收线程资源供下一 subtask
        threaded_notebook = result.get("_notebook", threaded_notebook)
        threaded_scene_graph = result.get("_scene_graph", threaded_scene_graph)
        cross_subtask_notes = result.get("cross_subtask_notes", cross_subtask_notes)
        pts = result.get("final_pts", pts)
        angle = result.get("final_angle", angle)

        # ... 现有评分逻辑（calc_agent_subtask_distance / snapshot match）不变 ...
```

**Step 3: 验证语法**

Run: `python -c "import ast; ast.parse(open('run_goatbench_two_tier_evaluation.py').read()); print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add run_goatbench_two_tier_evaluation.py
git commit -m "feat(runner): GOATBench runner uses langgraph entrypoint with cross-subtask threading"
```

---

## Task 11: AEQA 回归验证（变量隔离）

**Files:**
- Test: 手动跑 1-2 个 AEQA episode 对比

**Step 1: 跑 AEQA langgraph 1 episode**

在 server 2 上：
```bash
cd /root/tiernav
conda activate langgraph
python run_two_tier_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --engine langgraph --split 1 --start_ratio 0.0 --end_ratio 0.02
```

**Step 2: 确认 note_node 输出 task_type="question"**

检查日志含 `task_type="question"`，无 `is_terminal_task=True`，check_arrival 直通。

**Step 3: 对比 baseline 结果**

确认 LLM Match / Acc / SPL 与 baseline_langgraph_aeqa 一致（变量隔离：AEQA 行为不变）。

**Step 4: 若不一致，回滚检查**

若 AEQA 结果变化，问题在 note/check_arrival 插入。检查：
- note_node 是否误设 is_terminal_task
- check_arrival 是否误返 within_target
- executor_node 的 goal_type 判断是否影响 AEQA（goal_type=None 应跳过）

**Step 5: Commit（若需修复）**

```bash
git add -A
git commit -m "fix: preserve AEQA behavior after note/check_arrival insertion"
```

---

## Task 12: GOATBench 冒烟测试

**Files:**
- Test: 手动跑 1 个 GOATBench episode

**Step 1: 同步代码到 server 2**

```bash
scp -r src/two_tier_graph/ root@8.157.94.238:/root/tiernav/src/
scp run_goatbench_two_tier_evaluation.py root@8.157.94.238:/root/tiernav/
```

**Step 2: 跑 1 episode**

```bash
ssh root@8.157.94.238
cd /root/tiernav && conda activate langgraph
python run_goatbench_two_tier_evaluation.py -cf cfg/eval_goatbench.yaml --split 1 --start_ratio 0.0 --end_ratio 0.02
```

**Step 3: 验证日志**

确认：
- note_node 输出 `task_type="object_nav"` 等，`is_terminal_task=True`
- executor 追加 `observed_goal_positions`
- check_arrival 计算距离，within_target 在 < 1m 时为 True
- submit_node 走 GOATBench 分支，输出 `reached_goal`/`not_found`
- 跨 subtask: subtask 2 的 cross_subtask_notes 含 subtask 1 摘要

**Step 4: 修复问题**

按日志修复（目标匹配阈值、反投影坐标、线程资源回收等）。

**Step 5: Commit**

```bash
git add -A
git commit -m "fix: GOATBench smoke test fixes"
```

---

## Task 13: 全量 GOATBench 评估

**Step 1: 跑全量 split**

```bash
python run_goatbench_two_tier_evaluation.py -cf cfg/eval_goatbench.yaml --start_ratio 0.0 --end_ratio 1.0
```

**Step 2: 评分**

用现有 `calc_agent_subtask_distance`（测地）+ snapshot match 评分流程。

**Step 3: 对比 legacy baseline**

与之前 GOATBench legacy baseline 结果对比 SR / SPL。

**Step 4: 关机**

```bash
ssh root@8.157.94.238 "shutdown -h now"
```

**Step 5: 记录结果到 plan**

更新 `docs/plans/2026-06-28-tiernav-implementation-plan.md` B3 状态。
