# Claude Code Patterns for tiernav — Design

> 状态: Draft
> 日期: 2026-07-01
> 基线: main @ 014cde8
> 策略: 成功率优先（bundle changes，attribute decomposition deferred）
> 参考: `claude-code-analysis/` 项目对 Claude Code 泄露源码的静态分析

## 背景与目标

tiernav 已完成 LangGraph runtime redesign（014cde8），核心 runtime 稳定，178/177 tests passed。GOATBench smoke 验证通过（scene.objects 非空、room segmentation 生效、path_length 非零），但成功率仍低，工作流和 prompt 有进一步优化空间。

本设计研读 `claude-code-analysis/` 项目，提炼 Claude Code 在 context 工程、prompt 工程、memory 体系、multi-agent 协同上的工程设计，剪裁到 VLN benchmark 场景，分四阶段落地。

## Claude Code 的关键设计（借鉴面）

1. **分层 memory 体系** — Auto / Session / Agent / Team 四层分治，每层独立目录、独立更新策略、独立 scope。tiernav 当前只有 `MemorySession` + `SceneGraphMemory` 一层。
2. **Relevant Memory Recall** — 不全塞 prompt，用轻量选择器按 manifest 选 ≤5 个文件回灌。tiernav 的 `scene_graph_memory` 是全量 `get_summary_for_planner`。
3. **Context section 化 + cache boundary** — 静态段前置 + `DYNAMIC_BOUNDARY` + 动态段，section 级缓存。tiernav 已有 `cacheable=True/False` 分组，但无 boundary 标记和 section 缓存失效逻辑。
4. **Prompt 作为 runtime** — 6 层组装（override/coordinator/agent/custom/append + userContext/systemContext），专项 prompt 家族（compact/memory extraction）独立协议化。tiernav 的 prompt 全硬编码在 `ContextCompiler._render_task_instruction`。
5. **Compact 状态复灌** — 压缩后重建 FileAttachments / Plan / Deferred Tools 声明。tiernav 无 compact 机制，rounds 累积靠 `max_rounds` 截断。
6. **Multi-agent 三层** — subagent / coordinator / swarm teammate。tiernav 是单 planner 单轮。
7. **Session Memory forked subagent** — 后台沙箱化摘要代理，只允许 Edit 单文件。tiernav 无此层。
8. **Prompt 可观测** — `dump-prompts` JSONL + `/context` token 分析。tiernav 有 `event_log` 但无 prompt 级审计。

## 阶段总览

| 阶段 | 方向 | 依赖 | 落地批次 |
|------|------|------|----------|
| 1 | 可观测 + Prompt 骨架 | 无 | 1 批，4 项全做 |
| 2 | Memory 分层 + 模型驱动 Recall | 阶段 1 | 1 批 |
| 3 | Prompt 工程深化 + Compact 状态复灌 | 阶段 2 | 1 批 |
| 4 | Multi-agent（可选） | 阶段 1-3 稳定 | 仅在 SR 未达标时启动 |

策略：成功率优先。每阶段内部 bundle changes，不强制单变量隔离。每阶段结束跑 GOATBench benchmark 验证回归/提升。

---

## 阶段 1：可观测 + Prompt 骨架

四项改动全落在 `src/tiernav_runtime/` 现有模块，不新增顶层包。

### 1.1 文件变更

```
src/tiernav_runtime/
  contracts.py      + ContextSection.cache_break: bool = False
  context.py        + boundary 标记 + 模板外置钩子
  events.py         + PromptAuditEvent 事件类型
  recorder.py       + write_prompt_audit_jsonl()
  __init__.py       + dump_context_tokens() CLI 入口
  prompts/          (新目录)
    task_instruction.py   外置的策略文本段
    __init__.py
```

### 1.2 cache boundary（方案 B：字段标记）

`ContextSection` 加 `cache_break: bool = False`。

`ContextCompiler.compile()` 里：
- `task_instruction` / `action_schema` / `memory_index`：`cacheable=True, cache_break=False`
- `task_state`（第一个动态段）：`cacheable=False, cache_break=True` ← 边界
- 其余动态段：`cacheable=False, cache_break=False`

下游遍历 sections，第一个 `cache_break=True` 即为 cache 边界。无虚拟 section，`render_prompt` 无需特判。

### 1.3 JSONL 审计（含 content 全文）

新增 `PromptAuditEvent`，每轮 compile 后写一行到 `<output_dir>/prompt_audit/<episode_id>.jsonl`：

```json
{
  "round": 0,
  "step": 3,
  "sections": [
    {
      "name": "task_instruction",
      "content": "episode_id: ...\\nYou are a navigation planner...",
      "hash": "a1b2...",
      "tokens": 87,
      "cacheable": true,
      "cache_break": false
    },
    {
      "name": "task_state",
      "content": "continuous_context: enabled...",
      "hash": "c3d4...",
      "tokens": 24,
      "cacheable": false,
      "cache_break": true
    }
  ]
}
```

连 content 一起记，方便复盘时直接看完整 prompt，不用回查 event_log。

### 1.4 token 分析 CLI

`dump_context_tokens(episode_id, output_dir)` 读 JSONL，打印各 section token 占比表：

```
section                avg_tokens  pct   cacheable
task_instruction              87   38%   yes
action_schema                 45   20%   yes
memory_index                  12    5%   yes
task_state                    24   11%   no  ← boundary
recent_trace                   8    3%   no
current_observation           30   13%   no
...
```

### 1.5 prompt 模板外置

`_render_task_instruction` 的策略文本（"You are a navigation planner..." 那段）抽成 `prompts/task_instruction.py` 的 `STRATEGY_TEXT` 常量。compiler 调用常量拼装。语义零变化，只是文本不再硬编码在 compiler 里。

### 1.6 验证

- 178 tests 回归通过
- GOATBench smoke：planner 行为不变，输出格式不变
- `prompt_audit/<episode_id>.jsonl` 生成，含完整 content
- `dump_context_tokens` 输出各 section token 占比

---

## 阶段 2：Memory 分层 + 模型驱动 Recall

### 2.1 两层 memory scope

```
EpisodeMemory (对应 Claude Code Session Memory)
  - 生命周期: 单 episode 内
  - 内容: 当前 episode 的 trace 摘要、失败动作、已访问 room
  - 存储: 内存中（复用已有 MemorySession）
  - 更新: 每轮追加

SceneMemory (对应 Claude Code Agent Memory)
  - 生命周期: 跨 episode，同 scene 共享
  - 内容: 该 scene 的 room 拓扑、object 位置、导航经验
  - 存储: 本地文件 <output_dir>/scene_memory/<scene_id>.json
  - 更新: episode 结束时沉淀
```

`SceneGraphMemory` 不再每轮全量灌 prompt。降级为 SceneMemory 的底层数据源；planner 看到的是 manifest + recall 后的摘要。

### 2.2 SceneMemory 结构化存储

`<output_dir>/scene_memory/<scene_id>.json` 结构：

```json
{
  "scene_id": "scene_001",
  "rooms": {
    "kitchen": {
      "room_id": "kitchen",
      "visited_rounds": [0, 3],
      "objects_seen": ["chair", "table", "fridge"],
      "object_details": {
        "chair": {"count": 2, "poses": [...], "first_seen_round": 0}
      },
      "connectivity": ["living_room", "bathroom"],
      "notes": "fridge was open"
    }
  },
  "episodic_notes": [
    {"round": 0, "room": "kitchen", "event": "entered, found fridge"},
    {"round": 3, "room": "kitchen", "event": "revisited, fridge closed"}
  ],
  "last_updated": "2026-07-01T..."
}
```

结构清晰、可序列化、可增量更新。每个 room 是独立节点，object 按类聚合，episodic_notes 是时间线。

### 2.3 Manifest（给 planner/模型看目录）

`SceneGraphMemory.get_manifest()` 返回结构清单（不含详情）：

```
rooms: kitchen(visited), bedroom(visited), office(unvisited), bathroom(unvisited), living_room(visited)
objects: chair x4, table x2, fridge x1, bed x1, ...
episodic_notes: 4 entries (rounds 0,1,2,3)
```

manifest 全量进 prompt（体积小、稳定），让模型知道"记忆里有什么"。

### 2.4 模型驱动 Recall

recall 选择权交给模型，程序只负责结构化存储 + 提供 manifest。

recall 本身是一次 LLM 调用（现阶段复用 planner 同款模型，后续可抽独立轻量模型）。两种触发：

**a) subtask 开始自动 recall**

runtime 调用 recall model，输入：
- goal_description / question
- manifest（记忆目录）
- current room

recall model 输出要提取的节点列表：
```json
{"recall": [
  {"type": "room", "id": "kitchen", "reason": "goal mentions food"},
  {"type": "object", "id": "fridge", "reason": "likely food storage"}
]}
```

runtime 据此从 SceneMemory 取详情，注入 `scene_graph_memory` section。

**b) planner 主动 recall 工具**

`query_scene_memory(query: str)` —— planner 发自然语言 query，runtime 内部走同样的 recall model 逻辑，返回匹配节点详情。

prompt 约束（写进 task_instruction）：
```
scene_graph_memory 已含本 subtask 初始 recall。
仅在认为记忆中已有 goal/answer 相关线索但未在 prompt 显示时，调用 query_scene_memory。
已 recall 的内容会持续保留，勿重复查相同内容。
减少不必要的 recall 频率。
```

### 2.5 Compact 雏形

当 `round_index >= COMPACT_THRESHOLD`（如 5）：
- 早期轮（0 到 round-K）的 trace 压缩成单条 summary
- 保留最近 K 轮原文
- summary 注入 `memory_index` section（cacheable=False，因为每轮变）
- `available_targets` / `scene_graph_memory` / `tool_feedback` 在 compact 后照常重建（对应 Claude Code 的状态复灌）

### 2.6 不做的事

- 不做 Claude Code 的 forked subagent 摘要——benchmark 单线程足够，fork agent 引入 LLM 调用开销和不确定性
- 不做 Team Memory——benchmark 场景无团队共享需求
- 不做 Agent Memory Snapshot——scene_memory 文件本身已是可分发资产

### 2.7 验证

- 178 tests 回归通过
- GOATBench smoke：`scene_graph_memory` section 不再全量，只含 manifest + recall 结果
- `scene_memory/<scene_id>.json` 生成且结构正确
- `query_scene_memory` 工具可调用
- prompt token 占比：`scene_graph_memory` section 显著下降

---

## 阶段 3：Prompt 工程深化 + Compact 状态复灌

### 3.1 专项 prompt 家族

当前 `_render_task_instruction` 一套文本打天下。借鉴 Claude Code 的 compact/memory-extraction 专项 prompt，按阶段拆分：

```
prompts/
  task_instruction.py        # 通用骨架（身份 + 输出格式）
  strategy_explore.py        # explore 阶段策略
  strategy_navigate.py       # navigate 阶段策略
  strategy_submit.py         # submit 阶段策略
```

阶段判断依据 `state`：当前 room 是否含 goal object、是否有未访问 frontier、round 剩余量。`ContextCompiler` 按阶段选 strategy 段拼到 task_instruction 后。

**explore 阶段策略**（无 goal object 可见、有未访问 frontier）：
```
当前无 goal object 可见。优先 explore_frontier 扩展已知区域，
注意 available_targets 中的未访问 room。每轮换一个 frontier，勿重复。
```

**navigate 阶段策略**（goal object 可见或有线索）：
```
goal object 已在 scene_graph_memory 或 current_observation 出现。
调用 navigate_to_object 接近目标。到达后若距离满足，准备 submit。
```

**submit 阶段策略**（距离达标或 round 将尽）：
```
已接近目标或预算将尽。验证当前观测与 goal 一致后 submit_answer。
若 task_mode=question_answering，基于已观测信息给出答案。
```

### 3.2 Task classification note node

graph 里加一个轻量判断节点，在 planner 节点前：
- 输入：`EpisodeRequest.task_mode` + `GoalSpec.goal_type`
- 输出：`task_phase_hint`（explore / navigate / submit 的初始倾向）
- 不调用 LLM，纯规则判断，结果注入 `policy_hint` section

GOATBench goal-navigation 倾向 explore→navigate→submit；AEQA question-answering 倾向 explore→submit。

### 3.3 Compact 状态复灌

阶段 2 已有 compact 雏形（round ≥ 阈值压缩早期 trace）。阶段 3 补状态复灌：

compact 后新 prompt 必须重建这些动态 section，不能只留 summary：
- `available_targets` —— 重新从 tsdf_planner / scene 渲染
- `scene_graph_memory` —— 重新 recall（用 compact 后的 manifest）
- `tool_feedback` —— 清空（compact 是新起点，旧 feedback 不再相关）
- `current_observation` —— 保留 compact 前最后一轮的观测

对应 Claude Code compact 后重建 FileAttachments / Tools / Plan 的逻辑。

### 3.4 验证

- 178 tests 回归通过
- GOATBench smoke：不同阶段 prompt 文本正确切换
- task classification node 输出正确 phase hint
- compact 后 `available_targets` / `scene_graph_memory` 重建，`tool_feedback` 清空
- prompt_audit JSONL 可见 compact 事件前后的 section 变化

---

## 阶段 4：Multi-agent（可选，最后）

前三阶段稳定后才动。借鉴 Claude Code 的 coordinator/worker，剪裁到 VLN benchmark。

### 4.1 单 planner 现状的问题

单 planner 同时承担：观察决策、路径规划、记忆管理、终止判断。职责过载，prompt 膨胀，决策互相干扰。

### 4.2 coordinator/worker 拓扑

```
Coordinator (主 planner)
  ├─ ExploreWorker  — 纯观察，explore_panorama / explore_frontier
  ├─ NavigateWorker — 纯移动，navigate_to_object
  └─ (Coordinator 自己做 submit_answer + 记忆调度)
```

**分工**：
- Coordinator 看 manifest + scene_graph_memory + available_targets，决定"下一步该 explore 还是 navigate"
- ExploreWorker 接到指令后执行 explore，返回观测摘要（不决策）
- NavigateWorker 接到 object_name 后执行 navigate，返回路径结果 + 是否到达
- Coordinator 收到 worker 结果后更新记忆，决定下一步或 submit

### 4.3 通信机制

借鉴 Claude Code 的 task-notification，简化为单进程内函数调用（不起子进程，不引入 AsyncLocalStorage 级隔离）：

```python
# Coordinator 发工
worker_result = await run_worker(worker_type, instruction, state)
# worker_result 包含 observation + metrics + 是否成功

# Coordinator 收工后更新 state
state = update_state_with_worker_result(state, worker_result)
```

### 4.4 记忆隔离

- Coordinator 持有 SceneMemory + EpisodeMemory 全局视图
- Worker 不持有记忆，只拿 Coordinator 给的 instruction + 当前 observation
- Worker 结果回传后由 Coordinator 更新记忆

避免 worker 各自维护记忆导致状态分裂。

### 4.5 prompt 分离

- Coordinator prompt：策略层（该 explore 还是 navigate、何时 submit）+ manifest + scene_graph_memory
- ExploreWorker prompt：执行层（explore_panorama/explore_frontier 的调用格式）+ 当前 observation
- NavigateWorker prompt：执行层（navigate_to_object 的调用格式）+ target info

每层 prompt 更短更聚焦，对应 Claude Code 的 "专项 prompt 家族"。

### 4.6 不做的事

- 不做 swarm teammate（无 team file、mailbox、inbox poller）——benchmark 场景无需持久 team
- 不做 in-process teammate 的 AsyncLocalStorage 隔离——单进程函数调用足够
- 不做 leaderPermissionBridge——worker 无独立权限需求，Coordinator 全权决策

### 4.7 落地门槛

阶段 4 只在阶段 1-3 完成且成功率仍未达标时启动。若阶段 2-3 的 memory 分层 + prompt 工程已把成功率推到目标，阶段 4 可不做。

### 4.8 验证

- 178 tests 回归通过
- GOATBench smoke：Coordinator 正确派工，Worker 正确执行
- prompt_audit 可见 Coordinator / Worker 各自的 prompt 分离
- 整体 SR 对比阶段 3

---

## 整体验证计划

每阶段结束执行：
1. 本地 `pytest` 全量回归（基线 178 passed）
2. 服务器 `/root/tiernav` 3dmem 环境回归（基线 177 passed）
3. GOATBench smoke：验证 runtime 不崩溃 + 关键指标（scene.objects / room_seg / path_length / tool_result.room_id）
4. prompt_audit JSONL 审查：确认 section 结构、token 占比、cache boundary 符合预期

成功率对比在阶段 2、3、4 结束时各跑一次完整 GOATBench benchmark。

## 不做的事（全局）

- 不做 Claude Code 的 forked subagent 摘要（benchmark 单线程足够）
- 不做 Team Memory / Agent Memory Snapshot（benchmark 无团队需求）
- 不做 swarm teammate / mailbox / inbox poller（benchmark 无持久 team）
- 不做 AsyncLocalStorage 隔离（单进程足够）
- 不做 leaderPermissionBridge（worker 无独立权限）
- 不引入新依赖（复用现有 LangGraph + pydantic 栈）

## 设计依据

- `claude-code-analysis/analysis/04-agent-memory.md` — 分层 memory + relevant recall
- `claude-code-analysis/analysis/04f-context-management.md` — compact + 状态复灌
- `claude-code-analysis/analysis/04g-prompt-management.md` — prompt runtime + 专项 prompt 家族
- `claude-code-analysis/analysis/04h-multi-agent.md` — coordinator/worker 拓扑
- `claude-code-analysis/analysis/01-architecture-overview.md` — section 化 + cache boundary
- tiernav 当前 runtime: `src/tiernav_runtime/context.py` `contracts.py` `memory.py` `graph.py`
