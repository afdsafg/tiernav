# TierNav LangGraph Agent — Master Plan (融合版)

**日期**: 2026-06-28
**状态**: 完整实现 plan，分阶段执行
**融合来源**:
- `swift-forging-newton.md` (phase 1 LangGraph 形式化，已完成)
- `docs/plans/2026-06-28-claude-code-patterns-for-vln-design.md` (Claude Code 模式借鉴，P0-P6)
- phase 1 out-of-scope items 中"所有工作都要做"原则纳入的项目

**核心原则**:
1. **所有工作都要做，任何功能都不能丢** —— phase 1 out-of-scope 的 #1-#5 全部纳入最终阶段
2. **变量隔离** —— 每个 lever 可独立上线、独立 A/B 对照 baseline，单变量归因
3. **行为保持** —— phase 1 已完成的 baseline 行为不退化，后续 lever 不动则不改变 benchmark 数字
4. **分阶段实现，写全到最终完善版** —— Phase 1 (已完成) → Phase 2 (P0-P6) → Phase 3 (扩展完善)

---

## 0. 总览

### 0.1 三阶段结构

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase 1** | LangGraph 状态机形式化（7 nodes, 2 edges, mimo 默认, 5 tools） | ✅ 已完成（commit `49cfac8`, `55be6e8`） |
| **Phase 2** | Claude Code 模式借鉴（P0-P6：元模式 + 视觉记忆 + prompt cache + 行为验证 + fork） | 🚧 待实现 |
| **Phase 3** | 扩展完善（Claude provider + 新工具 + Critic node + LlamaIndex 整合 + 技术债） | 🚧 待实现 |

### 0.2 最终阶段（Phase 3 完成）的完善标准

- LangGraph runtime 作为默认 engine（`--engine langgraph` 默认）
- mimo + Claude 双 provider 可切换
- 5 个原生工具 + ForkSubagentTool + PixelNavigateTool 等新工具
- 视觉记忆三层（L0 索引 + L1 caption + L2 图片召回）闭合 CLIP gap
- Prompt cache 分段优化，cacheable 段跨轮命中
- 行为验证（stall 检测 + verification nudge）避免无效循环
- Critic node 作为可选 lever
- LlamaIndex 语义记忆与视觉记忆 L1/L2 深度整合
- 技术债清理（helper 去重、step_counter 全局状态）
- 完整 A/B 归因数据覆盖每个 lever

---

## 1. 锁定决策

### 1.1 Runtime
**LangGraph**（不是 Claude Agent SDK，不是 OpenAI Agents SDK）。理由：VLN 控制流复杂（5 actions, 3 guard types, entity-exhaustion skip semantics, backtracking via memory mutation），需要显式 graph，不是线性 perceive→act→verify loop。

### 1.2 LLM
- **Phase 1-2**: `mimo-v2.5` 默认（保持 benchmark 数字），`cfg.llm.provider: "mimo" | "claude"` 可切换
- **Phase 3**: `ClaudeProvider` 完整实现，Claude swap 作为独立 lever 可 A/B
- 两个 provider 返回相同 `PlannerAction` dataclass，所有下游 node provider-agnostic

### 1.3 HM-GE legacy
保持 as-is。不形式化、不删除。是对照 baseline 和 `ContextManager`/`SeedViewManager`/`STAGE*_PROMPT` 的唯一消费者。`run_episode` (`agent_workflow.py:410`) 和 `run_hmge_evaluation.py` 永久不动。

### 1.4 扩展性原则
- Node 可加/删/改，无需重新接线
- Heavy objects 通过 `RunnableConfig.configurable` 注入
- Tool 通过 registry 注册
- LLM 通过 provider interface

### 1.5 变量隔离原则（研究方法论）
- 每个 lever 独立上线、独立 A/B
- 一次只动一个 lever，保证 Δaccuracy 可归因
- 详见 §5 A/B 归因协议

### 1.6 设计原则：extensibility
Phase 1 的 graph 结构（7 nodes, 2 edges）是 baseline。后续 lever 扩展为 8 nodes 3 edges（P3）乃至更多（P6 fork, Phase 3 Critic）。每次扩展保持"可加/删/改"特性。

---

## 2. Phase 1 — 状态机形式化（已完成）

### 2.1 架构：7 nodes, 2 conditional edges

```
START → init → build_context → planner → loop_guard ──"submit"──→ submit → END
                          ↑                              │
                          │                              └──"execute"──→ executor → memory_update
                          │                                                     │
                          └───────────"continue"─────────────────────────────────┘
                                                (after_memory edge)
                                                  └──"fallback_submit"──→ submit → END
```

Backtracking (GD-fail → hypothesis_rejected) 是**数据流通过 memory，不是边**：`executor` 产生 `TrajectoryEvidence(outcome="detection_failed")`，`memory_update` 标记 room rejected，下一个 `build_context` 排除它。Graph 简单循环 via `continue`。

### 2.2 Nodes（每个 wrap 现有代码 verbatim）

| Node | Wraps | 职责 |
|---|---|---|
| `init` | `agent_workflow.py:1160-1251` | Episode setup + initial panorama; reset `silent_perception_step._step_counter` |
| `build_context` | `:1445-1532` | 组装 4-component planner prompt; active memory query; topdown map; notebook injection |
| `planner` | `:1538-1580` | LLM 决策 via `llm_provider.decide()`; Stage 6.5 frontier sub-selection via `decide_raw()` |
| `loop_guard` | `:1582-1612` | 3 guards (repeated panorama, visited seed, invalid object) via `_first_available_action`; emit `run_logger.log_decision`。**Node 不是 edge** —— mutates `current_action` + 必须 fire trace exactly once |
| `executor` | `:1633` | Dispatch action via `ToolRegistry`; update pose from `executor._pts/_angle` |
| `memory_update` | `:1637-1663` | Notebook + scene-graph + rejected-region marking; compute `exhausted_flag` (`agent_notebook.py:172`) |
| `submit` | `:1614-1630` (success) + `:1681-1702` (fallback) | Terminal answer; 两种 entry mode via 不同 edge |

### 2.3 Edges

**Static:** `START→init`, `init→build_context`, `build_context→planner`, `planner→loop_guard`, `executor→memory_update`, `submit→END`.

**Conditional `after_guard` (leaves `loop_guard`):**
```python
if current_action.action_type == "submit_answer": return "submit"   # :1614
return "execute"                                                     # :1633
```

**Conditional `after_memory` (leaves `memory_update`) — CRITICAL ordering:**
```python
def after_memory(state) -> str:
    if rounds_used >= max_planner_rounds:   return "fallback_submit"  # for-loop end → :1681
    if exhausted_flag:                       return "continue"          # entity-exhaustion → skip step-budget
    if steps_taken >= max_total_steps:      return "fallback_submit"  # step-budget break
    return "continue"
```
顺序很重要：`exhausted_flag` 必须在 step-budget 之前检查，以复现 `:1665-1677` 的 `continue` skip 语义。Round-budget 主导（for-loop 最外层）。

### 2.4 Guard → mechanism mapping

| 原始 guard | Mechanism |
|---|---|
| `:1582-1584` repeated panorama | Rewritten in `loop_guard_node` |
| `:1585-1589` visited seed | Rewritten in `loop_guard_node` |
| `:1590-1592` invalid object | Rewritten in `loop_guard_node` |
| `:1614` submit short-circuit | `after_guard` edge → `submit` |
| `:1665-1672` entity exhaustion `continue` | `after_memory` edge → `continue` (skip step-budget) |
| `:1675-1677` step-budget `break` | `after_memory` edge → `fallback_submit` |
| for-loop end | `after_memory` edge → `fallback_submit` (checked first) |
| `:1681-1702` `submit_best_guess` | `submit_node` fallback entry mode |

### 2.5 State schema（phase 1 baseline）

`TwoTierState` TypedDict —— node 间可序列化契约。Heavy objects（perception models, Habitat scene, TSDF volumetric map, planner/executor instances）留在 state 外，通过 `Resources` in `RunnableConfig.configurable` 注入。

**In-state:** episode identity (`scene_id`, `question_id`, `question`), budgets (`max_planner_rounds`, `max_total_steps`), method flags (`use_notebook`, `use_scene_graph`, `use_active_query`, `use_rejected_tracking`), mutable per-round (`pose`, `rounds_used`, `steps_taken`, `current_action`, `last_evidence`, `exhausted_flag`), accumulating history with `Annotated[list, operator.add]` reducers (`action_history`, `round_traces`), per-round prompt artifacts (`scene_analysis`, `history_text`, `progress_text`, `actions_text`, `current_views`, `topdown_b64`, `memory_summary`), terminal (`answer`, `success`, `error`, `terminal`, `failure_type`).

**Out-of-state (`Resources` dataclass):** `scene`, `tsdf_planner`, `memory_store`, `models`, `cfg`, `notebook`, `scene_graph`, `planner`, `executor`, `llm_provider`, `tool_registry`, `run_logger`.

**Tech debt flagged, not fixed (phase 1):** `silent_perception_step._step_counter` (`agent_tools.py:191`) 保持 function-attribute global。→ Phase 3 §4.5 处理。

### 2.6 LLM provider interface

```python
class LLMProvider(ABC):
    def decide(self, system_prompt, user_prompt, images_b64, *, max_tokens, temperature) -> PlannerAction: ...
    def decide_raw(self, messages, image_b64, *, max_tokens, temperature) -> str: ...
```

两个方法因为 codebase 有两种 call shape：action-deciding（需 `PlannerAction`）和 free-form（Stage 6.5 frontier, `submit_best_guess` fallback），各有自己的 tolerant parser。

- **`MimoProvider`** (phase 1 默认) —— thin adapter delegating to existing `Planner` class (`agent_planner.py:68`)。`decide()` 复用 `Planner.decide` + `Planner.parse_response` (`:139`) verbatim。Byte-identical behavior。
- **`ClaudeProvider`** (phase 3 实现) —— `anthropic` SDK with native tool-use; 5 actions 成为 tool definitions; `tool_use` block 直接映射到 `PlannerAction`。无 JSON-in-text parsing。

Config: `cfg.llm.provider: "mimo" | "claude"`. Swap 是 config change，不是 graph change。

### 2.7 Tool registry

替代 `agent_executor.py:291` 的 hard-coded if-chain。`ActionTool` ABC with `schema()` (returns `ToolSchema` with `name`, `arg_fields`, `prompt_description`, `is_terminal`) and `run(action, ctx) -> TrajectoryEvidence`. `ToolRegistry.dispatch(action, ctx)` looks up by `action.action_type`.

**5 default tools (phase 1)**, 各 wrap 现有 `Executor` method 1:1:
- `ExplorePanoramaTool` → `Executor.explore_panorama` (`:89`)
- `NavigateToObjectTool` → `Executor.navigate_to_object` (`:127`)
- `ExploreSeedTool` → `Executor.explore_seed` (`:201`)
- `ExploreFrontierTool` → `Executor.explore_frontier` (`:245`)
- `SubmitAnswerTool` → `is_terminal=True`, routed to `submit_node` via `after_guard` edge; `run()` no-op (unreachable)

`ToolRegistry.actions_prompt_text(ctx)` 连接每个 tool 的 `prompt_description` —— 替代 hardcoded `_build_actions` body (`:1360-1404`).

### 2.8 Phase 1 完成状态

- ✅ `src/two_tier_graph/` 完整实现（state.py, resources.py, nodes.py, edges.py, graph.py, providers.py, tools.py, entrypoint.py）
- ✅ `tests/test_two_tier_graph.py` 18 个 deterministic unit tests
- ✅ `--engine {legacy,langgraph}` flag 接入 `run_two_tier_aeqa_evaluation.py`
- ✅ Smoke test 通过（qwen3-vl-flash + hm3d）
- ✅ `call_vlm` signature 修复（`MODEL_NAME` fallback）
- ✅ `recursion_limit` 修复（`max_planner_rounds * 6 + 20`）
- ⏳ Full AEQA-41 benchmark run 待执行（`--engine langgraph --method ours_full`）
- ⏳ Side-by-side 10-question dev subset comparison 待执行

### 2.9 Phase 1 遗留待办

| 待办 | 说明 |
|---|---|
| Full AEQA-41 benchmark | `--engine langgraph --method ours_full`，确认无 regression |
| 10 题 dev subset 对照 | `action_history`/`rounds_used`/`steps_taken` 对比 legacy vs langgraph |
| `groundingdino/` 加入 server `.gitignore` | 694MB 权重不应进 git |
| `cfg/eval_aeqa.yaml` scene path | server 上 uncommitted，需决定 commit 还是 deployment-specific |

---

## 3. Phase 2 — Claude Code 模式借鉴（P0-P6）

### 3.0 P0 — 结构性元模式（基础设施，无行为改变）

P0 是纯重构，为后续 lever 铺路。**baseline 行为不变**（现有 4 个 reason 的语义和路由完全保留）。

#### 3.0.1 元模式 1：分层压缩 + 显式契约

**VLN baseline 现状**：`RoundTrace`（全量）→ `EvidenceNotebook`（压缩）→ 最终答案。链路存在但每层契约隐式、阈值 hard-coded、无中间观测点。

**Claude Code 借鉴**：5-pass ordered compaction（`applyToolResultBudget` → `snipCompact` → `microCompact` → `contextCollapse` → `autocompact`），每层有明确输入输出契约和阈值触发条件。

**应用到 VLN** —— 把现有压缩链重述为三层契约：

| 层 | 输入 | 输出 | 触发条件 | 实现 |
|---|---|---|---|---|
| **L_raw** | 每轮 planner 推理 + action + observation | `RoundTrace` 对象追加到 `round_traces` | 每轮结束（`memory_update_node`） | 现有，不变 |
| **L_compressed** | `round_traces` 全量 | `EvidenceNotebook` 条目 | 轮数 ≥ `compress_threshold`（默认 5，可配置） | 现有，加阈值显式化 |
| **L_index** | `EvidenceNotebook` 条目 | L0 索引行（P1 实现） | 每 `index_refresh_interval` 轮（默认 3，可配置） | P1 新增 |

**关键约束**（借鉴 Claude Code 每层契约）：
- 每层输出是**可序列化**的（不依赖 heavy resources），可独立测试
- 每层有**单调性**：L_compressed 输出条目数 ≤ L_raw 输入条目数；L_index 输出行数 ≤ L_compressed 输出条目数
- 每层失败**不阻塞上层**：L_index 构建失败时 fallback 到 L_compressed 全量注入

**`RunLogger` 观测点**：每轮记录每层的输入/输出条目数、token 估算、构建耗时。

**实现任务**：
1. `compress_threshold` 提取为 `cfg.memory.compress_threshold`（默认 5）
2. `index_refresh_interval` 提取为 `cfg.memory.index_refresh_interval`（默认 3，P1 使用）
3. `memory_update_node` 里显式化三层调用，每层有独立 try/except 和 fallback
4. `RunLogger` 加 `log_compression_layer(layer, input_count, output_count, token_est, duration)` 方法

#### 3.0.2 元模式 2：`transition.reason` 一等公民

**VLN baseline 现状**：`after_memory` 边返回 `round_budget / exhausted / step_budget / continue` 四路，已是不解析消息内容的枚举路由——**这其实就是 `transition.reason` 模式**，但没有系统化。

**Claude Code 借鉴**：`state.transition.reason` 字段标记恢复路径（`next_turn / max_output_tokens_recovery / reactive_compact_retry / stop_hook_blocking / token_budget_continuation`），测试可断言走了哪条路。

**应用到 VLN** —— 把 `after_memory` 的返回值升级为 state 字段：

```python
class TransitionReason(str, Enum):
    CONTINUE = "continue"
    ROUND_BUDGET = "round_budget"           # 现有
    EXHAUSTED = "exhausted"                 # 现有，跳过 step_budget
    STEP_BUDGET = "step_budget"             # 现有
    STALL_RECOVERY = "stall_recovery"       # P3 新增
    VERIFY_BEFORE_FALLBACK = "verify_before_fallback"  # P3 新增

@dataclass
class Transition:
    reason: TransitionReason
    from_node: str
    to_node: str
    round_idx: int
```

每轮在 `memory_update_node` 末尾写入 `state["last_transition"]`，`after_memory` 边基于此返回。`state["transition_log"]: Annotated[list, operator.add]` 累积所有 transition。

**实现任务**：
1. `state.py` 加 `TransitionReason` enum、`Transition` dataclass、`last_transition` 字段、`transition_log` 字段
2. `memory_update_node` 末尾写入 `last_transition`
3. `after_memory` 边从 `last_transition.reason` 读取（而非重新计算）
4. 测试可断言：`assert any(t.reason == TransitionReason.STALL_RECOVERY for t in final_state["transition_log"])`
5. **关键**：现有 4 个 reason 的语义和路由**完全保留**，P0 不引入新 reason

---

### 3.1 P1 — 视觉记忆层 L0 索引

**VLN baseline 缺口**：每轮 planner 看到 3 张当前视角 + topdown；历史信息以文本形式存在于 `RoundTrace`（全量）和 `EvidenceNotebook`（压缩）；历史 snapshots 写磁盘但从不进 prompt；`agent_memory.py:50` 算的 CLIP embedding 在 `:62` 的 `query()` 里被 keyword filter + linear scan 绕过，**算了没用**。

**Claude Code 借鉴**：typed memory + `MEMORY.md` ≤200 行索引（只有索引自动加载，正文按需读）+ `findRelevantMemories` 非阻塞预取 + `loadedNestedMemoryPaths` LRU 去重。

**关键差异**：Claude Code 是纯文本 memory，LLM "读"文件即可；VLM 必须把图片**直接放进 messages**才能"看到"。Claude Code 的"按需读正文"对 VLM 不直接适用——图片要么进 prompt 要么不进，没有"读"的中间态。

**设计：选项 C 分层混合**（P1 只做 L0 层，L1/L2 后续 P4/P5）：

**L0 索引层**（常驻 prompt）：
- 每条 ≤1 行，字段 = `[round, pose, object_class, one_line_desc]`
- 20 轮内 ≤20 行，类比 `MEMORY.md` ≤200 行约束
- 由 L_index 压缩层（元模式 1）生成，每 `index_refresh_interval` 轮（默认 3）更新一次
- 闭合 CLIP gap 的第一步：CLIP embedding 用于 L_index 层的检索排序

**非阻塞预取**（借鉴 `findRelevantMemories`）：
- `build_context_node` 开始时启动 CLIP 检索异步任务
- settle 才注入，超时 fallback 到 L0 索引常驻
- 不阻塞 planner 决策

**LRU 去重**（借鉴 `loadedNestedMemoryPaths`）：
- 维护 `loaded_snapshot_ids` set
- 避免同一 snapshot 跨轮重复注入

**实现任务**：
1. `src/two_tier_graph/visual_memory.py` 新模块：`VisualMemoryIndex` 类管理 L0 索引
2. `memory_update_node` 调用 `VisualMemoryIndex.update(round_trace, clip_embedding)` 构建 L0 行
3. `build_context_node` 读取 `VisualMemoryIndex.get_index_text()` 注入到 `memory_summary` 段
4. CLIP embedding 来源：复用 `agent_memory.py:50` 已有的计算（闭合 gap）
5. `loaded_snapshot_ids` set 存入 state（可序列化）
6. `cfg.memory.index_refresh_interval = 3`（可配置）
7. 测试：L0 索引构建、LRU 去重、超时 fallback

**A/B 对照**：纯文本增量，无视觉 token 改变。对比 `accuracy_0` (baseline) vs `accuracy_1` (baseline + L0)。

---

### 3.2 P2 — Prompt Cache 优化

**VLN baseline 缺口**：每轮 planner 调 `MimoProvider.decide()`，重建完整 prompt（系统指令 + 动作 schema + 当前视图 + 历史上下文 + 任务问题）。整包发 API，无 prompt cache 命中，VLM 调用成本高。

**Claude Code 借鉴**：
- `getSystemPrompt()` 返回 `string[]`（每段可独立缓存），不是单一字符串
- `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` 标记把 prompt 切成"静态前缀"（跨轮缓存）和"动态后缀"（每轮重算）
- `systemPromptSection` registry 区分 `systemPromptSection`（跨轮 memoized）vs `DANGEROUS_uncachedSystemPromptSection`（每轮重算，会破坏 cache）
- subagent fork 用 `CacheSafeParams` 保证 fork 命中父 agent 的 prompt cache

**实现前置条件**：provider 支持 prompt cache。当前主流 VLM provider 大部分支持（mimo/qwen3-vl-flash）。设计本身 provider-agnostic：无论 cache 是否生效，prompt 分段本身仍有价值（成本可观测、便于 A/B）。**先把功能做进去，provider 支持时自动受益**。

**设计**：

#### 3.2.1 Prompt 分段 registry

在 `build_context_node` 里把 planner prompt 从单一字符串改为有序段列表：

```python
@dataclass
class PromptSection:
    name: str
    content: str  # 或 list[ContentBlock] 含 image
    cacheable: bool  # 跨轮是否稳定

def build_planner_prompt(state, resources) -> list[PromptSection]:
    return [
        PromptSection("task_instruction", TASK_TEMPLATE, cacheable=True),
        PromptSection("action_schema", resources.tool_registry.actions_prompt_text(), cacheable=True),
        PromptSection("memory_index", L0_INDEX_TEMPLATE, cacheable=True),  # P1 L0
        PromptSection("reasoning_history", _build_reasoning_history(state), cacheable=False),
        PromptSection("current_views", [view["image_b64"] for view in state["current_views"]], cacheable=False),
        PromptSection("topdown", state["topdown_b64"], cacheable=False),
        PromptSection("active_query", state["question"], cacheable=True),
    ]
```

#### 3.2.2 Cache boundary 标记

`MimoProvider.decide()` 序列化时在最后一个 `cacheable=True` 段后插入边界标记（如 Anthropic 格式是 `cache_control: ephemeral`；非 cache provider 忽略）。静态前缀 = task_instruction + action_schema + memory_index + active_query，跨轮稳定。

#### 3.2.3 触发 cache miss 的字段白名单

借鉴 `DANGEROUS_uncachedSystemPromptSection` 命名警告，维护"会破坏 cache 的字段"白名单：
- `reasoning_history`（每轮增长）
- `current_views`（每轮位姿变化）
- `topdown`（每轮更新）
- `memory_index` 里 P1 L0 如果当轮有新 snapshot 也会变（trade-off：索引更新频率 vs cache 命中率，由 `index_refresh_interval` 控制）

#### 3.2.4 与 P1/P3 的耦合

- P1 L0 索引层纳入 `cacheable=True`——只在每 `index_refresh_interval` 轮更新一次，避免每轮 miss
- P3 的 `StallSignal.hint` 注入到 `reasoning_history` 段（`cacheable=False`），不破坏静态前缀
- P3 的 `RecoveryNote` 同理

#### 3.2.5 Fork 场景（与 P6 联动）

未来 multi-agent fork 时，fork child 必须复用父 agent 的静态前缀才能命中 cache。借鉴 `CacheSafeParams`：fork 时传 `{system_prompt_sections, tool_registry, model, message_prefix}`，child 必须用相同顺序、相同内容的 cacheable 段。

#### 3.2.6 成本可观测

`RunLogger` 记录每轮：cacheable token 数 / non-cacheable token 数 / cache hit 或 miss。

**实现任务**：
1. `src/two_tier_graph/prompt_sections.py` 新模块：`PromptSection` dataclass + `build_planner_prompt()`
2. `MimoProvider.decide()` 改造：接受 `list[PromptSection]` 而非单一字符串，插入 cache boundary
3. `build_context_node` 调用 `build_planner_prompt()` 替代原拼接逻辑
4. `RunLogger` 加 `log_prompt_cache(cacheable_tokens, non_cacheable_tokens, cache_hit)` 方法
5. `ClaudeProvider` (phase 3) 原生支持 `cache_control` 标记

**A/B 对照**：对比 `cacheable_tokens / total_tokens` 比率、cache hit rate（若 provider 支持）、accuracy。

---

### 3.3 P3 — 行为验证机制

**VLN baseline 缺口**：冒烟测试观察——agent 连续 5 轮调用 `explore_frontier(14)`，`steps_taken=0`，从未移动；Stage 6.5 报"no frontier images available"；最后被迫 fallback submit。整个流程没有任何结构性检查提醒"你在重复无效动作"。

**Claude Code 借鉴**：
- `TodoWriteTool` verification nudge：当 ≥3 任务关闭而无验证步骤时，工具返回里**结构性注入**提示
- `transition.reason` 一等公民：恢复路径用枚举标签标记
- `withMemoryCorrectionHint()`：取消/错误时给模型一个"刚才发生了什么"的注释

**关键差异**：Claude Code 是工具返回值里注入提示，VLN 的 planner 是 graph node，提示应注入到**下一轮 planner 的 prompt 上下文**里，而非工具返回。

**设计：三层结构**

#### 3.3.1 层 1：重复动作检测器（deterministic，在 `memory_update_node`）

`memory_update_node` 是每轮结束的汇聚点，已能看到 `action_history`。加纯函数 `detect_stall(action_history, steps_taken) -> Optional[StallSignal]`：

```python
@dataclass
class StallSignal:
    kind: Literal["repeated_action_no_progress", "no_valid_targets"]
    repeated_count: int
    last_action: str
    hint: str  # 给 planner 的文本提示
```

检测规则（deterministic，可单测）：
- 连续 ≥3 轮同一 action 同一参数，且 `steps_taken` 未增长 → `repeated_action_no_progress`
- 任何一轮工具返回 `exhausted_flag` 或 "no valid targets" → `no_valid_targets`

`StallSignal` 写入 state 字段 `stall_signal: Optional[StallSignal]`，供 `build_context_node` 读取注入。

#### 3.3.2 层 2：恢复路径标签（扩展 `after_memory` 边）

当前 `after_memory` 返回 4 路。扩展为：

```python
def after_memory(state) -> str:
    # 原有 4 路保留
    if rounds_used >= max_planner_rounds: return "fallback_submit"
    if state.get("exhausted_flag"): return "continue"  # 跳过 step_budget
    if steps_taken >= max_total_steps: return "fallback_submit"
    # 新增：stall 恢复路径
    if state.get("stall_signal") is not None:
        return "stall_recovery"
    return "continue"
```

`stall_recovery` 路由到新节点 `stall_recovery_node`（不做实际工作，只把 `stall_signal.hint` 转成 `RecoveryNote` 追加到 `round_traces`），然后回到 `build_context`。

#### 3.3.3 层 3：验证 nudge（在 `submit_node` 的 fallback 入口）

当 `after_memory` 路由到 `fallback_submit` 时，当前 `submit_node` 直接调 `submit_best_guess`。借鉴 TodoWrite verification nudge，在 fallback 路径加**前置检查**：

```python
def submit_node(state, resources):
    is_fallback = state.get("submit_reason") == "fallback"
    if is_fallback and not state.get("verification_attempted"):
        # 不直接提交，先走一次 verification
        return {"verification_attempted": True, ...}  # 路由回 build_context
    # 真正提交
```

verification 的内容 = 让 planner 显式回答"为什么现在提交而不是再探索一轮"，答案写进 `round_traces`，第二次 fallback 才真提交。**只在 fallback 路径触发，不在成功 submit 路径触发**。

#### 3.3.4 Graph 结构变更

phase 1 的 "7 nodes, 2 conditional edges" 升级为 **"8 nodes, 3 conditional edges"**：
- 新增节点：`stall_recovery_node`
- 新增条件边：`after_memory` 增加 `stall_recovery` 路由
- `submit_node` 内部分支增加 verification 前置检查（非新节点）

```
START → init → build_context → planner → loop_guard ──"submit"──→ submit → END
                          ↑                              │
                          │                              └──"execute"──→ executor → memory_update
                          │                                                     │
                          │              ┌──"stall_recovery"──→ stall_recovery ─┘
                          │              │                                      │
                          └──"continue"──┤                                      │
                                         └──"fallback_submit"──→ submit → END
                                          (after_memory edge, 3-way)
```

新增 `TransitionReason.STALL_RECOVERY` 和 `TransitionReason.VERIFY_BEFORE_FALLBACK`（P0 已预留 enum，P3 激活）。

**实现任务**：
1. `src/two_tier_graph/stall_detection.py` 新模块：`StallSignal` dataclass + `detect_stall()` 纯函数
2. `nodes.py` 加 `stall_recovery_node`
3. `edges.py` 扩展 `after_memory` 增加 `stall_recovery` 路由
4. `nodes.py` 的 `submit_node` 增加 verification 前置检查分支
5. `state.py` 加 `stall_signal`、`verification_attempted` 字段
6. `build_context_node` 读取 `stall_signal` 注入到 `reasoning_history` 段
7. 测试：stall 检测规则、stall_recovery 路由、verification nudge 流程

**A/B 对照**：纯 graph 结构改动，不动 prompt template。对比 stall 发生率、fallback submit 前的 verification 触发率、accuracy。

---

### 3.4 P4 — 视觉记忆层 L1 caption

在 P1 L0 索引层之上加 L1 caption 层。

**设计**：
- 每张 snapshot 离线生成 caption（VLM 单次调用，结果缓存磁盘）
- CLIP 检索 top-K caption 注入到 `reasoning_history` 段（`cacheable=False`）
- caption 缓存路径：`output_dir/captions/<snapshot_id>.txt`
- caption 生成时机：`memory_update_node` 时异步生成（不阻塞当前轮）

**实现任务**：
1. `src/two_tier_graph/visual_memory.py` 加 `CaptionStore` 类管理 caption 磁盘缓存
2. `memory_update_node` 异步调用 VLM 生成 caption（复用 `call_vlm`）
3. `build_context_node` 的 CLIP 检索返回 top-K caption，注入到 prompt
4. caption 生成失败时 fallback 到 L0 索引行
5. 测试：caption 缓存命中、CLIP 检索 top-K、fallback

**A/B 对照**：视觉 token 增量（caption 是文本，但描述了历史视觉内容）。对比 accuracy。

---

### 3.5 P5 — 视觉记忆层 L2 图片召回

在 P1 L0 + P4 L1 之上加 L2 图片召回层。

**设计**：
- 极少数情况（如"我刚才看到的是红色杯子还是蓝色？"）召回原始 snapshot
- 受 token 预算硬约束：`cfg.memory.l2_token_budget`（默认 3000 vision tokens，约 3 张图）
- 触发条件：planner 在 `reasoning_history` 里显式标注 `need_visual_recall: <snapshot_id>`（需扩展 PlannerAction schema 或用 `decide_raw` 单独请求）
- 召回的 snapshot 注入到 `current_views` 段之后，标记为 `recalled_view`

**实现任务**：
1. `src/two_tier_graph/visual_memory.py` 加 `ImageRecallStore` 类
2. `build_context_node` 检查 `need_visual_recall` 标志，加载 snapshot
3. token 预算硬约束检查
4. `loaded_snapshot_ids` LRU 去重（与 P1 共享）
5. 测试：召回触发、token 预算约束、LRU 去重

**A/B 对照**：视觉 token 增量。对比 accuracy，特别是需要细粒度视觉辨别的题目。

---

### 3.6 P6 — 多 agent fork 机制（stub）

**VLN baseline 缺口**：当前是单 agent 两层循环，所有决策在一个上下文里完成。plan §8 #2 提到"multi-agent split"作为未来 lever，但没有拆分机制——一旦拆分，子 agent 会从头重建 prompt，成本和延迟都高。

**Claude Code 借鉴**：
- `forkSubagent.ts` + `CacheSafeParams`：fork child 继承父 agent 的 system prompt、tools、model、message 前缀，**命中父 agent 的 prompt cache**
- 普通子 agent 用 `createSubagentContext` 建空上下文，不继承
- fork 是单向通信：child 的最终 assistant message 成为父 agent 的 tool result
- `recordSidechainTranscript`：fork 的完整 transcript 写磁盘，不进父上下文
- `tengu_slim_subagent_claudemd`：read-only 子 agent 剥离 CLAUDE.md 省 token

**关键差异**：
- Claude Code 是通用编码 agent，fork 用于"并行探索多个方案"；VLN 是具身决策，fork 的语义需要重新定义
- VLN 的"消息前缀"含视觉 token（current_views + topdown），fork 时若复用前缀必须连图片一起复用，cache 命中条件更严格

#### 3.6.1 VLN 场景下的 fork 语义

三个候选 fork 场景，**P6 只定义机制 + stub，不实现具体场景**（场景 A/B/C 进入 Phase 3）：

| 场景 | fork 时机 | child 职责 | 返回值 |
|---|---|---|---|
| **A. 子路线探索** | planner 选 `explore_frontier(k)` 时 | fork 出 N 个 child 各自模拟走 k 后再走 1-2 步，返回最 promising 的子路径 | 路径评分 + 文本描述 |
| **B. 区域验证** | planner 怀疑某对象在历史区域时 | fork child 加载该区域历史 snapshots，回答"X 是否出现过" | 是/否 + 证据 snapshot_id |
| **C. 子任务委托** | planner 拆分复杂任务时 | fork child 独立完成子任务（如"找到所有椅子"） | 子任务结果 |

#### 3.6.2 `CacheSafeParams` 在 VLN 的等价

```python
@dataclass
class CacheSafeParams:
    system_prompt_sections: list[PromptSection]  # P2 的 cacheable 段
    tool_registry: ToolRegistry
    llm_provider_config: LLMConfig  # 不传 provider 实例，传 config
    message_prefix: list[ContentBlock]  # 父 agent 当前轮的 prompt 前缀
    inherit_current_views: bool = True  # VLN 特有：是否复用父的当前视图
```

`inherit_current_views=True` 时 child 共享父的 current_views（同一位姿）；`=False` 时 child 用自己的虚拟位姿（场景 A/B）。**已确认可接受：场景 A 的 child 必须从父的当前位姿出发**。

#### 3.6.3 Fork 调度（伪代码 stub 接口）

借鉴 Claude Code 的 `AgentTool` 设计，`ToolRegistry` 注册 `ForkSubagentTool`。P6 用伪代码写假接口，作为纯结构性预留（类比 `ClaudeProvider` stub）：

```python
class ForkSubagentTool(ActionTool):
    """Fork 子 agent 执行子任务，命中父 agent 的 prompt cache。
    
    P6 阶段为结构性预留，run() 返回 NotImplementedError。
    场景 A/B/C 的具体 fork 逻辑在 Phase 3 实现。
    """
    
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="fork_subagent",
            arg_fields={
                "subagent_type": "str",  # 'route_explore' | 'area_verify' | 'task_delegate'
                "directive": "str",      # 给 child 的任务指令（说做什么，不是说当前情况）
                "max_turns": "int",      # child 的最大轮数
                "inherit_context": "bool",  # 是否复用父上下文（命中 cache）
            },
            prompt_description="Fork a subagent to explore a sub-route, verify a region, or delegate a subtask. The child shares your prompt cache.",
            is_terminal=False,
        )
    
    def run(self, args: dict, state: TwoTierState, resources: Resources) -> dict:
        # ===== P6 stub：纯结构性预留 =====
        # 真实实现流程（伪代码，Phase 3 填充）：
        # 1. cache_safe_params = self._build_cache_safe_params(state, resources)
        # 2. child_resources = self._create_child_resources(resources, cache_safe_params)
        # 3. child_state = self._init_child_state(args, cache_safe_params)
        # 4. child_graph = build_two_tier_graph()
        # 5. child_final_state = child_graph.invoke(
        #        child_state,
        #        config={
        #            "configurable": {"resources": child_resources},
        #            "recursion_limit": args["max_turns"] * 4 + 5,
        #        },
        #    )
        # 6. self._record_sidechain_transcript(child_final_state, parent_round=state["rounds_used"])
        # 7. return {"fork_result": child_final_state["round_traces"][-1]}
        raise NotImplementedError(
            "ForkSubagentTool is a structural placeholder. "
            "Scenarios A/B/C (route_explore/area_verify/task_delegate) "
            "are Phase 3 levers, not yet implemented."
        )
    
    def _build_cache_safe_params(self, state, resources) -> CacheSafeParams:
        """构造 CacheSafeParams 让 child 命中父的 prompt cache。"""
        cacheable_sections = [s for s in build_planner_prompt(state, resources) if s.cacheable]
        return CacheSafeParams(
            system_prompt_sections=cacheable_sections,
            tool_registry=resources.tool_registry,
            llm_provider_config=resources.llm_provider.config,
            message_prefix=state["message_prefix"],  # 含视觉 token
            inherit_current_views=True,
        )
```

#### 3.6.4 Sidechain transcript

借鉴 `recordSidechainTranscript`：fork 的完整 state（不含 heavy resources）写磁盘到 `output_dir/sidechains/<parent_round>_<fork_id>.jsonl`，**不进父 state**。父 agent 只看最终返回值。

#### 3.6.5 与 P1-P5 的耦合

- **P1/P4/P5**：child 可读父的 L0 索引（cacheable，cache 命中），但 L1/L2 按需召回——child 自己的检索结果不进父上下文
- **P3**：child 的 `transition.reason` 独立，父不感知 child 是否走了 `stall_recovery`
- **P2**：child 必须用 `CacheSafeParams` 复用父的 cacheable 段才能命中 cache，这是 fork 的核心价值
- **`tengu_slim_subagent_claudemd` 等价**：场景 B/C 的 child 若是 read-only（不调 navigate 工具），可剥离 action_schema 段省 token——但会破坏 cache，trade-off 需实测

**实现任务**：
1. `src/two_tier_graph/fork.py` 新模块：`CacheSafeParams` dataclass + `ForkSubagentTool` stub
2. `tools.py` 的 `build_default_tool_registry()` 注册 `ForkSubagentTool`
3. sidechain transcript 写磁盘逻辑（即使 stub 也先建好路径结构）
4. 测试：`ForkSubagentTool.run()` raises `NotImplementedError`、schema 正确、registry 注册成功

**A/B 对照**：P6 本身无行为改变（stub raises error），A/B 在 Phase 3 场景 A/B/C 实现时做。P6 与 P2 强耦合（fork 收益依赖 cache），Phase 3 场景实现时与 P2 绑定验证。

---

## 4. Phase 3 — 扩展完善（最终阶段）

Phase 3 实现 phase 1 out-of-scope 里被"所有工作都要做"原则纳入的项目，达到最终完善标准。

### 4.1 Claude Provider 实现

phase 1 out-of-scope #4 → Phase 3 实现。

**设计**：
- `anthropic` SDK with native tool-use
- 5 actions 成为 tool definitions（`explore_panorama`, `navigate_to_object`, `explore_seed`, `explore_frontier`, `submit_answer`）
- `tool_use` block 直接映射到 `PlannerAction`，无 JSON-in-text parsing
- 原生支持 `cache_control` 标记（P2 prompt cache 在 Claude provider 上自动生效）
- `decide_raw()` 用 Claude 的 free-form text mode（Stage 6.5 frontier, fallback submit）

**实现任务**：
1. `providers.py` 的 `ClaudeProvider` 从 stub 改为完整实现
2. 5 个 `ToolSchema` 转换为 Anthropic tool definition 格式
3. `decide()` 解析 `tool_use` block → `PlannerAction`
4. `decide_raw()` 处理 Claude text response
5. `cache_control: ephemeral` 标记插入（复用 P2 的 cacheable 段）
6. 测试：Claude provider 单测（mock anthropic SDK）、与 MimoProvider 行为等价性测试

**A/B 对照**：`cfg.llm.provider: "claude"` vs `"mimo"`，对比 accuracy、latency、cost。这是独立 lever，与 graph 结构无关。

### 4.2 新工具：PixelNavigateTool

phase 1 out-of-scope #5 → Phase 3 实现。

**设计**：
- planner 输出图上的像素坐标 `(x, y)` + 地图分辨率
- backproject 到 3D 世界坐标
- 调用现有 `Executor.navigate_to_object` 或新增 `navigate_to_point` 方法
- 注册到 `ToolRegistry`，planner 自动看到新 action（无需 graph edit）

```python
class PixelNavigateTool(ActionTool):
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="pixel_navigate",
            arg_fields={
                "pixel_x": "int",
                "pixel_y": "int",
                "reasoning": "str",
            },
            prompt_description="Navigate to a point on the topdown map specified by pixel coordinates. Use when you see a target location on the map.",
            is_terminal=False,
        )
    
    def run(self, args, ctx) -> TrajectoryEvidence:
        # 1. pixel → world coordinate via tsdf_planner.backproject
        # 2. navigate to world coordinate
        # 3. return TrajectoryEvidence
        ...
```

**实现任务**：
1. `src/two_tier_graph/tools.py` 加 `PixelNavigateTool`
2. `tsdf_planner.py` 加 `backproject(pixel_x, pixel_y) -> world_coord` 方法（若不存在）
3. `Executor` 加 `navigate_to_point(world_coord)` 方法（若不存在）
4. `build_default_tool_registry()` 注册 `PixelNavigateTool`
5. 测试：pixel→world 转换、navigation 执行

**A/B 对照**：新增 action 扩展 action space，对比 accuracy（特别是需要精确导航的题目）。

### 4.3 Critic Node

phase 1 out-of-scope #1 → Phase 3 实现。

**设计**：
- 新 node `critic_node` 插入在 `planner` 和 `loop_guard` 之间
- critic 评估 planner 的 `PlannerAction`：是否合理？是否有更好的选择？
- critic 可以 veto（强制重新决策）或 approve（放行）
- critic 本身调 `llm_provider.decide_raw()` 做自由文本评估

```
planner → critic ──"approve"──→ loop_guard
                └──"veto"──→ planner (重新决策，带 critic 反馈)
```

**实现任务**：
1. `nodes.py` 加 `critic_node`
2. `edges.py` 加 `after_critic` 条件边
3. `graph.py` 插入 `critic` node 和 `after_critic` edge
4. `state.py` 加 `critic_feedback` 字段
5. `cfg.critic.enabled` flag（默认 false，可 A/B 切换）
6. 测试：critic approve/veto 流程、veto 后重新决策

**A/B 对照**：`cfg.critic.enabled: true` vs `false`，对比 accuracy、rounds_used（critic 可能增加轮数但提升决策质量）。

### 4.4 LlamaIndex 语义记忆整合

phase 1 out-of-scope #3 → Phase 3 实现，与 P1/P4/P5 视觉记忆层深度整合。

**设计**：
- 替换 `MemoryStore.query` (`agent_memory.py:62`) 的 keyword filter + linear scan 为 LlamaIndex retriever
- CLIP embedding（`agent_memory.py:50` 已算但未用）成为 LlamaIndex 的索引基础
- `build_context_node` 的 active-query call site 是 hook point（phase 1 plan §8 #3 已标注）
- 与 P1 L0 / P4 L1 / P5 L2 整合：LlamaIndex 作为底层检索引擎，L0/L1/L2 是不同层级的检索结果呈现

**实现任务**：
1. 安装 `llama-index` 到 `langgraph` conda env（不动 `3dmem` env）
2. `src/two_tier_graph/semantic_memory.py` 新模块：`SemanticMemoryStore` 类包装 LlamaIndex retriever
3. `MemoryStore.query` 改为调用 `SemanticMemoryStore.query()`（保留原 keyword filter 作为 fallback）
4. CLIP embedding 索引到 LlamaIndex（闭合 gap）
5. P1 L0 索引层从 LlamaIndex 检索结果生成
6. P4 L1 caption 检索通过 LlamaIndex
7. P5 L2 图片召回通过 LlamaIndex
8. 测试：LlamaIndex 检索 vs keyword filter 对比、CLIP embedding 命中

**A/B 对照**：`cfg.memory.engine: "llamaindex" | "keyword"`，对比检索质量、accuracy。与 P1/P4/P5 绑定验证（LlamaIndex 是底层引擎，单独拉动无意义）。

### 4.5 技术债清理

phase 1 out-of-scope #7, #8 → Phase 3 处理。

#### 4.5.1 `silent_perception_step._step_counter` 全局状态

phase 1 out-of-scope #7。当前 `agent_tools.py:191` 是 function-attribute global，node 读写 verbatim。

**处理**：
- 移入 `TwoTierState.step_counter` 字段
- `silent_perception_step` 改为接受 `step_counter` 参数
- `_navigate_to_target_with_agent_step` 同步改造
- 风险评估：可能 perturb `silent_perception_step`/`_navigate_to_target_with_agent_step`，需充分回归测试

#### 4.5.2 HM-GE/Two-Tier helper 去重

phase 1 out-of-scope #8。`_NAV_OBJ_INVALID`, `_is_valid_object_desc`, `_build_messages` 在 `agent_workflow.py` 的 ~905 和 ~1735 各定义一次。

**处理**：
- 提取到 `src/shared_helpers.py`
- HM-GE 和 Two-Tier 都 import
- 确保行为 byte-identical（diff 对比）

#### 4.5.3 范围边界（永久不动）

以下 phase 1 out-of-scope items 永久不动：
- #6 changing the 5 actions or any prompt text —— `PLANNER_SYSTEM_PROMPT` (`agent_planner.py:39`), `_build_*` helpers (`:1255-1404`), `PlannerAction` schema preserved verbatim（除非 lever 显式要求，如 P3 stall hint 注入）
- #9 removing or formalizing HM-GE —— `run_episode` (`:410`) 和 `run_hmge_evaluation.py` 永久不动
- #10 changing return-dict contract of `run_episode_two_tier` —— LangGraph entrypoint 永远 reproduce exact result dict

---

## 5. A/B 归因协议

### 5.1 归因原则：一次只动一个 lever

**原则**：每次 A/B 实验只激活一个新 lever，其余保持 baseline 状态。

**为什么需要这条原则**：

假设 P1（L0 索引层）和 P2（prompt cache 分段）同时上线，跑 10 题 dev subset，accuracy 从 baseline 的 6/10 变成 7/10。问题是：
- 是 L0 索引让 agent 看到了历史线索，导致提升？
- 还是 prompt cache 分段让 VLM 在相同 token 预算下推理更稳定，导致提升？
- 还是两者协同？
- 或者其实是负负得正？

**无法回答**。这就是变量混淆——同时改了两个变量，无法把 Δaccuracy 归因到具体哪个 lever。研究场景下这等于白做实验。

**单变量对照的正确做法**：

```
Round 1: baseline (无任何新 lever)        → accuracy_0
Round 2: baseline + P1 only                → accuracy_1
对比 accuracy_1 vs accuracy_0 = P1 的归因
Round 3: baseline + P1 + P2                → accuracy_2  
对比 accuracy_2 vs accuracy_1 = P2 的归因（在 P1 已生效的前提下）
Round 4: baseline + P2 only (可选)          → accuracy_3
对比 accuracy_3 vs accuracy_0 = P2 的独立归因
```

**为什么这是研究方法论，不是工程惯例**：

工程上常见"一次合并多个改动，跑一次测试，过了就过"——因为工程目标是"系统工作"，不是"理解为什么工作"。研究目标是**理解每个设计选择的因果贡献**，才能在论文里说"L0 索引层贡献 +X，prompt cache 贡献 +Y"。这正是 feedback memory 里"variable isolation over bundled changes"的来源——benchmark 归因需要单变量。

**代价**：每个 lever 都要跑一次 10 题 subset，总实验次数 = lever 数 + 1（baseline）。Phase 2 P0-P6 七个阶段 + Phase 3 五个 lever = 13 次实验。但每次实验都有**可解释的结果**。

### 5.2 边界情况——何时可以破例

只有一个：**两个 lever 强耦合，单独拉动无意义**。已识别的强耦合对：
- **P6 fork 与 P2 prompt cache**：fork 收益依赖 cache 命中，单独上线 fork 无意义。Phase 3 场景 A/B/C 实现时与 P2 绑定验证
- **Phase 3 §4.4 LlamaIndex 与 P1/P4/P5**：LlamaIndex 是底层检索引擎，L0/L1/L2 是呈现层，单独拉动无意义。绑定验证

这种情况在实验记录里需要显式标注"X 与 Y 绑定验证"，而不是默认并行。

### 5.3 A/B 对照指标

每个 lever 上线后，跑同一个 10 题 dev subset，对比：

| 指标 | 来源 |
|---|---|
| `action_history` | state 字段 |
| `rounds_used` / `steps_taken` | state 字段 |
| `transition_log`（P0 后） | state 字段 |
| 每层 token 估算（P0 后） | `RunLogger` |
| cacheable / non-cacheable token 数（P2 后） | `RunLogger` |
| cache hit rate（P2 后，若 provider 支持） | API 响应 |
| 检索质量（Phase 3 §4.4 后） | LlamaIndex vs keyword 对比 |
| 最终答案 accuracy | 评测脚本 |

### 5.4 Phase 1 遗留 baseline 固化

在启动 P0 之前，必须先完成 phase 1 遗留待办（§2.9），固化 baseline 数字：
1. Full AEQA-41 benchmark run with `--engine langgraph --method ours_full`
2. 10 题 dev subset side-by-side comparison（legacy vs langgraph）
3. 确认无 regression 后，此数字成为 `accuracy_0` baseline

---

## 6. 实现顺序总览

```
Phase 1 (已完成)
  └── LangGraph 状态机形式化 (7 nodes, 2 edges)

Phase 2 (P0-P6)
  ├── P0: 元模式 1 + 元模式 2 (基础设施, 无行为改变)
  ├── P1: 视觉记忆 L0 索引 (纯文本增量)
  ├── P2: Prompt cache 优化 (cacheable 段分段)
  ├── P3: 行为验证机制 (stall 检测 + verification nudge, 8 nodes 3 edges)
  ├── P4: 视觉记忆 L1 caption (文本增量)
  ├── P5: 视觉记忆 L2 图片召回 (视觉 token 增量)
  └── P6: 多 agent fork 机制 (stub, 不实现场景)

Phase 3 (扩展完善)
  ├── §4.1: Claude Provider 实现
  ├── §4.2: PixelNavigateTool 新工具
  ├── §4.3: Critic Node
  ├── §4.4: LlamaIndex 语义记忆整合
  └── §4.5: 技术债清理 (step_counter, helper 去重)
```

### 依赖关系

```
P0 (元模式) ──┬──→ P1 (L0 索引) ──→ P2 (prompt cache) ──→ P6 (fork stub)
              │                        ↓
              │                    P3 (行为验证)   ←─ P0 (transition.reason)
              │                        ↓
              └──→ P4 (L1 caption) ──→ P5 (L2 图片召回)
                                         ↓
Phase 3 §4.4 (LlamaIndex) ←─ 绑定 P1/P4/P5

Phase 3 §4.1 (Claude) ── 独立 ──
Phase 3 §4.2 (PixelNavigate) ── 独立 ──
Phase 3 §4.3 (Critic) ── 独立 ──
Phase 3 §4.5 (技术债) ── 独立 ──
Phase 3 §4.1-§4.3 的场景 A/B/C (fork) ── 绑定 P2 + P6 ──
```

---

## 7. 关键 file:line 索引

### 7.1 phase 1 已实现的文件

| 文件 | 内容 |
|---|---|
| `src/two_tier_graph/state.py` | `TwoTierState` TypedDict + `CurrentPose` + `RoundTrace` |
| `src/two_tier_graph/resources.py` | `Resources` dataclass |
| `src/two_tier_graph/nodes.py` | 7 node functions |
| `src/two_tier_graph/edges.py` | `after_guard`, `after_memory` |
| `src/two_tier_graph/graph.py` | `build_two_tier_graph()` |
| `src/two_tier_graph/providers.py` | `LLMProvider` ABC + `MimoProvider` + `ClaudeProvider` stub |
| `src/two_tier_graph/tools.py` | `ToolRegistry` + `ActionTool` ABC + 5 default tools |
| `src/two_tier_graph/entrypoint.py` | `run_episode_two_tier_langgraph()` |
| `tests/test_two_tier_graph.py` | 18 deterministic unit tests |

### 7.2 phase 2-3 待创建的文件

| 文件 | Phase | 内容 |
|---|---|---|
| `src/two_tier_graph/visual_memory.py` | P1/P4/P5 | `VisualMemoryIndex` + `CaptionStore` + `ImageRecallStore` |
| `src/two_tier_graph/prompt_sections.py` | P2 | `PromptSection` + `build_planner_prompt()` |
| `src/two_tier_graph/stall_detection.py` | P3 | `StallSignal` + `detect_stall()` |
| `src/two_tier_graph/fork.py` | P6 | `CacheSafeParams` + `ForkSubagentTool` stub |
| `src/two_tier_graph/semantic_memory.py` | Phase 3 §4.4 | `SemanticMemoryStore` (LlamaIndex) |
| `src/shared_helpers.py` | Phase 3 §4.5 | 去重后的 shared helpers |

### 7.3 原 codebase file:line 索引（phase 1 保留）

| Component | Location |
|---|---|
| Two-Tier main loop | `src/agent_workflow.py:1087` (`run_episode_two_tier`) |
| For-loop | `:1442` |
| Planner call | `:1538` |
| Stage 6.5 frontier sub-call | `:1313` (`_select_frontier_with_vlm`), invoked `:1567` |
| Guards (3) | `:1582-1592` |
| `_first_available_action` fallback | `:1406-1435` |
| `_is_valid_object_desc` / `_NAV_OBJ_INVALID` | `:1745` / `:1735` (active defs; duplicates at `:915`/`:905`) |
| Submit success path | `:1614-1630` |
| Executor call | `:1633` |
| Memory update (notebook+scene_graph+rejected) | `:1637-1663` |
| Entity exhaustion `continue` | `:1665-1672` |
| Step-budget `break` | `:1675-1677` |
| Fallback `submit_best_guess` | `:1681-1702` |
| Planner class / prompt / parser | `src/agent_planner.py:68` / `:39` / `:139` |
| PlannerAction dataclass | `src/agent_planner.py:23` |
| Executor dispatch (5 actions) | `src/agent_executor.py:291` |
| Executor pose state | `src/agent_executor.py:42-43` |
| 5 memory stores | `agent_memory.py:25`, `scene_graph_memory.py:84`, `agent_notebook.py:96`, `agent_evidence.py:10`, `agent_context.py:20` |
| Notebook exhaustion (3-visit) | `agent_notebook.py:172` |
| Scene-graph active query | `scene_graph_memory.py:256` |
| TSDF occupancy state | `tsdf_planner.py:147-152` |
| Step counter global | `agent_tools.py:191` |
| `call_vlm` (HTTP) | `agent_workflow.py:179` |
| LLM config | `const.py:11-21` |
| HM-GE legacy loop | `agent_workflow.py:410` (`run_episode`) |
| HM-GE entry | `run_hmge_evaluation.py` |
| Two-Tier entry | `run_two_tier_aeqa_evaluation.py:311` (call), `:94` (`METHOD_CONFIGS`) |
| RunLogger trace API | `src/run_logger.py:163/176/259/294/328/363/372` |

---

## 8. Claude Code 源码位置索引

本文档借鉴的 Claude Code 机制在 `cc-haha/src/`（fork）和 `claude-code-analysis/analysis/`（中文解析）中的位置：

| 机制 | cc-haha 源码 | claude-code-analysis 解析 |
|---|---|---|
| 主循环 + transition.reason | `src/query.ts:244` `queryLoop()`, `:271` `state.transition` | `01-architecture-overview.md` §5.3 |
| 分层压缩（5 passes） | `src/services/compact/` (`autoCompact.ts`, `reactiveCompact.ts`, `microCompact.ts`, `snipCompact.ts`), `src/query/tokenBudget.ts` | `04f-context-management.md` §3-4 |
| Tool 系统 + buildTool fail-closed | `src/Tool.ts:362` `Tool` 接口, `:783` `buildTool()`, `src/services/tools/StreamingToolExecutor.ts` | `04b-tool-call-implementation.md` §2-6 |
| Fork subagent + CacheSafeParams | `src/tools/AgentTool/forkSubagent.ts`, `src/utils/forkedAgent.ts:57` | `04h-multi-agent.md` §4.2 |
| Typed memory + MEMORY.md + 非阻塞预取 | `src/memdir/` (`memdir.ts`, `findRelevantMemories.ts`, `memoryScan.ts`), `src/services/SessionMemory/`, `src/services/extractMemories/` | `04-agent-memory.md` §3-8 |
| System prompt section registry + cache boundary | `src/constants/prompts.ts:444` `getSystemPrompt()`, `src/constants/systemPromptSections.ts`, `src/utils/queryContext.ts` | `04g-prompt-management.md` §2-7 |
| TodoWrite verification nudge | `src/tools/TodoWriteTool/TodoWriteTool.ts` | （未独立成章，源码可读） |
| Sidechain transcript | `src/tools/AgentTool/runAgent.ts:248` `recordSidechainTranscript` | `04h-multi-agent.md` §4 |
| `tengu_slim_subagent_claudemd` | `src/tools/AgentTool/built-in/` | `04h-multi-agent.md` §4.2 |

---

## 9. 范围边界

### 9.1 永久不动

- `src/agent_workflow.py` 的 `run_episode_two_tier` 和 `run_episode` —— legacy fallback path，永久保留
- `run_hmge_evaluation.py` —— HM-GE legacy，永久不动
- `src/agent_planner.py:39` `PLANNER_SYSTEM_PROMPT`、`:1255-1404` `_build_*` helpers、`PlannerAction` schema —— 除非 lever 显式要求（如 P3 stall hint），preserved verbatim
- `run_episode_two_tier` 的 return-dict contract —— LangGraph entrypoint 永远 reproduce exact result dict
- `3dmem` conda env 依赖 —— 不动 cuda/torch 等配置困难项

### 9.2 各 Phase 不做（直到对应 Phase）

- Phase 2 P0-P6 不实现 Phase 3 的任何内容
- P6 不实现 fork 场景 A/B/C（Phase 3）
- P1/P4/P5 不实现 LlamaIndex 整合（Phase 3 §4.4）

### 9.3 明确永不做

- 不做 swarm（Claude Code 的多 child 并行 + mailbox），只做单 child fork
- 不做 child-to-parent 反向通信（SendMessage），fork 是单向的
- 不 formalize HM-GE legacy（保持 as-is）
- 不 changing the 5 original actions 的核心语义（除非 lever 显式要求）
