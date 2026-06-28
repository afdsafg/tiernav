# TierNav LangGraph Agent Master Plan 实施状态

**日期**: 2026-06-28
**对照文档**: `docs/plans/2026-06-28-tiernav-langgraph-agent-master-plan.md`

---

## 总览

| 阶段 | 状态 | 完成度 |
|---|---|---|
| **Phase 1** LangGraph 状态机形式化 | ✅ 完整 | 100% |
| **Phase 2** P0-P6 Claude Code 模式借鉴 | 🚧 部分 | ~40% |
| **Phase 3** 扩展完善 | 🚧 部分 | ~30% |

**baseline 固化**（Phase 1 遗留待办）:
- AEQA-41 langgraph baseline: LLM Match 50.6%, Acc 50.61%, SPL 36.82%
- AEQA-41 legacy baseline: LLM Match 50.6%, Acc 50.61%, SPL 48.33%
- Accuracy 一致（行为保持验证通过），SPL 差异源于 langgraph 路径更长

---

## Phase 1 — LangGraph 状态机形式化 ✅

**状态**: 完整实现，baseline 已固化

### 已完成
- `src/two_tier_graph/` 8 模块（state/resources/nodes/edges/graph/providers/tools/entrypoint）
- 7 nodes, 2 conditional edges → 后扩展为 9 nodes（+stall_recovery）, 3 edges（+after_critic）
- `--engine {legacy,langgraph}` flag 接入 `run_two_tier_aeqa_evaluation.py`
- `MimoProvider` 默认，`ClaudeProvider` stub
- 5 default tools via `ToolRegistry`
- AEQA-41 baseline scored

### Phase 1 遗留待办
| 待办 | 状态 |
|---|---|
| Full AEQA-41 benchmark | ✅ done |
| 10 题 dev subset 对照 | ⏳ 未做 |
| `groundingdino/` 加入 `.gitignore` | ⏳ 未做 |
| `cfg/eval_aeqa.yaml` scene path commit 决策 | ⏳ 未做 |

---

## Phase 2 — P0-P6 Claude Code 模式借鉴 🚧

### P0 — 结构性元模式 ✅ 完整

**元模式 1：分层压缩 + 显式契约**
- `compress_threshold`（默认 5）提取为 state 字段
- `index_refresh_interval`（默认 3）提取为 state 字段
- `memory_update_node` 显式化三层调用（L_raw / L_compressed / L_index）
- `compression_log` state 字段记录每层统计

**元模式 2：`transition.reason` 一等公民**
- `TransitionReason` enum（CONTINUE/ROUND_BUDGET/EXHAUSTED/STEP_BUDGET/STALL_RECOVERY/VERIFY_BEFORE_FALLBACK）
- `Transition` dataclass
- `last_transition` / `transition_log` state 字段
- `after_memory` 边基于 `last_transition.reason` 路由

### P1 — 视觉记忆 L0 索引 ✅ 完整

- `src/two_tier_graph/visual_memory.py` 含 `VisualMemoryIndex` 类
- `memory_update_node` 调用 `VisualMemoryIndex.update()` 构建 L0 行
- `build_context_node` 读取 `get_index_text()` 注入 `memory_summary` 段
- state 字段：`l0_index_text` / `visual_memory_state` / `loaded_snapshot_ids`
- CLIP embedding 复用 `agent_memory.py` 已有计算

### P2 — Prompt Cache 优化 🚧 部分

**已实现**:
- `src/two_tier_graph/prompt_sections.py` 含 `PromptSection` dataclass + `build_planner_prompt()`
- cacheable/non-cacheable 段区分

**未实现**:
- `MimoProvider.decide()` 仍接受单一字符串，未接 `list[PromptSection]`
- cache boundary 标记未插入（provider 侧未改造）
- `RunLogger` 未加 `log_prompt_cache()` 方法

### P3 — 行为验证机制 🚧 部分

**已实现**:
- `src/two_tier_graph/stall_detection.py` 含 `StallSignal` dataclass + `detect_stall()` 纯函数
- `stall_recovery_node` 存在
- `after_memory` 边有 `stall_recovery` 路由
- `submit_node` verification nudge（`verification_attempted` flag）
- state 字段：`stall_signal` / `verification_attempted`

**未实现**:
- `detect_stall()` 定义存在但 **未被 `memory_update_node` 调用** → stall 路由实际死代码
- `build_context_node` 未读取 `stall_signal` 注入 hint

### P4 — 视觉记忆 L1 caption 🚧 部分

**已实现**:
- `CaptionStore` 类骨架（磁盘缓存路径结构）

**未实现**:
- caption 生成（VLM 调用）未接线
- CLIP 检索 top-K caption 未实现
- `build_context_node` 未注入 caption

### P5 — 视觉记忆 L2 图片召回 🚧 部分

**已实现**:
- `ImageRecallStore` 类骨架
- state 字段 `need_visual_recall`

**未实现**:
- `build_context_node` 未检查 `need_visual_recall` 加载 snapshot
- token 预算硬约束未实现
- LRU 去重未与 P1 共享

### P6 — 多 agent fork 机制 📦 stub

- `src/two_tier_graph/fork.py` 含 `CacheSafeParams` dataclass + `ForkSubagentTool` stub
- `run()` raises `NotImplementedError`
- `ToolRegistry` 注册 `ForkSubagentTool`
- sidechain transcript 路径结构建好
- 场景 A/B/C 未实现（Phase 3 范围）

---

## Phase 3 — 扩展完善 🚧

### §4.1 Claude Provider ✅ 完整

- `ClaudeProvider` 从 stub 改为完整实现
- `anthropic` SDK with native tool-use
- 5 actions 转 Anthropic tool definition
- `decide()` 解析 `tool_use` block → `PlannerAction`
- `decide_raw()` 处理 Claude text response
- `cache_control: ephemeral` 标记支持

### §4.2 PixelNavigateTool 📦 stub

- `tools.py` 含 `PixelNavigateTool` stub
- `run()` 未实现（backproject 未接线）
- `tsdf_planner.backproject()` 未实现

### §4.3 Critic Node 🚧 部分

**已实现**:
- `critic_node` 存在（插入 planner → critic → loop_guard）
- `after_critic` 条件边（approve/veto）
- state 字段 `critic_veto` / `critic_feedback`
- `cfg.critic.enabled` flag（默认 false）

**未实现**:
- critic 评估逻辑是 stub（未调 `llm_provider.decide_raw()` 做自由文本评估）
- veto 后重新决策带反馈未完整测试

### §4.4 LlamaIndex 语义记忆整合 🚧 部分

**已实现**:
- `src/two_tier_graph/semantic_memory.py` 含 `SemanticMemoryStore` 类

**未实现**:
- `MemoryStore.query` 未改为调用 `SemanticMemoryStore.query()`（保留原 keyword filter）
- CLIP embedding 未索引到 LlamaIndex
- `build_context_node` active-query 未 hook 到 LlamaIndex
- LlamaIndex 未安装到 conda env

### §4.5 技术债清理 🚧 部分

**已实现**:
- `src/shared_helpers.py` 存在（HM-GE/Two-Tier helper 去重）

**未实现**:
- `silent_perception_step._step_counter` 仍是 function-attribute global，未移入 `TwoTierState`
- `_navigate_to_target_with_agent_step` 未改造

---

## A/B 归因协议执行情况

### §5.4 Phase 1 baseline 固化
- ✅ Full AEQA-41 benchmark run（langgraph + legacy）
- ⏳ 10 题 dev subset side-by-side 对照未执行

### 单变量对照实验
- 尚未启动任何 P0-P6 lever 的 A/B 实验
- 当前工作（GOATBench 统一工作流）是独立扩展，非 master plan 中的 lever

---

## 当前工作（独立于 master plan）

### GOATBench 统一工作流

**目标**: 将 GOATBench 评估接入 LangGraph Two-Tier 工作流，以 AEQA 工作流为基础扩展

**设计**: `docs/plans/2026-06-28-goatbench-unified-workflow-design.md`
**实现**: `docs/plans/2026-06-28-goatbench-unified-workflow-implementation.md`

**架构**: GOATBench 是完整工作流，AEQA 是子集。新增 `note_node`（任务分类）+ `check_arrival_node`（欧氏距离 < 1m 终止判定）。AEQA 路径行为不变（变量隔离）。

**进度**:
- ✅ 代码实现完成（10 tasks + 4 review fixes，14 commits）
- ✅ 服务器 import + 图编译验证通过（12 nodes: +note +check_arrival）
- 🚧 AEQA 回归验证中（确认变量隔离）
- ⏳ GOATBench 冒烟测试
- ⏳ 全量 GOATBench 评估

**关键修复**:
- `call_vlm` fallback 用 `QWEN_PLANNER_*`（完整 URL）避免 404
- `episode_payload` JSON 过滤 `_-prefixed` 非序列化字段
- `final_pts` ndarray → list 保证 JSON 安全
- `goal_metadata` 仅含描述文本，不 leak 真值坐标
- `observed_goal_positions` last-writer-wins 避免历史误匹配累积

---

## 文件清单

### Phase 1 已实现
| 文件 | 内容 |
|---|---|
| `src/two_tier_graph/state.py` | `TwoTierState` + `TransitionReason` + `Transition` + GOATBench 扩展字段 |
| `src/two_tier_graph/resources.py` | `Resources` + `goal_type`/`goal_metadata` |
| `src/two_tier_graph/nodes.py` | 11 nodes（init/build_context/planner/critic/loop_guard/executor/memory_update/stall_recovery/submit + note/check_arrival） |
| `src/two_tier_graph/edges.py` | 4 edges（after_guard/after_memory/after_submit/after_critic + after_check_arrival） |
| `src/two_tier_graph/graph.py` | `build_two_tier_graph()` 12 nodes |
| `src/two_tier_graph/providers.py` | `LLMProvider` + `MimoProvider` + `ClaudeProvider`（完整） |
| `src/two_tier_graph/tools.py` | `ToolRegistry` + 5 tools + `ForkSubagentTool` stub + `PixelNavigateTool` stub |
| `src/two_tier_graph/entrypoint.py` | `run_episode_two_tier_langgraph()` + GOATBench 参数 |
| `tests/test_two_tier_graph.py` | 18 deterministic unit tests |

### Phase 2-3 部分实现
| 文件 | 模块 | 状态 |
|---|---|---|
| `src/two_tier_graph/visual_memory.py` | P1/P4/P5 | `VisualMemoryIndex` ✅ / `CaptionStore` 🚧 / `ImageRecallStore` 🚧 |
| `src/two_tier_graph/prompt_sections.py` | P2 | `PromptSection` ✅ / provider 未接 🚧 |
| `src/two_tier_graph/stall_detection.py` | P3 | `StallSignal` + `detect_stall()` ✅ / 未调用 🚧 |
| `src/two_tier_graph/fork.py` | P6 | stub only 📦 |
| `src/two_tier_graph/semantic_memory.py` | §4.4 | `SemanticMemoryStore` ✅ / 未接线 🚧 |
| `src/shared_helpers.py` | §4.5 | helper 去重 ✅ / `_step_counter` 未迁移 🚧 |

---

## 下一步优先级

1. **完成 GOATBench 统一工作流**（当前工作）
   - AEQA 回归验证通过
   - GOATBench 冒烟测试
   - 全量 GOATBench 评估 + 评分

2. **P3 stall 检测激活**（最高性价比 lever）
   - `detect_stall()` 已实现，只需在 `memory_update_node` 调用
   - `build_context_node` 注入 hint
   - 可独立 A/B，不动 prompt template

3. **P2 prompt cache 接线**
   - `MimoProvider.decide()` 接 `list[PromptSection]`
   - 插入 cache boundary 标记

4. **Phase 1 遗留**: 10 题 dev subset 对照、`.gitignore` 清理
