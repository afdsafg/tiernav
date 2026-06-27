# Claude Code 模式借鉴 → VLN Agent 设计选项

**日期**: 2026-06-28
**状态**: 设计文档（已分节确认）
**基础**: 在 `2026-06-24-two-tier-refactor-design.md` 完成的 LangGraph phase 1 形式化之上扩展未来 lever
**动机**: Research capability —— 通过借鉴 Claude Code 的成熟 agent 机制，提升 tiernav VLN agent 的性能，同时保持变量隔离以支持 benchmark 归因

---

## 1. 总览与范围

### 1.1 本文档是什么

一份从 Claude Code 借鉴四组机制到 tiernav LangGraph runtime 的设计选项目录。每个 section 是**独立 lever**，保留变量隔离原则——任意子集可单独拉动、单独 A/B 对照 `--engine langgraph` baseline。

### 1.2 本文档不是什么

不是实现计划。代码级决策（文件路径、精确 API）后置。除非显式拉动某个 lever，不对现有 benchmark 数字产生行为改变。

### 1.3 四组 lever 与 plan §8 的对应

| # | Lever | 对应 plan §8 | 借鉴的 Claude Code 机制 |
|---|---|---|---|
| 1 | 视觉记忆层 | §8 #3（LlamaIndex 语义记忆） | `memdir/` typed memory + `MEMORY.md` 索引 + `findRelevantMemories` 非阻塞预取 + `loadedNestedMemoryPaths` LRU 去重 |
| 2 | 行为验证机制 | 新增，原 plan 无 | `TodoWriteTool` verification nudge + `transition.reason` 一等公民恢复标记 |
| 3 | Prompt cache 优化 | 新增，原 plan 无 | `systemPromptSection` registry + `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` 缓存分界 |
| 4 | 多 agent fork | §8 #2（multi-agent 拆分） | `forkSubagent` + `CacheSafeParams` 让 fork 命中父 agent prompt cache |

### 1.4 两个结构性元模式

贯穿所有 lever，单独成节（Section 6）作为基础设施，不作为可拉动的 lever：
- **分层压缩 + 显式契约**：替代 VLN 当前 ad-hoc 的 `RoundTrace → EvidenceNotebook → 最终答案` 链
- **`transition.reason` 一等公民**：扩展 VLN 现有 `after_memory` 边返回值为可测 state 字段

### 1.5 范围边界

明确不在范围内：
- Claude provider swap（plan §8 #1，独立）
- pixel→backproject→navigate 等新工具（plan §8 #4，独立）
- `3dmem` conda env 依赖修改
- 本文档阶段不实现任何 lever 的代码
- 不修改 phase 1 已完成的 LangGraph runtime

---

## 2. 视觉记忆层

### 2.1 VLN baseline 缺口

每轮 planner 看到 3 张当前视角 + topdown；历史信息以文本形式存在于 `RoundTrace`（全量）和 `EvidenceNotebook`（压缩）；历史 snapshots 写磁盘但从不进 prompt；`agent_memory.py:50` 算的 CLIP embedding 在 `:62` 的 `query()` 里被 keyword filter + linear scan 绕过，**算了没用**。

### 2.2 Claude Code 借鉴点

typed memory + `MEMORY.md` ≤200 行索引（只有索引自动加载，正文按需读）+ `findRelevantMemories` 非阻塞预取 + `loadedNestedMemoryPaths` LRU 去重。

### 2.3 关键差异

Claude Code 是纯文本 memory，LLM "读"文件即可；VLM 必须把图片**直接放进 messages**才能"看到"。Claude Code 的"按需读正文"对 VLM 不直接适用——图片要么进 prompt 要么不进，没有"读"的中间态。

### 2.4 设计选项：C（分层混合）

| 选项 | 注入物 | 检索 | token 成本 | 信息保真 |
|---|---|---|---|---|
| **C. 分层混合** | 索引常驻 + 按需召回 caption 或图片 | CLIP + 文本双层 | 可控（分层弹性） | 分层保真 |

**选 C 的理由**：
1. **最贴合 Claude Code 哲学**：索引常驻（对应 `MEMORY.md` ≤200 行）、正文按需（对应 `findRelevantMemories`），三层结构 = Claude Code 的"folder+index+正文"
2. **闭合 CLIP gap**：CLIP embedding 用在中层检索，不再算了没用
3. **变量隔离友好**：可分三步上线——先做索引层（A/B 对照 baseline，无新视觉 token），再加 caption 层（纯文本增量），再加图片召回层（视觉 token 增量）。每步独立可归因
4. **`build_context_node` 是天然 hook 点**：plan §8 #3 已标注，与 Claude Code 的 turn-start prefetch 同构

### 2.5 三层结构

- **L0 索引层**（常驻 prompt）：每条 ≤1 行，字段 = `[round, pose, object_class, one_line_desc]`。20 轮内 ≤20 行，类比 `MEMORY.md` ≤200 行约束
- **L1 caption 层**（按需召回）：每张 snapshot 离线生成 caption（VLM 单次调用，结果缓存磁盘）。CLIP 检索 top-K caption 注入
- **L2 图片层**（按需召回）：极少数情况（如"我刚才看到的是红色杯子还是蓝色？"）召回原始 snapshot。受 token 预算硬约束

### 2.6 非阻塞预取

借鉴 `findRelevantMemories`：`build_context_node` 开始时启动 CLIP 检索异步任务，settle 才注入，超时 fallback 到 L0 索引常驻——不阻塞 planner 决策。

### 2.7 LRU 去重

借鉴 `loadedNestedMemoryPaths`：维护 `loaded_snapshot_ids` set，避免同一 snapshot 跨轮重复注入。

### 2.8 L0 索引更新频率

`index_refresh_interval` 可配置，默认 3 轮。与 Section 4 prompt cache 耦合：L0 纳入 cacheable 段，更新过频会破坏 cache 命中。

---

## 3. 行为验证机制

### 3.1 VLN baseline 缺口

冒烟测试观察：agent 连续 5 轮调用 `explore_frontier(14)`，`steps_taken=0`，从未移动；Stage 6.5 报"no frontier images available"；最后被迫 fallback submit。整个流程没有任何结构性检查提醒"你在重复无效动作"。

### 3.2 Claude Code 借鉴点

- `TodoWriteTool` verification nudge：当 ≥3 任务关闭而无验证步骤时，工具返回里**结构性注入**提示（不是 system prompt 里的静态规则）
- `transition.reason` 一等公民：恢复路径用枚举标签标记，不靠解析消息内容
- `withMemoryCorrectionHint()`：取消/错误时给模型一个"刚才发生了什么"的注释

### 3.3 关键差异

Claude Code 是工具返回值里注入提示，VLN 的 planner 是 graph node，提示应注入到**下一轮 planner 的 prompt 上下文**里，而非工具返回。

### 3.4 设计：三层结构

#### 层 1：重复动作检测器（deterministic，在 `memory_update_node`）

`memory_update_node` 是每轮结束的汇聚点，已能看到 `action_history`。加一个纯函数 `detect_stall(action_history, steps_taken) -> Optional[StallSignal]`：

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

#### 层 2：恢复路径标签（扩展 `after_memory` 边）

当前 `after_memory` 返回 4 路：`round_budget / exhausted / step_budget / continue`。扩展为：

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

`stall_recovery` 路由到一个新的 `stall_recovery_node`（不做实际工作，只把 `stall_signal.hint` 转成 `RecoveryNote` 追加到 `round_traces`），然后回到 `build_context`。`transition.reason` 等价于"为什么走了这条路"的可测标签。

#### 层 3：验证 nudge（在 `submit_node` 的 fallback 入口）

当 `after_memory` 路由到 `fallback_submit` 时，当前 `submit_node` 直接调 `submit_best_guess`。借鉴 TodoWrite verification nudge，在 fallback 路径加一个**前置检查**：

```python
def submit_node(state, resources):
    is_fallback = state.get("submit_reason") == "fallback"
    if is_fallback and not state.get("verification_attempted"):
        # 不直接提交，先走一次 verification
        return {"verification_attempted": True, ...}  # 路由回 build_context
    # 真正提交
```

verification 的内容 = 让 planner 显式回答"为什么现在提交而不是再探索一轮"，答案写进 `round_traces`，第二次 fallback 才真提交。**只在 fallback 路径触发，不在成功 submit 路径触发**。

### 3.5 Graph 结构变更

plan §1 原设计 "7 nodes, 2 conditional edges" 升级为 **"8 nodes, 3 conditional edges"**：
- 新增节点：`stall_recovery_node`
- 新增条件边：`after_memory` 增加 `stall_recovery` 路由
- `submit_node` 内部分支增加 verification 前置检查（非新节点）

### 3.6 与 plan 的关系

plan §8 原本只有"multi-agent split"作为 graph 结构 lever，**行为验证机制是新增 lever**。但它是纯结构性改动，不动 prompt template、不动 LLM、不动工具实现，**变量隔离成立**——A/B 对照 baseline 只看 `stall_signal` 是否注入。

---

## 4. Prompt Cache 优化

### 4.1 VLN baseline 缺口

每轮 planner 调 `MimoProvider.decide()`，重建完整 prompt（系统指令 + 动作 schema + 当前视图 + 历史上下文 + 任务问题）。整包发 API，无 prompt cache 命中，VLM 调用成本高。

### 4.2 Claude Code 借鉴点

- `getSystemPrompt()` 返回 `string[]`（每段可独立缓存），不是单一字符串
- `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` 标记把 prompt 切成"静态前缀"（跨轮缓存）和"动态后缀"（每轮重算）
- `systemPromptSection` registry 区分 `systemPromptSection`（跨轮 memoized）vs `DANGEROUS_uncachedSystemPromptSection`（每轮重算，会破坏 cache）
- subagent fork 用 `CacheSafeParams` 保证 fork 命中父 agent 的 prompt cache：相同 system prompt、tools、model、message 前缀

### 4.3 关键差异与实现前置条件

Claude Code 用 Anthropic prompt cache（`cache_control` 标记）；VLN 后端是 mimo-v2.5 / qwen3-vl-flash。**实现前置条件：provider 支持 prompt cache**。当前主流 VLM provider 大部分支持 prompt cache。本节设计**与 provider 无关**：无论 cache 是否生效，prompt 分段本身仍有价值（成本可观测、便于 A/B）。先把功能做进去，provider 支持时自动受益。

### 4.4 设计

#### 1. Prompt 分段 registry

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
        PromptSection("memory_index", L0_INDEX_TEMPLATE, cacheable=True),  # Section 2 L0
        PromptSection("reasoning_history", _build_reasoning_history(state), cacheable=False),
        PromptSection("current_views", [view["image_b64"] for view in state["current_views"]], cacheable=False),
        PromptSection("topdown", state["topdown_b64"], cacheable=False),
        PromptSection("active_query", state["question"], cacheable=True),
    ]
```

#### 2. Cache boundary 标记

`MimoProvider.decide()` 序列化时在最后一个 `cacheable=True` 段后插入边界标记（如 Anthropic 格式是 `cache_control: ephemeral`；非 cache provider 忽略）。静态前缀 = task_instruction + action_schema + memory_index + active_query，跨轮稳定。

#### 3. 触发 cache miss 的字段白名单

借鉴 `DANGEROUS_uncachedSystemPromptSection` 命名警告，维护一份"会破坏 cache 的字段"白名单：
- `reasoning_history`（每轮增长）
- `current_views`（每轮位姿变化）
- `topdown`（每轮更新）
- `memory_index` 里 Section 2 的 L0 如果当轮有新 snapshot 也会变（trade-off：索引更新频率 vs cache 命中率，由 `index_refresh_interval` 控制）

#### 4. 与 Section 2/3 的耦合

- Section 2 L0 索引层应纳入 `cacheable=True`——但只在每 `index_refresh_interval` 轮（默认 3）更新一次（`memory_update_node` 控制），避免每轮 miss
- Section 3 的 `StallSignal.hint` 注入到 `reasoning_history` 段（`cacheable=False`），不破坏静态前缀
- Section 3 的 `RecoveryNote` 同理

#### 5. Fork 场景（与 Section 5 联动）

未来 multi-agent fork 时，fork child 必须复用父 agent 的静态前缀才能命中 cache。借鉴 `CacheSafeParams`：fork 时传 `{system_prompt_sections, tool_registry, model, message_prefix}`，child 必须用相同顺序、相同内容的 cacheable 段。

#### 6. 成本可观测

`RunLogger` 记录每轮：cacheable token 数 / non-cacheable token 数 / cache hit 或 miss。A/B 时可量化收益。

### 4.5 与 plan 的关系

plan 原本无 prompt cache lever。本节是新增，但**完全在 `build_context_node` 和 `MimoProvider` 内部**，不动 graph 结构、不动 state schema、不动工具。变量隔离成立——A/B 对照只看 `PromptSection` 分段是否启用。

---

## 5. 多 agent fork

### 5.1 VLN baseline 缺口

当前是单 agent 两层循环（planner + executor），所有决策在一个上下文里完成。plan §8 #2 提到"multi-agent split"作为未来 lever，但没有拆分机制——一旦拆分，子 agent 会从头重建 prompt，成本和延迟都高。

### 5.2 Claude Code 借鉴点

- `forkSubagent.ts` + `CacheSafeParams`（`forkedAgent.ts:57`）：fork child 继承父 agent 的 system prompt、tools、model、message 前缀，**命中父 agent 的 prompt cache**
- 普通子 agent 用 `createSubagentContext` 建空上下文，不继承
- fork 是单向通信：child 的最终 assistant message 成为父 agent 的 tool result
- `recordSidechainTranscript`：fork 的完整 transcript 写磁盘，不进父上下文
- `tengu_slim_subagent_claudemd`：read-only 子 agent（Explore/Plan）剥离 CLAUDE.md 省 token

### 5.3 关键差异

- Claude Code 是通用编码 agent，fork 用于"并行探索多个方案"；VLN 是具身决策，fork 的语义需要重新定义
- VLN 的"消息前缀"含视觉 token（current_views + topdown），fork 时若复用前缀必须连图片一起复用，cache 命中条件更严格

### 5.4 VLN 场景下的 fork 语义

三个候选 fork 场景，**本节只定义机制，不实现具体场景**：

| 场景 | fork 时机 | child 职责 | 返回值 |
|---|---|---|---|
| **A. 子路线探索** | planner 选 `explore_frontier(k)` 时 | fork 出 N 个 child 各自模拟走 k 后再走 1-2 步，返回最 promising 的子路径 | 路径评分 + 文本描述 |
| **B. 区域验证** | planner 怀疑某对象在历史区域时 | fork child 加载该区域历史 snapshots，回答"X 是否出现过" | 是/否 + 证据 snapshot_id |
| **C. 子任务委托** | planner 拆分复杂任务时 | fork child 独立完成子任务（如"找到所有椅子"） | 子任务结果 |

### 5.5 `CacheSafeParams` 在 VLN 的等价

fork 时传：

```python
@dataclass
class CacheSafeParams:
    system_prompt_sections: list[PromptSection]  # Section 4 的 cacheable 段
    tool_registry: ToolRegistry
    llm_provider_config: LLMConfig  # 不传 provider 实例，传 config
    message_prefix: list[ContentBlock]  # 父 agent 当前轮的 prompt 前缀
    # VLN 特有：是否复用父的当前视图
    inherit_current_views: bool = True
```

`message_prefix` 包含父 agent 的 `task_instruction + action_schema + memory_index + reasoning_history + current_views + topdown + active_query`。child 用相同顺序、相同内容的 cacheable 段（Section 4），命中父的 prompt cache。

`inherit_current_views=True` 时 child 共享父的 current_views（同一位姿）；`=False` 时 child 用自己的虚拟位姿（场景 A/B）。**已确认可接受：场景 A 的 child 必须从父的当前位姿出发**。

### 5.6 Fork 调度（伪代码 stub 接口）

借鉴 Claude Code 的 `AgentTool` 设计，`ToolRegistry` 注册一个新工具 `ForkSubagentTool`。本阶段用伪代码写假接口，作为纯结构性预留（类比 `ClaudeProvider` stub）：

```python
class ForkSubagentTool(ActionTool):
    """Fork 子 agent 执行子任务，命中父 agent 的 prompt cache。
    
    本阶段为结构性预留，run() 返回 NotImplementedError。
    场景 A/B/C 的具体 fork 逻辑后续单独拉动。
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
        # ===== 本阶段 stub：纯结构性预留 =====
        # 真实实现流程（伪代码）：
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
            "are future levers, not yet implemented."
        )
    
    def _build_cache_safe_params(self, state, resources) -> CacheSafeParams:
        """构造 CacheSafeParams 让 child 命中父的 prompt cache。"""
        # 复用父的 cacheable 段（Section 4）
        cacheable_sections = [s for s in build_planner_prompt(state, resources) if s.cacheable]
        return CacheSafeParams(
            system_prompt_sections=cacheable_sections,
            tool_registry=resources.tool_registry,
            llm_provider_config=resources.llm_provider.config,
            message_prefix=state["message_prefix"],  # 含视觉 token
            inherit_current_views=True,
        )
```

### 5.7 Sidechain transcript

借鉴 `recordSidechainTranscript`：fork 的完整 state（不含 heavy resources）写磁盘到 `output_dir/sidechains/<parent_round>_<fork_id>.jsonl`，**不进父 state**。父 agent 只看最终返回值。调试时可读 sidechain。

### 5.8 与 Section 2/3/4 的耦合

- **Section 2**：child 可读父的 L0 索引（cacheable，cache 命中），但 L1/L2 按需召回——child 自己的检索结果不进父上下文
- **Section 3**：child 的 `transition.reason` 独立，父不感知 child 是否走了 `stall_recovery`
- **Section 4**：child 必须用 `CacheSafeParams` 复用父的 cacheable 段才能命中 cache，这是 fork 的核心价值
- **`tengu_slim_subagent_claudemd` 等价**：场景 B/C 的 child 若是 read-only（不调 navigate 工具），可剥离 action_schema 段省 token——但会破坏 cache，trade-off 需实测

### 5.9 与 plan 的关系

plan §8 #2 原本只标注"multi-agent split"为未来 lever。本节给出**具体机制**（fork + CacheSafeParams + sidechain），但**不实现任何场景**——场景 A/B/C 留给后续单独拉动。本节是纯结构性预留：`ForkSubagentTool` 注册到 `ToolRegistry`，但 `run()` 在本阶段返回 `NotImplementedError`（类比 `ClaudeProvider` stub）。

### 5.10 范围边界

明确不做：
- 不实现场景 A/B/C 的具体 fork 逻辑
- 不做 swarm（Claude Code 的多 child 并行 + mailbox），只做单 child fork
- 不做 child-to-parent 反向通信（SendMessage），fork 是单向的

---

## 6. 结构性元模式

这两个模式贯穿前四个 lever，单独成节是为了把它们从"隐式设计选择"提升为"显式工程原则"。它们是**基础设施**，不作为可拉动的 lever。

### 6.1 元模式 1：分层压缩 + 显式契约

**VLN baseline 现状**：`RoundTrace`（全量）→ `EvidenceNotebook`（压缩）→ 最终答案。链路存在但每层契约隐式、阈值 hard-coded、无中间观测点。

**Claude Code 借鉴**：5-pass ordered compaction（`applyToolResultBudget` → `snipCompact` → `microCompact` → `contextCollapse` → `autocompact`），每层有明确输入输出契约和阈值触发条件。

**应用到 VLN**：把现有压缩链重述为三层契约：

| 层 | 输入 | 输出 | 触发条件 | 实现 |
|---|---|---|---|---|
| **L_raw** | 每轮 planner 推理 + action + observation | `RoundTrace` 对象追加到 `round_traces` | 每轮结束（`memory_update_node`） | 现有，不变 |
| **L_compressed** | `round_traces` 全量 | `EvidenceNotebook` 条目 | 轮数 ≥ `compress_threshold`（默认 5） | 现有，加阈值显式化 |
| **L_index** | `EvidenceNotebook` 条目 | L0 索引行（Section 2） | 每 `index_refresh_interval` 轮（默认 3） | 新增（Section 2 L0） |

**关键约束**（借鉴 Claude Code 每层契约）：
- 每层输出是**可序列化**的（不依赖 heavy resources），可独立测试
- 每层有**单调性**：L_compressed 输出条目数 ≤ L_raw 输入条目数；L_index 输出行数 ≤ L_compressed 输出条目数
- 每层失败**不阻塞上层**：L_index 构建失败时 fallback 到 L_compressed 全量注入

**`RunLogger` 观测点**：每轮记录每层的输入/输出条目数、token 估算、构建耗时。A/B 时可定位"哪层是瓶颈"。

### 6.2 元模式 2：`transition.reason` 一等公民

**VLN baseline 现状**：`after_memory` 边返回 `round_budget / exhausted / step_budget / continue` 四路，已经是不解析消息内容的枚举路由——**这其实就是 `transition.reason` 模式**，但没有系统化。

**Claude Code 借鉴**：`state.transition.reason` 字段标记恢复路径（`next_turn / max_output_tokens_recovery / reactive_compact_retry / stop_hook_blocking / token_budget_continuation`），测试可断言走了哪条路。

**应用到 VLN**：把 `after_memory` 的返回值和 Section 3 新增路径统一升级为 state 字段：

```python
class TransitionReason(str, Enum):
    CONTINUE = "continue"
    ROUND_BUDGET = "round_budget"           # 现有
    EXHAUSTED = "exhausted"                 # 现有，跳过 step_budget
    STEP_BUDGET = "step_budget"             # 现有
    STALL_RECOVERY = "stall_recovery"       # Section 3 新增
    VERIFY_BEFORE_FALLBACK = "verify_before_fallback"  # Section 3 新增

@dataclass
class Transition:
    reason: TransitionReason
    from_node: str
    to_node: str
    round_idx: int
```

每轮在 `memory_update_node` 末尾写入 `state["last_transition"]`，`after_memory` 边基于此返回。测试可断言：

```python
def test_stall_triggers_recovery():
    final_state = graph.invoke({...})
    assert any(t.reason == TransitionReason.STALL_RECOVERY 
               for t in final_state["transition_log"])
```

### 6.3 元模式与四个 lever 的关系

| Lever | 元模式 1（分层压缩） | 元模式 2（transition.reason） |
|---|---|---|
| Section 2 视觉记忆层 | L_index 层就是 L0 索引 | 无直接关系 |
| Section 3 行为验证 | 无直接关系 | 新增 `stall_recovery / verify_before_fallback` 两个 reason |
| Section 4 Prompt cache | L0 纳入 cacheable 段，L1/L2 不纳入 | 无直接关系 |
| Section 5 多 agent fork | child 的压缩链独立，不进父 | child 的 transition 独立 |

### 6.4 与 plan 的关系

这两个元模式是**对 plan §1-§7 已有设计的重述和升级**，不是新增 lever：
- 元模式 1 把 `RoundTrace → EvidenceNotebook` 链显式化，加阈值可配置
- 元模式 2 把 `after_memory` 四路返回升级为 state 字段 `last_transition`，并新增两个 reason

**变量隔离成立**：baseline 行为不变（现有 4 个 reason 的语义和路由完全保留），新增的 2 个 reason 只在 Section 3 lever 拉动时激活。

---

## 7. 实现顺序与 A/B 协议

### 7.1 实现顺序原则

遵循变量隔离原则（feedback memory）：每个 lever 可独立上线、独立 A/B 对照 `--engine langgraph` baseline。但 lever 间有**依赖关系**，不是完全平行：

```
元模式 1, 元模式 2 (基础设施, 无行为改变)
    ↓
Section 2 视觉记忆层 (L0 索引层依赖元模式 1 的 L_index 契约)
    ↓ (可并行)
Section 4 Prompt cache (L0 纳入 cacheable 段依赖 Section 2)
    ↓ (可并行)
Section 3 行为验证 (依赖元模式 2 的 transition.reason)
    ↓
Section 5 多 agent fork (依赖 Section 4 的 CacheSafeParams)
```

### 7.2 建议上线顺序

| 阶段 | Lever | 理由 |
|---|---|---|
| **P0** | 元模式 1 + 元模式 2 | 纯重构，无行为改变，为后续 lever 铺路。先做 `transition.reason` 升级（小），再做压缩链显式化（中） |
| **P1** | Section 2 L0 索引层 | 闭合 CLIP gap 的第一步，纯文本增量，无视觉 token 改变，可对照 baseline |
| **P2** | Section 4 Prompt cache 分段 | L0 纳入 cacheable 段，收益可观测（cacheable token 数 / hit rate）。**前置条件：provider 支持 prompt cache**（当前主流 VLM provider 大部分支持） |
| **P3** | Section 3 行为验证 | 纯 graph 结构改动，不动 prompt template。先上 stall 检测器（deterministic，可单测），再上 verification nudge |
| **P4** | Section 2 L1 caption 层 | VLM 离线生成 caption，磁盘缓存。视觉 token 增量，需 A/B |
| **P5** | Section 2 L2 图片召回层 | 极少数情况召回原始 snapshot，受 token 预算硬约束 |
| **P6** | Section 5 多 agent fork | 机制预留（`ForkSubagentTool` stub），场景 A/B/C 后续单独拉动 |

### 7.3 A/B 对照协议

每个 lever 上线后，跑同一个 10 题 dev subset，对比：

| 指标 | 来源 |
|---|---|
| `action_history` | state 字段 |
| `rounds_used` / `steps_taken` | state 字段 |
| `transition_log`（元模式 2 后） | state 字段 |
| 每层 token 估算（元模式 1 后） | `RunLogger` |
| cacheable / non-cacheable token 数（Section 4 后） | `RunLogger` |
| cache hit rate（Section 4 后，若 provider 支持） | API 响应 |
| 最终答案 accuracy | 评测脚本 |

### 7.4 归因原则：一次只动一个 lever

**原则**：每次 A/B 实验只激活一个新 lever，其余保持 baseline 状态。

**为什么需要这条原则**：

假设 P1（L0 索引层）和 P2（prompt cache 分段）同时上线，跑 10 题 dev subset，accuracy 从 baseline 的 6/10 变成 7/10。问题是：

- 是 L0 索引让 agent 看到了历史线索，导致提升？
- 还是 prompt cache 分段让 VLM 在相同 token 预算下推理更稳定，导致提升？
- 还是两者协同？
- 或者其实是负负得正（L0 单独会降 accuracy，cache 单独也会降，但合起来反而升）？

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

这样每个 lever 都有**干净的 Δ 归因**。

**为什么这是研究方法论，不是工程惯例**：

工程上常见"一次合并多个改动，跑一次测试，过了就过"——因为工程目标是"系统工作"，不是"理解为什么工作"。研究目标是**理解每个设计选择的因果贡献**，才能在论文里说"L0 索引层贡献 +X，prompt cache 贡献 +Y"。这正是 feedback memory 里"variable isolation over bundled changes"的来源——benchmark 归因需要单变量。

**代价**：每个 lever 都要跑一次 10 题 subset，总实验次数 = lever 数 + 1（baseline）。P0-P6 七个阶段 = 8 次实验。但每次实验都有**可解释的结果**，不会出现"不知道为什么变了"。

**边界情况——何时可以破例**：

只有一个：**两个 lever 强耦合，单独拉动无意义**。比如 Section 5 fork 强依赖 Section 4 的 CacheSafeParams——如果 fork 单独上线而 cache 没做，fork 的收益无法体现（cache miss 成本高）。这种情况在文档里需要显式标注"P5 与 P4 绑定验证"，而不是默认并行。

### 7.5 文档计划

本设计文档即最终产物，commit 到 `docs/plans/2026-06-28-claude-code-patterns-for-vln-design.md`。

**不写**：代码级实现计划（文件路径、精确 API）、具体 fork 场景 A/B/C 的实现、Claude provider 实现。

### 7.6 与现有 plan 的关系

- `docs/plans/2026-06-24-two-tier-refactor-design.md`（phase 1 LangGraph 形式化）是**已完成**的 baseline，本文档在其之上扩展未来 lever
- 本文档不修改 phase 1 plan，只补充 plan §8 的"未来 lever"具体机制
- plan §8 原有 #1（Claude provider swap）和 #4（新工具）**不在本文档范围**，保持独立

---

## 附录：Claude Code 源码位置索引

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
