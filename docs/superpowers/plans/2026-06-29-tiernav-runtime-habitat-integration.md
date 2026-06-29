# TierNav Runtime Habitat Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut TierNav over to the new `tiernav_runtime` as the only supported execution backend for real AEQA and GOATBench experiments, with real Habitat, TSDFPlanner, Executor, VLM, and scene-graph services behind OpenAI-compatible planner configuration.

**Architecture:** Keep `tiernav_runtime` graph-first and contract-first, but replace the current fake/dev-only wiring with explicit production services: a runtime environment service, a configurable OpenAI-compatible planner client, real tools wrapping `Executor`, a memory bridge, and task session managers. AEQA remains one question per episode with no cross-question memory; GOATBench becomes one long-lived episode session with persisted memory across subtasks and explicit 1m success validation.

**Tech Stack:** Python 3.9+, Pydantic, LangGraph, pytest, Habitat, TSDFPlanner, existing TierNav `Scene`/`Executor`/`SceneGraphMemory`/`MemoryStore`, OpenAI-compatible chat APIs.

---

## Scope Check

This plan covers the runtime cutover, benchmark adapters, and runner migration needed to actually run AEQA and GOATBench on the new backend. It deliberately does not redesign the research problem, change benchmark scoring definitions, or invent new model architectures.

Two hard gates apply:

1. The deterministic runtime unit tests must stay green while the production wiring is added.
2. The supported runner path after cutover must not expose `legacy` or old `langgraph` as selectable execution engines.

## File Structure

Create or modify these files:

```text
src/tiernav_runtime/
  contracts.py
  context.py
  memory.py
  planner.py
  tools.py
  graph.py
  entrypoint.py
  adapters.py
  env.py
  success.py
  config.py

run_two_tier_aeqa_evaluation.py
run_goatbench_evaluation.py

archive/legacy_runtime/
  two_tier_graph/...
  goatbench_graph/...

tests/runtime/
  test_contracts.py
  test_context_compiler.py
  test_memory_service.py
  test_planner_client.py
  test_tools.py
  test_graph_runtime.py
  test_adapters.py
  test_entrypoint_compat.py
  test_success_evaluator.py
  test_config.py
  test_default_path_no_stubs.py
  test_schema_snapshots.py
```

Responsibilities:

- `contracts.py`: Pydantic contracts for AEQA/GOATBench runtime state, requests, results, memory scope, and benchmark rules.
- `context.py`: sectioned prompt compiler with cacheable/dynamic split.
- `memory.py`: room-snapshot-object memory graph and active query packs.
- `planner.py`: OpenAI-compatible planner client and legacy planner adapter.
- `tools.py`: real tool registry plus deterministic test defaults.
- `graph.py`: LangGraph nodes and routing.
- `entrypoint.py`: runtime entrypoint and legacy-compatible result mapping.
- `adapters.py`: AEQA and GOATBench task adapters.
- `env.py`: Habitat/TSDF/scene/model session service.
- `success.py`: benchmark-specific success evaluation and distance checks.
- `config.py`: provider/runtime configuration models.

## Implementation Order

```text
Task 1  Contracts, config, and schema snapshots
Task 2  Planner client and OpenAI-compatible provider injection
Task 3  Runtime environment service for Habitat, TSDFPlanner, and models
Task 4  Real tools wrapping Executor
Task 5  Memory bridge and task-scoped session semantics
Task 6  Success evaluator for AEQA and GOATBench
Task 7  LangGraph runtime wiring against real services
Task 8  AEQA and GOATBench adapters
Task 9  Runner cutover to runtime-only engines
Task 10 Archive old runtime code and add import/backstop audits
Task 11 End-to-end smoke and regression gates
```

---

## Task 1: Contracts, Config, and JSON Schema

**Files:**
- Modify: `src/tiernav_runtime/contracts.py`
- Create: `src/tiernav_runtime/config.py`
- Modify: `tests/runtime/test_contracts.py`
- Create: `tests/runtime/test_config.py`
- Modify: `tests/runtime/test_schema_snapshots.py`

- [ ] **Step 1: Write failing tests for the new benchmark contracts and runtime config**

```python
from src.tiernav_runtime.config import ProviderConfig, RuntimeConfig
from src.tiernav_runtime.contracts import GoalSpec, MemoryScope, BenchmarkRule


def test_runtime_config_keeps_provider_settings_injected():
    cfg = RuntimeConfig(
        provider=ProviderConfig(
            api_key_env="TEST_KEY",
            base_url_env="TEST_BASE_URL",
            model_env="TEST_MODEL",
        )
    )
    assert cfg.provider.api_key_env == "TEST_KEY"
    assert cfg.provider.base_url_env == "TEST_BASE_URL"
    assert cfg.provider.model_env == "TEST_MODEL"


def test_goal_spec_separates_planner_and_scoring_fields():
    goal = GoalSpec(
        goal_type="object",
        goal_description="chair",
        goal_object_ids_for_scoring=["obj-1"],
        subtask_index=2,
        subtask_total=5,
    )
    assert goal.goal_description == "chair"
    assert goal.goal_object_ids_for_scoring == ["obj-1"]


def test_benchmark_rule_exposes_memory_scope_and_success_distance():
    rule = BenchmarkRule(
        success_distance_m=1.0,
        requires_explicit_stop=True,
        memory_scope=MemoryScope.SUBTASK_SEQUENCE,
        scoring_mode="distance",
    )
    assert rule.success_distance_m == 1.0
```

- [ ] **Step 2: Run the tests to confirm the new models are missing**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_contracts.py tests/runtime/test_config.py -q
```

Expected: FAIL because `ProviderConfig`, `RuntimeConfig`, `GoalSpec`, `MemoryScope`, or `BenchmarkRule` are not fully defined yet.

- [ ] **Step 3: Implement the minimal contracts and config models**

Add Pydantic models for:

- `MemoryScope`
- `BenchmarkRule`
- `GoalSpec`
- `ProviderConfig`
- `RuntimeConfig`

Keep provider settings configurable through environment variable names or explicit values, not hard-coded URLs/models.

- [ ] **Step 4: Run the tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_contracts.py tests/runtime/test_config.py \
  tests/runtime/test_schema_snapshots.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/contracts.py src/tiernav_runtime/config.py tests/runtime/test_contracts.py tests/runtime/test_config.py tests/runtime/test_schema_snapshots.py
git commit -m "feat(runtime): add benchmark contracts and provider config"
```

---

## Task 2: Planner Client and OpenAI-Compatible Provider Injection

**Files:**
- Modify: `src/tiernav_runtime/planner.py`
- Modify: `src/agent_planner.py`
- Modify: `src/const.py`
- Modify: `tests/runtime/test_tools.py`
- Create: `tests/runtime/test_planner_client.py`

- [ ] **Step 1: Write failing tests for configurable planner transport**

```python
from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.planner import PlannerClient


def test_planner_client_uses_injected_provider_settings():
    cfg = ProviderConfig(
        api_key_env="PLAN_API_KEY",
        base_url_env="PLAN_BASE_URL",
        model_env="PLAN_MODEL",
    )
    client = PlannerClient(provider=cfg)
    assert client.provider.api_key_env == "PLAN_API_KEY"
```

- [ ] **Step 2: Run the planner client test**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_planner_client.py -q
```

Expected: FAIL because `PlannerClient` does not yet accept injected provider config.

- [ ] **Step 3: Implement a provider-agnostic planner client**

Requirements:

- `PlannerClient` must read `api_key`, `base_url`, and `model` from injected config, environment, or explicit constructor args.
- The client must call the existing OpenAI-compatible transport, not hard-code vendor-specific endpoints.
- The old `src.agent_planner.Planner` may remain as a bridge, but it must use injected settings rather than module-level constants.
- `src/const.py` should stop pretending a single model/base URL is the canonical runtime choice; keep env names only as defaults, not as hard-coded runtime state.

- [ ] **Step 4: Run the tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_planner_client.py tests/runtime/test_tools.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/planner.py src/agent_planner.py src/const.py tests/runtime/test_planner_client.py tests/runtime/test_tools.py
git commit -m "feat(runtime): make planner transport configurable"
```

---

## Task 3: Runtime Environment Service

**Files:**
- Create: `src/tiernav_runtime/env.py`
- Modify: `src/tiernav_runtime/entrypoint.py`
- Modify: `src/tiernav_runtime/graph.py`
- Modify: `tests/runtime/test_graph_runtime.py`
- Modify: `tests/runtime/test_entrypoint_compat.py`

- [ ] **Step 1: Write failing tests for environment construction and session ownership**

```python
def test_environment_service_can_build_aeqa_session():
    env = RuntimeEnvironmentService.for_aeqa(...)
    assert env.task_mode == "question_answering"


def test_environment_service_can_build_goatbench_session():
    env = RuntimeEnvironmentService.for_goatbench(...)
    assert env.task_mode == "goal_navigation"
```

- [ ] **Step 2: Run the tests to confirm the service is missing**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_graph_runtime.py tests/runtime/test_entrypoint_compat.py -q
```

Expected: FAIL because `RuntimeEnvironmentService` is not implemented yet.

- [ ] **Step 3: Implement the environment service**

`RuntimeEnvironmentService` should own:

- `Scene`
- `TSDFPlanner`
- detection model
- SAM predictor
- CLIP model
- CLIP preprocess/tokenizer
- logger handles
- current pose/path length

It must support:

- fresh session per AEQA question
- long-lived session per GOATBench episode
- scene cleanup on teardown
- pose threading across GOATBench subtasks

- [ ] **Step 4: Run the tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_graph_runtime.py tests/runtime/test_entrypoint_compat.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/env.py src/tiernav_runtime/entrypoint.py src/tiernav_runtime/graph.py tests/runtime/test_graph_runtime.py tests/runtime/test_entrypoint_compat.py
git commit -m "feat(runtime): add habitat-backed environment service"
```

---

## Task 4: Real Tools Wrapping Executor

**Files:**
- Modify: `src/tiernav_runtime/tools.py`
- Modify: `src/agent_executor.py`
- Modify: `tests/runtime/test_tools.py`
- Modify: `tests/runtime/test_default_path_no_stubs.py`

- [ ] **Step 1: Write failing tests for tool dispatch against Executor**

```python
def test_runtime_tools_wrap_executor_methods():
    registry = build_real_tool_registry(...)
    assert "navigate_to_object" in registry.names()
    assert "fork_subagent" not in registry.names()
```

- [ ] **Step 2: Run the tool tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_tools.py tests/runtime/test_default_path_no_stubs.py -q
```

Expected: FAIL because the real registry is not wired to `Executor` yet.

- [ ] **Step 3: Wire tool registry to the real Executor**

Requirements:

- Keep the deterministic noop registry for fake tests only.
- Add a production registry that calls `Executor.explore_panorama`, `navigate_to_object`, `explore_seed`, `explore_frontier`, and `submit_answer`.
- Convert tool outputs into `ToolResult` with pose/path metrics and JSON-safe observations.
- Preserve existing `Executor` behavior for the underlying navigation stack.

- [ ] **Step 4: Run the tool tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_tools.py tests/runtime/test_default_path_no_stubs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/tools.py src/agent_executor.py tests/runtime/test_tools.py tests/runtime/test_default_path_no_stubs.py
git commit -m "feat(runtime): route production tools through executor"
```

---

## Task 5: Memory Bridge and Session Semantics

**Files:**
- Modify: `src/tiernav_runtime/memory.py`
- Modify: `src/tiernav_runtime/context.py`
- Modify: `src/tiernav_runtime/graph.py`
- Modify: `tests/runtime/test_memory_service.py`
- Modify: `tests/runtime/test_context_compiler.py`

- [ ] **Step 1: Write failing tests for AEQA episode-local memory and GOATBench cross-subtask memory**

```python
def test_aeqa_memory_resets_per_question():
    ...


def test_goatbench_memory_persists_across_subtasks():
    ...
```

- [ ] **Step 2: Run the memory and context tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_memory_service.py tests/runtime/test_context_compiler.py -q
```

Expected: FAIL until the runtime session semantics are added.

- [ ] **Step 3: Implement runtime memory bridging**

Requirements:

- AEQA gets a fresh `MemoryService` per question.
- GOATBench reuses `MemoryService`, `Notebook`, and `SceneGraphMemory` across subtasks in one episode.
- The memory bridge must update room, snapshot, and object layers from real observations.
- The context compiler must keep scoring-only GOATBench fields out of planner-visible prompt content.

- [ ] **Step 4: Run the memory and context tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_memory_service.py tests/runtime/test_context_compiler.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/memory.py src/tiernav_runtime/context.py src/tiernav_runtime/graph.py tests/runtime/test_memory_service.py tests/runtime/test_context_compiler.py
git commit -m "feat(runtime): bridge memory into continuous context"
```

---

## Task 6: Success Evaluator

**Files:**
- Create: `src/tiernav_runtime/success.py`
- Modify: `src/tiernav_runtime/contracts.py`
- Modify: `tests/runtime/test_policy.py`
- Create: `tests/runtime/test_success_evaluator.py`

- [ ] **Step 1: Write failing tests for AEQA and GOATBench success rules**

```python
def test_aeqa_success_requires_answer_submission():
    ...


def test_goatbench_success_requires_explicit_submit_and_distance():
    ...
```

- [ ] **Step 2: Run the success tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_success_evaluator.py tests/runtime/test_policy.py -q
```

Expected: FAIL because the evaluator is not implemented yet.

- [ ] **Step 3: Implement benchmark-specific success evaluation**

Requirements:

- AEQA: runtime completion comes from answer submission; official answer quality is external LLM-Match.
- GOATBench: explicit terminal submit plus distance within `cfg.success_distance` (local default 1.0) is required.
- Add structured distance fields to `EpisodeResult`.
- Do not mark GOATBench success from snapshot presence alone.

- [ ] **Step 4: Run the success tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_success_evaluator.py tests/runtime/test_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/success.py src/tiernav_runtime/contracts.py tests/runtime/test_success_evaluator.py tests/runtime/test_policy.py
git commit -m "feat(runtime): add benchmark success evaluation"
```

---

## Task 7: LangGraph Runtime Against Real Services

**Files:**
- Modify: `src/tiernav_runtime/graph.py`
- Modify: `src/tiernav_runtime/entrypoint.py`
- Modify: `tests/runtime/test_graph_runtime.py`

- [ ] **Step 1: Write failing integration tests for real-service graph nodes**

```python
def test_runtime_graph_uses_real_services_when_injected():
    ...
```

- [ ] **Step 2: Run the graph tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_graph_runtime.py -q
```

Expected: FAIL until the graph consumes the new environment/planner/tool/memory services.

- [ ] **Step 3: Rewire the runtime graph**

Requirements:

- The graph must use injected services only.
- Node transitions must support AEQA and GOATBench task modes.
- Production graph paths must call real tools, planner, memory, and success evaluator.
- The deterministic fake-service path must remain available for tests and replay.

- [ ] **Step 4: Run the graph tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_graph_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/graph.py src/tiernav_runtime/entrypoint.py tests/runtime/test_graph_runtime.py
git commit -m "feat(runtime): wire langgraph to real runtime services"
```

---

## Task 8: AEQA and GOATBench Adapters

**Files:**
- Modify: `src/tiernav_runtime/adapters.py`
- Modify: `run_two_tier_aeqa_evaluation.py`
- Modify: `run_goatbench_evaluation.py`
- Modify: `tests/runtime/test_adapters.py`

- [ ] **Step 1: Write failing adapter tests for AEQA and GOATBench payload shapes**

```python
def test_aeqa_adapter_builds_episode_request_without_cross_episode_memory():
    ...


def test_goatbench_adapter_threads_subtask_context_and_goal_metadata():
    ...
```

- [ ] **Step 2: Run the adapter tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_adapters.py -q
```

Expected: FAIL until the adapters are updated for the new runtime contracts.

- [ ] **Step 3: Update task adapters and runner payload mapping**

Requirements:

- AEQA adapter emits one request per question, no cross-question memory.
- GOATBench adapter emits one request per subtask inside a long-lived episode session.
- Runners must preserve official output shape while forwarding runtime event log paths and success fields.

- [ ] **Step 4: Run the adapter tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_adapters.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/adapters.py run_two_tier_aeqa_evaluation.py run_goatbench_evaluation.py tests/runtime/test_adapters.py
git commit -m "feat(runtime): adapt aeqa and goatbench to runtime contracts"
```

---

## Task 9: Runner Cutover to Runtime-Only Engines

**Files:**
- Modify: `run_two_tier_aeqa_evaluation.py`
- Modify: `run_goatbench_evaluation.py`
- Modify: `run_goatbench_two_tier_evaluation.py`
- Modify: `src/const.py`

- [ ] **Step 1: Write failing tests or import audits for old engine removal**

```python
def test_aeqa_runner_accepts_runtime_only():
    ...
```

- [ ] **Step 2: Run the existing smoke/compat tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_entrypoint_compat.py tests/runtime/test_default_path_no_stubs.py -q
```

Expected: FAIL until runners are cut over.

- [ ] **Step 3: Remove legacy engine choices from runners**

Requirements:

- AEQA runner `--engine` should be `runtime` only.
- GOATBench runner `--engine` should be `runtime` only.
- Any remaining old `langgraph` or `legacy` path must move to archive or be removed from default execution.
- Planner/provider settings must come from configuration/env injection, not hard-coded values.

- [ ] **Step 4: Run the smoke/compat tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_entrypoint_compat.py tests/runtime/test_default_path_no_stubs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run_two_tier_aeqa_evaluation.py run_goatbench_evaluation.py run_goatbench_two_tier_evaluation.py src/const.py
git commit -m "feat(runtime): cut runners over to runtime-only mode"
```

---

## Task 10: Archive Old Runtime Code and Add Backstop Audits

**Files:**
- Create: `archive/legacy_runtime/two_tier_graph/...`
- Create: `archive/legacy_runtime/goatbench_graph/...`
- Modify: `tests/runtime/test_default_path_no_stubs.py`
- Modify: `tests/runtime/test_schema_snapshots.py`

- [ ] **Step 1: Add audit tests for archived code and default import boundaries**

```python
def test_default_runners_do_not_import_archived_legacy_runtime():
    ...
```

- [ ] **Step 2: Run the audit tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_default_path_no_stubs.py tests/runtime/test_schema_snapshots.py -q
```

Expected: FAIL until the archive and import boundaries are in place.

- [ ] **Step 3: Move old runtime source into archive**

Requirements:

- Preserve source backups under `archive/legacy_runtime/`.
- Keep them out of the supported runner import path.
- Leave a Git-level backup tag/branch in place as well.

- [ ] **Step 4: Run the audit tests again**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/runtime/test_default_path_no_stubs.py tests/runtime/test_schema_snapshots.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add archive/legacy_runtime tests/runtime/test_default_path_no_stubs.py tests/runtime/test_schema_snapshots.py
git commit -m "chore(runtime): archive legacy runtime paths"
```

---

## Task 11: End-to-End Smoke and Regression Gates

**Files:**
- Modify: `tests/runtime/test_graph_runtime.py`
- Modify: `tests/runtime/test_entrypoint_compat.py`
- Modify: `tests/runtime/test_success_evaluator.py`
- Optional: add short smoke scripts under `docs/` if needed for operator clarity

- [ ] **Step 1: Write the smoke assertions for AEQA and GOATBench**

```python
def test_aeqa_runtime_smoke_with_fake_services():
    ...


def test_goatbench_runtime_smoke_with_fake_services():
    ...
```

- [ ] **Step 2: Run the full runtime test suite**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime -q
```

Expected: PASS.

- [ ] **Step 3: Run focused legacy-regression tests only if still relevant**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/test_two_tier_graph.py \
  tests/test_prompt_sections.py \
  tests/test_stall_detection.py \
  tests/test_aeqa_output_format.py \
  -q
```

Expected: either PASS if the archive still keeps them green, or clearly documented legacy failures if those paths are intentionally retired. Do not let these block the new runtime gate.

- [ ] **Step 4: Commit the final runtime gate**

```bash
git add tests/runtime
git commit -m "test(runtime): add final smoke and regression gates"
```

---

## Self-Review Checklist

- [ ] Every spec requirement has a corresponding task.
- [ ] No task relies on hard-coded provider base URLs or model names.
- [ ] AEQA memory resets per question.
- [ ] GOATBench memory persists across subtasks inside one episode.
- [ ] GOATBench success requires explicit submit/stop plus distance within 1m under local config.
- [ ] Old runtime code is backed up before removal from default execution.
- [ ] `tests/runtime` remains the primary gate.
- [ ] No placeholder text remains in any step.
- [ ] File paths exist in the current repo or are created explicitly in the plan.
