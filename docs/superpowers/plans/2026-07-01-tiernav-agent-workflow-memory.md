# TierNav Agent Workflow Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire continuous runtime memory, local scene graph persistence, and richer planner context without changing the legacy TSDF/detection stack.

**Architecture:** Use `MemorySession` as the runtime memory service when present, mirror tool observations into optional `SceneGraphMemory`, persist that graph beside event logs, and extend `ContextCompiler` with dynamic agent-state sections.

**Tech Stack:** Python, Pydantic contracts, LangGraph runtime, existing TierNav executor/scene graph classes, pytest.

---

### Task 1: Runtime Memory Session Wiring

**Files:**
- Modify: `src/tiernav_runtime/entrypoint.py`
- Test: `tests/runtime/test_memory_service.py`

- [x] Add failing test proving `with_real_services(..., memory_scope_adapter=session)` sets `services.memory is session`.
- [x] Implement the minimal factory change.
- [x] Run the targeted memory test.

### Task 2: Scene Graph Sink

**Files:**
- Modify: `src/tiernav_runtime/memory.py`
- Modify: `src/scene_graph_memory.py`
- Test: `tests/runtime/test_memory_service.py`

- [x] Add failing tests for scene graph mirroring and JSON persistence.
- [x] Add lightweight `persist_json(path)` to `SceneGraphMemory`.
- [x] Teach `MemorySession.update_from_observation()` to mirror observations into optional `scene_graph`.
- [x] Run targeted tests.

### Task 3: Context Sections

**Files:**
- Modify: `src/tiernav_runtime/context.py`
- Test: `tests/runtime/test_context_compiler.py`

- [x] Add failing tests for `task_state`, `scene_graph_memory`, and `tool_feedback`.
- [x] Remove diagnostic stderr from available target rendering.
- [x] Implement dynamic section rendering.
- [x] Run context compiler tests.

### Task 4: Runner Wiring

**Files:**
- Modify: `run_goatbench_evaluation.py`
- Modify: `run_two_tier_aeqa_evaluation.py`
- Test: `tests/runtime/test_goatbench_logger_integration.py`

- [x] Wire GOATBench `MemorySession` with `SceneGraphMemory`.
- [x] Persist scene graph to the runtime workflow output directory.
- [x] Keep AEQA per-question memory independent.
- [x] Run targeted runtime tests.

### Task 5: Verification

- [x] Run runtime unit tests covering memory, context, graph, tools, path length, success, and GOATBench logger integration.
- [x] Report any legacy test failures separately from runtime regressions.
