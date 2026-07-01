# TierNav Agent Workflow Memory Design

## Goal

Improve the current LangGraph runtime as a continuous-context navigation agent while preserving the legacy TSDF, detection, scene update, and executor stack.

## Scope

This change only touches the runtime layer and runner wiring. It must not rewrite `scene_aeqa.py`, `scene_goatbench.py`, `tsdf_planner.py`, `agent_tools.py`, or the ConceptGraph detection pipeline.

## Design

The runtime will use `MemorySession` as the active memory service whenever a runner provides one. AEQA keeps per-question memory; GOATBench keeps subtask-sequence memory across subtasks.

GOATBench will create a `SceneGraphMemory` for each episode and pass it into `MemorySession`. The runtime will mirror tool observations into this scene graph and persist it locally as JSON after updates. This creates a concrete room-view-object memory artifact without changing the detection stack.

The context compiler will render task state, scene-graph summary, tool feedback, and available targets as separate sections. Static task/schema sections remain cacheable; dynamic sections carry current observations and recovery hints. The planner prompt remains a single rendered string for current providers, but the section boundaries stay explicit and testable.

Tool failures will be treated as agent context, not just errors. The next planning round should see the failed action, target, and reason, so it can change strategy rather than repeat invalid calls.

## Acceptance Criteria

- `RuntimeEntrypoint.with_real_services()` uses the provided `MemorySession` for graph queries and updates.
- GOATBench subtask memory persists through the runtime memory path.
- `SceneGraphMemory` can be serialized to a stable local JSON file.
- The planner prompt includes dynamic scene graph and tool feedback sections when available.
- The prompt does not contain diagnostic stderr output or fake target IDs.
- Runtime tests pass for the changed behavior.
