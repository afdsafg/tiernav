# TierNav Production Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Wire the tiernav_runtime into true production mode: real VLM planner, real Habitat-backed tools via Executor, AEQA/GOATBench benchmark rules, and per-benchmark memory sessions. Replace all fake service stubs in the runner scripts.

**Architecture:** The tiernav_runtime structural scaffolding (Tasks 1-11) is complete. Three gaps block production use: (A) `PlannerClient` lacks a `decide()` method that the graph calls; (B) both AEQA and GOATBench runners use `with_fake_services()` and a fake planner; (C) real tool registry, environment service, success evaluator, and benchmark rules are never wired into the entrypoint. We close these gaps in four tasks.

**Tech Stack:** Python 3.9+, Pydantic, LangGraph, pytest, habitat-sim, OpenAI-compatible chat APIs, existing `src.agent_workflow.call_vlm`, `src.agent_planner.Planner`, `src.agent_executor.Executor`.

---

## Pre-implementation Verification

Run the full runtime test suite to confirm the starting state:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q
```

Expected: 293 passed. This is the starting baseline.

---

### Task A: Add PlannerClient.decide() — Real VLM Planner

**Files:**
- Create: `tests/runtime/test_planner_decide.py`
- Modify: `src/tiernav_runtime/planner.py`

**Step A1: Write failing test for PlannerClient.decide()**

```python
"""Tests for PlannerClient.decide() — production VLM planner path."""
import pytest
from unittest.mock import patch

from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.contracts import PlannerDecision
from src.tiernav_runtime.planner import PlannerClient


class TestPlannerClientDecide:
    def test_decide_calls_vlm_and_returns_decision(self):
        """decide() calls call_vlm with the prompt, parses the response into PlannerDecision."""
        cfg = ProviderConfig(
            api_key_env="TEST_KEY",
            base_url_env="TEST_BASE_URL",
            model_env="TEST_MODEL",
        )
        client = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="test-model")

        fake_response = (
            '{"action_type": "explore_panorama", '
            '"reason": "Need to observe surroundings", '
            '"expected": "Get room layout", '
            '"object_name": "chair"}'
        )

        with patch(
            "src.tiernav_runtime.planner._call_vlm",
            return_value=fake_response,
        ) as mock_vlm:
            decision = client.decide("Test prompt")

        mock_vlm.assert_called_once()
        call_args = mock_vlm.call_args[0][0]
        assert any("Test prompt" in msg.get("content", "") for msg in call_args)

        assert isinstance(decision, PlannerDecision)
        assert decision.action_type == "explore_panorama"
        assert decision.reasoning == "Need to observe surroundings"
        assert decision.arguments.get("object_name") == "chair"

    def test_decide_handles_invalid_json(self):
        """decide() returns a fallback submit_answer on unparseable VLM output."""
        cfg = ProviderConfig(
            api_key_env="TEST_KEY", base_url_env="TEST_BASE_URL", model_env="TEST_MODEL",
        )
        client = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="test-model")

        with patch(
            "src.tiernav_runtime.planner._call_vlm",
            return_value="not valid json {{{",
        ):
            decision = client.decide("Test prompt")

        assert decision.action_type == "submit_answer"
        assert decision.confidence == 0.0
        assert "planner_parse_error" in decision.arguments.get("failure_reason", "")

    def test_decide_handles_missing_action_type(self):
        """decide() falls back when action_type is missing from parsed JSON."""
        cfg = ProviderConfig(
            api_key_env="TEST_KEY", base_url_env="TEST_BASE_URL", model_env="TEST_MODEL",
        )
        client = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="test-model")

        with patch(
            "src.tiernav_runtime.planner._call_vlm",
            return_value='{"reason": "no action"}',
        ):
            decision = client.decide("Test prompt")

        assert decision.action_type == "submit_answer"
        assert decision.confidence == 0.0
```

**Step A2: Run the failing test**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_planner_decide.py -q
```

Expected: FAIL — `PlannerClient` object has no attribute `decide`.

**Step A3: Implement PlannerClient.decide()**

Add this method to `PlannerClient` in `src/tiernav_runtime/planner.py`:

```python
def decide(self, prompt: str) -> PlannerDecision:
    """Call the VLM with the compiled prompt and return a PlannerDecision.
    
    Builds a single user message from ``prompt``, calls the
    OpenAI-compatible transport, parses the JSON response, and maps it
    through ``planner_action_to_decision`` for legacy PlannerAction
    compatibility.
    
    On JSON parse errors or missing ``action_type``, returns a terminal
    ``submit_answer`` with confidence 0.0 to avoid infinite loops on
    unparseable planner output.
    """
    import json as _json

    messages = [{"role": "user", "content": prompt}]
    try:
        raw = self.call_vlm(messages)
    except Exception:
        return PlannerDecision(
            action_type="submit_answer",
            reasoning="planner_call_failed",
            confidence=0.0,
            arguments={"failure_reason": "planner_call_failed"},
        )

    try:
        parsed = _json.loads(raw.strip())
    except _json.JSONDecodeError:
        return PlannerDecision(
            action_type="submit_answer",
            reasoning="planner_parse_error",
            confidence=0.0,
            arguments={"failure_reason": "planner_parse_error", "raw": raw[:500]},
        )

    if not isinstance(parsed, dict):
        return PlannerDecision(
            action_type="submit_answer",
            reasoning="planner_response_not_dict",
            confidence=0.0,
            arguments={"failure_reason": "planner_response_not_dict"},
        )

    if not parsed.get("action_type"):
        return PlannerDecision(
            action_type="submit_answer",
            reasoning="planner_missing_action_type",
            confidence=0.0,
            arguments={"failure_reason": "planner_missing_action_type"},
        )

    return planner_action_to_decision(
        type("PlannerAction", (), parsed)()
    )
```

Also add the import at the top of the file (it's already present - verify):

```python
from .contracts import PlannerDecision  # already imported
```

**Step A4: Run tests to verify**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_planner_decide.py tests/runtime/test_planner_client.py -q
```

Expected: PASS (3 decide tests + existing planner client tests).

**Step A5: Commit**

```bash
git add tests/runtime/test_planner_decide.py src/tiernav_runtime/planner.py
git commit -m "feat(runtime): add PlannerClient.decide() with VLM call and fallback"
```

---

### Task B: Wire AEQA Runner with Production Services

**Files:**
- Modify: `run_two_tier_aeqa_evaluation.py` (the `_run_aeqa_runtime` function)
- Modify: `tests/runtime/test_entrypoint_compat.py` (add AEQA dispatch test)

**Context:** The current `_run_aeqa_runtime` at line 47-137 creates a fake planner and uses `with_fake_services`. We need to replace this with a real `PlannerClient`, build a `RuntimeEnvironmentService` from the scene/models the runner already constructs, wire a `BenchmarkRule` for AEQA, and use `with_real_services`.

**Step B1: Write failing test for AEQA dispatch with real planner**

Add to `tests/runtime/test_entrypoint_compat.py`:

```python
def test_aeqa_production_dispatch_shape():
    """_run_aeqa_runtime with real Entrypoint produces the legacy dict shape."""
    from unittest.mock import MagicMock, patch
    from src.tiernav_runtime.adapters import AEQATaskAdapter
    from src.tiernav_runtime.contracts import RunSpec, BenchmarkRule, MemoryScope, PlannerDecision
    from src.tiernav_runtime.entrypoint import RuntimeEntrypoint, episode_result_to_legacy_dict
    from src.tiernav_runtime.tools import ToolRegistry
    from src.tiernav_runtime.memory import MemoryService
    from src.tiernav_runtime.policy import WorkflowPolicy

    adapter = AEQATaskAdapter()
    request = adapter.to_request(
        scene_id="test_scene",
        question_id="test_q",
        question="What is the room color?",
        output_dir="/tmp/test_aeqa_dispatch",
    )
    spec = RunSpec(
        run_id="test_q",
        task_name="aeqa",
        dataset_split="aeqa",
        output_dir="/tmp/test_aeqa_dispatch",
        planner_provider="http://test",
        planner_model="test-model",
        max_rounds=5,
        max_steps=20,
    )

    rule = BenchmarkRule(
        success_distance_m=0.0,
        requires_explicit_stop=False,
        memory_scope=MemoryScope.PER_QUESTION,
        scoring_mode="answer_quality",
    )

    real_planner = MagicMock()
    real_planner.decide.return_value = PlannerDecision(
        action_type="submit_answer",
        arguments={"answer": "blue"},
        confidence=0.9,
        reasoning="test",
    )

    env = MagicMock()
    executor = MagicMock()
    executor.path_length = 0.0

    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=real_planner,
        environment=env,
        rule=rule,
        executor=executor,
    )
    result = entrypoint.run(spec, request)
    legacy = episode_result_to_legacy_dict(result, question="What is the room color?")

    assert legacy["question_id"] == "test_q"
    assert "answer" in legacy
    assert "success" in legacy
    assert legacy["answer"] == "blue"
    assert legacy["success"] is True
```

**Step B2: Run the new test to see it fail**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_entrypoint_compat.py::test_aeqa_production_dispatch_shape -q
```

Note: This test may actually PASS because `with_real_services()` already exists and works with mock services. The real gap is in the runner script, not in the runtime framework. If it passes, confirm it and skip the "fail" step.

**Step B3: Rewire `_run_aeqa_runtime` with production services**

Replace the fake planner block (lines 122-137 of `run_two_tier_aeqa_evaluation.py`) with:

```python
    # --- Real production services (Task B: production wiring) ---
    from src.tiernav_runtime.contracts import BenchmarkRule, MemoryScope
    from src.tiernav_runtime.planner import PlannerClient
    from src.tiernav_runtime.env import RuntimeEnvironmentService
    from src.tiernav_runtime.memory import MemorySession
    from src.tiernav_runtime.entrypoint import RuntimeEntrypoint

    # Build real VLM planner client
    planner = PlannerClient(provider_config)

    # Build environment service wrapping the caller's scene + tsdf_planner
    env_service = RuntimeEnvironmentService.for_aeqa(
        scene=scene,
        tsdf_planner=tsdf_planner,
        executor=None,  # Executor constructed lazily by graph via tools
        detection_model=detection_model,
        sam_predictor=sam_predictor,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
    )

    # Build Executor from the environment's objects
    from src.agent_executor import Executor
    executor = Executor(
        scene=scene,
        tsdf_planner=tsdf_planner,
        detection_model=detection_model,
        sam_predictor=sam_predictor,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        logger=None,
    )

    # AEQA rule: per-question memory, no distance check, answer quality scoring
    rule = BenchmarkRule(
        success_distance_m=0.0,
        requires_explicit_stop=False,
        memory_scope=MemoryScope.PER_QUESTION,
        scoring_mode="answer_quality",
    )

    # Per-question memory session (fresh each call)
    memory_session = MemorySession(scope=MemoryScope.PER_QUESTION)
    memory_session.start_session(episode_id=question_id)

    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=planner,
        environment=env_service,
        rule=rule,
        executor=executor,
        memory_scope_adapter=memory_session,
    )
    result = entrypoint.run(spec, request)
    return episode_result_to_legacy_dict(result, question=question)
```

Also remove the `FakePlanner` and `_fake_planner` code (lines 122-137).

**Step B4: Run the existing tests**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_entrypoint_compat.py tests/runtime/ -q
```

Expected: All pass (the entrypoint tests use fake services; production wiring is only in the runner).

**Step B5: Commit**

```bash
git add run_two_tier_aeqa_evaluation.py tests/runtime/test_entrypoint_compat.py
git commit -m "feat(runtime): wire AEQA runner with real VLM planner, environment service, and benchmark rules"
```

---

### Task C: Wire GOATBench Runner with Production Services

**Files:**
- Modify: `run_goatbench_evaluation.py` (the `_run_goatbench_runtime` function)

**Context:** The GOATBench runtime function at line 437 also uses a fake planner and `with_fake_services`. We need to wire real services with GOATBench-specific semantics: subtask-sequence memory, 1m success distance, and explicit-stop requirement.

**Step C1: Read the current `_run_goatbench_runtime` function**

Understand the full function signature and body to know what scene/planner/models are available.

**Step C2: Rewire `_run_goatbench_runtime` with real services**

The GOATBench runner already has the scene, tsdf_planner, models, and executor constructed. The key changes:

1. Build a real `PlannerClient` from `ProviderConfig`
2. Build `RuntimeEnvironmentService.for_goatbench()`
3. Build a real `Executor` from existing objects
4. Create `BenchmarkRule` with `success_distance_m=1.0`, `requires_explicit_stop=True`, `memory_scope=SUBTASK_SEQUENCE`
5. Create `MemorySession(scope=MemoryScope.SUBTASK_SEQUENCE)` that persists across subtasks
6. Use `RuntimeEntrypoint.with_real_services()`

The key difference from AEQA: the GOATBench `MemorySession` must be created **once per episode** (before the subtask loop) and reused across subtasks. The `start_session` method with the same `episode_id` will preserve memory across subtasks.

**Step C3: Verify with existing tests**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q
```

Expected: All pass.

**Step C4: Commit**

```bash
git add run_goatbench_evaluation.py
git commit -m "feat(runtime): wire GOATBench runner with real VLM planner, 1m success distance, and cross-subtask memory"
```

---

### Task D: End-to-End Verification

**Files:**
- Modify: `tests/runtime/test_entrypoint_compat.py` (add end-to-end test)
- Modify: `docs/plans/2026-06-30-tiernav-production-wiring.md` (this file)

**Step D1: Write end-to-end test for full production service stack**

Add an integration-style test that:
1. Builds real `RuntimeServices` with all real components (not fakes)
2. Runs the graph with mock VLM responses
3. Verifies the full path: planner -> tools -> memory -> success eval

```python
def test_full_production_stack_with_mocked_vlm():
    """End-to-end: real ToolRegistry + MemoryService + SuccessEvaluator + Environment."""
    from unittest.mock import MagicMock, patch
    from src.tiernav_runtime.config import ProviderConfig
    from src.tiernav_runtime.contracts import (
        BenchmarkRule, EpisodeRequest, MemoryScope, PlannerDecision,
        RunSpec, TaskMode,
    )
    from src.tiernav_runtime.env import RuntimeEnvironmentService
    from src.tiernav_runtime.memory import MemorySession
    from src.tiernav_runtime.planner import PlannerClient
    from src.tiernav_runtime.policy import WorkflowPolicy
    from src.tiernav_runtime.entrypoint import RuntimeEntrypoint, episode_result_to_legacy_dict

    provider_config = ProviderConfig(
        api_key_env="TEST",
        base_url_env="TEST_BASE",
        model_env="TEST_MODEL",
    )
    planner = PlannerClient(provider_config, api_key="sk-test", base_url="http://test", model="t")

    env = RuntimeEnvironmentService(
        task_mode=TaskMode.GOAL_NAVIGATION,
        scene=MagicMock(),
        tsdf_planner=MagicMock(),
        executor=MagicMock(),
    )
    env.set_goal_pose({"x": 0.0, "y": 0.0})

    executor = MagicMock()
    executor.path_length = 0.0
    executor.explore_panorama.return_value = type("Evidence", (), {
        "progress": "Explored room",
        "room_id": 1,
        "key_frames": [],
        "objects_nearby": [],
        "outcome": "success",
        "gd_quality": "good",
        "subgoal": "explore",
        "salient": [],
    })()
    executor.navigate_to_object.return_value = type("Evidence", (), {
        "progress": "Navigated to chair",
        "room_id": 2,
        "key_frames": ["img1"],
        "objects_nearby": ["obj1"],
        "outcome": "arrived",
        "gd_quality": "good",
        "subgoal": "navigate",
        "salient": ["chair"],
    })()

    rule = BenchmarkRule(
        success_distance_m=1.0,
        requires_explicit_stop=True,
        memory_scope=MemoryScope.SUBTASK_SEQUENCE,
        scoring_mode="distance",
    )
    memory_session = MemorySession(scope=MemoryScope.SUBTASK_SEQUENCE)
    memory_session.start_session(episode_id="ep1")

    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=planner,
        environment=env,
        rule=rule,
        executor=executor,
        memory_scope_adapter=memory_session,
    )

    spec = RunSpec(
        run_id="prod_test",
        task_name="goatbench",
        dataset_split="test",
        output_dir="/tmp/prod_test",
        planner_provider="http://test",
        planner_model="test",
    )
    request = EpisodeRequest(
        episode_id="ep1",
        scene_id="scene1",
        task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION,
        prompt="Navigate to chair",
        output_dir="/tmp/prod_test",
    )

    # Mock the planner to first explore, then navigate, then submit
    call_count = [0]
    def mock_decide(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return PlannerDecision(action_type="explore_panorama", reasoning="explore")
        elif call_count[0] == 2:
            return PlannerDecision(
                action_type="navigate_to_object",
                arguments={"object_name": "chair"},
                reasoning="found chair",
            )
        else:
            return PlannerDecision(
                action_type="submit_answer",
                arguments={"answer": "found chair"},
                reasoning="done",
            )
    planner.decide = mock_decide

    result = entrypoint.run(spec, request)
    legacy = episode_result_to_legacy_dict(result)

    assert legacy["steps_taken"] >= 2
    assert legacy["success"] is True
    assert legacy["answer"] == "found chair"
```

**Step D2: Run the end-to-end test**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_entrypoint_compat.py::test_full_production_stack_with_mocked_vlm -v
```

Expected: PASS — the full production stack (real tools, real memory, real success eval) runs through the graph with mocked VLM.

**Step D3: Run the full test suite**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q
```

Expected: All pass (294+ tests).

**Step D4: Commit**

```bash
git add tests/runtime/test_entrypoint_compat.py docs/plans/2026-06-30-tiernav-production-wiring.md
git commit -m "test(runtime): add full production stack end-to-end test"
```

---

## Completion Checklist

- [ ] PlannerClient has `decide()` method calling real VLM
- [ ] AEQA runner uses real PlannerClient + RuntimeEnvironmentService + BenchmarkRule + MemorySession(PER_QUESTION)
- [ ] GOATBench runner uses real services with SUBTASK_SEQUENCE memory + 1m distance rule
- [ ] All tests pass (including the new decide tests and production stack test)
- [ ] No fake_planner or with_fake_services in production code paths
- [ ] Legacy runner code paths preserved in archive (already done in Task 10)
