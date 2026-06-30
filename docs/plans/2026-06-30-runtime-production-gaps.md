# TierNav Runtime Production Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Close the 6 remaining design-doc gaps: GOATBench Logger wiring, real path_length, 9 missing event types, LLM round logs, SPL metric, success_by_snapshot + planner retry.

**Architecture:** All changes stay inside `src/tiernav_runtime/` (graph nodes, entrypoint, planner, contracts) plus the GOATBench runner. No new external deps. Event logging threads through the existing `EpisodeRecorder` via a recorder handle on `RuntimeServices`. The GOATBench `Logger` gets fed from `EpisodeResult` + executor state in the runner's post-subtask block. Planner retry adds one bounded retry in `decide()` controlled by `BenchmarkRule`.

**Tech Stack:** Python 3.9+, Pydantic, LangGraph, pytest, numpy, existing `src.logger_goatbench.Logger`.

---

## Pre-implementation Verification

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q
```

Expected: 298 passed (current baseline).

---

### Task A: Fix path_length to use Executor's real `_path_length` (P0)

**Files:**
- Modify: `src/tiernav_runtime/entrypoint.py:169-184`
- Modify: `src/tiernav_runtime/graph.py:181-243` (execute_tool_node already syncs `env._path_length`; expose it on state)
- Test: `tests/runtime/test_path_length.py`

**Root cause:** `entrypoint.py:179` sets `path_length=float(state.step_index)`. The graph's `execute_tool_node` already syncs `env._path_length` from `executor._path_length` (graph.py:207), but that value never flows into `EpisodeState` or `EpisodeResult`.

**Step A1: Write failing test**

```python
"""path_length must reflect real executor distance, not step_index."""
from unittest.mock import MagicMock
from src.tiernav_runtime.contracts import (
    BenchmarkRule, EpisodeRequest, MemoryScope, RunSpec, TaskMode,
)
from src.tiernav_runtime.env import RuntimeEnvironmentService
from src.tiernav_runtime.entrypoint import RuntimeEntrypoint
from src.tiernav_runtime.memory import MemorySession
from src.tiernav_runtime.planner import PlannerClient
from src.tiernav_runtime.config import ProviderConfig


def test_path_length_uses_env_service_path_length(tmp_path):
    """path_length in EpisodeResult comes from env_service.path_length."""
    cfg = ProviderConfig(
        api_key_env="TEST_KEY", base_url_env="TEST_BASE_URL", model_env="TEST_MODEL",
    )
    planner = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="m")

    env = RuntimeEnvironmentService.for_aeqa(
        scene=None, tsdf_planner=None, executor=MagicMock(),
    )
    env._path_length = 12.5  # simulate accumulated distance

    rule = BenchmarkRule(
        success_distance_m=1.0, requires_explicit_stop=False,
        memory_scope=MemoryScope.PER_QUESTION, scoring_mode="answer",
    )
    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=planner, environment=env, rule=rule,
        executor=MagicMock(),
        memory_scope_adapter=MemorySession(scope=MemoryScope.PER_QUESTION),
    )

    spec = RunSpec(
        run_id="pl", task_name="aeqa", dataset_split="test",
        output_dir=str(tmp_path), planner_provider="http://t", planner_model="m",
    )
    request = EpisodeRequest(
        episode_id="ep-pl", scene_id="s", task_name="aeqa",
        task_mode=TaskMode.QUESTION_ANSWERING, prompt="q", output_dir=str(tmp_path),
    )

    # Mock planner to submit immediately
    from src.tiernav_runtime.contracts import PlannerDecision
    planner.decide = lambda prompt: PlannerDecision(
        action_type="submit_answer", arguments={"answer": "yes"}, reasoning="done",
    )

    result = entrypoint.run(spec, request)
    assert result.path_length == 12.5
    assert result.path_length != result.steps_taken
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_path_length.py -v`
Expected: FAIL — `result.path_length` equals `steps_taken`, not 12.5.

**Step A2: Fix entrypoint to read env_service.path_length**

In `src/tiernav_runtime/entrypoint.py`, replace line 179:

```python
        path_length=float(state.step_index),
```

with:

```python
        path_length=self._resolve_path_length(state),
```

Add method to `RuntimeEntrypoint`:

```python
    def _resolve_path_length(self, state: EpisodeState) -> float:
        """Prefer the environment service's real path_length; fall back to 0.0."""
        env = self.services.environment
        if env is not None and hasattr(env, "path_length"):
            try:
                return float(env.path_length)
            except (TypeError, ValueError):
                pass
        return 0.0
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_path_length.py -v`
Expected: PASS.

**Step A3: Run full suite**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q
```

Expected: 299 passed.

**Step A4: Commit**

```bash
git add tests/runtime/test_path_length.py src/tiernav_runtime/entrypoint.py
git commit -m "fix(runtime): path_length from env service instead of step_index"
```

---

### Task B: Add 9 missing event types + LLM round logs (P1)

**Files:**
- Modify: `src/tiernav_runtime/graph.py` (add recorder to RuntimeServices, emit events in nodes)
- Modify: `src/tiernav_runtime/entrypoint.py` (pass recorder to services)
- Modify: `src/tiernav_runtime/replay.py` (parse new event types)
- Test: `tests/runtime/test_event_logging.py`

**Design:** Per spec `docs/superpowers/specs/2026-06-29-tiernav-runtime-habitat-integration-design.md:377-389`, the runtime must emit 11 event types. Currently only `episode_started`/`episode_ended` are emitted. Add: `subtask_started`, `context_compiled`, `planner_called`, `planner_decision`, `tool_called`, `tool_result`, `memory_query`, `memory_updated`, `success_evaluated`.

The recorder lives on `RuntimeServices` so any node can append. The entrypoint creates it and seeds it. LLM round logs are the `planner_called`+`planner_decision` pair (they carry the raw prompt and parsed decision).

**Step B1: Write failing test for new events**

```python
"""Production graph emits all 11 design-spec event types."""
import json
from pathlib import Path
from unittest.mock import MagicMock

from src.tiernav_runtime.contracts import (
    BenchmarkRule, EpisodeRequest, MemoryScope, RunSpec, TaskMode,
)
from src.tiernav_runtime.env import RuntimeEnvironmentService
from src.tiernav_runtime.entrypoint import RuntimeEntrypoint
from src.tiernav_runtime.memory import MemorySession
from src.tiernav_runtime.planner import PlannerClient
from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.contracts import PlannerDecision


def test_emits_planner_and_tool_events(tmp_path):
    cfg = ProviderConfig(
        api_key_env="T", base_url_env="T", model_env="T",
    )
    planner = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")
    planner.decide = lambda prompt: PlannerDecision(
        action_type="submit_answer", arguments={"answer": "yes"}, reasoning="r",
    )

    env = RuntimeEnvironmentService.for_aeqa(
        scene=None, tsdf_planner=None, executor=MagicMock(),
    )
    rule = BenchmarkRule(
        success_distance_m=1.0, requires_explicit_stop=False,
        memory_scope=MemoryScope.PER_QUESTION, scoring_mode="answer",
    )
    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=planner, environment=env, rule=rule, executor=MagicMock(),
        memory_scope_adapter=MemorySession(scope=MemoryScope.PER_QUESTION),
    )

    spec = RunSpec(
        run_id="ev", task_name="aeqa", dataset_split="test",
        output_dir=str(tmp_path), planner_provider="http://t", planner_model="m",
    )
    request = EpisodeRequest(
        episode_id="ep-ev", scene_id="s", task_name="aeqa",
        task_mode=TaskMode.QUESTION_ANSWERING, prompt="q", output_dir=str(tmp_path),
    )

    entrypoint.run(spec, request)

    log = Path(request.event_log_path(tmp_path))
    events = [json.loads(l) for l in log.read_text().splitlines()]
    types = [e["event_type"] for e in events]

    assert "episode_started" in types
    assert "context_compiled" in types
    assert "planner_called" in types
    assert "planner_decision" in types
    assert "tool_called" in types
    assert "tool_result" in types
    assert "memory_updated" in types
    assert "success_evaluated" in types
    assert "episode_ended" in types
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_event_logging.py -v`
Expected: FAIL — only `episode_started`/`episode_ended` present.

**Step B2: Add recorder to RuntimeServices**

In `src/tiernav_runtime/graph.py`, add to `RuntimeServices`:

```python
from .recorder import EpisodeRecorder
from .events import make_event

@dataclass
class RuntimeServices:
    ...
    recorder: EpisodeRecorder | None = None
```

Add a helper:

```python
def _emit(services: RuntimeServices, episode_id: str, event_type: str,
          sequence: int, payload: dict | None = None) -> None:
    """Append an event if a recorder is wired."""
    if services.recorder is not None:
        services.recorder.append(
            make_event(episode_id, event_type, sequence, payload or {})
        )
```

Sequences: `episode_started`=1 and `episode_ended`=2 stay at the entrypoint. Intra-episode events use a per-episode monotonic counter starting at 3. Store the counter on `RuntimeServices`:

```python
    _event_seq: int = 2  # episode_started=1; intra-episode starts at 3
```

Wait — dataclass mutable counter is awkward. Instead, have `_emit` read the current recorder line count + 1. Simpler: `RuntimeServices` holds `event_seq: list[int] = field(default_factory=lambda: [2])` (boxed int). Update `_emit`:

```python
def _emit(services, episode_id, event_type, payload=None):
    if services.recorder is None:
        return
    services.event_seq[0] += 1
    services.recorder.append(
        make_event(episode_id, event_type, services.event_seq[0], payload or {})
    )
```

The entrypoint resets `services.event_seq[0] = 2` before `graph.invoke` and appends `episode_ended` with `sequence = services.event_seq[0] + 1`.

**Step B3: Emit events in nodes**

In `compile_context_node`, after `sections = services.context.compile(...)`:

```python
    _emit(services, episode.episode_id, "context_compiled", {
        "sections": [s.model_dump(mode="json") for s in sections],
        "memory_query_used": episode.memory_pack is not None,
    })
    if episode.memory_pack is not None:
        _emit(services, episode.episode_id, "memory_query", {
            "summary": episode.memory_pack.summary,
        })
```

In `plan_node`, before calling planner:

```python
    _emit(services, episode.episode_id, "planner_called", {
        "prompt": episode.prompt,
        "round_index": episode.round_index + 1,
    })
```

After decision parsed:

```python
    _emit(services, episode.episode_id, "planner_decision", {
        "action_type": decision.action_type,
        "reasoning": decision.reasoning,
        "arguments": decision.arguments,
        "round_index": episode.round_index,
    })
```

In `execute_tool_node`, before dispatch:

```python
    _emit(services, episode.episode_id, "tool_called", {
        "call_id": call.call_id, "action_type": call.action_type,
        "arguments": call.arguments, "step_index": episode.step_index,
    })
```

After dispatch + memory update:

```python
    _emit(services, episode.episode_id, "tool_result", {
        "observation": result.observation.model_dump(mode="json"),
        "ok": result.ok, "terminal": result.terminal,
        "step_index": episode.step_index,
    })
    _emit(services, episode.episode_id, "memory_updated", {
        "action_type": result.action_type,
        "round_index": episode.round_index,
    })
```

In `finalize_node`, after verdict (evaluator path):

```python
    _emit(services, episode.episode_id, "success_evaluated", {
        "success": episode.success, "answer": episode.answer,
        "submitted_explicitly": submitted_explicitly,
        "distance_to_goal": episode.distance_to_goal,
        "failure_type": episode.failure_type,
    })
```

**Step B4: Wire recorder in entrypoint**

In `RuntimeEntrypoint.run`, after creating recorder, before `graph.invoke`:

```python
        self.services.recorder = recorder
        self.services.event_seq[0] = 2
```

Replace the `episode_ended` sequence=2 with:

```python
                sequence=self.services.event_seq[0] + 1,
```

Clear recorder after invoke (so the services object is reusable):

```python
        finally:
            self.services.recorder = None
```

**Step B5: Update replay to parse new events (lenient)**

In `src/tiernav_runtime/replay.py`, in the replay loop, add a branch that ignores (or lightly applies) the new event types so replay doesn't crash on production logs. Add to the `else` branch:

```python
            elif event.event_type in {
                "subtask_started", "context_compiled", "planner_called",
                "planner_decision", "tool_called", "tool_result",
                "memory_query", "memory_updated", "success_evaluated",
            }:
                # Informational events; replay applies tool_result observations
                # and planner_decision for richer state reconstruction.
                if event.event_type == "tool_result":
                    payload = ToolResultReceivedPayload.model_validate(event.payload)
                    state.last_observation = payload.observation
                    if payload.step_index is not None:
                        state.step_index = payload.step_index
                elif event.event_type == "planner_decision":
                    from .contracts import PlannerDecision
                    state.current_decision = PlannerDecision.model_validate(
                        {k: v for k, v in event.payload.items()
                         if k in {"action_type", "reasoning", "expected",
                                  "confidence", "arguments"}}
                    )
                # other event types: no state change, accepted for forward-compat
```

Note: the existing `tool_result_received` branch stays for back-compat with old logs; the new `tool_result` is the production name. Keep both.

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_event_logging.py -v`
Expected: PASS.

**Step B6: Run full suite + fix regressions**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q
```

Expected: all pass. Existing event-log tests that assert exact sequence numbers (1, 2) may need updating to the new scheme — update them to assert `episode_started` has the lowest sequence and `episode_ended` the highest, not hard-coded 1/2.

**Step B7: Commit**

```bash
git add src/tiernav_runtime/graph.py src/tiernav_runtime/entrypoint.py \
        src/tiernav_runtime/replay.py tests/runtime/test_event_logging.py
git commit -m "feat(runtime): emit 9 missing event types + LLM round logs"
```

---

### Task C: GOATBench Logger integration (P0)

**Files:**
- Modify: `run_goatbench_evaluation.py:547-635` (`_run_goatbench_runtime`)
- Test: `tests/runtime/test_goatbench_logger_integration.py`

**Root cause:** `_run_goatbench_runtime` runs subtasks through the runtime entrypoint but never calls `logger.init_subtask`, `logger.log_step`, or `logger.log_subtask_result`. Stats stay NaN.

**Step C1: Write failing test (mocked, no habitat)**

```python
"""GOATBench runner feeds EpisodeResult + executor state into legacy Logger."""
from unittest.mock import MagicMock
import numpy as np

from src.tiernav_runtime.contracts import (
    EpisodeResult, TaskMode, PlannerDecision,
)


def test_log_subtask_result_called_with_runtime_outputs():
    """After each subtask, logger.log_subtask_result gets success/distance/snapshots."""
    from run_goatbench_evaluation import _feed_result_to_logger

    logger = MagicMock()
    logger.subtask_explore_dist = 2.0
    logger.success_by_snapshot = {}
    logger.success_by_distance = {}
    logger.spl_by_snapshot = {}
    logger.spl_by_distance = {}
    logger.success_by_task = {}
    logger.spl_by_task = {}
    logger.n_filtered_snapshots_list = {}
    logger.n_total_snapshots_list = {}
    logger.n_total_frames_list = {}

    result = EpisodeResult(
        episode_id="ep1_0", scene_id="s", task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION, success=True,
        distance_to_goal=0.5, submit_was_explicit=True,
        path_length=3.0, steps_taken=5,
    )
    executor = MagicMock()
    executor._pts = np.array([1.0, 2.0])
    executor._path_length = 3.0
    scene = MagicMock()
    scene.snapshots = {"a": 1, "b": 2}
    scene.frames = list(range(10))
    tsdf_planner = MagicMock()

    _feed_result_to_logger(
        logger=logger, result=result, executor=executor, scene=scene,
        tsdf_planner=tsdf_planner, subtask_id="ep1_0",
        goal_type="object", subtask_goal=[{
            "object_category": "chair", "object_id": "obj_0",
            "position": [1, 1, 1],
            "view_points": [{"agent_state": {"position": [1, 1, 1]}}],
        }],
        floor_height=0.5,
    )

    logger.log_subtask_result.assert_called_once()
    call = logger.log_subtask_result.call_args.kwargs
    assert call["success_by_distance"] is True
    assert call["subtask_id"] == "ep1_0"
    assert call["gt_subtask_explore_dist"] > 0
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_goatbench_logger_integration.py -v`
Expected: FAIL — `_feed_result_to_logger` doesn't exist.

**Step C2: Extract `_feed_result_to_logger` helper**

In `run_goatbench_evaluation.py`, add a module-level function near `_run_goatbench_runtime`:

```python
def _feed_result_to_logger(
    *, logger, result, executor, scene, tsdf_planner,
    subtask_id, goal_type, subtask_goal, floor_height,
):
    """Translate a runtime EpisodeResult into legacy Logger calls.

    Mirrors the legacy path: init_subtask (for gt distance + metadata),
    log_step (path), log_subtask_result (verdict + SPL + snapshots).
    """
    import numpy as np
    import habitat_sim

    pts = executor._pts if hasattr(executor, "_pts") and executor._pts is not None else np.array([0.0, 0.0])
    pts_3d = np.array([float(pts[0]), float(pts[1]), float(floor_height)], dtype=np.float32)

    subtask_metadata = logger.init_subtask(
        subtask_id, goal_type, subtask_goal, pts_3d, scene, tsdf_planner,
    )

    # log_step along the executor's traversed path (approx: single segment)
    try:
        start_voxel = tsdf_planner.habitat2voxel(pts_3d)[:2]
        logger.log_step(pts_voxel=start_voxel)
    except Exception:
        pass

    success_by_distance = bool(result.success and result.submit_was_explicit
                               and (result.distance_to_goal or 0.0) <= 1.0)
    # success_by_snapshot: legacy heuristic — target object observed in a snapshot.
    # Runtime doesn't track object-id matches; default to success_by_distance.
    success_by_snapshot = success_by_distance

    n_filtered = 0
    n_total = len(getattr(scene, "snapshots", {}) or {})
    n_frames = len(getattr(scene, "frames", []) or [])

    logger.log_subtask_result(
        success_by_snapshot=success_by_snapshot,
        success_by_distance=success_by_distance,
        subtask_id=subtask_id,
        gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
        goal_type=goal_type,
        n_filtered_snapshots=n_filtered,
        n_total_snapshots=n_total,
        n_total_frames=n_frames,
    )
```

**Step C3: Call helper in `_run_goatbench_runtime`**

After `results.append(result); global_step += 1` (around line 629-630), add:

```python
            try:
                _feed_result_to_logger(
                    logger=logger, result=result, executor=executor,
                    scene=scene, tsdf_planner=tsdf_planner,
                    subtask_id=f"{episode_id}_{subtask_idx}",
                    goal_type=goal_type, subtask_goal=subtask_goal,
                    floor_height=floor_height,
                )
            except Exception as log_e:
                logging.warning("Logger feed failed for subtask %d: %s", subtask_idx, log_e)
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_goatbench_logger_integration.py -v`
Expected: PASS.

**Step C4: Commit**

```bash
git add run_goatbench_evaluation.py tests/runtime/test_goatbench_logger_integration.py
git commit -m "feat(goatbench): wire runtime EpisodeResult into legacy Logger for stats"
```

---

### Task D: SPL metric (P1)

**Files:**
- Already implemented by `Logger.log_subtask_result` (logger_goatbench.py:389-398) once Task C feeds it.
- Verify: `tests/runtime/test_goatbench_logger_integration.py` extended.

**Step D1: Extend Task C test to assert SPL populated**

Add to the test in Task C1:

```python
    # SPL computed by logger.log_subtask_result
    assert "ep1_0" in logger.spl_by_distance
```

Since `log_subtask_result` is mocked, this assertion requires running the real method. Change the test to use a real `Logger` instance against a temp dir instead of MagicMock:

```python
def test_spl_computed_after_logger_feed(tmp_path):
    from src.logger_goatbench import Logger
    import habitat_sim  # only for type; Logger __init__ doesn't import it

    logger = Logger(str(tmp_path), 0.0, 1.0, 1, voxel_size=0.05)
    logger.init_episode("ep1")
    # ... same feed as above ...
    assert "ep1_0" in logger.spl_by_distance
    assert logger.spl_by_distance["ep1_0"] >= 0.0
```

If `Logger.__init__` requires habitat resources not available in unit test, fall back to asserting the call args include `gt_subtask_explore_dist > 0` (which drives SPL). Keep the MagicMock version and assert:

```python
    assert call["gt_subtask_explore_dist"] > 0.0  # SPL numerator
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_goatbench_logger_integration.py -v`
Expected: PASS.

**Step D2: Commit (if separate from Task C)**

If the SPL assertion was added in Task C, no separate commit. Otherwise:

```bash
git add tests/runtime/test_goatbench_logger_integration.py
git commit -m "test(goatbench): assert SPL inputs fed to Logger"
```

---

### Task E: success_by_snapshot (P2)

**Files:**
- Modify: `run_goatbench_evaluation.py` (`_feed_result_to_logger` from Task C)
- Modify: `src/tiernav_runtime/contracts.py` (add optional field to EpisodeResult)
- Test: `tests/runtime/test_goatbench_logger_integration.py`

**Design:** `success_by_snapshot` in legacy code checks whether the target object was observed in a filtered snapshot near the agent. The runtime doesn't do object-id matching in the planner loop. Add an optional `target_observed: Optional[bool]` to `EpisodeResult`, set by the runner when it can check the executor's last observation against the goal object ids. Default None → fall back to `success_by_distance`.

**Step E1: Add optional field to EpisodeResult**

In `src/tiernav_runtime/contracts.py`, `EpisodeResult`:

```python
    # GOATBench: was the target object observed in a snapshot? None when
    # unchecked; the runner sets it from executor last-observation.
    target_observed: Optional[bool] = None
```

**Step E2: Use it in `_feed_result_to_logger`**

In Task C's helper, replace:

```python
    success_by_snapshot = success_by_distance
```

with:

```python
    success_by_snapshot = (
        result.target_observed if result.target_observed is not None
        else success_by_distance
    )
```

**Step E3: Runner sets target_observed when possible**

In `_run_goatbench_runtime`, after the distance-computation block (around line 627), before `results.append`:

```python
            # Check if target object appears in executor's last observation.
            try:
                goal_obj_ids = [int(g["object_id"].split("_")[-1])
                                for g in subtask_goal if "object_id" in g]
                last_obs = getattr(executor, "last_observation", None)
                if last_obs is not None and hasattr(last_obs, "objects_nearby"):
                    seen = set(last_obs.objects_nearby or [])
                    result.target_observed = any(oid in seen for oid in goal_obj_ids)
            except Exception:
                pass
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_goatbench_logger_integration.py -v`
Expected: PASS.

**Step E4: Commit**

```bash
git add src/tiernav_runtime/contracts.py run_goatbench_evaluation.py \
        tests/runtime/test_goatbench_logger_integration.py
git commit -m "feat(goatbench): success_by_snapshot from target_observed in EpisodeResult"
```

---

### Task F: Planner retry (P2)

**Files:**
- Modify: `src/tiernav_runtime/contracts.py` (add `planner_retries` to BenchmarkRule)
- Modify: `src/tiernav_runtime/planner.py` (`decide` accepts retry budget)
- Modify: `src/tiernav_runtime/graph.py` (plan_node passes retry budget)
- Test: `tests/runtime/test_planner_retry.py`

**Design:** Spec line 414: "Policy may retry once if configured." Add `planner_retries: int = 0` to `BenchmarkRule`. `decide()` retries on parse errors up to `planner_retries` times before falling back to submit_answer. The graph's `plan_node` reads the budget from the rule via services.

**Step F1: Add field to BenchmarkRule**

In `src/tiernav_runtime/contracts.py`, find `BenchmarkRule` and add:

```python
    # Bounded planner retry count on parse/call failures. 0 = no retry
    # (immediate fallback submit). Default 0 preserves current behavior.
    planner_retries: NonNegativeInt = 0
```

**Step F2: Write failing test**

```python
"""PlannerClient.decide retries on parse errors up to planner_retries."""
from unittest.mock import patch
from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.planner import PlannerClient


def test_decide_retries_on_parse_error():
    cfg = ProviderConfig(api_key_env="T", base_url_env="T", model_env="T")
    client = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")

    calls = {"n": 0}
    def fake_vlm(messages, **kw):
        calls["n"] += 1
        return "not json {{{" if calls["n"] == 1 else '{"action_type": "explore_panorama", "reason": "ok"}'

    with patch("src.tiernav_runtime.planner._call_vlm", side_effect=fake_vlm):
        decision = client.decide("p", retries=1)

    assert calls["n"] == 2
    assert decision.action_type == "explore_panorama"


def test_decide_falls_back_after_retry_budget_exhausted():
    cfg = ProviderConfig(api_key_env="T", base_url_env="T", model_env="T")
    client = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")

    with patch("src.tiernav_runtime.planner._call_vlm",
               return_value="not json {{{"):
        decision = client.decide("p", retries=2)

    assert decision.action_type == "submit_answer"
    assert "planner_parse_error" in decision.arguments.get("failure_reason", "")
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_planner_retry.py -v`
Expected: FAIL — `decide()` doesn't accept `retries`.

**Step F3: Implement retry in decide()**

In `src/tiernav_runtime/planner.py`, change `decide` signature and loop:

```python
    def decide(self, prompt: str, *, retries: int = 0) -> PlannerDecision:
        """Call the VLM; on parse failure, retry up to `retries` times before fallback."""
        import json as _json

        messages = [{"role": "user", "content": prompt}]
        last_error_decision = None

        for attempt in range(retries + 1):
            try:
                raw = self.call_vlm(messages)
            except Exception:
                last_error_decision = PlannerDecision(
                    action_type="submit_answer",
                    reasoning="planner_call_failed",
                    confidence=0.0,
                    arguments={"failure_reason": "planner_call_failed",
                               "attempt": attempt + 1},
                )
                continue  # retry on call failure too

            try:
                parsed = _json.loads(raw.strip())
            except _json.JSONDecodeError:
                last_error_decision = PlannerDecision(
                    action_type="submit_answer",
                    reasoning="planner_parse_error",
                    confidence=0.0,
                    arguments={"failure_reason": "planner_parse_error",
                               "raw": raw[:500], "attempt": attempt + 1},
                )
                continue

            if not isinstance(parsed, dict) or not parsed.get("action_type"):
                last_error_decision = PlannerDecision(
                    action_type="submit_answer",
                    reasoning="planner_response_not_dict" if not isinstance(parsed, dict)
                              else "planner_missing_action_type",
                    confidence=0.0,
                    arguments={"failure_reason": "planner_bad_response", "attempt": attempt + 1},
                )
                continue

            return planner_action_to_decision(type("PlannerAction", (), parsed)())

        return last_error_decision  # type: ignore[return-value]
```

Run: `/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_planner_retry.py -v`
Expected: PASS.

**Step F4: Wire retry budget through plan_node**

In `src/tiernav_runtime/graph.py`, `plan_node`, change:

```python
    raw = services.planner.decide(episode.prompt)
```

to:

```python
    rule = getattr(services, "_active_rule", None)
    retries = getattr(rule, "planner_retries", 0) if rule is not None else 0
    raw = services.planner.decide(episode.prompt, retries=retries)
```

Expose the rule on services. In `RuntimeServices`, add:

```python
    # Active BenchmarkRule, set by with_real_services, read by plan_node
    # for planner_retries. None for fake-services path.
    rule: Any = None
```

In `entrypoint.py`, `with_real_services`, set `services.rule = rule` before returning. (Add `rule=rule,` to the `RuntimeServices(...)` call.)

**Step F5: Run full suite**

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q
```

Expected: all pass. Existing `test_planner_decide.py` tests call `decide(prompt)` without `retries` — the default 0 preserves their behavior.

**Step F6: Commit**

```bash
git add src/tiernav_runtime/contracts.py src/tiernav_runtime/planner.py \
        src/tiernav_runtime/graph.py src/tiernav_runtime/entrypoint.py \
        tests/runtime/test_planner_retry.py
git commit -m "feat(runtime): bounded planner retry driven by BenchmarkRule.planner_retries"
```

---

## Completion Checklist

- [ ] Task A: `EpisodeResult.path_length` reads `env_service.path_length` (not `step_index`)
- [ ] Task B: 9 missing event types emitted + LLM round logs (planner_called/planner_decision)
- [ ] Task C: GOATBench Logger fed from runtime EpisodeResult (stats no longer NaN)
- [ ] Task D: SPL metric computed (via Task C's `log_subtask_result` call)
- [ ] Task E: `success_by_snapshot` from `EpisodeResult.target_observed`
- [ ] Task F: `BenchmarkRule.planner_retries` drives bounded retry in `decide()`
- [ ] All tests pass (`/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/ -q`)
- [ ] Smoke test on server still runs end-to-end (AEQA 5q + GOATBench 1 episode)
