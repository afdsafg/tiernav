"""Tests for the LangGraph runtime graph skeleton.

Deterministic fakes only: no external services, no network. Exercises the
planned examples (tool-then-submit, round-budget fallback) plus extra
coverage for context compilation, memory query, tool-failure resilience,
and the default ContextCompiler on RuntimeServices.

The graph is driven through the planned LangGraph invoke contract:
``build_runtime_graph()`` takes no arguments and services are injected via
``config={"configurable": {"services": services}}``. The final graph output
is a plain JSON dict, so assertions read ``final_state["state"]["..."]``.
"""
from __future__ import annotations

from typing import Optional

from src.tiernav_runtime.context import ContextCompiler
from src.tiernav_runtime.contracts import (
    AblationConfig,
    EpisodeRequest,
    Observation,
    PlannerDecision,
    RunSpec,
    ToolCall,
    ToolResult,
)
from src.tiernav_runtime.graph import (
    RuntimeGraphState,
    RuntimeServices,
    build_runtime_graph,
)
from src.tiernav_runtime.memory import MemoryService
from src.tiernav_runtime.policy import WorkflowPolicy
from src.tiernav_runtime.tools import RuntimeTool, ToolRegistry, with_stable_defaults


# ── Fakes ─────────────────────────────────────────────────────────────────


class FakePlanner:
    """Deterministic planner that replays a scripted action sequence.

    Each call to decide returns the next PlannerDecision in ``script``. The
    final decision is repeated once the script is exhausted so the policy
    can route to a terminal node.
    """

    def __init__(self, script: list[PlannerDecision]) -> None:
        if not script:
            raise ValueError("script must be non-empty")
        self._script = list(script)
        self._i = 0
        self.prompts: list[str] = []

    def decide(self, prompt: str, **kwargs) -> PlannerDecision:
        self.prompts.append(prompt)
        decision = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return decision


def _request(prompt: str = "where is the mug?") -> EpisodeRequest:
    return EpisodeRequest(
        episode_id="ep-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt=prompt,
    )


def _spec(
    *,
    max_rounds: int = 10,
    max_steps: int = 50,
    ablation: Optional[AblationConfig] = None,
) -> RunSpec:
    return RunSpec(
        run_id="run-1",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
        seed=0,
        max_rounds=max_rounds,
        max_steps=max_steps,
        ablation=ablation if ablation is not None else AblationConfig(),
    )


def _services(planner: FakePlanner) -> RuntimeServices:
    return RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
    )


def _run(
    services: RuntimeServices, spec: RunSpec, request: EpisodeRequest
) -> dict:
    """Invoke the compiled graph via the config contract; return final dict."""
    graph = build_runtime_graph()
    final_state = graph.invoke(
        {"spec": spec.model_dump(mode="json"), "request": request.model_dump(mode="json")},
        config={"configurable": {"services": services}},
    )
    assert isinstance(final_state, dict)
    return final_state


# ── Plan examples ─────────────────────────────────────────────────────────


def test_tool_then_submit_returns_answer():
    """FakePlanner: explore_frontier then submit_answer -> terminal, mug, step_index 1."""
    planner = FakePlanner(
        [
            PlannerDecision(
                action_type="explore_frontier",
                arguments={"frontier_id": "f1"},
            ),
            PlannerDecision(
                action_type="submit_answer",
                arguments={"answer": "mug"},
            ),
        ]
    )
    final_state = _run(_services(planner), _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["answer"] == "mug"
    assert state["step_index"] == 1


def test_single_submit_answer_step_index_is_one():
    """A single submit_answer route (no tool dispatch) leaves step_index at 0.

    submit_answer routes policy->finalize without running execute_tool, so
    no step is taken. This complements test_tool_then_submit_returns_answer
    which asserts step_index == 1 after exactly one tool dispatch.
    """
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "mug"})]
    )
    final_state = _run(_services(planner), _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["answer"] == "mug"
    assert state["step_index"] == 0


def test_round_budget_fallback():
    """max_rounds=1 -> terminal, success False, failure_type round_budget."""
    planner = FakePlanner(
        [PlannerDecision(action_type="explore_frontier")]
    )
    final_state = _run(_services(planner), _spec(max_rounds=1), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is False
    assert state["answer"] == "unanswerable"
    assert state["failure_type"] == "round_budget"
    # policy key is carried into the final graph output.
    assert final_state["policy"]["reason"] == "round_budget"


# ── Extra coverage ────────────────────────────────────────────────────────


def test_context_sections_written_and_prompt_nonempty():
    task_prompt = "where is the mug?"
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "sofa"})]
    )
    final_state = _run(_services(planner), _spec(), _request(prompt=task_prompt))
    state = final_state["state"]

    assert state["context_sections"], "context_sections must be populated"
    names = [s["name"] for s in state["context_sections"]]
    assert "task_instruction" in names
    assert "action_schema" in names
    assert state["prompt"] == task_prompt
    assert final_state["prompt"], "rendered graph prompt must be non-empty"
    assert "## task_instruction" in final_state["prompt"]
    assert final_state["prompt"] != state["prompt"]


def test_task_prompt_does_not_recursively_embed_rendered_context():
    """EpisodeState.prompt must stay as the task text across planner rounds."""
    task_prompt = "where is the mug?"
    planner = FakePlanner(
        [
            PlannerDecision(action_type="explore_panorama"),
            PlannerDecision(
                action_type="submit_answer",
                arguments={"answer": "mug"},
            ),
        ]
    )
    final_state = _run(_services(planner), _spec(), _request(prompt=task_prompt))
    state = final_state["state"]

    assert len(planner.prompts) >= 2
    second_round_prompt = planner.prompts[1]
    assert second_round_prompt.count("## task_instruction") == 1
    assert f"prompt: {task_prompt}" in second_round_prompt
    assert state["prompt"] == task_prompt
    assert final_state["prompt"] == second_round_prompt


def test_memory_pack_present_when_active_memory_query_true():
    """active_memory_query=True -> memory_pack is populated before planning."""
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    # Seed memory so query() returns a non-empty pack.
    services = _services(planner)
    services.memory.update_from_observation(
        Observation(
            image_ids=["img-1"], summary="a mug on the counter", object_ids=["mug"]
        ),
        action_type="explore_panorama",
        round_index=0,
    )
    spec = _spec(ablation=AblationConfig(active_memory_query=True, spatial_memory=True))
    final_state = _run(services, spec, _request(prompt="mug"))
    state = final_state["state"]

    assert state["memory_pack"] is not None
    assert state["memory_pack"]["query"] == "mug"
    assert state["memory_pack"]["summary"], "memory_pack summary should be non-empty with a match"


def test_memory_pack_absent_when_active_memory_query_false():
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    spec = _spec(ablation=AblationConfig(active_memory_query=False, spatial_memory=False))
    final_state = _run(_services(planner), spec, _request())
    state = final_state["state"]

    assert state["memory_pack"] is None


def test_tool_failure_from_registry_does_not_crash():
    """A tool returning a structured error must not raise; graph continues to fallback."""

    class AlwaysFailTool(RuntimeTool):
        name = "explore_frontier"

        def run(self, call: ToolCall) -> ToolResult:
            return ToolResult(
                call_id=call.call_id,
                action_type=call.action_type,
                ok=False,
                terminal=False,
                error="simulated tool failure",
            )

    registry = ToolRegistry()
    registry.register(AlwaysFailTool())
    registry.register(with_stable_defaults()._tools["submit_answer"])

    planner = FakePlanner(
        [
            PlannerDecision(action_type="explore_frontier"),
            PlannerDecision(action_type="submit_answer", arguments={"answer": "ok"}),
        ]
    )
    services = RuntimeServices(
        planner=planner,
        tools=registry,
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
    )
    # Should not raise; the failed tool yields a non-terminal error result and
    # the graph continues to the next round, which submits an answer.
    final_state = _run(services, _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["answer"] == "ok"


def test_runtime_services_default_context_is_context_compiler():
    """RuntimeServices fills ContextCompiler by default when context=None."""
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        # context intentionally omitted
    )
    assert isinstance(services.context, ContextCompiler)


def test_runtime_services_explicit_context_preserved():
    """An explicitly supplied ContextCompiler is not replaced."""
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    custom = ContextCompiler()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        context=custom,
    )
    assert services.context is custom


def test_graph_state_typeddict_has_required_keys():
    """RuntimeGraphState declares spec/request/state/policy/prompt at least."""
    # TypedDict __annotations__ reveal the declared keys.
    ann = RuntimeGraphState.__annotations__
    for key in ("spec", "request", "state", "policy", "prompt"):
        assert key in ann, f"RuntimeGraphState missing key: {key}"


# ── RuntimeEnvironmentService (Task 3) ───────────────────────────────────


from src.tiernav_runtime.env import RuntimeEnvironmentService


class FakeScene:
    """Minimal scene double with a cleanup flag."""

    def __init__(self) -> None:
        self.cleanup_called = False

    def cleanup(self) -> None:
        self.cleanup_called = True


class FakeExecutor:
    """Executor double tracking path_length and set_state calls."""

    def __init__(self, path_length: float = 0.0) -> None:
        self._path_length = path_length
        self.set_state_calls: list[tuple] = []

    @property
    def path_length(self) -> float:
        return self._path_length

    def set_state(self, pts, angle, step_counter: int) -> None:
        self.set_state_calls.append((pts, angle, step_counter))


def _aeqa_env(executor: FakeExecutor | None = None) -> RuntimeEnvironmentService:
    return RuntimeEnvironmentService.for_aeqa(
        scene=FakeScene(),
        tsdf_planner=object(),
        executor=executor or FakeExecutor(),
        detection_model=None,
        sam_predictor=None,
        clip_model=None,
        clip_preprocess=None,
        clip_tokenizer=None,
        logger=None,
    )


def _goat_env(executor: FakeExecutor | None = None) -> RuntimeEnvironmentService:
    return RuntimeEnvironmentService.for_goatbench(
        scene=FakeScene(),
        tsdf_planner=object(),
        executor=executor or FakeExecutor(),
        detection_model=None,
        sam_predictor=None,
        clip_model=None,
        clip_preprocess=None,
        clip_tokenizer=None,
        logger=None,
    )


def test_environment_service_can_build_aeqa_session():
    env = _aeqa_env()
    assert env.task_mode == "question_answering"


def test_environment_service_can_build_goatbench_session():
    env = _goat_env()
    assert env.task_mode == "goal_navigation"


def test_aeqa_start_session_resets_pose_and_path_length():
    env = _aeqa_env()
    # Prime state with a non-default pose and path length.
    env.start_session("q-1", initial_pose={"x": 1.0, "y": 2.0, "theta": 0.5})
    assert env.current_pose == {"x": 1.0, "y": 2.0, "theta": 0.5}

    # Second question: fresh session must reset pose/path_length.
    env.start_session("q-2", initial_pose={"x": 9.0, "y": 8.0, "theta": 0.0})
    assert env.current_pose == {"x": 9.0, "y": 8.0, "theta": 0.0}
    assert env.path_length == 0.0


def test_goatbench_start_session_threads_pose_across_subtasks():
    env = _goat_env()
    env.start_session("ep-1", initial_pose={"x": 0.0, "y": 0.0, "theta": 0.0})
    # Simulate movement during subtask 1 by priming internal state directly
    # (no public setter exists by design — pose is advanced by the graph).
    env._current_pose = {"x": 3.0, "y": 4.0, "theta": 1.0}
    env._path_length = 5.0

    # Subtask 2 within same episode: pose must thread, NOT reset.
    env.start_session("ep-1", initial_pose={"x": 0.0, "y": 0.0, "theta": 0.0})
    assert env.current_pose == {"x": 3.0, "y": 4.0, "theta": 1.0}
    assert env.path_length == 5.0


def test_goatbench_start_session_resets_on_new_episode():
    env = _goat_env()
    env.start_session("ep-1", initial_pose={"x": 1.0, "y": 2.0, "theta": 0.5})
    env._path_length = 7.0

    # New episode: fresh start must reset pose/path_length.
    env.start_session("ep-2", initial_pose={"x": 9.0, "y": 8.0, "theta": 0.0})
    assert env.current_pose == {"x": 9.0, "y": 8.0, "theta": 0.0}
    assert env.path_length == 0.0


def test_teardown_session_calls_scene_cleanup_and_marks_torn_down():
    env = _aeqa_env()
    env.start_session("q-1")
    env.teardown_session()
    assert env.scene.cleanup_called is True
    assert env.is_torn_down is True


def test_teardown_session_idempotent():
    env = _aeqa_env()
    env.start_session("q-1")
    env.teardown_session()
    # Second teardown must not re-call cleanup.
    env.teardown_session()
    assert env.scene.cleanup_called is True


# ── Real-service graph node tests (Task 7) ────────────────────────────────


from src.tiernav_runtime.contracts import BenchmarkRule, MemoryScope, TaskMode
from src.tiernav_runtime.success import SuccessEvaluator


# ── Helpers ───────────────────────────────────────────────────────────────


def _aeqa_evaluator() -> SuccessEvaluator:
    """SuccessEvaluator for AEQA (QUESTION_ANSWERING). Distance irrelevant."""
    return SuccessEvaluator(
        BenchmarkRule(
            success_distance_m=1.0,
            requires_explicit_stop=False,
            memory_scope=MemoryScope.PER_QUESTION,
            scoring_mode="aeqa",
        )
    )


def _goat_evaluator(distance_m: float = 1.0) -> SuccessEvaluator:
    """SuccessEvaluator for GOATBench with a configurable success distance."""
    return SuccessEvaluator(
        BenchmarkRule(
            success_distance_m=distance_m,
            requires_explicit_stop=True,
            memory_scope=MemoryScope.SUBTASK_SEQUENCE,
            scoring_mode="goatbench_spl",
        )
    )


def _services_with_evaluator(
    planner: FakePlanner,
    evaluator: SuccessEvaluator,
    *,
    environment: object | None = None,
) -> RuntimeServices:
    return RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        success_evaluator=evaluator,
        environment=environment,
    )


def _fake_env_with_distance(
    current_pose: dict | None = None,
    goal_pose: dict | None = None,
) -> RuntimeEnvironmentService:
    """Create a RuntimeEnvironmentService with preset poses for testing."""
    env = _goat_env()
    if current_pose is not None:
        env._current_pose = dict(current_pose)
    if goal_pose is not None:
        env._goal_pose = dict(goal_pose)
        env._goal_poses = [dict(goal_pose)]
    return env


# ── Success evaluator tests ───────────────────────────────────────────────


def test_runtime_graph_uses_success_evaluator_when_injected():
    """AEQA: submit_answer with evaluator → success based on verdict."""
    evaluator = _aeqa_evaluator()
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "mug"})]
    )
    services = _services_with_evaluator(planner, evaluator)
    final_state = _run(services, _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["answer"] == "mug"
    assert state["submitted_explicitly"] is True


def test_runtime_graph_evaluator_empty_answer_is_failure():
    """AEQA: submit_answer with empty answer → evaluator rejects."""
    evaluator = _aeqa_evaluator()
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={})]
    )
    services = _services_with_evaluator(planner, evaluator)
    final_state = _run(services, _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is False
    assert state["failure_type"] == "no_answer"


def test_runtime_graph_falls_back_to_aeqa_logic_when_no_evaluator():
    """Without evaluator, old AEQA answer-nonempty logic still works."""
    planner = FakePlanner(
        [
            PlannerDecision(
                action_type="explore_frontier", arguments={"frontier_id": "f1"}
            ),
            PlannerDecision(
                action_type="submit_answer", arguments={"answer": "mug"}
            ),
        ]
    )
    services = _services(planner)  # no evaluator
    final_state = _run(services, _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["answer"] == "mug"
    # submitted_explicitly not set in old path
    assert state.get("submitted_explicitly", False) is False


def test_goatbench_finalize_node_uses_distance():
    """GOATBench: distance below threshold → success; above → failure."""
    evaluator = _goat_evaluator(distance_m=1.0)
    # Set up an env with a computed distance. The fake env computes
    # Euclidean distance from current_pose to goal_pose.
    env = _fake_env_with_distance(
        current_pose={"x": 0.0, "y": 0.0, "theta": 0.0},
        goal_pose={"x": 0.5, "y": 0.0, "theta": 0.0},
    )
    # distance = sqrt((0.5)^2 + 0) = 0.5, which is ≤ 1.0 → success.
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "arrived"})]
    )
    services = _services_with_evaluator(planner, evaluator, environment=env)
    # Set task_mode to GOAL_NAVIGATION for the evaluator.
    request = EpisodeRequest(
        episode_id="ep-goat-1",
        scene_id="scene-1",
        task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION,
        prompt="navigate to the chair",
    )
    final_state = _run(services, _spec(), request)
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["distance_to_goal"] == 0.5


def test_goatbench_distance_exceeded_is_failure():
    """GOATBench: distance above threshold → failure."""
    evaluator = _goat_evaluator(distance_m=1.0)
    env = _fake_env_with_distance(
        current_pose={"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0},
        goal_pose={"x": 3.0, "y": 0.0, "z": 4.0, "theta": 0.0},
    )
    # distance = sqrt(3^2 + 4^2) = 5.0 > 1.0 → failure.
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "arrived"})]
    )
    services = _services_with_evaluator(planner, evaluator, environment=env)
    request = EpisodeRequest(
        episode_id="ep-goat-2",
        scene_id="scene-1",
        task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION,
        prompt="navigate to the chair",
    )
    final_state = _run(services, _spec(), request)
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is False
    assert state["failure_type"] == "distance_exceeded"


def test_execute_tool_node_computes_distance_from_env_when_available():
    """execute_tool_node updates distance_to_goal from env after each tool."""
    evaluator = _goat_evaluator(distance_m=2.0)
    # Habitat coords: y is up; floor-plane distance uses x, z.
    env = _fake_env_with_distance(
        current_pose={"x": 1.0, "y": 1.0, "z": 1.0, "theta": 0.0},
        goal_pose={"x": 4.0, "y": 1.0, "z": 5.0, "theta": 0.0},
    )
    # distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9+16) = 5.0
    planner = FakePlanner(
        [
            PlannerDecision(
                action_type="navigate_to_object",
                arguments={"object_name": "chair"},
            ),
            PlannerDecision(
                action_type="submit_answer", arguments={"answer": "arrived"}
            ),
        ]
    )
    services = _services_with_evaluator(planner, evaluator, environment=env)
    request = EpisodeRequest(
        episode_id="ep-goat-3",
        scene_id="scene-1",
        task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION,
        prompt="navigate to the chair",
    )
    final_state = _run(services, _spec(), request)
    state = final_state["state"]

    # Distance should have been computed in execute_tool_node after the nav tool.
    # The final submit_answer also refreshes distance in finalize_node, giving the
    # same value (pose hasn't changed since no real movement).
    assert state["terminal"] is True
    assert state["distance_to_goal"] == 5.0
    assert state["success"] is False  # 5.0 > 2.0
    assert state["failure_type"] == "distance_exceeded"


def test_execute_tool_skips_distance_when_no_environment():
    """When environment is None, no distance is computed (fake path)."""
    planner = FakePlanner(
        [
            PlannerDecision(
                action_type="navigate_to_object",
                arguments={"object_name": "chair"},
            ),
            PlannerDecision(
                action_type="submit_answer", arguments={"answer": "mug"}
            ),
        ]
    )
    services = _services(planner)  # no environment
    final_state = _run(services, _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["answer"] == "mug"
    assert state["distance_to_goal"] is None


def test_fallback_node_sets_submitted_explicitly_false():
    """fallback_node marks the episode as not explicitly submitted."""
    planner = FakePlanner(
        [PlannerDecision(action_type="explore_frontier")]
    )
    services = _services(planner)
    final_state = _run(services, _spec(max_rounds=1), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is False
    assert state["answer"] == "unanswerable"
    assert state.get("submitted_explicitly", True) is False


# ── End-to-end smoke tests ────────────────────────────────────────────────


def test_aeqa_runtime_smoke_with_fake_services():
    """Full AEQA smoke: FakePlanner explore_frontier -> submit_answer("chair") -> success.

    Exercises the complete runtime graph topology for AEQA with a
    SuccessEvaluator (BenchmarkRule). Verifies the full path: tool dispatch,
    observe, plan, submit, finalize, including distance (irrelevant for AEQA
    but the node should set it to None).
    """
    evaluator = _aeqa_evaluator()
    planner = FakePlanner(
        [
            PlannerDecision(
                action_type="explore_frontier",
                arguments={"frontier_id": "f1"},
            ),
            PlannerDecision(
                action_type="submit_answer",
                arguments={"answer": "chair"},
            ),
        ]
    )
    services = _services_with_evaluator(planner, evaluator)
    final_state = _run(services, _spec(), _request())
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["answer"] == "chair"
    assert state["step_index"] == 1
    assert state["submitted_explicitly"] is True
    # AEQA distance is irrelevant; should remain None.
    assert state["distance_to_goal"] is None


def test_goatbench_runtime_smoke_with_fake_services():
    """GOATBench smoke: FakePlanner navigate -> submit, distance check within threshold.

    Exercises the complete runtime graph topology for GOATBench with a
    SuccessEvaluator (BenchmarkRule, success_distance_m=1.0). Sets up a fake
    env with known poses so the distance is computed inside the graph.
    Verifies the evaluator's distance check runs and marks success.
    """
    evaluator = _goat_evaluator(distance_m=1.0)
    env = _fake_env_with_distance(
        current_pose={"x": 0.0, "y": 0.0, "theta": 0.0},
        goal_pose={"x": 0.5, "y": 0.0, "theta": 0.0},
    )
    # distance = sqrt((0.5-0)^2 + (0-0)^2) = 0.5 <= 1.0 -> success.
    planner = FakePlanner(
        [
            PlannerDecision(
                action_type="navigate_to_object",
                arguments={"object_name": "chair"},
            ),
            PlannerDecision(
                action_type="submit_answer",
                arguments={"answer": "arrived"},
            ),
        ]
    )
    services = _services_with_evaluator(planner, evaluator, environment=env)
    request = EpisodeRequest(
        episode_id="ep-goat-smoke",
        scene_id="scene-1",
        task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION,
        prompt="navigate to the chair",
    )
    final_state = _run(services, _spec(), request)
    state = final_state["state"]

    assert state["terminal"] is True
    assert state["success"] is True
    assert state["distance_to_goal"] == 0.5
    assert state["submitted_explicitly"] is True


# ── Phase 2: compact trace tests ──────────────────────────────────────────


from src.tiernav_runtime.context import ContextCompiler
from src.tiernav_runtime.graph import (
    COMPACT_THRESHOLD,
    _compact_trace,
    compile_context_node,
)
from src.tiernav_runtime.contracts import EpisodeState


class CallVlmPlanner:
    """Planner double exposing call_vlm for compact/recall paths.

    decide() is inherited-style: returns a terminal submit so the graph
    can complete if invoked. call_vlm returns the scripted raw text.
    """

    def __init__(self, vlm_text: str = "compact summary text", *, raise_on_vlm: bool = False) -> None:
        self._vlm_text = vlm_text
        self._raise_on_vlm = raise_on_vlm
        self.vlm_calls: list[list[dict]] = []

    def call_vlm(self, messages, **kwargs) -> str:
        self.vlm_calls.append(messages)
        if self._raise_on_vlm:
            raise RuntimeError("simulated planner failure")
        return self._vlm_text

    def decide(self, prompt: str, **kwargs) -> PlannerDecision:
        return PlannerDecision(
            action_type="submit_answer", arguments={"answer": "x"}
        )


def _compact_services(planner: CallVlmPlanner) -> RuntimeServices:
    return RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
    )


def _state_for_compact(round_index: int = COMPACT_THRESHOLD) -> EpisodeState:
    return EpisodeState(
        episode_id="ep-c1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="where is the mug?",
        round_index=round_index,
        step_index=round_index,
        last_observation=Observation(summary="mug visible on counter", room_id="kitchen"),
        distance_to_goal=0.5,
        failure_type="",
    )


def test_compact_trace_returns_string_when_planner_returns_text():
    planner = CallVlmPlanner(vlm_text="Rooms visited: kitchen. Mug found.")
    services = _compact_services(planner)
    episode = _state_for_compact()

    result = _compact_trace(services, episode)

    assert result == "Rooms visited: kitchen. Mug found."
    assert len(planner.vlm_calls) == 1
    # The compact prompt includes progress data from the episode.
    sent = planner.vlm_calls[0][0]["content"]
    assert "where is the mug?" in sent
    assert "kitchen" in sent


def test_compact_trace_returns_empty_on_planner_exception():
    planner = CallVlmPlanner(raise_on_vlm=True)
    services = _compact_services(planner)
    episode = _state_for_compact()

    result = _compact_trace(services, episode)

    assert result == ""


def test_compact_trace_returns_empty_on_blank_response():
    planner = CallVlmPlanner(vlm_text="   \n  ")
    services = _compact_services(planner)
    episode = _state_for_compact()

    result = _compact_trace(services, episode)

    assert result == ""


def _compile_context_invoke(services: RuntimeServices, episode: EpisodeState, spec: RunSpec) -> dict:
    """Invoke compile_context_node directly with a synthetic graph_state."""
    return compile_context_node(
        {
            "spec": spec.model_dump(mode="json"),
            "state": episode.model_dump(mode="json"),
        },
        config={"configurable": {"services": services}},
    )


def test_compile_context_node_triggers_compact_when_round_threshold_met():
    planner = CallVlmPlanner(vlm_text="compact: kitchen visited")
    services = _compact_services(planner)
    episode = _state_for_compact(round_index=COMPACT_THRESHOLD)

    out = _compile_context_invoke(services, episode, _spec())
    out_state = EpisodeState.model_validate(out["state"])

    assert out_state.compact_summary == "compact: kitchen visited"
    assert len(planner.vlm_calls) == 1


def test_compile_context_node_does_not_recompact_when_summary_already_set():
    planner = CallVlmPlanner(vlm_text="should not be called again")
    services = _compact_services(planner)
    episode = _state_for_compact(round_index=COMPACT_THRESHOLD + 2)
    episode.compact_summary = "pre-existing summary"

    out = _compile_context_invoke(services, episode, _spec())
    out_state = EpisodeState.model_validate(out["state"])

    assert out_state.compact_summary == "pre-existing summary"
    assert planner.vlm_calls == []


def test_compile_context_node_skips_compact_below_threshold():
    planner = CallVlmPlanner(vlm_text="should not be called")
    services = _compact_services(planner)
    episode = _state_for_compact(round_index=COMPACT_THRESHOLD - 1)

    out = _compile_context_invoke(services, episode, _spec())
    out_state = EpisodeState.model_validate(out["state"])

    assert out_state.compact_summary == ""
    assert planner.vlm_calls == []


# ── Phase 2: scene memory sediment tests ─────────────────────────────────


from src.tiernav_runtime.graph import _sediment_scene_memory, finalize_node
from src.tiernav_runtime.memory import MemorySession, ObjectNode, RoomNode
from src.tiernav_runtime.contracts import MemoryScope


class FakeSceneMemoryStore:
    """Records update_room / add_episodic_note calls for assertions."""

    def __init__(self) -> None:
        self.room_updates: list[dict] = []
        self.notes: list[dict] = []

    def update_room(self, room_id, objects_seen, visited_round, connectivity=None, notes=""):
        self.room_updates.append({
            "room_id": room_id,
            "objects_seen": list(objects_seen),
            "visited_round": visited_round,
            "connectivity": connectivity,
            "notes": notes,
        })

    def add_episodic_note(self, round, room, event):
        self.notes.append({"round": round, "room": room, "event": event})


def _sediment_episode(success: bool = True, answer: str = "mug") -> EpisodeState:
    return EpisodeState(
        episode_id="ep-sed-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="where is the mug?",
        round_index=2,
        step_index=2,
        last_observation=Observation(summary="done", room_id="kitchen"),
        terminal=True,
        success=success,
        answer=answer,
        failure_type="" if success else "no_answer",
    )


def _session_with_rooms() -> MemorySession:
    """Build a MemorySession with one room and one object in that room."""
    session = MemorySession(scope=MemoryScope.SUBTASK_SEQUENCE)
    session.start_session(episode_id="ep-sed-1")
    mem = session.current_memory
    mem.rooms["kitchen"] = RoomNode(room_id="kitchen")
    mem.objects["mug"] = ObjectNode(object_id="mug", room_id="kitchen")
    return session


def test_sediment_calls_update_room_and_add_episodic_note():
    store = FakeSceneMemoryStore()
    session = _session_with_rooms()
    planner = CallVlmPlanner()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        memory_session=session,
        scene_memory_store=store,
    )
    episode = _sediment_episode()

    _sediment_scene_memory(services, episode)

    assert len(store.room_updates) == 1
    ru = store.room_updates[0]
    assert ru["room_id"] == "kitchen"
    assert ru["objects_seen"] == ["mug"]
    assert ru["visited_round"] == 2
    assert ru["notes"] == "success"
    assert len(store.notes) == 1
    note = store.notes[0]
    assert note["room"] == "kitchen"
    assert "episode=ep-sed-1" in note["event"]
    assert "success=True" in note["event"]
    assert "answer=mug" in note["event"]


def test_sediment_failure_note_when_episode_failed():
    store = FakeSceneMemoryStore()
    session = _session_with_rooms()
    planner = CallVlmPlanner()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        memory_session=session,
        scene_memory_store=store,
    )
    episode = _sediment_episode(success=False, answer="")
    episode.failure_type = "no_answer"

    _sediment_scene_memory(services, episode)

    assert store.room_updates[0]["notes"] == "failed: no_answer"
    assert "success=False" in store.notes[0]["event"]


def test_sediment_is_noop_when_store_is_none():
    session = _session_with_rooms()
    planner = CallVlmPlanner()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        memory_session=session,
        scene_memory_store=None,
    )
    episode = _sediment_episode()

    # Must not raise.
    _sediment_scene_memory(services, episode)


def test_sediment_is_non_fatal_when_session_not_started():
    """session.current_memory raises RuntimeError before any session is started."""
    store = FakeSceneMemoryStore()
    session = MemorySession(scope=MemoryScope.SUBTASK_SEQUENCE)
    planner = CallVlmPlanner()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        memory_session=session,
        scene_memory_store=store,
    )
    episode = _sediment_episode()

    _sediment_scene_memory(services, episode)

    # No session -> no rooms iterated -> no calls.
    assert store.room_updates == []
    assert store.notes == []


def test_sediment_is_non_fatal_when_memory_session_is_none():
    store = FakeSceneMemoryStore()
    planner = CallVlmPlanner()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        memory_session=None,
        scene_memory_store=store,
    )
    episode = _sediment_episode()

    _sediment_scene_memory(services, episode)
    assert store.room_updates == []
    assert store.notes == []


def test_finalize_node_invokes_sediment_on_terminal_episode():
    """finalize_node early-exit path must still call sediment."""
    store = FakeSceneMemoryStore()
    session = _session_with_rooms()
    planner = CallVlmPlanner()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        memory_session=session,
        scene_memory_store=store,
    )
    # Episode already terminal (e.g. set by fallback_node or execute_tool).
    episode = _sediment_episode()
    graph_state = {"state": episode.model_dump(mode="json")}

    finalize_node(graph_state, config={"configurable": {"services": services}})

    assert len(store.room_updates) == 1, "sediment must run on terminal early-exit"
    assert len(store.notes) == 1


def test_finalize_node_invokes_sediment_on_legacy_path():
    """finalize_node legacy (no evaluator) path must call sediment."""
    store = FakeSceneMemoryStore()
    session = _session_with_rooms()
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "mug"})]
    )
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        memory_session=session,
        scene_memory_store=store,
    )
    # Drive the full graph so finalize runs through the legacy path.
    final_state = _run(services, _spec(), _request())

    assert final_state["state"]["terminal"] is True
    assert len(store.room_updates) >= 1, "sediment must run on legacy path"
    assert len(store.notes) >= 1

