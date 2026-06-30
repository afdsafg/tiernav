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
    planner.decide = lambda prompt, **kw: PlannerDecision(
        action_type="submit_answer", arguments={"answer": "yes"}, reasoning="done",
    )

    result = entrypoint.run(spec, request)
    assert result.path_length == 12.5
    assert result.path_length != result.steps_taken
