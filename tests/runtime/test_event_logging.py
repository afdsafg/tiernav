"""Production graph emits all 11 design-spec event types."""
import json
from pathlib import Path
from unittest.mock import MagicMock

from src.tiernav_runtime.contracts import (
    BenchmarkRule, EpisodeRequest, MemoryScope, RunSpec, TaskMode, PlannerDecision,
)
from src.tiernav_runtime.env import RuntimeEnvironmentService
from src.tiernav_runtime.entrypoint import RuntimeEntrypoint
from src.tiernav_runtime.memory import MemorySession
from src.tiernav_runtime.planner import PlannerClient
from src.tiernav_runtime.config import ProviderConfig


def test_emits_planner_and_tool_events(tmp_path):
    cfg = ProviderConfig(
        api_key_env="T", base_url_env="T", model_env="T",
    )
    planner = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")
    # Two-round script: a navigation action (exercises execute_tool and the
    # tool_called/tool_result/memory_updated events), then submit_answer to
    # finalize. submit_answer alone routes policy->finalize and skips
    # execute_tool, so it cannot cover the tool events.
    decisions = iter([
        PlannerDecision(
            action_type="explore_panorama", arguments={}, reasoning="look around",
        ),
        PlannerDecision(
            action_type="submit_answer", arguments={"answer": "yes"}, reasoning="r",
        ),
    ])
    planner.decide = lambda prompt: next(decisions)

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

    log_path = Path(str(tmp_path)) / request.episode_id / "events.jsonl"
    events = [json.loads(l) for l in log_path.read_text().splitlines()]
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
