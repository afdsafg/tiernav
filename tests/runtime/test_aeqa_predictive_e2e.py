from src.tiernav_runtime.contracts import (
    BenchmarkRule,
    EpisodeRequest,
    MemoryScope,
    PlannerDecision,
    RunSpec,
)
from src.tiernav_runtime.entrypoint import RuntimeEntrypoint
from src.tiernav_runtime.policy import WorkflowPolicy
from src.tiernav_runtime.tools import SubmitAnswerTool, ToolRegistry


class NoopPlanner:
    def call_vlm(self, messages, **kwargs):
        return "Answer: a towel (Evidence: Snapshot 0)"

    def decide(self, prompt, **kwargs):
        raise AssertionError("AEQA predictive runtime should not call text decide")


class FakeController:
    def decide(self, *, episode, context_text, env, planner, prompt_audit=None):
        raw = planner.call_vlm([{"role": "user", "content": context_text}])
        assert "towel" in raw
        return PlannerDecision(
            action_type="submit_answer",
            reasoning="fake controller answer",
            arguments={"answer": "a towel"},
            confidence=0.9,
        )


def test_entrypoint_runs_aeqa_controller_to_answer(tmp_path):
    tools = ToolRegistry()
    tools.register(SubmitAnswerTool(task_mode="question_answering"))
    rule = BenchmarkRule(
        success_distance_m=0.0,
        memory_scope=MemoryScope.PER_QUESTION,
        scoring_mode="aeqa",
    )
    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=NoopPlanner(),
        environment=None,
        rule=rule,
        executor=None,
        policy=WorkflowPolicy(),
        task_mode="question_answering",
        tools=tools,
        aeqa_controller=FakeController(),
    )
    spec = RunSpec(
        run_id="run-aeqa-fake",
        task_name="aeqa",
        dataset_split="unit",
        output_dir=str(tmp_path),
        planner_provider="fake",
        planner_model="fake",
        max_rounds=3,
        max_steps=5,
    )
    request = EpisodeRequest(
        episode_id="ep-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is hanging on the oven handle?",
        output_dir=str(tmp_path),
    )

    result = entrypoint.run(spec, request)

    assert result.success is True
    assert result.answer == "a towel"
    assert result.rounds_used == 1
