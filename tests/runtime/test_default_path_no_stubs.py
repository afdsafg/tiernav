"""Default-path no-stub audit: dispatch every default tool with minimal calls."""

import os
import re

import pytest

from src.agent_evidence import TrajectoryEvidence
from src.tiernav_runtime.contracts import ToolCall, ToolResult
from src.tiernav_runtime.tools import ToolRegistry, build_real_tool_registry


def test_default_registry_excludes_stubs():
    registry = ToolRegistry.with_stable_defaults()
    names = set(registry.names())
    assert "fork_subagent" not in names
    assert "pixel_navigate" not in names


@pytest.mark.parametrize("action_type", list(ToolRegistry.with_stable_defaults().names()))
def test_default_tools_dispatch_without_not_implemented(action_type):
    registry = ToolRegistry.with_stable_defaults()
    arguments = {"answer": "test"} if action_type == "submit_answer" else {}
    call = ToolCall(call_id="audit-1", action_type=action_type, arguments=arguments)
    result = registry.dispatch(call)
    assert isinstance(result, ToolResult)
    assert result.call_id == "audit-1"
    assert result.action_type == action_type
    # Name blacklist (test_default_registry_excludes_stubs) and behavior check
    # here are complementary: the name filter blocks known stubs by name, while
    # these assertions ensure dispatch actually succeeds rather than returning a
    # ToolResult that merely wraps a swallowed NotImplementedError.
    assert result.ok is True, f"{action_type} returned error: {result.error!r}"
    assert "NotImplementedError" not in result.error


class _FakeExecutor:
    def __init__(self) -> None:
        self._path_length = 0.5

    @property
    def path_length(self) -> float:
        return self._path_length

    def _ev(self) -> TrajectoryEvidence:
        return TrajectoryEvidence(
            subgoal="s", task_mode="m", progress="p", outcome="ok"
        )

    def explore_panorama(self, config=None):
        return self._ev()

    def navigate_to_object(self, object_name, view_idx=None):
        return self._ev()

    def explore_seed(self, seed_id):
        return self._ev()

    def explore_frontier(self, frontier_id):
        return self._ev()


def test_real_registry_excludes_stubs():
    registry = build_real_tool_registry(_FakeExecutor())
    names = set(registry.names())
    assert "fork_subagent" not in names
    assert "pixel_navigate" not in names


@pytest.mark.parametrize("action_type", list(build_real_tool_registry(_FakeExecutor()).names()))
def test_real_registry_dispatches_without_not_implemented(action_type):
    registry = build_real_tool_registry(_FakeExecutor())
    arguments = {"answer": "test"} if action_type == "submit_answer" else {}
    if action_type == "navigate_to_object":
        arguments = {"object_name": "chair"}
    elif action_type == "explore_seed":
        arguments = {"seed_id": "s1"}
    elif action_type == "explore_frontier":
        arguments = {"frontier_id": "f1"}
    call = ToolCall(call_id="audit-real-1", action_type=action_type, arguments=arguments)
    result = registry.dispatch(call)
    assert isinstance(result, ToolResult)
    assert result.ok is True, f"{action_type} returned error: {result.error!r}"
    assert "NotImplementedError" not in result.error


# ── Task 9: runner cutover audit ─────────────────────────────────────────
# The runners import habitat/torch and cannot be imported in lightweight
# tests.  Instead we audit the source text to confirm legacy/langgraph
# paths have been removed.


_CMA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_runner_text(filename: str) -> str:
    path = os.path.join(_CMA_DIR, filename)
    assert os.path.isfile(path), f"runner {filename} not found at {path}"
    with open(path, "r") as f:
        return f.read()


RUNNER_FILES = [
    "run_two_tier_aeqa_evaluation.py",
    "run_goatbench_evaluation.py",
]


@pytest.mark.parametrize("runner_file", RUNNER_FILES)
def test_runner_has_no_legacy_or_langgraph_imports(runner_file):
    """Runners must not import legacy agent_workflow/langgraph modules (Task 9)."""
    text = _read_runner_text(runner_file)
    assert "from src.agent_workflow import" not in text, \
        f"{runner_file}: remove import of src.agent_workflow"
    assert "from src.two_tier_graph.entrypoint import" not in text, \
        f"{runner_file}: remove import of src.two_tier_graph.entrypoint"
    assert "from src.goatbench_graph.entrypoint import" not in text, \
        f"{runner_file}: remove import of src.goatbench_graph.entrypoint"


@pytest.mark.parametrize("runner_file", RUNNER_FILES)
def test_runner_has_no_legacy_or_langgraph_in_engines(runner_file):
    """_ENGINES dict (if present) must not contain 'legacy' or 'langgraph' keys."""
    text = _read_runner_text(runner_file)
    # If _ENGINES still exists, it should only have "runtime"
    engines_match = re.search(r'_ENGINES\s*=\s*\{([^}]+)\}', text)
    if engines_match:
        engines_body = engines_match.group(1)
        assert '"legacy"' not in engines_body, \
            f"{runner_file}: _ENGINES must not contain 'legacy'"
        assert '"langgraph"' not in engines_body, \
            f"{runner_file}: _ENGINES must not contain 'langgraph'"


@pytest.mark.parametrize("runner_file", RUNNER_FILES)
def test_runner_has_no_engine_cli_arg(runner_file):
    """--engine CLI argument must have been removed (Task 9)."""
    text = _read_runner_text(runner_file)
    assert 'add_argument("--engine"' not in text and \
           "add_argument('--engine'" not in text, \
        f"{runner_file}: remove --engine CLI argument"


@pytest.mark.parametrize("runner_file", RUNNER_FILES)
def test_runner_uses_provider_config_for_planner(runner_file):
    """Runtime functions must build ProviderConfig from env-var names (Task 9)."""
    text = _read_runner_text(runner_file)
    assert "ProviderConfig(" in text, \
        f"{runner_file}: must use ProviderConfig for planner injection"
    assert "QWEN_PLANNER_API_KEY" in text, \
        f"{runner_file}: ProviderConfig must reference QWEN_PLANNER_API_KEY env var"
    assert "QWEN_PLANNER_BASE_URL" in text, \
        f"{runner_file}: ProviderConfig must reference QWEN_PLANNER_BASE_URL env var"
    assert "QWEN_PLANNER_MODEL" in text, \
        f"{runner_file}: ProviderConfig must reference QWEN_PLANNER_MODEL env var"
