# AEQA Predictive Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AEQA-first predictive runtime demo that sends real images to the VLM, uses a Pred-EQA style controller, and preserves the current GOATBench text-planner path.

**Architecture:** Add small multimodal message contracts and transport compatibility first. Then add a runtime-native AEQA predictive controller that builds image evidence, calls answer/explore prompts, and returns ordinary `PlannerDecision` objects to the LangGraph graph. Wire the AEQA runner to this controller and an AEQA-only tool registry while leaving GOATBench on the existing planner/tool path.

**Tech Stack:** Python, Pydantic v2, LangGraph, pytest, OpenAI-compatible chat content lists, existing TSDF/executor/frontier tools.

---

## Scope And Worktree

Conclusion: current scope. This plan implements only the AEQA predictive runtime demo from `docs/superpowers/specs/2026-07-01-aeqa-predictive-runtime-design.md`.

Execution worktree: use `/home/afdsafg/下载/new/tiernav`. Before implementation execution, create or confirm an isolated worktree via `superpowers:using-git-worktrees` if the executor chooses subagent-driven or inline implementation. This plan file itself is written in the current worktree.

Do not add or commit the currently untracked `Pred-EQA/`, `RoboClaw/`, or `ep-chair/` directories unless the user explicitly asks.

## File Structure

Create:

- `src/tiernav_runtime/aeqa_predictive.py`: AEQA visual-state models, multimodal content helpers, prompt builders, parsers, and `AEQAPredictiveController`.
- `tests/runtime/test_aeqa_predictive.py`: unit tests for visual state building, content formatting, parsers, and controller decisions.
- `tests/runtime/test_aeqa_runner_wiring.py`: tests for AEQA runner pose conversion and runtime wiring helpers.

Modify:

- `src/tiernav_runtime/contracts.py`: add public multimodal message/content contracts.
- `src/tiernav_runtime/planner.py`: allow `PlannerClient.decide()` to accept text prompts or full message lists without breaking text-only planner behavior.
- `src/tiernav_runtime/recorder.py`: add multimodal prompt audit sanitation that records image metadata instead of full base64.
- `src/tiernav_runtime/graph.py`: add optional `aeqa_controller` service and route AEQA planning through it when configured.
- `src/tiernav_runtime/tools.py`: add `build_aeqa_tool_registry()` and include current image base64 in observations.
- `src/tiernav_runtime/entrypoint.py`: allow caller-provided tool registry and AEQA controller in real services.
- `run_two_tier_aeqa_evaluation.py`: wire the AEQA controller, AEQA tool registry, and correct `x/y/z/theta` initial pose.
- Runtime tests that assert schema names, graph behavior, tool registry names, and planner transport behavior.

Reference but do not import directly:

- `Pred-EQA/src/pred_eqa.py`
- `Pred-EQA/src/query_vlm.py`
- `Pred-EQA/src/scene_vlm_only.py`
- `Pred-EQA/src/plan_extraction_utils.py`
- `/home/afdsafg/下载/new/3D-Mem/src/agent_workflow.py`

## Task 1: Multimodal Runtime Contracts

**Files:**
- Modify: `src/tiernav_runtime/contracts.py`
- Modify: `tests/runtime/test_contracts.py`

- [ ] **Step 1: Write failing contract tests**

Append these tests near the existing `ContextSection` tests in `tests/runtime/test_contracts.py`:

```python
from src.tiernav_runtime.contracts import (
    ImageURL,
    ImageURLContentBlock,
    PlannerMessage,
    TextContentBlock,
)


def test_multimodal_message_round_trips_text_and_image_blocks():
    msg = PlannerMessage(
        role="user",
        content=[
            TextContentBlock(text="Question: what is hanging on the oven handle?"),
            ImageURLContentBlock(
                image_url=ImageURL(
                    url="data:image/png;base64,abc123",
                    detail="high",
                )
            ),
        ],
    )

    payload = msg.model_dump(mode="json")

    assert payload["role"] == "user"
    assert payload["content"][0] == {
        "type": "text",
        "text": "Question: what is hanging on the oven handle?",
    }
    assert payload["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,abc123",
            "detail": "high",
        },
    }
    assert PlannerMessage.model_validate(payload).content[1].image_url.url.endswith("abc123")


def test_planner_message_accepts_legacy_string_content():
    msg = PlannerMessage(role="user", content="plain text prompt")
    assert msg.model_dump(mode="json") == {
        "role": "user",
        "content": "plain text prompt",
    }


def test_image_url_rejects_non_string_url():
    with pytest.raises(ValidationError) as exc:
        ImageURL(url=123)
    assert "url" in str(exc.value)
```

Update `test_json_schema_dump_contains_all_public_models()` expected set to include:

```python
        "TextContentBlock",
        "ImageURL",
        "ImageURLContentBlock",
        "PlannerMessage",
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/runtime/test_contracts.py::test_multimodal_message_round_trips_text_and_image_blocks tests/runtime/test_contracts.py::test_planner_message_accepts_legacy_string_content tests/runtime/test_contracts.py::test_image_url_rejects_non_string_url -q
```

Expected: FAIL with an import error for `ImageURL`, `ImageURLContentBlock`, `PlannerMessage`, and `TextContentBlock`.

- [ ] **Step 3: Add multimodal contracts**

In `src/tiernav_runtime/contracts.py`, update the imports:

```python
from typing import Annotated, Any, Literal, Optional, Union
```

Add these models after `ContextSection`:

```python
class TextContentBlock(RuntimeModel):
    """OpenAI-compatible text block for multimodal planner messages."""

    type: Literal["text"] = "text"
    text: str


class ImageURL(RuntimeModel):
    """Image URL payload for OpenAI-compatible image_url blocks."""

    url: str
    detail: Optional[str] = None


class ImageURLContentBlock(RuntimeModel):
    """OpenAI-compatible image block for multimodal planner messages."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentBlock = Union[TextContentBlock, ImageURLContentBlock]


class PlannerMessage(RuntimeModel):
    """A chat message accepted by the planner transport."""

    role: str
    content: Union[str, list[ContentBlock]]
```

Add the new public model literals:

```python
    "TextContentBlock",
    "ImageURL",
    "ImageURLContentBlock",
    "PlannerMessage",
```

Add the new public model registry entries:

```python
    "TextContentBlock": TextContentBlock,
    "ImageURL": ImageURL,
    "ImageURLContentBlock": ImageURLContentBlock,
    "PlannerMessage": PlannerMessage,
```

- [ ] **Step 4: Run contract tests**

Run:

```bash
pytest tests/runtime/test_contracts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/contracts.py tests/runtime/test_contracts.py
git commit -m "feat: add multimodal planner contracts"
```

## Task 2: Planner Transport Compatibility

**Files:**
- Modify: `src/tiernav_runtime/planner.py`
- Modify: `tests/runtime/test_planner_client.py`
- Modify: `tests/runtime/test_planner_decide.py`

- [ ] **Step 1: Write failing planner transport tests**

Append this test to `tests/runtime/test_planner_decide.py`:

```python
    def test_decide_accepts_multimodal_messages(self):
        cfg = ProviderConfig(
            api_key_env="TEST_KEY",
            base_url_env="TEST_BASE_URL",
            model_env="TEST_MODEL",
        )
        client = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="test-model")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Choose an action."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]

        with patch(
            "src.tiernav_runtime.planner._call_vlm",
            return_value='{"action_type": "submit_answer", "answer": "towel"}',
        ) as mock_vlm:
            decision = client.decide(messages)

        sent_messages = mock_vlm.call_args[0][0]
        assert sent_messages == messages
        assert sent_messages[0]["content"][1]["image_url"]["url"].endswith("abc")
        assert decision.action_type == "submit_answer"
        assert decision.arguments["answer"] == "towel"
```

Append this test to `tests/runtime/test_planner_client.py`:

```python
def test_call_vlm_preserves_multimodal_content_list(monkeypatch):
    captured: dict = {}

    def fake_call_vlm(messages, **kwargs):
        captured["messages"] = messages
        return "ok"

    cfg = ProviderConfig(
        api_key_env="PLAN_API_KEY",
        base_url_env="PLAN_BASE_URL",
        model_env="PLAN_MODEL",
    )
    monkeypatch.setenv("PLAN_API_KEY", "sk-transport")
    monkeypatch.setenv("PLAN_BASE_URL", "https://transport.example.com/v1")
    monkeypatch.setenv("PLAN_MODEL", "transport-model")

    import src.tiernav_runtime.planner as planner_mod
    monkeypatch.setattr(planner_mod, "_call_vlm", fake_call_vlm)

    client = PlannerClient(provider=cfg)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]

    response = client.call_vlm(messages)

    assert response == "ok"
    assert captured["messages"] is messages
```

- [ ] **Step 2: Run the failing planner tests**

Run:

```bash
pytest tests/runtime/test_planner_client.py::test_call_vlm_preserves_multimodal_content_list tests/runtime/test_planner_decide.py::TestPlannerClientDecide::test_decide_accepts_multimodal_messages -q
```

Expected: `test_call_vlm_preserves_multimodal_content_list` may already pass. `test_decide_accepts_multimodal_messages` fails because `decide()` treats message lists as string prompts.

- [ ] **Step 3: Implement message normalization**

In `src/tiernav_runtime/planner.py`, import the new contract:

```python
from .contracts import PlannerDecision, PlannerMessage
```

Add this helper near `_call_vlm()`:

```python
def _coerce_messages(prompt_or_messages: Any) -> list[dict]:
    """Return OpenAI-compatible messages from text, dicts, or PlannerMessage models."""
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]

    if isinstance(prompt_or_messages, PlannerMessage):
        return [prompt_or_messages.model_dump(mode="json", exclude_none=True)]

    if isinstance(prompt_or_messages, list):
        messages: list[dict] = []
        for item in prompt_or_messages:
            if isinstance(item, PlannerMessage):
                messages.append(item.model_dump(mode="json", exclude_none=True))
            elif isinstance(item, dict):
                messages.append(dict(item))
            else:
                raise TypeError(
                    "planner messages must be dict or PlannerMessage, got "
                    + type(item).__name__
                )
        return messages

    raise TypeError(
        "prompt must be str, PlannerMessage, or list of messages, got "
        + type(prompt_or_messages).__name__
    )
```

Change the `decide()` signature and message construction:

```python
    def decide(self, prompt: Any, *, retries: int = 0) -> PlannerDecision:
```

Replace:

```python
        messages = [{"role": "user", "content": prompt}]
```

with:

```python
        messages = _coerce_messages(prompt)
```

- [ ] **Step 4: Run planner tests**

Run:

```bash
pytest tests/runtime/test_planner_client.py tests/runtime/test_planner_decide.py tests/runtime/test_planner_retry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/planner.py tests/runtime/test_planner_client.py tests/runtime/test_planner_decide.py
git commit -m "feat: preserve multimodal planner messages"
```

## Task 3: Prompt Audit Sanitization

**Files:**
- Modify: `src/tiernav_runtime/recorder.py`
- Create: `tests/runtime/test_prompt_audit_multimodal.py`

- [ ] **Step 1: Write failing audit tests**

Create `tests/runtime/test_prompt_audit_multimodal.py`:

```python
import json

from src.tiernav_runtime.recorder import PromptAuditRecorder, sanitize_multimodal_messages


def test_sanitize_multimodal_messages_replaces_base64_with_metadata():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + ("a" * 80)},
                },
            ],
        }
    ]

    sanitized = sanitize_multimodal_messages(messages)

    image_url = sanitized[0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,<omitted chars=80 sha256=")
    assert "aaaaaaaaaa" not in image_url


def test_prompt_audit_records_multimodal_without_raw_base64(tmp_path):
    recorder = PromptAuditRecorder(tmp_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + ("b" * 64)},
                },
            ],
        }
    ]

    recorder.record_multimodal(
        episode_id="ep-1",
        round_index=2,
        step_index=3,
        label="answerer",
        messages=messages,
    )

    path = tmp_path / "prompt_audit" / "ep-1.multimodal.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    url = payload["messages"][0]["content"][1]["image_url"]["url"]

    assert payload["label"] == "answerer"
    assert payload["round"] == 2
    assert payload["step"] == 3
    assert "<omitted chars=64 sha256=" in url
    assert "bbbbbbbbbb" not in json.dumps(payload)
```

- [ ] **Step 2: Run the failing audit tests**

Run:

```bash
pytest tests/runtime/test_prompt_audit_multimodal.py -q
```

Expected: FAIL with import error for `sanitize_multimodal_messages` or missing `record_multimodal`.

- [ ] **Step 3: Add sanitizer and multimodal audit method**

In `src/tiernav_runtime/recorder.py`, add `hashlib` import:

```python
import hashlib
```

Add these helpers above `PromptAuditRecorder`:

```python
def _sanitize_image_url(url: str) -> str:
    prefix = "base64,"
    if not isinstance(url, str) or prefix not in url:
        return url
    before, b64 = url.split(prefix, 1)
    digest = hashlib.sha256(b64.encode("utf-8")).hexdigest()
    return before + prefix + f"<omitted chars={len(b64)} sha256={digest}>"


def sanitize_multimodal_messages(messages: list[dict]) -> list[dict]:
    """Copy messages and replace inline base64 image payloads with metadata."""
    sanitized: list[dict] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            clean_content = []
            for block in content:
                if not isinstance(block, dict):
                    clean_content.append(block)
                    continue
                clean_block = dict(block)
                if clean_block.get("type") == "image_url":
                    image_url = dict(clean_block.get("image_url") or {})
                    image_url["url"] = _sanitize_image_url(str(image_url.get("url", "")))
                    clean_block["image_url"] = image_url
                clean_content.append(clean_block)
            copied["content"] = clean_content
        sanitized.append(copied)
    return sanitized
```

Add this method to `PromptAuditRecorder`:

```python
    def record_multimodal(
        self,
        episode_id: str,
        round_index: int,
        step_index: int,
        label: str,
        messages: list[dict],
    ) -> None:
        path = self.dir / f"{episode_id}.multimodal.jsonl"
        entry = {
            "round": round_index,
            "step": step_index,
            "label": label,
            "messages": sanitize_multimodal_messages(messages),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run audit tests**

Run:

```bash
pytest tests/runtime/test_prompt_audit.py tests/runtime/test_prompt_audit_multimodal.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/recorder.py tests/runtime/test_prompt_audit_multimodal.py
git commit -m "feat: sanitize multimodal prompt audits"
```

## Task 4: AEQA Visual State And Parsers

**Files:**
- Create: `src/tiernav_runtime/aeqa_predictive.py`
- Create: `tests/runtime/test_aeqa_predictive.py`

- [ ] **Step 1: Write failing visual-state and parser tests**

Create `tests/runtime/test_aeqa_predictive.py` with these initial tests:

```python
from src.tiernav_runtime.aeqa_predictive import (
    AEQAFrontier,
    AEQAImage,
    AEQAVisualState,
    build_content,
    parse_answer_response,
    parse_frontier_response,
    parse_retain_indices,
)


def test_build_content_interleaves_text_and_images():
    content = build_content([
        ("Question: what is on the handle?", None),
        ("Snapshot 0:", "abc123"),
    ])

    assert content == [
        {"type": "text", "text": "Question: what is on the handle?"},
        {"type": "text", "text": "Snapshot 0:"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123", "detail": "high"},
        },
    ]


def test_parse_answer_response_extracts_answer_and_evidence():
    parsed = parse_answer_response(
        "I can answer.\nAnswer: a white towel (Evidence: Snapshot 2)"
    )

    assert parsed["action"] == "answer"
    assert parsed["answer"] == "a white towel"
    assert parsed["evidence_snapshot"] == 2


def test_parse_answer_response_continue_exploration():
    parsed = parse_answer_response("Continue Exploration")
    assert parsed == {
        "action": "continue_exploration",
        "answer": "",
        "evidence_snapshot": None,
    }


def test_parse_frontier_response_accepts_next_step_format():
    assert parse_frontier_response("Reasoning...\nNext Step: Frontier 7", ["3", "7"]) == "7"


def test_parse_frontier_response_falls_back_to_first_valid_frontier():
    assert parse_frontier_response("Next Step: Frontier 99", ["3", "7"]) == "3"


def test_parse_retain_indices_accepts_braces_and_filters_bounds():
    assert parse_retain_indices("Retain Snapshots: {0, 2, 9}.", max_count=3) == [0, 2]


def test_visual_state_is_json_serializable():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        current_step=4,
        snapshots=[AEQAImage(image_id="snap-1", image_b64="abc", label="Snapshot 0")],
        frontiers=[AEQAFrontier(frontier_id="5", image_b64="def")],
        egocentric_views=[],
        memory_text="plan line",
        tool_feedback="last action failed",
    )

    payload = state.model_dump(mode="json")
    assert payload["snapshots"][0]["image_id"] == "snap-1"
    assert payload["frontiers"][0]["frontier_id"] == "5"
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/runtime/test_aeqa_predictive.py::test_build_content_interleaves_text_and_images tests/runtime/test_aeqa_predictive.py::test_parse_answer_response_extracts_answer_and_evidence -q
```

Expected: FAIL with import error for `src.tiernav_runtime.aeqa_predictive`.

- [ ] **Step 3: Add visual-state models and parsers**

Create `src/tiernav_runtime/aeqa_predictive.py` with this code:

```python
"""AEQA predictive controller helpers.

This module is runtime-native and keeps Habitat/GPU objects behind duck-typed
environment adapters so unit tests can use fakes.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import Field

from .contracts import PlannerDecision, RuntimeModel


class AEQAImage(RuntimeModel):
    image_id: str
    image_b64: str
    label: str = ""
    source: str = "snapshot"


class AEQAFrontier(RuntimeModel):
    frontier_id: str
    image_b64: str
    label: str = ""


class AEQAVisualState(RuntimeModel):
    question: str
    current_step: int = 0
    snapshots: list[AEQAImage] = Field(default_factory=list)
    frontiers: list[AEQAFrontier] = Field(default_factory=list)
    egocentric_views: list[AEQAImage] = Field(default_factory=list)
    memory_text: str = ""
    tool_feedback: str = ""


class AEQAPredictiveMemory(RuntimeModel):
    prediction_items: list[dict[str, str]] = Field(default_factory=list)
    retained_snapshot_ids: list[str] = Field(default_factory=list)
    pruned_snapshot_ids: list[str] = Field(default_factory=list)
    step_summaries: list[str] = Field(default_factory=list)
    last_answerer_decision: str = ""
    last_explorer_decision: str = ""


def build_content(pairs: list[tuple[str, Optional[str]]]) -> list[dict]:
    """Build OpenAI-compatible content blocks from text and optional base64 images."""
    content: list[dict] = []
    for text, image_b64 in pairs:
        content.append({"type": "text", "text": text})
        if image_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    },
                }
            )
    return content


def parse_retain_indices(response: str, max_count: int, prefix: str = "Retain Snapshots") -> list[int]:
    if not response or max_count <= 0:
        return []
    lines = [line.strip() for line in response.splitlines() if prefix in line]
    target = lines[-1] if lines else response
    numbers = [int(n) for n in re.findall(r"\d+", target)]
    return [n for n in numbers if 0 <= n < max_count]


def parse_answer_response(response: str) -> dict[str, Any]:
    text = (response or "").strip()
    if not text:
        return {"action": "continue_exploration", "answer": "", "evidence_snapshot": None}
    if "continue exploration" in text.lower():
        return {"action": "continue_exploration", "answer": "", "evidence_snapshot": None}

    matches = re.findall(
        r"Answer:\s*(.+?)(?:\s*\(Evidence:\s*Snapshot\s*(\d+)\s*\))?\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        return {"action": "continue_exploration", "answer": "", "evidence_snapshot": None}

    answer, snap_idx = matches[-1]
    answer = answer.strip().strip(".").strip().strip('"').strip("'")
    return {
        "action": "answer" if answer else "continue_exploration",
        "answer": answer,
        "evidence_snapshot": int(snap_idx) if snap_idx else None,
    }


def parse_frontier_response(response: str, valid_frontier_ids: list[str]) -> Optional[str]:
    if not valid_frontier_ids:
        return None
    text = response or ""
    match = re.search(r"Next\s+Step\s*:\s*Frontier\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
    if match and match.group(1) in valid_frontier_ids:
        return match.group(1)
    match = re.search(r"\bFrontier\s+([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    if match and match.group(1) in valid_frontier_ids:
        return match.group(1)
    return valid_frontier_ids[0]
```

- [ ] **Step 4: Run visual-state and parser tests**

Run:

```bash
pytest tests/runtime/test_aeqa_predictive.py -q
```

Expected: PASS for the tests added in this task.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/aeqa_predictive.py tests/runtime/test_aeqa_predictive.py
git commit -m "feat: add AEQA predictive helpers"
```

## Task 5: AEQA Evidence Builder And Prompt Builders

**Files:**
- Modify: `src/tiernav_runtime/aeqa_predictive.py`
- Modify: `tests/runtime/test_aeqa_predictive.py`

- [ ] **Step 1: Add failing evidence and prompt tests**

Append these tests to `tests/runtime/test_aeqa_predictive.py`:

```python
from src.tiernav_runtime.aeqa_predictive import (
    AEQAVisualStateBuilder,
    build_answer_messages,
    build_explore_messages,
)


class FakeVisualEnv:
    def get_aeqa_visual_state(self, episode):
        return {
            "question": episode.prompt,
            "current_step": episode.step_index,
            "snapshots": [
                {"image_id": "snap-0", "image_b64": "aaa", "label": "Snapshot 0"},
            ],
            "frontiers": [
                {"frontier_id": "4", "image_b64": "bbb", "label": "Frontier 4"},
            ],
            "egocentric_views": [],
            "memory_text": "High-Level Plan:\n * find the oven",
            "tool_feedback": "last action: none",
        }


class FakeEpisode:
    prompt = "What is hanging on the oven handle?"
    step_index = 5
    last_observation = None


def test_visual_state_builder_uses_environment_adapter():
    builder = AEQAVisualStateBuilder()
    state = builder.build(FakeEpisode(), FakeVisualEnv())

    assert state.question == "What is hanging on the oven handle?"
    assert state.snapshots[0].image_b64 == "aaa"
    assert state.frontiers[0].frontier_id == "4"
    assert "find the oven" in state.memory_text


def test_build_answer_messages_include_snapshot_image():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[AEQAImage(image_id="snap-0", image_b64="aaa", label="Snapshot 0")],
        frontiers=[],
        egocentric_views=[],
        memory_text="High-Level Plan:\n * inspect oven",
    )

    messages = build_answer_messages(state)

    assert messages[0]["role"] == "system"
    assert "sufficient to answer" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert any(block.get("type") == "image_url" for block in messages[1]["content"])
    assert "Snapshot 0" in messages[1]["content"][2]["text"]


def test_build_explore_messages_include_frontier_image():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[],
        frontiers=[AEQAFrontier(frontier_id="4", image_b64="bbb", label="Frontier 4")],
        egocentric_views=[],
    )

    messages = build_explore_messages(state)

    assert messages[0]["role"] == "system"
    assert "PHYSICALLY NAVIGATE" in messages[0]["content"]
    assert any(block.get("type") == "image_url" for block in messages[1]["content"])
    assert any("Frontier 4" in block.get("text", "") for block in messages[1]["content"])
```

- [ ] **Step 2: Run the failing evidence tests**

Run:

```bash
pytest tests/runtime/test_aeqa_predictive.py::test_visual_state_builder_uses_environment_adapter tests/runtime/test_aeqa_predictive.py::test_build_answer_messages_include_snapshot_image tests/runtime/test_aeqa_predictive.py::test_build_explore_messages_include_frontier_image -q
```

Expected: FAIL with import error for `AEQAVisualStateBuilder`, `build_answer_messages`, or `build_explore_messages`.

- [ ] **Step 3: Add builder and prompts**

Append this implementation to `src/tiernav_runtime/aeqa_predictive.py`:

```python
class AEQAVisualStateBuilder:
    """Build AEQA visual state from a duck-typed runtime environment."""

    def build(self, episode: Any, env: Any) -> AEQAVisualState:
        if env is not None and hasattr(env, "get_aeqa_visual_state"):
            return AEQAVisualState.model_validate(env.get_aeqa_visual_state(episode))

        question = str(getattr(episode, "prompt", "") or "")
        step = int(getattr(episode, "step_index", 0) or 0)
        return AEQAVisualState(
            question=question,
            current_step=step,
            snapshots=[],
            frontiers=[],
            egocentric_views=[],
            memory_text="",
            tool_feedback="",
        )


ANSWER_SYSTEM_PROMPT = """Task: You are an indoor agent that needs to determine if the current collected visual information is sufficient to answer the question.

Instructions:
1. Carefully analyze the question's required object, attribute, relationship, or state.
2. Carefully inspect all available snapshots.
3. If any snapshot contains enough visual evidence, output Answer.
4. If the evidence is insufficient, output Continue Exploration.
"""


EXPLORE_SYSTEM_PROMPT = """Task: You are an indoor agent that needs to PHYSICALLY NAVIGATE through sequential frontier selections to find information needed for answering the question.

Instructions:
1. Use common room-object relationships to infer where the needed evidence may be.
2. Use previous visual clues and the high-level plan to avoid repeated irrelevant areas.
3. Choose exactly one available frontier when exploration is needed.
4. Keep selecting frontiers until visual evidence is sufficient to answer.
"""


def build_answer_messages(state: AEQAVisualState) -> list[dict]:
    pairs: list[tuple[str, Optional[str]]] = [
        (f"Question: {state.question}\n", None),
    ]
    if state.memory_text:
        pairs.append((state.memory_text + "\n", None))
    pairs.append(("Available Snapshots:\n", None))
    if not state.snapshots:
        pairs.append(("No snapshots available\n", None))
    for idx, snapshot in enumerate(state.snapshots):
        label = snapshot.label or f"Snapshot {idx}"
        pairs.append((f"{label}: ", snapshot.image_b64))
        pairs.append(("\n", None))
    pairs.append((
        'Output Format:\n'
        'If answerable: "Answer: [concise answer] (Evidence: Snapshot [index])"\n'
        'If not answerable: "Continue Exploration"',
        None,
    ))
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": build_content(pairs)},
    ]


def build_explore_messages(state: AEQAVisualState) -> list[dict]:
    pairs: list[tuple[str, Optional[str]]] = [
        (f"Target Question: {state.question}\n", None),
    ]
    if state.memory_text:
        pairs.append((state.memory_text + "\n", None))
    if state.tool_feedback:
        pairs.append(("Tool Feedback:\n" + state.tool_feedback + "\n", None))
    pairs.append(("Previously Observed Clues:\n", None))
    for idx, snapshot in enumerate(state.snapshots):
        label = snapshot.label or f"Snapshot {idx}"
        pairs.append((f"{label}: ", snapshot.image_b64))
        pairs.append(("\n", None))
    if not state.snapshots:
        pairs.append(("No snapshots available\n", None))
    pairs.append(("\nAvailable Exploration Directions:\n", None))
    if not state.frontiers:
        pairs.append(("No frontiers available\n", None))
    for frontier in state.frontiers:
        label = frontier.label or f"Frontier {frontier.frontier_id}"
        pairs.append((f"{label}: ", frontier.image_b64))
        pairs.append(("\n", None))
    valid = ", ".join(f.frontier_id for f in state.frontiers) or "none"
    pairs.append((
        "Output Format:\n"
        "First explain briefly. Then provide exactly: \"Next Step: Frontier i\".\n"
        f"Available Frontier ids: {valid}",
        None,
    ))
    return [
        {"role": "system", "content": EXPLORE_SYSTEM_PROMPT},
        {"role": "user", "content": build_content(pairs)},
    ]
```

- [ ] **Step 4: Run evidence and prompt tests**

Run:

```bash
pytest tests/runtime/test_aeqa_predictive.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/aeqa_predictive.py tests/runtime/test_aeqa_predictive.py
git commit -m "feat: build AEQA multimodal evidence prompts"
```

## Task 6: AEQA Predictive Controller

**Files:**
- Modify: `src/tiernav_runtime/aeqa_predictive.py`
- Modify: `tests/runtime/test_aeqa_predictive.py`

- [ ] **Step 1: Add failing controller tests**

Append these tests to `tests/runtime/test_aeqa_predictive.py`:

```python
from src.tiernav_runtime.aeqa_predictive import AEQAPredictiveController
from src.tiernav_runtime.contracts import EpisodeState


class ScriptedVLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def call_vlm(self, messages, **kwargs):
        self.messages.append(messages)
        return self.responses.pop(0)


class RecordingAudit:
    def __init__(self):
        self.calls = []

    def record_multimodal(self, **kwargs):
        self.calls.append(kwargs)


def _episode():
    return EpisodeState(
        episode_id="ep-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is hanging on the oven handle?",
    )


def test_controller_submits_when_answerer_finds_answer():
    controller = AEQAPredictiveController()
    planner = ScriptedVLM(["Answer: a white towel (Evidence: Snapshot 0)"])
    audit = RecordingAudit()
    env = FakeVisualEnv()

    decision = controller.decide(
        episode=_episode(),
        context_text="compiled text",
        env=env,
        planner=planner,
        prompt_audit=audit,
    )

    assert decision.action_type == "submit_answer"
    assert decision.arguments["answer"] == "a white towel"
    assert decision.arguments["evidence_snapshot"] == 0
    assert len(planner.messages) == 1
    assert audit.calls[0]["label"] == "aeqa_answerer"


def test_controller_explores_frontier_when_answerer_says_continue():
    controller = AEQAPredictiveController()
    planner = ScriptedVLM([
        "Continue Exploration",
        "The kitchen is likely beyond this opening.\nNext Step: Frontier 4",
    ])
    env = FakeVisualEnv()

    decision = controller.decide(
        episode=_episode(),
        context_text="compiled text",
        env=env,
        planner=planner,
    )

    assert decision.action_type == "explore_frontier"
    assert decision.arguments["frontier_id"] == "4"
    assert len(planner.messages) == 2


def test_controller_submits_unanswerable_when_no_frontier_and_no_answer():
    class NoFrontierEnv:
        def get_aeqa_visual_state(self, episode):
            return {
                "question": episode.prompt,
                "current_step": 0,
                "snapshots": [],
                "frontiers": [],
                "egocentric_views": [],
                "memory_text": "",
                "tool_feedback": "",
            }

    controller = AEQAPredictiveController()
    planner = ScriptedVLM(["Continue Exploration"])

    decision = controller.decide(
        episode=_episode(),
        context_text="compiled text",
        env=NoFrontierEnv(),
        planner=planner,
    )

    assert decision.action_type == "submit_answer"
    assert decision.arguments["answer"] == "unanswerable"
    assert decision.confidence == 0.0
```

- [ ] **Step 2: Run failing controller tests**

Run:

```bash
pytest tests/runtime/test_aeqa_predictive.py::test_controller_submits_when_answerer_finds_answer tests/runtime/test_aeqa_predictive.py::test_controller_explores_frontier_when_answerer_says_continue tests/runtime/test_aeqa_predictive.py::test_controller_submits_unanswerable_when_no_frontier_and_no_answer -q
```

Expected: FAIL with import error for `AEQAPredictiveController`.

- [ ] **Step 3: Implement controller**

Append this class to `src/tiernav_runtime/aeqa_predictive.py`:

```python
class AEQAPredictiveController:
    """Pred-EQA style AEQA controller that returns runtime PlannerDecision objects."""

    def __init__(self, builder: Optional[AEQAVisualStateBuilder] = None) -> None:
        self.builder = builder or AEQAVisualStateBuilder()
        self._memory_by_episode: dict[str, AEQAPredictiveMemory] = {}

    def memory_for(self, episode_id: str) -> AEQAPredictiveMemory:
        return self._memory_by_episode.setdefault(episode_id, AEQAPredictiveMemory())

    def decide(
        self,
        *,
        episode: Any,
        context_text: str,
        env: Any,
        planner: Any,
        prompt_audit: Any = None,
    ) -> PlannerDecision:
        state = self.builder.build(episode, env)
        memory = self.memory_for(str(getattr(episode, "episode_id", "")))
        if memory.step_summaries:
            state.memory_text = (state.memory_text + "\n" + "\n".join(memory.step_summaries)).strip()

        answer_messages = build_answer_messages(state)
        self._audit(prompt_audit, episode, "aeqa_answerer", answer_messages)
        answer_raw = planner.call_vlm(answer_messages, max_tokens=1024, temperature=0.3)
        parsed_answer = parse_answer_response(answer_raw)
        memory.last_answerer_decision = answer_raw or ""

        if parsed_answer["action"] == "answer" and parsed_answer["answer"]:
            args: dict[str, Any] = {"answer": parsed_answer["answer"]}
            if parsed_answer["evidence_snapshot"] is not None:
                args["evidence_snapshot"] = parsed_answer["evidence_snapshot"]
            return PlannerDecision(
                action_type="submit_answer",
                reasoning="AEQA answerer found sufficient visual evidence.",
                confidence=0.8,
                arguments=args,
            )

        valid_frontier_ids = [frontier.frontier_id for frontier in state.frontiers]
        if not valid_frontier_ids:
            return PlannerDecision(
                action_type="submit_answer",
                reasoning="AEQA answerer could not answer and no frontier is available.",
                confidence=0.0,
                arguments={"answer": "unanswerable"},
            )

        explore_messages = build_explore_messages(state)
        self._audit(prompt_audit, episode, "aeqa_explorer", explore_messages)
        explore_raw = planner.call_vlm(explore_messages, max_tokens=1024, temperature=0.3)
        selected = parse_frontier_response(explore_raw, valid_frontier_ids)
        memory.last_explorer_decision = explore_raw or ""

        return PlannerDecision(
            action_type="explore_frontier",
            reasoning="AEQA explorer selected a frontier for more visual evidence.",
            confidence=0.6,
            arguments={"frontier_id": selected},
        )

    @staticmethod
    def _audit(prompt_audit: Any, episode: Any, label: str, messages: list[dict]) -> None:
        if prompt_audit is None or not hasattr(prompt_audit, "record_multimodal"):
            return
        prompt_audit.record_multimodal(
            episode_id=str(getattr(episode, "episode_id", "")),
            round_index=int(getattr(episode, "round_index", 0) or 0),
            step_index=int(getattr(episode, "step_index", 0) or 0),
            label=label,
            messages=messages,
        )
```

- [ ] **Step 4: Run controller tests**

Run:

```bash
pytest tests/runtime/test_aeqa_predictive.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/aeqa_predictive.py tests/runtime/test_aeqa_predictive.py
git commit -m "feat: add AEQA predictive controller"
```

## Task 7: Graph Routing For AEQA Controller

**Files:**
- Modify: `src/tiernav_runtime/graph.py`
- Modify: `tests/runtime/test_graph_runtime.py`

- [ ] **Step 1: Add failing graph routing tests**

Append these tests to `tests/runtime/test_graph_runtime.py`:

```python
class FakeAEQAController:
    def __init__(self, decision: PlannerDecision) -> None:
        self.decision = decision
        self.calls = []

    def decide(self, *, episode, context_text, env, planner, prompt_audit=None):
        self.calls.append(
            {
                "episode_id": episode.episode_id,
                "context_text": context_text,
                "env": env,
                "planner": planner,
                "prompt_audit": prompt_audit,
            }
        )
        return self.decision


def test_graph_uses_aeqa_controller_for_question_answering():
    planner = FakePlanner([PlannerDecision(action_type="submit_answer", arguments={"answer": "wrong"})])
    controller = FakeAEQAController(
        PlannerDecision(action_type="submit_answer", arguments={"answer": "controller answer"})
    )
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        aeqa_controller=controller,
    )

    final_state = _run(services, _spec(), _request())

    assert final_state["state"]["answer"] == "controller answer"
    assert len(controller.calls) == 1
    assert planner.prompts == []


def test_graph_keeps_text_planner_for_goal_navigation():
    request = EpisodeRequest(
        episode_id="ep-goat",
        scene_id="scene-1",
        task_name="goatbench",
        task_mode="goal_navigation",
        prompt="Navigate to refrigerator",
    )
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={})]
    )
    controller = FakeAEQAController(
        PlannerDecision(action_type="submit_answer", arguments={"answer": "unused"})
    )
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        aeqa_controller=controller,
    )

    final_state = _run(services, _spec(), request)

    assert len(controller.calls) == 0
    assert len(planner.prompts) == 1
    assert final_state["state"]["task_mode"] == "goal_navigation"
```

- [ ] **Step 2: Run failing graph tests**

Run:

```bash
pytest tests/runtime/test_graph_runtime.py::test_graph_uses_aeqa_controller_for_question_answering tests/runtime/test_graph_runtime.py::test_graph_keeps_text_planner_for_goal_navigation -q
```

Expected: FAIL because `RuntimeServices` has no `aeqa_controller` field.

- [ ] **Step 3: Add service field and route in plan node**

In `src/tiernav_runtime/graph.py`, import `TaskMode`:

```python
    TaskMode,
```

Add this field to `RuntimeServices`:

```python
    # AEQA-only predictive controller. When set, question_answering episodes
    # use it instead of the legacy one-call JSON planner path.
    aeqa_controller: object | None = None
```

In `plan_node`, replace:

```python
    raw = services.planner.decide(prompt, retries=retries)
```

with:

```python
    if (
        episode.task_mode is TaskMode.QUESTION_ANSWERING
        and services.aeqa_controller is not None
    ):
        raw = services.aeqa_controller.decide(
            episode=episode,
            context_text=prompt,
            env=services.environment,
            planner=services.planner,
            prompt_audit=services.prompt_audit,
        )
    else:
        raw = services.planner.decide(prompt, retries=retries)
```

- [ ] **Step 4: Run graph tests**

Run:

```bash
pytest tests/runtime/test_graph_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/graph.py tests/runtime/test_graph_runtime.py
git commit -m "feat: route AEQA planning through predictive controller"
```

## Task 8: AEQA Tool Registry And Observation Image Payloads

**Files:**
- Modify: `src/tiernav_runtime/tools.py`
- Modify: `tests/runtime/test_tools.py`

- [ ] **Step 1: Add failing tool tests**

Append these tests to `tests/runtime/test_tools.py`:

```python
from src.tiernav_runtime.tools import build_aeqa_tool_registry


def test_build_aeqa_tool_registry_exposes_only_frontier_and_submit():
    reg = build_aeqa_tool_registry(FakeExecutor(), task_mode="question_answering")
    assert reg.names() == ["explore_frontier", "submit_answer"]
    schema = reg.action_schema_text()
    assert "explore_frontier" in schema
    assert "submit_answer" in schema
    assert "navigate_to_object" not in schema
    assert "explore_seed" not in schema
    assert "explore_panorama" not in schema


def test_evidence_observation_preserves_current_image_base64():
    evidence = TrajectoryEvidence(
        subgoal="Explore frontier 4",
        task_mode="explore_frontier",
        progress="Arrived at frontier 4",
        outcome="arrived_near_target",
        current_image_b64="abc123",
        key_frames=["frontier-step"],
        room_id=2,
        objects_nearby=["oven"],
    )
    reg = build_aeqa_tool_registry(FakeExecutor(), task_mode="question_answering")
    tool = reg._tools["explore_frontier"]

    class ImageExecutor(FakeExecutor):
        def explore_frontier(self, frontier_id):
            return evidence

    reg = build_aeqa_tool_registry(ImageExecutor(), task_mode="question_answering")
    result = reg.dispatch(
        ToolCall(call_id="c-img", action_type="explore_frontier", arguments={"frontier_id": "4"})
    )

    assert result.ok is True
    assert result.observation.image_ids == ["frontier-step"]
    assert result.observation.raw["current_image_b64"] == "abc123"
```

- [ ] **Step 2: Run failing tool tests**

Run:

```bash
pytest tests/runtime/test_tools.py::test_build_aeqa_tool_registry_exposes_only_frontier_and_submit tests/runtime/test_tools.py::test_evidence_observation_preserves_current_image_base64 -q
```

Expected: FAIL because `build_aeqa_tool_registry` does not exist and `current_image_b64` is not copied into `Observation.raw`.

- [ ] **Step 3: Add AEQA registry and image payload copying**

In `_evidence_to_observation()` in `src/tiernav_runtime/tools.py`, add `current_image_b64` to `raw`:

```python
            "current_image_b64": str(getattr(evidence, "current_image_b64", "") or ""),
```

Add this function below `build_real_tool_registry()`:

```python
def build_aeqa_tool_registry(
    executor: _ExecutorLike,
    *,
    task_mode: str = "question_answering",
) -> ToolRegistry:
    """Return the AEQA demo tool registry: frontier exploration plus answer submit."""
    registry = ToolRegistry()
    registry.register(ExploreFrontierTool(executor))
    registry.register(SubmitAnswerTool(task_mode=task_mode))
    return registry
```

- [ ] **Step 4: Run tool tests**

Run:

```bash
pytest tests/runtime/test_tools.py tests/runtime/test_default_path_no_stubs.py -q
```

Expected: PASS. Existing default and real registries must still include the old GOATBench-compatible tools.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/tools.py tests/runtime/test_tools.py
git commit -m "feat: add AEQA frontier-only tool registry"
```

## Task 9: Entrypoint Injection

**Files:**
- Modify: `src/tiernav_runtime/entrypoint.py`
- Modify: `tests/runtime/test_entrypoint_compat.py`

- [ ] **Step 1: Add failing entrypoint injection test**

Append this test to `tests/runtime/test_entrypoint_compat.py`:

```python
from src.tiernav_runtime.tools import ToolRegistry, SubmitAnswerTool


def test_with_real_services_accepts_custom_tools_and_aeqa_controller():
    class Planner:
        def decide(self, prompt, **kwargs):
            return PlannerDecision(action_type="submit_answer", arguments={"answer": "unused"})

    class Controller:
        pass

    custom_tools = ToolRegistry()
    custom_tools.register(SubmitAnswerTool(task_mode="question_answering"))
    controller = Controller()
    rule = BenchmarkRule(
        success_distance_m=0.0,
        memory_scope=MemoryScope.PER_QUESTION,
        scoring_mode="aeqa",
    )

    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=Planner(),
        environment=None,
        rule=rule,
        executor=None,
        task_mode="question_answering",
        tools=custom_tools,
        aeqa_controller=controller,
    )

    assert entrypoint.services.tools is custom_tools
    assert entrypoint.services.aeqa_controller is controller
```

Ensure these imports exist at the top of the file:

```python
from src.tiernav_runtime.contracts import BenchmarkRule, MemoryScope, PlannerDecision
from src.tiernav_runtime.entrypoint import RuntimeEntrypoint
```

- [ ] **Step 2: Run failing entrypoint test**

Run:

```bash
pytest tests/runtime/test_entrypoint_compat.py::test_with_real_services_accepts_custom_tools_and_aeqa_controller -q
```

Expected: FAIL because `with_real_services()` does not accept `tools` or `aeqa_controller`.

- [ ] **Step 3: Add injection parameters**

In `src/tiernav_runtime/entrypoint.py`, change the `with_real_services()` signature to include:

```python
        tools: ToolRegistry | None = None,
        aeqa_controller: Any = None,
```

Change the `RuntimeServices` construction from:

```python
            tools=build_real_tool_registry(executor, task_mode=task_mode),
```

to:

```python
            tools=tools if tools is not None else build_real_tool_registry(executor, task_mode=task_mode),
```

Add this field in the same constructor call:

```python
            aeqa_controller=aeqa_controller,
```

- [ ] **Step 4: Run entrypoint tests**

Run:

```bash
pytest tests/runtime/test_entrypoint_compat.py tests/runtime/test_path_length.py tests/runtime/test_event_logging.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiernav_runtime/entrypoint.py tests/runtime/test_entrypoint_compat.py
git commit -m "feat: inject AEQA controller and tools"
```

## Task 10: AEQA Runner Wiring And Pose Fix

**Files:**
- Modify: `run_two_tier_aeqa_evaluation.py`
- Create: `tests/runtime/test_aeqa_runner_wiring.py`

- [ ] **Step 1: Add failing runner wiring tests**

Create `tests/runtime/test_aeqa_runner_wiring.py`:

```python
import numpy as np

from run_two_tier_aeqa_evaluation import _initial_pose_from_pts


def test_initial_pose_from_3d_pts_preserves_z_axis():
    pose = _initial_pose_from_pts(np.array([1.0, 2.0, 3.0]), 0.75)
    assert pose == {"x": 1.0, "y": 2.0, "z": 3.0, "theta": 0.75}


def test_initial_pose_from_2d_pts_defaults_z_to_zero():
    pose = _initial_pose_from_pts([4.0, 5.0], 1.25)
    assert pose == {"x": 4.0, "y": 5.0, "z": 0.0, "theta": 1.25}
```

- [ ] **Step 2: Run failing runner wiring tests**

Run:

```bash
pytest tests/runtime/test_aeqa_runner_wiring.py -q
```

Expected: FAIL because `_initial_pose_from_pts` does not exist.

- [ ] **Step 3: Add pose helper**

In `run_two_tier_aeqa_evaluation.py`, add this helper near the runtime workflow function:

```python
def _initial_pose_from_pts(start_pts, start_angle: float) -> dict[str, float]:
    pts_array = getattr(start_pts, "tolist", None)
    if pts_array is not None:
        pts_list = pts_array()
    else:
        pts_list = list(start_pts) if start_pts is not None else [0.0, 0.0, 0.0]
    x = float(pts_list[0]) if len(pts_list) > 0 else 0.0
    y = float(pts_list[1]) if len(pts_list) > 1 else 0.0
    z = float(pts_list[2]) if len(pts_list) > 2 else 0.0
    return {
        "x": x,
        "y": y,
        "z": z,
        "theta": float(start_angle),
    }
```

Replace the inline `pts_array` and `initial_pose` construction with:

```python
    initial_pose = _initial_pose_from_pts(start_pts, start_angle)
    pts_list = [initial_pose["x"], initial_pose["y"], initial_pose["z"]]
```

- [ ] **Step 4: Wire AEQA controller and registry**

In `run_two_tier_aeqa_evaluation.py`, add imports near the real services imports:

```python
    from src.tiernav_runtime.aeqa_predictive import AEQAPredictiveController
    from src.tiernav_runtime.tools import build_aeqa_tool_registry
```

Before `RuntimeEntrypoint.with_real_services(...)`, add:

```python
    aeqa_tools = build_aeqa_tool_registry(
        executor,
        task_mode=request.task_mode.value,
    )
    aeqa_controller = AEQAPredictiveController()
```

Change the entrypoint call to:

```python
    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=planner,
        environment=env,
        rule=rule,
        executor=executor,
        memory_scope_adapter=memory_session,
        task_mode=request.task_mode.value,
        tools=aeqa_tools,
        aeqa_controller=aeqa_controller,
    )
```

- [ ] **Step 5: Run runner wiring tests**

Run:

```bash
pytest tests/runtime/test_aeqa_runner_wiring.py tests/runtime/test_entrypoint_compat.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add run_two_tier_aeqa_evaluation.py tests/runtime/test_aeqa_runner_wiring.py
git commit -m "feat: wire AEQA predictive runtime"
```

## Task 11: Runtime Integration Verification

**Files:**
- Modify: `tests/runtime/test_graph_runtime.py`
- Modify: `tests/runtime/test_planner_client.py`
- Modify: `tests/runtime/test_tools.py`
- Modify: `tests/runtime/test_contracts.py`

- [ ] **Step 1: Run targeted runtime tests**

Run:

```bash
pytest \
  tests/runtime/test_contracts.py \
  tests/runtime/test_planner_client.py \
  tests/runtime/test_planner_decide.py \
  tests/runtime/test_planner_retry.py \
  tests/runtime/test_prompt_audit.py \
  tests/runtime/test_prompt_audit_multimodal.py \
  tests/runtime/test_aeqa_predictive.py \
  tests/runtime/test_graph_runtime.py \
  tests/runtime/test_tools.py \
  tests/runtime/test_entrypoint_compat.py \
  tests/runtime/test_aeqa_runner_wiring.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run all runtime tests**

Run:

```bash
pytest tests/runtime -q
```

Expected: PASS. If a failure is unrelated to AEQA predictive runtime, record the failing test name and error in the final handoff before changing any unrelated code.

- [ ] **Step 3: Run schema snapshot tests if present in the runtime subset**

Run:

```bash
pytest tests/runtime/test_schema_snapshots.py -q
```

Expected: PASS or an intentional schema snapshot update for the new public multimodal models. If the snapshot test requires updating a checked-in schema file, inspect the diff and commit only the schema change produced by the contract additions.

- [ ] **Step 4: Verify GOATBench text path was not converted to AEQA controller**

Run:

```bash
pytest tests/runtime/test_graph_runtime.py::test_graph_keeps_text_planner_for_goal_navigation tests/runtime/test_tools.py::test_with_stable_defaults_names_exact -q
```

Expected: PASS. This confirms GOATBench still uses the existing text planner and stable default tools.

- [ ] **Step 5: Commit any verification-only test adjustments**

Only run this if Step 3 required a schema snapshot update:

```bash
git add tests/runtime/test_schema_snapshots.py
git commit -m "test: update runtime schema snapshots"
```

If no files changed, do not create an empty commit.

## Task 12: End-To-End Smoke With Fake AEQA Controller

**Files:**
- Create: `tests/runtime/test_aeqa_predictive_e2e.py`

- [ ] **Step 1: Add fake end-to-end smoke test**

Create `tests/runtime/test_aeqa_predictive_e2e.py`:

```python
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
```

- [ ] **Step 2: Run fake end-to-end smoke test**

Run:

```bash
pytest tests/runtime/test_aeqa_predictive_e2e.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit smoke test**

```bash
git add tests/runtime/test_aeqa_predictive_e2e.py
git commit -m "test: cover AEQA predictive runtime smoke path"
```

## Task 13: Final Verification

**Files:**
- No planned source edits.

- [ ] **Step 1: Check git status**

Run:

```bash
git status --short --branch
```

Expected: only unrelated pre-existing untracked directories may remain, such as `Pred-EQA/`, `RoboClaw/`, or `ep-chair/`. There should be no unstaged changes from this implementation.

- [ ] **Step 2: Run runtime suite**

Run:

```bash
pytest tests/runtime -q
```

Expected: PASS.

- [ ] **Step 3: Run relevant non-runtime tests if time permits**

Run:

```bash
pytest tests/test_aeqa_output_format.py tests/test_aeqa_vlm_config.py tests/test_e2e_smoke.py -q
```

Expected: PASS. If these tests fail due to pre-existing external dependencies, capture the exact failure and do not mask it with unrelated edits.

- [ ] **Step 4: Produce handoff summary**

Summarize:

```text
Implemented:
- multimodal planner contracts and transport compatibility
- sanitized multimodal prompt audit
- AEQA predictive visual-state builder, prompts, parsers, and controller
- AEQA controller graph route
- AEQA-only frontier/submit tool registry
- AEQA runner wiring and initial pose z-axis fix

Verified:
- pytest tests/runtime -q
- pytest tests/test_aeqa_output_format.py tests/test_aeqa_vlm_config.py tests/test_e2e_smoke.py -q

Known remaining scope:
- GOATBench hybrid memory/navigation redesign is separate
- full Pred-EQA snapshot/frontier pruning quality tuning is separate from this demo
```

## Self-Review

Spec coverage:

- Multimodal content contract: Task 1.
- Planner transport accepts image content: Task 2.
- Prompt audit avoids raw base64: Task 3.
- Pred-EQA style parsing and image prompts: Tasks 4 and 5.
- AEQA predictive controller: Task 6.
- LangGraph runtime integration: Task 7.
- AEQA tool reduction to `explore_frontier` and `submit_answer`: Task 8.
- Entrypoint and AEQA runner wiring: Tasks 9 and 10.
- GOATBench compatibility: Tasks 7, 8, 11, and 13.
- Fake end-to-end AEQA runtime proof: Task 12.

Red-flag scan:

- No empty or fill-in-later sections remain.
- The plan uses concrete paths, test names, commands, and code snippets.

Type consistency:

- `AEQAVisualState`, `AEQAImage`, and `AEQAFrontier` are defined before controller use.
- `AEQAPredictiveController.decide()` accepts keyword-only arguments matching the graph route.
- `PlannerDecision.arguments` uses JSON-compatible strings and ints only.
- `build_aeqa_tool_registry()` returns the existing `ToolRegistry` type.
