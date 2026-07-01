"""AEQA predictive controller helpers.

This module is runtime-native and keeps Habitat/GPU objects behind duck-typed
environment adapters so unit tests can use fakes.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import Field

from .contracts import RuntimeModel


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

    matches = re.findall(
        r"^\s*Answer:\s*(.+?)(?:\s*\(Evidence:\s*Snapshot\s*(\d+)\s*\))?\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if matches:
        answer, snap_idx = matches[-1]
        answer = answer.strip().strip(".").strip().strip('"').strip("'")
        if answer:
            return {
                "action": "answer",
                "answer": answer,
                "evidence_snapshot": int(snap_idx) if snap_idx else None,
            }

    if re.search(r"^\s*Continue\s+Exploration\s*\.?\s*$", text, flags=re.IGNORECASE | re.MULTILINE):
        return {"action": "continue_exploration", "answer": "", "evidence_snapshot": None}

    return {"action": "continue_exploration", "answer": "", "evidence_snapshot": None}


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
    snapshots = [snapshot for snapshot in state.snapshots if snapshot.image_b64]
    if not snapshots:
        pairs.append(("No snapshots available\n", None))
    for idx, snapshot in enumerate(snapshots):
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
    snapshots = [snapshot for snapshot in state.snapshots if snapshot.image_b64]
    for idx, snapshot in enumerate(snapshots):
        label = snapshot.label or f"Snapshot {idx}"
        pairs.append((f"{label}: ", snapshot.image_b64))
        pairs.append(("\n", None))
    if not snapshots:
        pairs.append(("No snapshots available\n", None))
    pairs.append(("\nAvailable Exploration Directions:\n", None))
    frontiers = [frontier for frontier in state.frontiers if frontier.image_b64]
    if not frontiers:
        pairs.append(("No frontiers available\n", None))
    for frontier in frontiers:
        label = frontier.label or f"Frontier {frontier.frontier_id}"
        pairs.append((f"{label}: ", frontier.image_b64))
        pairs.append(("\n", None))
    valid = ", ".join(f.frontier_id for f in frontiers) or "none"
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
