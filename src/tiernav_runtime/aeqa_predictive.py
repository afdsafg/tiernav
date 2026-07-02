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
    """Lightweight Pred-EQA memory surface for the controller.

    Task 6 reads `step_summaries` and records the last answerer/explorer raw
    responses in `last_answerer_decision` / `last_explorer_decision`. The
    prediction/pruning fields (`prediction_items`, `retained_snapshot_ids`,
    `pruned_snapshot_ids`) are intentionally retained as the scaffolding for the
    snapshot-pruning / predictive-memory lane planned for this file and its
    tests; that lane is not wired into the controller path in Task 6.
    """

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


def _valid_images(images: list[AEQAImage]) -> list[AEQAImage]:
    return [image for image in images if image.image_b64]


def _append_image_section(
    pairs: list[tuple[str, Optional[str]]],
    *,
    title: str,
    images: list[AEQAImage],
    empty_text: str,
) -> None:
    pairs.append((title, None))
    valid_images = _valid_images(images)
    if not valid_images:
        pairs.append((empty_text, None))
        return
    for idx, image in enumerate(valid_images):
        label = image.label or f"Image {idx}"
        pairs.append((f"{label}: ", image.image_b64))
        pairs.append(("\n", None))


def parse_retain_indices(response: str, max_count: int, prefix: str = "Retain Snapshots") -> list[int]:
    """Parse snapshot-retention indices out of a planner response.

    This supports the snapshot-pruning / predictive-memory lane planned for this
    file: it parses which snapshot indices a retain/prune response should keep.
    The first Task 6 controller path does not call it yet, but it is kept here
    so the lane can be wired in without re-introducing parsing surface.
    """
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
2. Carefully inspect all available snapshots and current/egocentric views.
3. If any image contains enough visual evidence, output Answer.
4. If the evidence is insufficient, output Continue Exploration.
"""


EXPLORE_SYSTEM_PROMPT = """Task: You are an indoor agent that needs to PHYSICALLY NAVIGATE through sequential frontier selections to find information needed for answering the question.

Instructions:
1. Use common room-object relationships to infer where the needed evidence may be.
2. Use previous visual clues and the high-level plan to avoid repeated irrelevant areas.
3. Choose exactly one available frontier when exploration is needed.
4. Keep selecting frontiers until visual evidence is sufficient to answer.
"""


# Heuristic placeholder confidence values for the controller's PlannerDecision
# outputs. These are NOT calibrated scores; they exist so downstream consumers
# receive a consistent ordering signal (answer > explore > unanswerable) until a
# calibrated confidence source is wired in.
ANSWER_HEURISTIC_CONFIDENCE = 0.8
EXPLORE_HEURISTIC_CONFIDENCE = 0.6
UNANSWERABLE_CONFIDENCE = 0.0


def build_answer_messages(state: AEQAVisualState) -> list[dict]:
    pairs: list[tuple[str, Optional[str]]] = [
        (f"Question: {state.question}\n", None),
    ]
    if state.memory_text:
        pairs.append((state.memory_text + "\n", None))
    _append_image_section(
        pairs,
        title="Current / Egocentric Views:\n",
        images=state.egocentric_views,
        empty_text="No current egocentric views available\n",
    )
    pairs.append(("Available Snapshots:\n", None))
    snapshots = _valid_images(state.snapshots)
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
    _append_image_section(
        pairs,
        title="Current / Egocentric Views:\n",
        images=state.egocentric_views,
        empty_text="No current egocentric views available\n",
    )
    pairs.append(("Available Snapshots:\n", None))
    snapshots = _valid_images(state.snapshots)
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


class AEQAPredictiveController:
    """Pred-EQA style AEQA controller that returns runtime PlannerDecision objects."""

    def __init__(
        self,
        builder: Optional[AEQAVisualStateBuilder] = None,
        max_memory_episodes: int = 128,
    ) -> None:
        self.builder = builder or AEQAVisualStateBuilder()
        # Bounded insertion-ordered cache: evicts the oldest episode memory once
        # the cap is reached so long-running sessions do not grow unbounded.
        self.max_memory_episodes = max(1, int(max_memory_episodes))
        self._memory_by_episode: dict[str, AEQAPredictiveMemory] = {}

    def memory_for(self, episode_id: str) -> AEQAPredictiveMemory:
        existing = self._memory_by_episode.get(episode_id)
        if existing is not None:
            return existing
        if len(self._memory_by_episode) >= self.max_memory_episodes:
            # dict preserves insertion order; drop the oldest entry.
            oldest = next(iter(self._memory_by_episode))
            del self._memory_by_episode[oldest]
        memory = AEQAPredictiveMemory()
        self._memory_by_episode[episode_id] = memory
        return memory

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
        answer_raw: Optional[str] = None
        try:
            answer_raw = planner.call_vlm(answer_messages, max_tokens=1024, temperature=0.3)
        finally:
            self._audit(prompt_audit, episode, "aeqa_answerer", answer_messages, response=answer_raw)
        parsed_answer = parse_answer_response(answer_raw)
        memory.last_answerer_decision = answer_raw or ""

        if parsed_answer["action"] == "answer" and parsed_answer["answer"]:
            args: dict[str, Any] = {"answer": parsed_answer["answer"]}
            if parsed_answer["evidence_snapshot"] is not None:
                args["evidence_snapshot"] = parsed_answer["evidence_snapshot"]
            return PlannerDecision(
                action_type="submit_answer",
                reasoning="AEQA answerer found sufficient visual evidence.",
                confidence=ANSWER_HEURISTIC_CONFIDENCE,
                arguments=args,
            )

        valid_frontier_ids = [
            frontier.frontier_id for frontier in state.frontiers if frontier.image_b64
        ]
        if not valid_frontier_ids:
            return PlannerDecision(
                action_type="submit_answer",
                reasoning="AEQA answerer could not answer and no frontier is available.",
                confidence=UNANSWERABLE_CONFIDENCE,
                arguments={"answer": "unanswerable"},
            )

        explore_messages = build_explore_messages(state)
        explore_raw: Optional[str] = None
        try:
            explore_raw = planner.call_vlm(explore_messages, max_tokens=1024, temperature=0.3)
        finally:
            self._audit(prompt_audit, episode, "aeqa_explorer", explore_messages, response=explore_raw)
        selected = parse_frontier_response(explore_raw, valid_frontier_ids)
        memory.last_explorer_decision = explore_raw or ""

        return PlannerDecision(
            action_type="explore_frontier",
            reasoning="AEQA explorer selected a frontier for more visual evidence.",
            confidence=EXPLORE_HEURISTIC_CONFIDENCE,
            arguments={"frontier_id": selected},
        )

    @staticmethod
    def _audit(
        prompt_audit: Any,
        episode: Any,
        label: str,
        messages: list[dict],
        response: str | None = None,
    ) -> None:
        if prompt_audit is None or not hasattr(prompt_audit, "record_multimodal"):
            return
        prompt_audit.record_multimodal(
            episode_id=str(getattr(episode, "episode_id", "")),
            round_index=int(getattr(episode, "round_index", 0) or 0),
            step_index=int(getattr(episode, "step_index", 0) or 0),
            label=label,
            messages=messages,
            response=response,
        )
