"""Upper Planner: Structured prompt + mim-v2.5 API + action parsing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.const import (
    QWEN_PLANNER_API_KEY,
    QWEN_PLANNER_BASE_URL,
    QWEN_PLANNER_MODEL,
)

logger = logging.getLogger(__name__)


# ── Data class ────────────────────────────────────────────────────────────

@dataclass
class PlannerAction:
    """Structured decision from the Planner."""
    action_type: str
    reason: str = ""
    confidence: float = 0.0
    snapshot_id: Optional[str] = None
    object_name: Optional[str] = None
    seed_id: Optional[str] = None
    frontier_id: Optional[str] = None
    view_idx: Optional[int] = None
    answer: Optional[str] = None
    expected: Optional[str] = None


# ── System prompt ─────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = (
    "You are an embodied AI agent exploring indoor scenes.\n"
    "You make strategic decisions about where to navigate next.\n\n"
    "Available Actions:\n"
    "1. explore_panorama: Take an 8-view panorama to re-orient. Arguments: {}\n"
    "2. navigate_to_object: Navigate toward an object you can SEE in one of the attached current-view snapshots.\n"
    '   Arguments: {{"snapshot_id": "step12_view1", "object_name": "<target object>"}}\n'
    "3. explore_seed: Navigate to a seed viewpoint.\n"
    '   Arguments: {{"seed_id": "3"}}\n'
    "4. explore_frontier: Navigate to an unexplored frontier.\n"
    '   Arguments: {{"frontier_id": "14"}}\n'
    "5. submit_answer: Submit the final answer to the question.\n"
    '   Arguments: {{"snapshot_id": "step12_view1", "answer": "<answer grounded in the cited snapshot>"}}\n\n'
    "Decision Rules:\n"
    "- The attached current-view snapshots are the visual observations after the previous action.\n"
    "- If the current-view snapshots or history contain enough evidence to answer, choose submit_answer immediately.\n"
    "- For navigate_to_object, return the snapshot_id that shows the object, object_name, and reasoning.\n"
    "- For submit_answer, return the snapshot_id supporting the answer, answer, and reasoning.\n"
    "- Exploration/navigation actions must include expected: what you expect the selected action to verify or accomplish next.\n"
    "- submit_answer is terminal and does not need expected.\n"
    "- Do not call navigate_to_object again for the same object after arriving near it unless a different view is explicitly needed.\n\n"
    "Response Format:\n"
    '{{"reasoning": "why this action", "expected": "omit for submit_answer", "action": "<action_name>", "arguments": {{...}}, "confidence": 0.8}}\n\n'
    "Always output valid JSON. Only use actions from the list above."
)


# Navigation variant — goal-driven navigation instead of QA
PLANNER_NAV_SYSTEM_PROMPT = (
    "You are an embodied AI agent navigating indoor scenes to reach a goal.\n"
    "You make strategic decisions about where to navigate next.\n\n"
    "Available Actions:\n"
    "1. explore_panorama: Take an 8-view panorama to re-orient. Arguments: {}\n"
    "2. navigate_to_object: Navigate toward an object you can SEE in one of the attached current-view snapshots.\n"
    '   Arguments: {{"snapshot_id": "step12_view1", "object_name": "<target object>"}}\n'
    "3. explore_seed: Navigate to a seed viewpoint.\n"
    '   Arguments: {{"seed_id": "3"}}\n'
    "4. explore_frontier: Navigate to an unexplored frontier.\n"
    '   Arguments: {{"frontier_id": "14"}}\n'
    "5. submit_answer: Indicate that you have reached the goal location. This is terminal.\n"
    '   Arguments: {{"snapshot_id": "step12_view1", "answer": "reached_goal"}}\n\n'
    "Decision Rules:\n"
    "- The attached current-view snapshots are the visual observations after the previous action.\n"
    "- If you believe you have reached the goal, choose submit_answer immediately.\n"
    "- For navigate_to_object, return the snapshot_id that shows the object, object_name, and reasoning.\n"
    "- For submit_answer, return the snapshot_id showing the goal and reasoning.\n"
    "- Exploration/navigation actions must include expected: what you expect the selected action to verify or accomplish next.\n"
    "- submit_answer is terminal and does not need expected.\n"
    "- Do not call navigate_to_object again for the same object after arriving near it unless a different view is explicitly needed.\n\n"
    "Response Format:\n"
    '{{"reasoning": "why this action", "expected": "omit for submit_answer", "action": "<action_name>", "arguments": {{...}}, "confidence": 0.8}}\n\n'
    "Always output valid JSON. Only use actions from the list above."
)


# ── Planner class ─────────────────────────────────────────────────────────

class Planner:
    """Calls mimo-v2.5 via proven call_vlm and returns PlannerAction."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = QWEN_PLANNER_MODEL,
        goal_type: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.goal_type = goal_type
        self.system_prompt = PLANNER_NAV_SYSTEM_PROMPT if goal_type else PLANNER_SYSTEM_PROMPT

    def decide(
        self,
        question: str,
        history: str,
        scene: str,
        progress: str,
        actions: str,
        image_b64: Optional[str] = None,
        extra_image_b64s: Optional[list[str]] = None,
        image_b64s: Optional[list[str]] = None,
    ) -> PlannerAction:
        """Call VLM with 4-component prompt, parse action."""
        prompt = self.build_prompt(question, history, scene, progress, actions)
        messages = [{"role": "system", "content": self.system_prompt}]

        images = []
        if image_b64s:
            images.extend([img for img in image_b64s if img])
        elif image_b64:
            images.append(image_b64)
        if extra_image_b64s:
            images.extend([img for img in extra_image_b64s if img])

        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                })
            messages.append({
                "role": "user",
                "content": content,
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response_text = self._call_api(messages)
        return self.parse_response(response_text)

    def build_prompt(
        self,
        question: str,
        history: str,
        scene: str,
        progress: str,
        actions: str,
    ) -> str:
        """Build 4-component prompt with selected actions."""
        return (
            f"Question: {question}\n\n"
            f"{history}\n\n"
            f"{scene}\n\n"
            f"{progress}\n\n"
            f"{actions}\n\n"
            "Decide the next action and respond with JSON."
        )

    def parse_response(self, response: str) -> PlannerAction:
        """Parse VLM response: try JSON first, then keyword fallback."""
        if not response:
            return PlannerAction(action_type="explore_panorama",
                                reason="Empty VLM response", confidence=0.0)

        import re
        raw = response.strip()

        # Try code fence first
        data = None
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try raw { ... }
        if data is None and "{" in raw:
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                data = json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

        if data is not None:
            action_type = data.get("action", "explore_panorama")
            args = data.get("arguments", {}) or {}
            answer = args.get("answer") or data.get("answer")
            if action_type == "submit_answer" and not answer:
                logger.info("Planner JSON submit_answer missing answer; raw[:200]=%s", raw[:200])
                return PlannerAction(
                    action_type="explore_panorama",
                    reason="Invalid submit_answer: missing answer",
                    confidence=0.0,
                )
            return PlannerAction(
                action_type=action_type,
                reason=data.get("reasoning", data.get("reason", "")),
                confidence=float(data.get("confidence", 0.5)),
                snapshot_id=args.get("snapshot_id") or data.get("snapshot_id"),
                object_name=args.get("object_name") or data.get("object_name"),
                seed_id=str(args["seed_id"]) if args.get("seed_id") is not None else str(data.get("seed_id")) if data.get("seed_id") is not None else None,
                frontier_id=str(args.get("frontier_id")) if args.get("frontier_id") is not None else None,
                view_idx=args["view_idx"] if args.get("view_idx") is not None else data.get("view_idx"),
                answer=answer,
                expected=data.get("expected") or args.get("expected"),
            )

        # Keyword fallback with ID extraction
        raw_l = raw.lower()
        for kw, action in [
            ("submit_answer", "submit_answer"),
            ("navigate_to_object", "navigate_to_object"),
            ("explore_seed", "explore_seed"),
            ("explore_frontier", "explore_frontier"),
            ("explore_panorama", "explore_panorama"),
        ]:
            if kw in raw_l:
                if action == "submit_answer":
                    logger.info("Planner ignored non-JSON submit_answer fallback, raw[:200]=%s", raw[:200])
                    return PlannerAction(
                        action_type="explore_panorama",
                        reason="Invalid non-JSON submit_answer response",
                        confidence=0.0,
                    )
                if action == "explore_seed":
                    m = re.search(r'(?:seed|room)[_\s]*(\d+)', raw_l)
                    sid = str(m.group(1)) if m else None
                    return PlannerAction(action_type=action, seed_id=sid, reason="Inferred", confidence=0.4)
                if action == "explore_frontier":
                    m = re.search(r'frontier[_\s]*(\d+)', raw_l)
                    fid = str(m.group(1)) if m else None
                    return PlannerAction(action_type=action, frontier_id=fid, reason="Inferred", confidence=0.4)
                if action == "navigate_to_object":
                    m = re.search(r'(?:navigate_to_object|navigate to|go to|find)\s*["\']?(\w+)["\']?', raw_l)
                    obj = m.group(1) if m else "object"
                    return PlannerAction(action_type=action, object_name=obj, reason="Inferred", confidence=0.4)
                return PlannerAction(action_type=action, reason="Inferred", confidence=0.4)

        logger.info("Planner parse fallback, raw[:200]=%s", raw[:200])
        return PlannerAction(action_type="explore_panorama",
                            reason="Parse failed", confidence=0.0)

    def _call_api(self, messages: list[dict]) -> str:
        """Use the proven call_vlm from agent_workflow (mimo-v2.5 compatible)."""
        from src.agent_workflow import call_vlm
        return call_vlm(
            messages,
            max_tokens=4096,
            temperature=0.3,
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
        )
