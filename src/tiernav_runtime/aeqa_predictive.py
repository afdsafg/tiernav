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
