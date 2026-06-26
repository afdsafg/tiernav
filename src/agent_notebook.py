"""Evidence Notebook — persistent cross-stage memory for two-tier agent."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NotebookEntry:
    """单个证据条目。"""
    step: int
    timestamp: str = ""
    entry_type: str = ""  # room_explored | object_observed | hypothesis_rejected | seed_visited | frontier_visited
    content: str = ""
    negation: bool = False
    confidence: float = 0.0
    key_frame_id: Optional[str] = None


@dataclass
class StructuredNotebook:
    """Contribution 1 — structured persistent task state.

    Extends the evidence-tracking notebook with explicit:
      - hypotheses: current hypotheses about where the answer/target is
      - todo: pending verification actions
      - rejected_regions: rooms/areas ruled out with reasons (Contribution: rejected tracking)
      - evidence_ids: evidence viewed so far
      - cross_subtask: entries carried across subtasks (GOATBench reuse, V2)

    Kept separate from the legacy entry-list API so existing code/tests keep working.
    """
    hypotheses: list[str] = field(default_factory=list)
    todo: list[str] = field(default_factory=list)
    rejected_regions: dict[str, str] = field(default_factory=dict)  # region_id -> reason
    evidence_ids: list[str] = field(default_factory=list)
    cross_subtask: list[dict] = field(default_factory=list)

    def add_hypothesis(self, h: str) -> None:
        if h and h not in self.hypotheses:
            self.hypotheses.append(h)
            # Keep most recent 8
            if len(self.hypotheses) > 8:
                self.hypotheses = self.hypotheses[-8:]

    def add_todo(self, t: str) -> None:
        if t and t not in self.todo:
            self.todo.append(t)
            if len(self.todo) > 6:
                self.todo = self.todo[-6:]

    def mark_rejected(self, region_id: str, reason: str) -> None:
        self.rejected_regions[str(region_id)] = reason

    def is_rejected(self, region_id: str) -> bool:
        return str(region_id) in self.rejected_regions

    def add_evidence(self, evidence_id: str) -> None:
        if evidence_id and evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)
            if len(self.evidence_ids) > 32:
                self.evidence_ids = self.evidence_ids[-32:]

    def to_dict(self) -> dict:
        return {
            "hypotheses": list(self.hypotheses),
            "todo": list(self.todo),
            "rejected_regions": dict(self.rejected_regions),
            "evidence_ids": list(self.evidence_ids),
            "cross_subtask": list(self.cross_subtask),
        }

    def get_injection_text(self) -> str:
        """Compact text for Planner prompt injection."""
        lines = ["## Structured Notebook"]
        if self.hypotheses:
            lines.append("Hypotheses:")
            for h in self.hypotheses[-5:]:
                lines.append(f"  - {h}")
        if self.todo:
            lines.append("TODO:")
            for t in self.todo[-4:]:
                lines.append(f"  - {t}")
        if self.rejected_regions:
            lines.append("Rejected regions:")
            for rid, reason in list(self.rejected_regions.items())[-5:]:
                lines.append(f"  - region {rid}: {reason[:80]}")
        if self.evidence_ids:
            lines.append(f"Evidence viewed: {len(self.evidence_ids)} items ({', '.join(self.evidence_ids[-6:])})")
        return "\n".join(lines)


class EvidenceNotebook:
    """Agent 证据笔记本：记录探索历史、检测循环、注入 Planner prompt。"""

    def __init__(self):
        self.entries: list[NotebookEntry] = []
        # entity_id (lowercase) -> visit count
        self._exhausted_ids: dict[str, int] = defaultdict(int)
        # entity_id (lowercase) -> list of outcome content strings
        self._last_outcomes: dict[str, list[str]] = defaultdict(list)
        # Contribution 1: structured persistent task state
        self.structured = StructuredNotebook()

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def add_entry(
        self,
        step: int,
        entry_type: str,
        content: str,
        negation: bool = False,
        confidence: float = 0.0,
        key_frame_id: Optional[str] = None,
    ) -> NotebookEntry:
        """添加一条证据记录并更新循环检测状态。"""
        entry = NotebookEntry(
            step=step,
            entry_type=entry_type,
            content=content,
            negation=negation,
            confidence=confidence,
            key_frame_id=key_frame_id,
        )
        self.entries.append(entry)

        # 追踪 seed / frontier 访问次数以检测循环
        if entry_type == "seed_visited":
            entity_id = self._extract_id(content, "Seed_")
            if entity_id:
                self._last_outcomes[entity_id].append(content)
                self._exhausted_ids[entity_id] += 1
        elif entry_type == "frontier_visited":
            entity_id = self._extract_id(content, "Frontier_")
            if entity_id:
                self._last_outcomes[entity_id].append(content)
                self._exhausted_ids[entity_id] += 1

        # Feed structured notebook (Contribution 1)
        if entry_type == "hypothesis_rejected":
            # Extract room/seed id from content for rejected_regions
            rid = self._extract_id(content, "Room_") or self._extract_id(content, "Seed_")
            if rid:
                self.structured.mark_rejected(rid, content[:120])
        if key_frame_id:
            self.structured.add_evidence(key_frame_id)

        return entry

    def update_from_evidence(self, evidence, step: int):
        """将 TrajectoryEvidence 转换为 NotebookEntry 并添加（复用追踪逻辑）。"""
        entry = evidence.to_notebook_entry(step)
        # 复用 add_entry 的追踪逻辑，不要直接 append
        self.add_entry(
            step=entry.step,
            entry_type=entry.entry_type,
            content=entry.content,
            negation=entry.negation,
            confidence=entry.confidence,
            key_frame_id=entry.key_frame_id,
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def is_exhausted(self, entity_id: str) -> bool:
        """检测同一实体是否已访问 3 次以上。"""
        return self._exhausted_ids.get(entity_id.lower(), 0) >= 3

    def get_injection_text(self, max_entries: int = 10) -> str:
        """生成注入 Planner prompt 的历史文本。"""
        recent = self.entries[-max_entries:]
        lines: list[str] = []
        for e in recent:
            marker = "NOT " if e.negation else ""
            line = f"- [Step {e.step}] {marker}{e.content}"
            lines.append(line)
        return "## History\nYou have explored the following:\n" + "\n".join(lines)

    def get_visited_seeds(self) -> set[str]:
        """返回已访问过的 seed ID 集合。"""
        return {
            entity_id
            for e in self.entries
            if e.entry_type == "seed_visited"
            for entity_id in [self._extract_id(e.content, "seed")]
            if entity_id
        }

    def get_visited_frontiers(self) -> set[str]:
        """返回已访问过的 frontier ID 集合。"""
        return {
            entity_id
            for e in self.entries
            if e.entry_type == "frontier_visited"
            for entity_id in [self._extract_id(e.content, "frontier")]
            if entity_id
        }

    # ------------------------------------------------------------------
    # Structured notebook integration (Contribution 1)
    # ------------------------------------------------------------------

    def apply_planner_update(self, update: dict) -> None:
        """Apply a planner-emitted notebook_update dict.

        Expected keys (all optional):
          hypotheses: list[str]
          todo: list[str]
          rejected: list[str] or list[{"region_id":..., "reason":...}]
          evidence_ids: list[str]
        """
        if not update:
            return
        for h in update.get("hypotheses") or []:
            self.structured.add_hypothesis(str(h))
        for t in update.get("todo") or []:
            self.structured.add_todo(str(t))
        for r in update.get("rejected") or []:
            if isinstance(r, dict):
                self.structured.mark_rejected(str(r.get("region_id", "")), str(r.get("reason", "")))
            else:
                self.structured.mark_rejected(str(r), "planner rejected")
        for eid in update.get("evidence_ids") or []:
            self.structured.add_evidence(str(eid))

    def to_dict(self) -> dict:
        """Serialize current notebook state for decision_trace.jsonl."""
        return {
            "num_entries": len(self.entries),
            "recent_entries": [
                {"step": e.step, "type": e.entry_type, "content": e.content[:200], "negation": e.negation}
                for e in self.entries[-5:]
            ],
            "structured": self.structured.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_id(content: str, prefix: str) -> str:
        """Extract IDs from forms like Seed_3, seed 3, or Navigate to seed 3."""
        kind = prefix.rstrip("_").lower()
        match = re.search(rf"\b{re.escape(kind)}[_\s-]*(\d+)\b", content, re.IGNORECASE)
        if match:
            return f"{kind}_{match.group(1)}"
        return ""
