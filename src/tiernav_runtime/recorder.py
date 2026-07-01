"""Append-only JSONL recorder for runtime events."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import ContextSection
from .events import EpisodeEvent


class EpisodeRecorder:
    """Write episode events as append-only JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: EpisodeEvent) -> None:
        """Append one event without rewriting existing lines."""

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")


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


class PromptAuditRecorder:
    """Write per-round prompt sections (with full content) as JSONL.

    Each line is one compile round, containing the episode's sections with
    full content for post-hoc prompt auditing (token analysis, cache-break
    inspection, debugging).
    """

    def __init__(self, output_dir: str | Path):
        self.dir = Path(output_dir) / "prompt_audit"
        self.dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        episode_id: str,
        round_index: int,
        step_index: int,
        sections: list[ContextSection],
    ) -> None:
        path = self.dir / f"{episode_id}.jsonl"
        entry = {
            "round": round_index,
            "step": step_index,
            "sections": [
                {
                    "name": s.name,
                    "content": s.content,
                    "hash": s.content_hash,
                    "tokens": s.token_estimate,
                    "cacheable": s.cacheable,
                    "cache_break": s.cache_break,
                }
                for s in sections
            ],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
