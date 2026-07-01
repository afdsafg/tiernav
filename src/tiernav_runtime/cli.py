"""CLI utilities for tiernav runtime prompt auditing."""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict


def dump_context_tokens(episode_id: str, output_dir: str) -> str:
    """Read prompt_audit/<episode_id>.jsonl and return a token analysis table.

    Aggregates per-round section token estimates, prints a table showing
    each section's average tokens, percentage of total, and cache/boundary
    markers. Returns the table string (and prints to stdout).
    """
    path = Path(output_dir) / "prompt_audit" / f"{episode_id}.jsonl"
    if not path.exists():
        return f"no prompt audit log found at {path}"

    rounds = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rounds.append(json.loads(line))

    if not rounds:
        return f"empty prompt audit log at {path}"

    # Aggregate: per section name, collect token counts and cache flags.
    # Section order from the last round (most recent structure).
    last_sections = rounds[-1]["sections"]
    section_names = [s["name"] for s in last_sections]

    token_sums: dict[str, float] = defaultdict(float)
    cacheable: dict[str, bool] = {}
    cache_break: dict[str, bool] = {}
    for r in rounds:
        for s in r["sections"]:
            name = s["name"]
            token_sums[name] += s["tokens"]
            cacheable[name] = s.get("cacheable", False)
            cache_break[name] = s.get("cache_break", False)

    n_rounds = len(rounds)
    avg_tokens = {name: token_sums[name] / n_rounds for name in section_names}
    total = sum(avg_tokens.values()) or 1.0

    # Build table.
    lines = []
    header = f"{'section':<28} {'avg_tokens':>10} {'pct':>6} {'cacheable':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in section_names:
        avg = avg_tokens[name]
        pct = avg / total * 100
        cache_str = "yes" if cacheable.get(name) else "no"
        marker = "  <- boundary" if cache_break.get(name) else ""
        lines.append(f"{name:<28} {avg:>10.1f} {pct:>5.1f}% {cache_str:>9}{marker}")
    lines.append("")
    lines.append(f"total avg tokens: {sum(avg_tokens.values()):.1f} over {n_rounds} rounds")
    result = "\n".join(lines)
    print(result)
    return result
