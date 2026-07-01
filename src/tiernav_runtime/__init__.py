"""TierNav runtime package.

The runtime is the contract-first, LangGraph-only execution layer used by
AEQA, GOATBench, replay, and ablation runs.
"""
from .cli import dump_context_tokens

__all__ = ["dump_context_tokens"]
