"""LangGraph state-machine formalization of the Two-Tier Planner-Executor loop.

Phase 1: behavior-preserving port of `run_episode_two_tier` (agent_workflow.py:1087)
into a LangGraph StateGraph. Nodes are thin wrappers around existing helpers;
the LLM provider and tool registry are abstracted for future swaps.

See /home/afdsafg/.codebuddy/plans/swift-forging-newton.md for the design.
"""
