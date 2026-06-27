"""build_two_tier_graph — compile the LangGraph StateGraph.

Graph topology (7 nodes, 2 conditional edges):

    START → init → build_context → planner → loop_guard ──"submit"──→ submit → END
                              ↑                              │
                              │                              └──"execute"──→ executor → memory_update
                              │                                                     │
                              └───────────"continue"─────────────────────────────────┘
                                                    (after_memory edge)
                                                      └──"fallback_submit"──→ submit → END

Backtracking (GD-fail → hypothesis_rejected) is data flow through memory, not
an edge: executor produces TrajectoryEvidence(outcome="detection_failed"),
memory_update marks room rejected, next build_context excludes it. The graph
simply loops via "continue".
"""
from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from .edges import after_guard, after_memory
from .nodes import (
    build_context_node,
    executor_node,
    init_node,
    loop_guard_node,
    memory_update_node,
    planner_node,
    submit_node,
)
from .state import TwoTierState


def build_two_tier_graph():
    """Build and compile the Two-Tier StateGraph.

    Returns a compiled graph that can be invoked via:
        graph.invoke(initial_state, config={"configurable": {"resources": ...}})

    The graph is provider-agnostic and tool-registry-agnostic — swapping LLM
    providers or adding tools requires no graph edit. Nodes are addable/
    removable/modifiable via LangGraph's native API (g.add_node, g.add_edge,
    etc.) for future levers (Critic, multi-agent, etc.).
    """
    g: StateGraph = StateGraph(TwoTierState)

    # ── Add 7 nodes ──
    g.add_node("init", init_node)
    g.add_node("build_context", build_context_node)
    g.add_node("planner", planner_node)
    g.add_node("loop_guard", loop_guard_node)
    g.add_node("executor", executor_node)
    g.add_node("memory_update", memory_update_node)
    g.add_node("submit", submit_node)

    # ── Static edges ──
    g.add_edge(START, "init")
    g.add_edge("init", "build_context")
    g.add_edge("build_context", "planner")
    g.add_edge("planner", "loop_guard")
    g.add_edge("executor", "memory_update")
    g.add_edge("submit", END)

    # ── Conditional edge: after_guard (leaves loop_guard) ──
    # "submit" → submit_node (action is submit_answer, wraps :1614)
    # "execute" → executor_node (wraps :1633)
    g.add_conditional_edges(
        "loop_guard",
        after_guard,
        {
            "submit": "submit",
            "execute": "executor",
        },
    )

    # ── Conditional edge: after_memory (leaves memory_update) ──
    # "continue" → build_context (next round, back edge — the main loop)
    # "fallback_submit" → submit_node (budget exhausted, wraps :1681)
    #
    # CRITICAL: the ordering in after_memory (round-budget → exhausted →
    # step-budget) must be preserved to reproduce the :1665-1677 skip semantics.
    g.add_conditional_edges(
        "memory_update",
        after_memory,
        {
            "continue": "build_context",
            "fallback_submit": "submit",
        },
    )

    return g.compile()
