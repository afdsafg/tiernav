"""build_two_tier_graph — compile the LangGraph StateGraph.

Graph topology (8 nodes, 3 conditional edges):

    START → init → build_context → planner → loop_guard ──"submit"──→ submit → END
                              ↑                              │
                              │                              └──"execute"──→ executor → memory_update
                              │                                                     │
                              ├──"continue"──────────────────────────────────────────┘
                              │   (after_memory edge)
                              ├──"stall_recovery"──→ stall_recovery ──────────────────┘
                              │                          (P3: hint injected, retry)
                              └──"fallback_submit"──→ submit → END

Backtracking (GD-fail → hypothesis_rejected) is data flow through memory, not
an edge: executor produces TrajectoryEvidence(outcome="detection_failed"),
memory_update marks room rejected, next build_context excludes it. The graph
simply loops via "continue".

P3 adds stall_recovery_node (8th node) + stall_recovery route. When
memory_update detects repeated-action-no-progress, after_memory routes to
stall_recovery which injects a hint, then loops back to build_context.
"""
from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from .edges import after_check_arrival, after_critic, after_guard, after_memory, after_submit
from .nodes import (
    build_context_node,
    check_arrival_node,
    critic_node,
    executor_node,
    init_node,
    loop_guard_node,
    memory_update_node,
    note_node,
    planner_node,
    stall_recovery_node,
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

    # ── Add 10 nodes ──
    g.add_node("note", note_node)
    g.add_node("init", init_node)
    g.add_node("build_context", build_context_node)
    g.add_node("planner", planner_node)
    g.add_node("critic", critic_node)  # D3: planner→critic→loop_guard
    g.add_node("loop_guard", loop_guard_node)
    g.add_node("executor", executor_node)
    g.add_node("check_arrival", check_arrival_node)
    g.add_node("memory_update", memory_update_node)
    g.add_node("stall_recovery", stall_recovery_node)  # P3
    g.add_node("submit", submit_node)

    # ── Static edges ──
    g.add_edge(START, "note")
    g.add_edge("note", "init")
    g.add_edge("init", "build_context")
    g.add_edge("build_context", "planner")
    g.add_edge("planner", "critic")  # D3: critic between planner & loop_guard
    g.add_edge("executor", "check_arrival")
    g.add_edge("stall_recovery", "build_context")  # P3: hint injected, retry

    # ── Conditional edge: after_check_arrival (GOATBench proximity) ──
    # GOATBench: within_target → submit (终止); 否则 → memory_update (继续)
    # AEQA: is_terminal_task=False → 恒 memory_update
    g.add_conditional_edges(
        "check_arrival",
        after_check_arrival,
        {
            "submit": "submit",
            "memory_update": "memory_update",
        },
    )

    # ── Conditional edge: after_critic (leaves critic_node, D3) ──
    # planner→loop_guard replaced by planner→critic→(after_critic)→loop_guard|planner
    # "planner"    → planner (veto: force re-decision with feedback)
    # "loop_guard" → loop_guard (no veto: original flow)
    g.add_conditional_edges(
        "critic",
        after_critic,
        {
            "planner": "planner",
            "loop_guard": "loop_guard",
        },
    )

    # ── Conditional edge: after_submit (leaves submit_node) ──
    # P3 verification nudge: on first fallback entry, route back to
    # build_context for one more round; on second entry, commit to END.
    g.add_conditional_edges(
        "submit",
        after_submit,
        {
            "verify": "build_context",
            "end": END,
        },
    )

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
    # "stall_recovery" → stall_recovery_node (P3: repeated-action-no-progress)
    # "fallback_submit" → submit_node (budget exhausted, wraps :1681)
    #
    # CRITICAL: the ordering in after_memory (stall → round-budget → exhausted →
    # step-budget) must be preserved to reproduce the :1665-1677 skip semantics.
    g.add_conditional_edges(
        "memory_update",
        after_memory,
        {
            "continue": "build_context",
            "stall_recovery": "stall_recovery",  # P3
            "fallback_submit": "submit",
        },
    )

    return g.compile()
