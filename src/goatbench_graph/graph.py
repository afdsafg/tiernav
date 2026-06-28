"""build_goatbench_graph — compile LangGraph StateGraph.

Graph topology (5 nodes, 1 conditional edge):

  START → observe → update_memory → vlm_decide → navigate → check_arrival
                                                            │
                                                            ├──"continue"──→ observe
                                                            └──"end"──→ END

The main loop is the back-edge from check_arrival to observe. check_arrival
sets terminal=True on success or step-budget exhaustion.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .edges import after_check_arrival, after_vlm_decide
from .nodes import (
    check_arrival_node,
    navigate_node,
    observe_node,
    update_memory_node,
    vlm_decide_node,
)
from .state import GoatbenchState


def build_goatbench_graph():
    """Build and compile the GOATBench StateGraph.

    Topology:
      START → observe → update_memory → vlm_decide ─┬─"navigate"─→ navigate → check_arrival
                                                   └─"check_arrival"─→ check_arrival
      check_arrival ─┬─"continue"─→ observe (loop)
                     └─"end"─→ END

    The vlm_decide→check_arrival short-circuit skips navigate when VLM fails
    (terminal=True), mirroring the original `break` that skipped agent_step.
    """
    g = StateGraph(GoatbenchState)
    g.add_node("observe", observe_node)
    g.add_node("update_memory", update_memory_node)
    g.add_node("vlm_decide", vlm_decide_node)
    g.add_node("navigate", navigate_node)
    g.add_node("check_arrival", check_arrival_node)
    g.add_edge(START, "observe")
    g.add_edge("observe", "update_memory")
    g.add_edge("update_memory", "vlm_decide")
    g.add_conditional_edges(
        "vlm_decide",
        after_vlm_decide,
        {"navigate": "navigate", "check_arrival": "check_arrival"},
    )
    g.add_edge("navigate", "check_arrival")
    g.add_conditional_edges(
        "check_arrival",
        after_check_arrival,
        {"continue": "observe", "end": END},
    )
    return g.compile()
