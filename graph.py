from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import CapstoneState
from nodes import (
    memory_node,
    router_node,
    retrieval_node,
    skip_node,
    tool_node,
    answer_node,
    eval_node,
    save_node,
)

FAITH_THRESHOLD = 0.7
MAX_RETRIES     = 2


def routing_decision(state: CapstoneState) -> str:
    nav = state.get("nav_route", "kb")
    if nav not in {"kb", "calc", "none"}:
        return "kb"
    return nav


def eval_decision(state: CapstoneState) -> str:
    if state.get("faith_score", 1.0) < FAITH_THRESHOLD and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "finish"


def bump_retry(state: CapstoneState) -> CapstoneState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    g = StateGraph(CapstoneState)

    g.add_node("n_memory",   memory_node)
    g.add_node("n_router",   router_node)
    g.add_node("n_retrieve", retrieval_node)
    g.add_node("n_skip",     skip_node)
    g.add_node("n_tool",     tool_node)
    g.add_node("n_generate", answer_node)
    g.add_node("n_eval",     eval_node)
    g.add_node("n_retry",    bump_retry)
    g.add_node("n_save",     save_node)

    g.set_entry_point("n_memory")
    g.add_edge("n_memory", "n_router")

    g.add_conditional_edges(
        "n_router",
        routing_decision,
        {"kb": "n_retrieve", "calc": "n_tool", "none": "n_skip"},
    )

    g.add_edge("n_retrieve", "n_generate")
    g.add_edge("n_skip",     "n_generate")
    g.add_edge("n_tool",     "n_generate")
    g.add_edge("n_generate", "n_eval")

    g.add_conditional_edges(
        "n_eval",
        eval_decision,
        {"retry": "n_retry", "finish": "n_save"},
    )

    g.add_edge("n_retry", "n_generate")
    g.add_edge("n_save",  END)

    return g


_graph  = None
_memory = None


def get_graph():
    global _graph, _memory
    if _graph is None:
        _memory = MemorySaver()
        _graph  = build_graph().compile(checkpointer=_memory)
    return _graph


def ask(question: str, thread_id: str = "default") -> dict:
    graph  = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial: CapstoneState = {
        "q_text":       question,
        "chat_history": [],
        "nav_route":    "",
        "kb_context":   "",
        "kb_sources":   [],
        "tool_output":  "",
        "ai_response":  "",
        "faith_score":  1.0,
        "retry_count":  0,
        "user_goal":    "",
        "user_name":    "",
    }

    result = graph.invoke(initial, config=config)

    return {
        "answer":       result.get("ai_response", ""),
        "route":        result.get("nav_route", ""),
        "faithfulness": result.get("faith_score", 0.0),
        "sources":      result.get("kb_sources", []),
        "user_name":    result.get("user_name", ""),
        "user_goal":    result.get("user_goal", ""),
    }
