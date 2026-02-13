# demo_langgraph_routing.py
from __future__ import annotations

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# --------------------------
# Demo A: add_conditional_edges
# --------------------------
class StateA(TypedDict):
    query: str
    intent: str
    answer: str


def router_node(state: StateA) -> dict:
    """
    这是一个“普通节点”：它只负责更新 state（不负责路由）。
    """
    q = state["query"]
    intent = "qa" if "?" in q else "chitchat"
    return {"intent": intent}


def route_after_router(state: StateA) -> Literal["qa_node", "chat_node"]:
    """
    这是一个“路由函数”：它不会更新 state，只返回下一步去哪个节点。
    """
    return "qa_node" if state["intent"] == "qa" else "chat_node"


def qa_node(state: StateA) -> dict:
    return {"answer": f"[QA] I think your question is: {state['query']}"}


def chat_node(state: StateA) -> dict:
    return {"answer": f"[CHAT] You said: {state['query']}"}


def build_graph_a():
    builder = StateGraph(StateA)
    builder.add_node("router", router_node)
    builder.add_node("qa_node", qa_node)
    builder.add_node("chat_node", chat_node)

    builder.add_edge(START, "router")

    # 核心：router 跑完后，不用写死 edge，而是动态决定下一节点
    builder.add_conditional_edges("router", route_after_router)

    builder.add_edge("qa_node", END)
    builder.add_edge("chat_node", END)
    return builder.compile()


# --------------------------
# Demo B: Command(update + goto)
# --------------------------
class StateB(TypedDict):
    query: str
    model_tier: str
    answer: str


def choose_path(state: StateB) -> Command[Literal["fast_path", "strong_path"]]:
    """
    这个节点返回 Command：既更新 state，又决定 goto 到哪个节点。
    注意：不要给这个节点再加静态出边，否则会“静态边 + goto 都执行”。
    """
    q = state["query"]
    if len(q) < 30:
        return Command(update={"model_tier": "fast"}, goto="fast_path")
    else:
        return Command(update={"model_tier": "strong"}, goto="strong_path")


def fast_path(state: StateB) -> dict:
    # 假装 fast 模型：回答更简短
    return {"answer": f"[FAST] short answer for: {state['query']} (tier={state['model_tier']})"}


def strong_path(state: StateB) -> dict:
    # 假装 strong 模型：回答更详细
    return {"answer": f"[STRONG] detailed answer for: {state['query']} (tier={state['model_tier']})"}


def build_graph_b():
    builder = StateGraph(StateB)
    builder.add_node("choose_path", choose_path)  # 返回 Command 的节点
    builder.add_node("fast_path", fast_path)
    builder.add_node("strong_path", strong_path)

    builder.add_edge(START, "choose_path")

    # 注意：这里不要 add_edge("choose_path", ...) ——让 Command 完全控制路由
    builder.add_edge("fast_path", END)
    builder.add_edge("strong_path", END)
    return builder.compile()


def run_with_stream(graph, init_state: dict, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("init_state:", init_state)

    # 有些版本支持 stream_mode="values"：每一步打印 state
    try:
        print("\n-- stream (values) --")
        for s in graph.stream(init_state, stream_mode="values"):
            print(s)
    except TypeError:
        # 兼容旧版本：没有 stream_mode 参数
        print("\n-- stream (default) --")
        for s in graph.stream(init_state):
            print(s)

    print("\n-- invoke (final) --")
    final_state = graph.invoke(init_state)
    print(final_state)
    print("=" * 80)


if __name__ == "__main__":
    g_a = build_graph_a()
    g_b = build_graph_b()

    # Demo A：问句 -> qa_node；陈述句 -> chat_node
    run_with_stream(
        g_a,
        {"query": "What is LangGraph?", "intent": "", "answer": ""},
        "Demo A: add_conditional_edges (router decides next node)"
    )
    run_with_stream(
        g_a,
        {"query": "Hello there", "intent": "", "answer": ""},
        "Demo A: add_conditional_edges (router decides next node)"
    )

    # Demo B：短 query -> fast_path；长 query -> strong_path
    run_with_stream(
        g_b,
        {"query": "Explain LangGraph", "model_tier": "", "answer": ""},
        "Demo B: Command(update + goto) (choose_path decides model + next node)"
    )
    run_with_stream(
        g_b,
        {"query": "Please explain how LangGraph schedules nodes in supersteps with fan-out and join edges.",
         "model_tier": "", "answer": ""},
        "Demo B: Command(update + goto) (choose_path decides model + next node)"
    )
