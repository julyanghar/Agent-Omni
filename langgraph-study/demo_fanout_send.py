from __future__ import annotations

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# 1) 定义图的全局 State
class State(TypedDict):
    query: str
    # router 产出的多个任务（动态数量）
    tasks: list[dict]  # e.g. [{"agent":"text_agent","question":"..."}, ...]
    # 并行分支返回的结果，使用 reducer 累加
    results: Annotated[list[str], operator.add]
    final: str


# 2) router：根据 query 动态决定进入哪些分支（这里模拟：总是进 3 个“agent任务”）
def router(state: State):
    q = state["query"]
    tasks = [
        {"agent": "text_agent", "question": f"[text] summarize: {q}"},
        {"agent": "tool_agent", "question": f"[tool] extract keywords: {q}"},
        {"agent": "judge_agent", "question": f"[judge] classify intent: {q}"},
    ]
    return {"tasks": tasks}


# 3) fan-out planner：把 tasks 变成 Send 列表（动态进入多个分支）
def fanout(state: State):
    sends = []
    for t in state["tasks"]:
        # 每个 Send 都给 worker 一个“局部state”，相互独立
        sends.append(Send("worker", {"task": t}))
    return sends


# 4) worker：模拟 subagent 执行（真实情况你会在这里调用不同 LLM/工具/子图）
def worker(state):
    t = state["task"]
    agent = t["agent"]
    question = t["question"]
    # 模拟不同 agent 的输出
    out = f"{agent} -> {question} -> (result)"
    return {"results": [out]}


# 5) reduce：所有并行结果汇总成最终答案
def reduce_node(state: State):
    # 注意：results 是 reducer 累加出来的 list
    joined = "\n".join(state["results"])
    return {"final": f"FINAL SUMMARY:\n{joined}"}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("router", router)
    builder.add_node("worker", worker)
    builder.add_node("reduce", reduce_node)

    builder.add_edge(START, "router")

    # 关键：router 后用 conditional edges 做动态 fan-out
    # 返回 Send 列表（map-reduce模式）
    builder.add_conditional_edges("router", fanout)

    # worker 执行完，把结果汇聚到 reduce
    builder.add_edge("worker", "reduce")
    builder.add_edge("reduce", END)

    return builder.compile()


if __name__ == "__main__":
    g = build_graph()

    init = {"query": "Explain how to route to multiple agents.", "tasks": [], "results": [], "final": ""}
    # stream 方便你观察每一步 state
    for s in g.stream(init, stream_mode="values"):
        print("---- step ----")
        print(s)

    print("\n=== FINAL ===")
    print(g.invoke(init)["final"])
