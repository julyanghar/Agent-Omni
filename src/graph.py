from .state import State
from langgraph.graph import StateGraph, START, END
from .nodes import master_reasoning, master_dispatcher, master_decision, master_dispatcher_1_node
from .subagents.text_agent import text_agent, text_summarize_node
# NOTE: 当前版本未在图中使用 image_agent / image_summarize_node，如需恢复请在此重新导入并注册节点。
# from .subagents.image_agent import image_agent, image_summarize_node
from .subagents.video_agent import video_agent, video_summarize_node
from .subagents.audio_agent import audio_agent, audio_summarize_node
from .config import config

SUBAGENT_NAMES = config["agents"]["names"]

# 总结类节点的名称常量，供 add_node / add_edge 复用
TEXT_SUMMARIZE_NODE = "text_summarize"
# IMAGE_SUMMARIZE_NODE = "image_summarize"  # 图像模态已禁用
VIDEO_SUMMARIZE_NODE = "video_summarize"
AUDIO_SUMMARIZE_NODE = "audio_summarize"
MASTER_DISPATCHER_1_NODE = "master_dispatcher_1"


def agent_selected(agent_name: str):
    def condition(state: State) -> bool:
        # return agent_name in state.get("reasoning_result_json", {}).get("agent_questions", {}).keys()
        return True

    return condition


def next_round(state):
    # return state["decision_result"]["is_final"] == False and state["max_round_num"] > state["cur_round_num"]
    return state["max_round_num"] > state["cur_round_num"]


def _build_graph():
    builder = StateGraph(State)

    # 仍然使用函数形式的节点
    for node in [
        master_reasoning,
        master_dispatcher,
        master_decision,
        text_agent,
        # image_agent,  # 图像模态已禁用
        video_agent,
        audio_agent,
    ]:
        builder.add_node(node)

    # 使用类实例作为节点，并指定节点名
    # 注意：LangGraph 的 add_node(name, node) 需要先传节点名，再传节点对象
    builder.add_node(TEXT_SUMMARIZE_NODE, text_summarize_node)
    builder.add_node(VIDEO_SUMMARIZE_NODE, video_summarize_node)
    builder.add_node(AUDIO_SUMMARIZE_NODE, audio_summarize_node)
    builder.add_node(MASTER_DISPATCHER_1_NODE, master_dispatcher_1_node)

    # 从 START 到各个 summarize 节点
    builder.add_edge(START, TEXT_SUMMARIZE_NODE)
    builder.add_edge(START, VIDEO_SUMMARIZE_NODE)
    builder.add_edge(START, AUDIO_SUMMARIZE_NODE)

    # summarize 节点到 master_dispatcher_1
    # 分析意图只需要text_summary，不需要video_summary和audio_summary
    builder.add_edge(
        [TEXT_SUMMARIZE_NODE, VIDEO_SUMMARIZE_NODE, AUDIO_SUMMARIZE_NODE],
        MASTER_DISPATCHER_1_NODE,
    )

    builder.add_edge(MASTER_DISPATCHER_1_NODE, master_reasoning.__name__)

    for agent_name in SUBAGENT_NAMES:
        builder.add_edge(
            master_reasoning.__name__, agent_name
        )  # TODO: Consider conditional edge
        builder.add_edge(agent_name, master_dispatcher.__name__)

    builder.add_edge(master_dispatcher.__name__, master_decision.__name__)
    # builder.add_edge(master_decision.__name__, END)

    builder.add_conditional_edges(
        master_decision.__name__,
        next_round,
        {
            True: master_reasoning.__name__,
            False: END,
        }
    )

    return builder.compile()


_graph = _build_graph()
