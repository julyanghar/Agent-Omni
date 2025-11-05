from .state import State
from langgraph.graph import StateGraph, START, END
from .nodes import master_reasoning, master_dispatcher, master_decision, master_dispatcher_1
from .subagents.text_agent import text_agent, text_summarize
from .subagents.image_agent import image_agent, image_summarize
from .subagents.video_agent import video_agent, video_summarize
from .subagents.audio_agent import audio_agent, audio_summarize
from .config import config

SUBAGENT_NAMES = config["agents"]["names"]

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

    for node in [
        master_reasoning,
        master_dispatcher,
        master_dispatcher_1,
        master_decision,
        text_agent,
        image_agent,
        video_agent,
        audio_agent,
        text_summarize,
        image_summarize,
        video_summarize,
        audio_summarize,
    ]:
        builder.add_node(node)

    builder.add_edge(START, text_summarize.__name__)
    builder.add_edge(START, image_summarize.__name__)
    builder.add_edge(START, video_summarize.__name__)
    builder.add_edge(START, audio_summarize.__name__)

    builder.add_edge(text_summarize.__name__, master_dispatcher_1.__name__)
    builder.add_edge(image_summarize.__name__, master_dispatcher_1.__name__)
    builder.add_edge(video_summarize.__name__, master_dispatcher_1.__name__)
    builder.add_edge(audio_summarize.__name__, master_dispatcher_1.__name__)

    builder.add_edge(master_dispatcher_1.__name__, master_reasoning.__name__)

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
