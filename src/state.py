from langgraph.graph import MessagesState
from typing import List, Any, Type, Optional
from pydantic import BaseModel


class State(MessagesState):
    query: str
    cur_round_num: int
    max_round_num: int

    final_answer_structure: Optional[Type[BaseModel]]
    field_by_field: bool

    reasoning_result: Any
    decision_result: Any

    text_summary: Any
    image_summary: Any
    video_summary: Any
    audio_summary: Any

    text_agent_result: Any
    image_agent_result: Any
    video_agent_result: Any
    audio_agent_result: Any

    text_agent_result_list: Any
    image_agent_result_list: Any
    video_agent_result_list: Any
    audio_agent_result_list: Any

    reasoning_result_list: Any
    decision_result_list: Any
    reasoning_prompt_list: Any
    decision_prompt_list: Any

    text: List[Any]
    image: List[Any]
    video: List[Any]
    audio: List[Any]
