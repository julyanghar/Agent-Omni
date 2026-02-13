from ..models import text_model as model
from ..config import config
from ..state import State
from ..utils import postprocessing
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_aws.chat_models.bedrock import convert_messages_to_prompt_llama3
import json
import re


def generate(state, questions):
    if questions is None or len(questions) == 0:
        return questions

    system_prompt = """You are a specialized Text Agent in a multi-agent system.
Your task is to read and analyze a long text document and accurately answer a question based solely on the content of that document.

Instructions:

1. You will receive:
   - A text document (optional, may be very long)
   - A question related to the document or topic

2. YOUR JOB IS TO:
   - If a document is provided:
     - Carefully read and comprehend the entire document or the most relevant parts
     - Identify evidence or passages most relevant to the question
     - Generate a clear, concise, and accurate answer based solely on the information in the document
     - If the answer cannot be found in the document, explicitly state: “The document does not contain enough information to answer this question.”

   - If no related document is provided:
     - Use your own internal knowledge to answer the question to the best of your ability
     - Be clear and accurate; avoid speculation

3. Constraints:
   - Do NOT use external knowledge or assumptions when a document is provided
   - Do NOT fabricate or hallucinate facts not supported by the text
   - Keep your answer concise and precise, unless otherwise instructed
"""
    messages_batch = []
    for question in questions:
        content = {
            # "system_prompt": system_prompt,
            "question": question,
            "text": state["text"]
        }
        messages_batch.append(content)

    results = model.media_batch(messages_batch)
    return results


def text_agent(state: State):
    import time
    start = time.time()

    if ("text" not in state.keys() or
        state["text"] is None or 
        len(state["text"]) == 0):
        return {
            "text_agent_result": None,
            "text_agent_result_list": None,
        }


    result_list = []
    for instruction in state.get("reasoning_result").get("agent_instructions", []):
        if instruction.get("agent_name", None) != "text_agent":
            continue

        questions = instruction.get("questions", [])
    
        results = generate(state, questions)

        result_list = [{
            "question": question,
            "answer": postprocessing(answer)
        } for question, answer in zip(questions, results)]

#     print("=" * 10, "text_agent result", "=" * 10)
#     print(result_list)
    end = time.time()
    # print(f"[text_agent] Time taken: {end - start:.3f}s")

    return {
        "text_agent_result": result_list,
        "text_agent_result_list": [result_list] if "text_agent_result_list" not in state.keys() else [*state["text_agent_result_list"], result_list]
    }


class TextSummarizeNode:
    """
    类节点版本的文本摘要节点，实现 __call__ 以便用于 StateGraph。
    """

    def __init__(self, cfg):
        self.config = cfg

    def __call__(self, state: State):
        import time
        start = time.time()

        if ("text" not in state.keys() or
            state["text"] is None or 
            len(state["text"]) == 0):
            return {
                "text_summary": None
            }

        questions = ["Summarize the provided text."]

        result = generate(state, questions)[0]

#         print("=" * 10, "text_agent result", "=" * 10)
#         print(result.content)
        end = time.time()
        # print(f\"[text_summary] Time taken: {end - start:.3f}s\")

        return {
            "text_summary": postprocessing(result),
        }


# 单例实例，供 graph 使用
text_summarize_node = TextSummarizeNode(config)


def text_summarize(state: State):
    """
    兼容旧接口的函数封装，内部委托给 TextSummarizeNode 实例。
    """
    return text_summarize_node(state)

