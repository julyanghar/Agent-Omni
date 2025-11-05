from ..models import audio_model as model
from ..state import State
from ..config import config
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage

def generate(state, questions):
    if questions is None or len(questions) == 0:
        return questions

    system_prompt = """You are a specialized Audio Agent in a multi-agent system.  
Your task is to analyze provided audio and accurately answer a question based solely on the content of that audio.

Instructions:

1. You will receive:
   - audio (recording, voice note, podcast, etc.)
   - A question related to the content of the audio 

2. Your job is to:
   - Carefully examine audios in detail or focus on the most relevant parts
   - Identify audio evidence or regions that are most relevant to the question
   - Generate a clear, concise, and accurate answer based only on what is visible in audios
   - Unless otherwise instructed, keep your answer as concise and precise as possible

3. Constraints:
   - Do NOT use external knowledge beyond what is visible in audios
   - Do NOT speculate or hallucinate information not supported by audios
   - If the answer cannot be found from audios, clearly state that in the answer field
"""

    results = None
    if not isinstance(state["audio"], list):
        state["audio"] = [state["audio"]]
    audio_batch_size = config["model"]["audio_agent"].get("max_audio_input", 1) # some models has max limit for audio input
    for audio_batch_begin in range(0, len(state['audio']), audio_batch_size):
        messages_batch = []
        for question in questions:
            content = {
                # "system_prompt": system_prompt,
                "system_prompt": "You are a helpful assistant.",
                "question": question,
                "audio": state['audio'][audio_batch_begin:audio_batch_begin + audio_batch_size]
            }
            messages_batch.append(content)

        responses = model.media_batch(messages_batch)
        if results is None:
            results = responses
        else:
            for i, response in enumerate(responses):
                results[i].content += response.content
    return results


def audio_agent(state: State):
    if ("audio" not in state.keys() or
        state["audio"] is None or 
        len(state["audio"]) == 0 or
        state["audio"][0] == None):
        return {
            "audio_agent_result": None,
            "audio_agent_result_list": None,
        }


    result_list = []
    for instruction in state.get("reasoning_result").get("agent_instructions", []):
        if instruction.get("agent_name", None) != "audio_agent":
            continue

        questions = instruction.get("questions", [])
        results = generate(state, questions)
    
        result_list = [{
            "question": question,
            "answer": answer.content
        } for question, answer in zip(questions, results)]

#     print("=" * 10, "audio_agent result", "=" * 10)
#     print(result_list)

    return {
        "audio_agent_result": result_list,
        "audio_agent_result_list": [result_list] if "audio_agent_result_list" not in state.keys() else [*state["audio_agent_result_list"], result_list]
    }


def audio_summarize(state: State):
    if ("audio" not in state.keys() or
        state["audio"] is None or 
        len(state["audio"]) == 0 or
        state["audio"][0] == None):
        return {
            "audio_summary": None
        }


    questions = ["Summarize the provided audio."]
    result = generate(state, questions)[0]

#     print("=" * 10, "sumarize audio_agent", "=" * 10)
#     print(result)

    return {
        "audio_summary": result.content,
    }

