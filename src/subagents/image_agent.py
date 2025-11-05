from ..models import image_model as model
from ..state import State
from ..config import config
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage

def generate(state, questions):
    if questions is None or len(questions) == 0:
        return questions

    system_prompt = """You are a specialized Image Agent in a multi-agent system.  
Your task is to analyze provided images and accurately answer a question based solely on the visual content of that image.

Instructions:

1. You will receive:
   - Images (photograph, diagram, screenshot, etc.)
   - A question related to the content of the image

2. Your job is to:
   - Carefully examine images in detail or focus on the most relevant parts
   - Identify visual evidence or regions that are most relevant to the question
   - Generate a clear, concise, and accurate answer based only on what is visible in images
   - Unless otherwise instructed, keep your answer as concise and precise as possible

3. Constraints:
   - Do NOT use external knowledge beyond what is visible in images
   - Do NOT speculate or hallucinate information not supported by images
   - If the answer cannot be found from images, clearly state that in the answer field
"""

    results = None
    image_batch_size = config["model"]["image_agent"].get("max_image_input", len(state['image'])) # some models has max limit for image input
    if not isinstance(state["image"], list):
        state["image"] = [state["image"]]
    for image_batch_begin in range(0, len(state['image']), image_batch_size):
        messages_batch = []
        for question in questions:
            content = {
                # "system_prompt": system_prompt,
                "question": question,
                "image": state['image'][image_batch_begin:image_batch_begin + image_batch_size]
            }
            messages_batch.append(content)

        responses = model.media_batch(messages_batch)
        if results is None:
            results = responses
        else:
            for i, response in enumerate(responses):
                results[i].content += response.content
        return results


def image_agent(state: State):
    if ("image" not in state.keys() or
        state["image"] is None or 
        len(state["image"]) == 0 or
        state["image"][0] == None):
        return {
            "image_agent_result": None,
            "image_agent_result_list": None,
        }


    result_list = []
    for instruction in state.get("reasoning_result").get("agent_instructions", []):
        if instruction.get("agent_name", None) != "image_agent":
            continue

        questions = instruction.get("questions", [])
        results = generate(state, questions)
    
        result_list = [{
            "question": question,
            "answer": answer.content
        } for question, answer in zip(questions, results)]

#     print("=" * 10, "image_agent result", "=" * 10)
#     print(result_list)

    return {
        "image_agent_result": result_list,
        "image_agent_result_list": [result_list] if "image_agent_result_list" not in state.keys() else [*state["image_agent_result_list"], result_list]
    }


def image_summarize(state: State):
    if ("image" not in state.keys() or
        state["image"] is None or 
        len(state["image"]) == 0 or
        state["image"][0] == None):
        return {
            "image_summary": None
        }


    questions = ["Summarize the provided image."]
    result = generate(state, questions)[0]

    return {
        "image_summary": result.content,
    }

