import os
from ..models import video_model as model
from ..state import State
from ..config import config
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from decord import VideoReader

def image_to_base64(image_array):
    image = Image.fromarray(image_array)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def sample_frames(vr, video_config={}, fps=1, force_sample=False):
    if isinstance(vr, str):
        if not os.path.exists(vr):
            raise FileNotFoundError(f"Video file not found: {vr}")
        vr = VideoReader(vr)

    max_frames_num = video_config.get("max_frames_num", 60)
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    raw_frames = vr.get_batch(frame_idx).asnumpy()
    resized_frames = [
        Image.fromarray(frame).resize(video_config.get("image_size", (336, 336)), Image.BICUBIC)
        for frame in raw_frames
    ]
    spare_frames = np.stack(resized_frames)
    spare_frames = [Image.fromarray(image_array) for image_array in spare_frames]
    return spare_frames, frame_time, video_time

def generate(state, questions):
    if questions is None or len(questions) == 0:
        return questions

    system_prompt = """You are a specialized Video Agent in a multi-agent system.  
Your task is to analyze provided video and accurately answer a question based solely on the visual content of that video.

Instructions:

1. You will receive:
   - video (photograph, diagram, screenshot, etc.)
   - A question related to the content of the video 

2. Your job is to:
   - Carefully examine video in detail or focus on the most relevant parts
   - Identify visual evidence or regions that are most relevant to the question
   - Generate a clear, concise, and accurate answer based only on what is visible in video 
   - Unless otherwise instructed, keep your answer as concise and precise as possible

3. Constraints:
   - Do NOT use external knowledge beyond what is visible in videos 
   - Do NOT speculate or hallucinate information not supported by videos 
   - If the answer cannot be found from videos, clearly state that in the answer field
"""

    results = None
    if not isinstance(state["video"], list):
        state["video"] = [state["video"]]
    for video in state['video']:
        frames, _, _  = sample_frames(video, config["model"]["video_agent"])
        image_batch_size = config["model"]["video_agent"].get("max_image_input", len(frames)) # some models has max limit for image input
        for image_batch_begin in range(0, len(frames), image_batch_size):
            messages_batch = []
            for question in questions:
                content = {
                    # "system_prompt": system_prompt,
                    "system_prompt": "You are a helpful assistant.",
                    "question": question,
                    "image": frames[image_batch_begin:image_batch_begin + image_batch_size]
                }
                messages_batch.append(content)
        
            responses = model.media_batch(messages_batch)
            if results is None:
                results = responses
            else:
                for i, response in enumerate(responses):
                    results[i].content += response.content
        return results


def video_agent(state: State):
    if ("video" not in state.keys() or
        state["video"] is None or 
        len(state["video"]) == 0 or
        state["video"][0] == None):
        return {
            "video_agent_result": None,
            "video_agent_result_list": None,
        }


    result_list = []
    for instruction in state.get("reasoning_result").get("agent_instructions", []):
        if instruction.get("agent_name", None) != "video_agent":
            continue

        questions = instruction.get("questions", [])
        results = generate(state, questions)
    
        result_list = [{
            "question": question,
            "answer": answer.content
        } for question, answer in zip(questions, results)]

    # print("=" * 10, "video_agent result", "=" * 10)
    # print(result_list)

    return {
        "video_agent_result": result_list,
        "video_agent_result_list": [result_list] if "video_agent_result_list" not in state.keys() else [*state["video_agent_result_list"], result_list]
    }


def video_summarize(state: State):
    if ("video" not in state.keys() or
        state["video"] is None or 
        len(state["video"]) == 0 or
        state["video"][0] == None):
        return {
            "video_summary": None
        }


    questions = ["Summarize the provided video."]
    result = generate(state, questions)[0]

    # print("=" * 10, "video_agent summary result", "=" * 10)
    # print(result.content)

    return {
        "video_summary": result.content,
    }

