import sys
sys.path.append("../")
import os
os.environ["CONFIG_PATH"] = "./config.yaml"

from src.state import State
from src.graph import _graph
from src.config import config

modality_list = ["text", "image", "video", "audio", "omni"]


text_path = None
image_path = None
video_path = None
audio_path = None

text_path = "./media/text.txt"
# image_path = "./media/image.jpeg"
video_path = "./media/video.mp4"
# audio_path = "./media/audio.wav"

system_prompt = "You are a helpful assistant."
question = "Please summarize what is described in the provided materials."

state = State(messages=[], max_round_num=3)
state["query"] = question

for modality in modality_list:
    print("---"*20)
    content = None
    print(f"{modality} Modelity:")

    text = None
    image = None
    video = None
    audio = None

    if modality in ["text", "omni"]:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            state["text"] = text

    if modality in ["image", "omni"]:
        image = image_path
        state["image"] = image

    if modality in ["video", "omni"]:
        video = video_path
        state["video"] = video

    if modality in ["audio", "omni"]:
        audio = audio_path
        state["audio"] = audio




end_state = _graph.batch([state])
# print(end_state)
print(end_state[0]["decision_result"])
