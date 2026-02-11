import sys
sys.path.append("../")
from src.models import load_model
from src.config import load_config
from src.subagents.video_agent import sample_frames
import base64
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import argparse

parser = argparse.ArgumentParser(description="Run multimodal model with configurable model path")
parser.add_argument("--config_path", type=str, required=True, help="Path to the model config YAML file")
args = parser.parse_args()

config_path = args.config_path

modality_list = ["text", "image", "video", "audio"]

# modality_list = ["audio"]

text_path = "./media/text.txt"
image_path = "./media/image.jpeg"
video_path = "./media/video.mp4"
audio_path = "./media/audio.wav"

system_prompt = "You are a helpful assistant."
question = "Please summarize what is described in the provided materials."

for modality in modality_list:
    print("---"*20)
    content = None
    print(f"{modality} Modelity:")
    if modality == "text":
        model_config = load_config(config_path)
        model = load_model(model_config)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        content = {
            "system_prompt": system_prompt,
            "question": question,
            "text": [text]
        }

    if modality == "image":
        model_config = load_config(config_path)
        model = load_model(model_config)

        content = {
            "system_prompt": system_prompt,
            "question": question,
            "image": [image_path]
        }

    if modality == "video":
        model_config = load_config(config_path)
        model = load_model(model_config)

        frames, _, _ = sample_frames(video_path, model_config)
        content = {
            "system_prompt": system_prompt,
            "question": question,
            "image": frames
        }

    if modality == "audio":
        model_config = load_config(config_path)
        model = load_model(model_config)

        content = {
            "system_prompt": system_prompt,
            "question": question,
            "audio": audio_path
        }

    responses = model.media_invoke(content)
    print(responses)

    responses = model.media_batch([content])
    print(responses)
