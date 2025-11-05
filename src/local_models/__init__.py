from .qwen2_audio import Qwen2AudioModel
from .text_model import TextModel


MODEL_MAP = {
    "Qwen/Qwen2-Audio-7B-Instruct": Qwen2AudioModel,
    "openai/gpt-oss-20b": TextModel,
    "Qwen/Qwen3-4B-Instruct-2507": TextModel,
    "Qwen/Qwen3-4B-Thinking-2507": TextModel,
}
