import os
from .utils import normalize_image_to_data_url, normalize_audio_to_data_url
from .local_models import MODEL_MAP
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import time
from .config import config, load_config
from PIL import Image

def load_model(model_config):
    if model_config is None:
        return None

    api_provider = model_config.get("api_provider", None)
    model_id = model_config.get("model_id", None)
    base_url = model_config.get("base_url", None)
    api_key_name = model_config.get("api_key_name", None)
    params = model_config.get("params", None)

    if "openai" in api_provider:
        from langchain_openai import ChatOpenAI
        return ModelInvokeWrapper(ChatOpenAI(
            base_url=base_url,
            api_key="None" if api_key_name is None else os.environ.get(api_key_name, None),
            model_name=model_id,
            **(params or {}),
        ))
    elif "local" in api_provider:
        return ModelInvokeWrapper(MODEL_MAP[model_id](
            model_id=model_id,
            **(params or {})
        ))
    elif "bedrock" in api_provider:
        from langchain_aws import ChatBedrockConverse
        import boto3
        return ModelInvokeWrapper(ChatBedrockConverse(
            client=boto3.client("bedrock-runtime", region_name="us-east-1"),
            model=model_id,
            **(params or {})
        ))
    else:
        raise ValueError("Incorrect api provider")



class ModelInvokeWrapper:
    def __init__(self, model):
        self.model = model

    def construct_message(self, system_prompt, question, text, image, audio):
        messages = []
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))
        
        content = []
        if text is not None:
            content.append({
                "type": "text",
                "text": str(text)
            })

        if image is not None:
            image = [image] if not isinstance(image, list) else image
            for img in image:
                data_url = normalize_image_to_data_url(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                )

        if audio is not None:
            audio = [audio] if not isinstance(audio, list) else audio
            for aud in audio:
                data_url = normalize_audio_to_data_url(aud)
                content.append(
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": data_url
                        }
                    }
                )

        content.append(
            {
                "type": "text",
                "text": question
            },
        )

        messages.append(HumanMessage(content=content))
        return messages

    def media_invoke(self, content, *args, **kwargs):
        system_prompt = content.get("system_prompt", None)
        question = content.get("question", None)
        text = content.get("text", None)
        image = content.get("image", None)
        audio = content.get("audio", None)
        messages = self.construct_message(system_prompt, question, text, image, audio)

        retry = config["system"]["retry_times"]
        while retry:
            try:
                response = self.model.invoke(messages)
                break
            except Exception as e:
                retry -= 1
                print(e, "remaining retry time:", retry)
                if retry <= 0:
                    raise e
                time.sleep(30)
        return response


    def media_batch(self, contents, *args, **kwargs):
        messages_batch = []
        for content in contents:
            system_prompt = content.get("system_prompt", None)
            question = content.get("question", None)
            text = content.get("text", None)
            image = content.get("image", None)
            audio = content.get("audio", None)
            messages = self.construct_message(system_prompt, question, text, image, audio)
            messages_batch.append(messages)

        retry = config["system"]["retry_times"]
        while retry:
            try:
                responses = self.model.batch(messages_batch)
                break
            except Exception as e:
                retry -= 1
                print(e, "remaining retry time:", retry)
                if retry <= 0:
                    raise e
                time.sleep(30)
        return responses

    def invoke(self, *args, **kwargs):
        return self.model.invoke(*args, **kwargs)

    def batch(self, *args, **kwargs):
        return self.model.batch(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

master_model = load_model(config["model"]["master_agent"])
text_model = load_model(config["model"].get("text_agent", None))
image_model = load_model(config["model"].get("image_agent", None))
video_model = load_model(config["model"].get("video_agent", None))
audio_model = load_model(config["model"].get("audio_agent", None))

