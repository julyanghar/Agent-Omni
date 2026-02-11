"""
模型加载和调用模块
该模块负责加载和管理不同的AI模型，支持多种API提供商（OpenAI、本地模型、AWS Bedrock、Google等）
并提供统一的调用接口来处理多模态输入（文本、图像、音频）
"""

import os
import base64
from .utils import normalize_image_to_data_url, normalize_audio_to_data_url
from .local_models import MODEL_MAP
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import time
from .config import config, load_config
from PIL import Image

def extract_base64_from_data_url(data_url: str) -> str:
    """
    从 data URL 中提取 base64 字符串
    
    参数:
        data_url: data URL 字符串，格式如 "data:audio/wav;base64,<base64_string>"
    
    返回:
        base64 字符串（不包含前缀）
    """
    if data_url.startswith("data:"):
        # 找到 base64 数据部分
        if "," in data_url:
            return data_url.split(",", 1)[1]
    # 如果不是 data URL，直接返回（可能是 URL 或其他格式）
    return data_url

def load_model(model_config):
    """
    根据配置加载相应的AI模型

    参数:
        model_config (dict): 模型配置字典，包含以下键：
            - api_provider: API提供商（openai/local/bedrock/google）
            - model_id: 模型标识符
            - base_url: API基础URL（可选）
            - api_key_name: 环境变量中API密钥的名称（可选）
            - params: 额外的模型参数（可选）

    返回:
        ModelInvokeWrapper: 包装后的模型对象，如果配置为None则返回None
    """
    if model_config is None:
        return None

    # 从配置中提取模型参数
    api_provider = model_config.get("api_provider", None)
    model_id = model_config.get("model_id", None)
    base_url = model_config.get("base_url", None)
    api_key_name = model_config.get("api_key_name", None)
    params = model_config.get("params", None)

    # 根据API提供商类型加载对应的模型
    if "openai" in api_provider:
        # 加载OpenAI兼容的模型（包括OpenAI官方API和兼容接口）
        from langchain_openai import ChatOpenAI
        return ModelInvokeWrapper(ChatOpenAI(
            base_url=base_url,
            api_key="None" if api_key_name is None else os.environ.get(api_key_name, None),
            model_name=model_id,
            **(params or {}),
        ), api_provider=api_provider)
    elif "local" in api_provider:
        # 加载本地模型（从MODEL_MAP中获取）
        return ModelInvokeWrapper(MODEL_MAP[model_id](
            model_id=model_id,
            **(params or {})
        ), api_provider=api_provider)
    elif "bedrock" in api_provider:
        # 加载AWS Bedrock模型
        from langchain_aws import ChatBedrockConverse
        import boto3
        return ModelInvokeWrapper(ChatBedrockConverse(
            client=boto3.client("bedrock-runtime", region_name="us-east-1"),
            model=model_id,
            **(params or {})
        ), api_provider=api_provider)
    elif "google" in api_provider:
        # 加载Google Generative AI模型（如Gemini）
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get(api_key_name, None) if api_key_name else None
        if api_key is None:
            raise ValueError(f"API key not found in environment variable: {api_key_name}")
        return ModelInvokeWrapper(ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_id,
            **(params or {})
        ), api_provider=api_provider)
    else:
        raise ValueError("Incorrect api provider")



class ModelInvokeWrapper:
    """
    模型调用包装器类
    为不同的AI模型提供统一的调用接口，支持多模态输入（文本、图像、音频）
    并提供重试机制以提高调用的可靠性
    """

    def __init__(self, model, api_provider=None):
        """
        初始化模型包装器

        参数:
            model: 底层的AI模型实例（来自langchain或其他框架）
            api_provider: API提供商类型（openai/local/bedrock/google），用于确定消息格式
        """
        self.model = model
        self.api_provider = api_provider

    def construct_message(self, system_prompt, question, text, image, audio):
        """
        构建多模态消息列表

        参数:
            system_prompt (str): 系统提示词，用于设定AI的行为和角色
            question (str): 用户的问题或指令
            text (str): 额外的文本内容
            image (str/list): 图像路径或图像路径列表
            audio (str/list): 音频路径或音频路径列表

        返回:
            list: 包含SystemMessage和HumanMessage的消息列表
        """
        messages = []
        # 如果有系统提示词，添加到消息列表
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))

        # 构建用户消息的内容列表（支持多模态）
        content = []

        # 添加文本内容
        if text is not None:
            content.append({
                "type": "text",
                "text": str(text)
            })

        # 添加图像内容（支持单个或多个图像）
        if image is not None:
            # 如果不是列表，转换为列表以统一处理
            image = [image] if not isinstance(image, list) else image
            for img in image:
                # 将图像转换为data URL格式
                data_url = normalize_image_to_data_url(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                )

        # 添加音频内容（支持单个或多个音频）
        if audio is not None:
            # 如果不是列表，转换为列表以统一处理
            audio = [audio] if not isinstance(audio, list) else audio
            for aud in audio:
                # 将音频转换为data URL格式
                data_url = normalize_audio_to_data_url(aud)
                
                # 对于 Gemini/Google 模型，使用 file 格式（符合 langchain_google_genai 的要求）
                if self.api_provider and "google" in self.api_provider:
                    # 从 data URL 中提取 base64 字符串
                    base64_data = extract_base64_from_data_url(data_url)
                    content.append(
                        {
                            "type": "file",
                            "source_type": "base64",
                            "mime_type": "audio/wav",
                            "data": base64_data
                        }
                    )
                else:
                    # 对于其他模型，使用 audio_url 格式
                    content.append(
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": data_url
                            }
                        }
                    )

        # 最后添加用户的问题
        content.append(
            {
                "type": "text",
                "text": question
            },
        )

        # 将所有内容封装为HumanMessage
        messages.append(HumanMessage(content=content))
        return messages

    def media_invoke(self, content, *args, **kwargs):
        """
        调用模型处理多模态输入（单次调用）

        参数:
            content (dict): 包含多模态内容的字典，可包含以下键：
                - system_prompt: 系统提示词
                - question: 用户问题
                - text: 文本内容
                - image: 图像内容
                - audio: 音频内容

        返回:
            response: 模型的响应结果

        异常:
            如果重试次数用尽仍然失败，则抛出最后一次的异常
        """
        # 从content字典中提取各个组件
        system_prompt = content.get("system_prompt", None)
        question = content.get("question", None)
        text = content.get("text", None)
        image = content.get("image", None)
        audio = content.get("audio", None)
        # 构建消息列表
        messages = self.construct_message(system_prompt, question, text, image, audio)

        # 实现重试机制，从配置中获取重试次数
        retry = config["system"]["retry_times"]
        while retry:
            try:
                # 尝试调用模型
                response = self.model.invoke(messages)
                break  # 成功则跳出循环
            except Exception as e:
                retry -= 1
                print(e, "remaining retry time:", retry)
                if retry <= 0:
                    raise e  # 重试次数用尽，抛出异常
                time.sleep(15)  # 等待15秒后重试
        return response


    def media_batch(self, contents, *args, **kwargs):
        """
        批量调用模型处理多模态输入（批处理）

        参数:
            contents (list): 包含多个content字典的列表，每个字典格式同media_invoke

        返回:
            responses: 模型的批量响应结果列表

        异常:
            如果重试次数用尽仍然失败，则抛出最后一次的异常
        """
        # 为每个content构建消息列表
        messages_batch = []
        for content in contents:
            system_prompt = content.get("system_prompt", None)
            question = content.get("question", None)
            text = content.get("text", None)
            image = content.get("image", None)
            audio = content.get("audio", None)
            messages = self.construct_message(system_prompt, question, text, image, audio)
            messages_batch.append(messages)

        # 实现重试机制，从配置中获取重试次数
        retry = config["system"]["retry_times"]
        while retry:
            try:
                # 尝试批量调用模型
                responses = self.model.batch(messages_batch)
                break  # 成功则跳出循环
            except Exception as e:
                retry -= 1
                print(e, "remaining retry time:", retry)
                if retry <= 0:
                    raise e  # 重试次数用尽，抛出异常
                time.sleep(30)  # 等待30秒后重试
        return responses

    def invoke(self, *args, **kwargs):
        """
        直接调用底层模型的invoke方法（不经过多模态处理）

        返回:
            模型的原始响应
        """
        return self.model.invoke(*args, **kwargs)

    def batch(self, *args, **kwargs):
        """
        直接调用底层模型的batch方法（不经过多模态处理）

        返回:
            模型的批量响应
        """
        return self.model.batch(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        使包装器对象可直接调用，转发到底层模型

        返回:
            模型的响应
        """
        return self.model(*args, **kwargs)


# 从配置文件中加载各种专用模型
# master_model: 主控模型，负责任务分解和协调
master_model = load_model(config["model"]["master_agent"])
# text_model: 文本处理专用模型
text_model = load_model(config["model"].get("text_agent", None))
# image_model: 图像处理专用模型
image_model = load_model(config["model"].get("image_agent", None))
# video_model: 视频处理专用模型
video_model = load_model(config["model"].get("video_agent", None))
# audio_model: 音频处理专用模型
audio_model = load_model(config["model"].get("audio_agent", None))


