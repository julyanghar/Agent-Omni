# Gemini API 音频格式问题解决方案

## 问题描述

在使用 `langchain_google_genai` 调用 Gemini API 处理音频输入时，遇到以下错误：

```
ValueError: Unrecognized message part type: audio_url.
```

或

```
ValueError: Unrecognized message part type: inline_data.
```

## 问题原因

`langchain_google_genai` 库不支持以下音频格式：
- ❌ `audio_url` 格式（OpenAI 风格）
- ❌ `inline_data` 格式（直接内联数据）

Gemini API 需要使用特定的 `file` 格式来处理音频输入。

## 解决方案

### 正确的音频格式

对于 Gemini/Google 模型，音频应使用以下格式：

```python
{
    "type": "file",
    "source_type": "base64",
    "mime_type": "audio/wav",
    "data": base64_string  # 纯 base64 字符串，不包含 data URL 前缀
}
```

### 实现步骤

#### 1. 修改 `load_model` 函数

在创建 `ModelInvokeWrapper` 时传递 `api_provider` 参数：

```python
def load_model(model_config):
    # ... 其他代码 ...
    api_provider = model_config.get("api_provider", None)
    
    if "google" in api_provider:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # ... 创建模型 ...
        return ModelInvokeWrapper(
            ChatGoogleGenerativeAI(...),
            api_provider=api_provider  # 传递 api_provider
        )
    # ... 其他提供商类似处理 ...
```

#### 2. 修改 `ModelInvokeWrapper.__init__` 方法

接收并存储 `api_provider` 信息：

```python
class ModelInvokeWrapper:
    def __init__(self, model, api_provider=None):
        self.model = model
        self.api_provider = api_provider  # 存储 api_provider
```

#### 3. 添加辅助函数

从 data URL 中提取 base64 字符串：

```python
def extract_base64_from_data_url(data_url: str) -> str:
    """
    从 data URL 中提取 base64 字符串
    
    参数:
        data_url: data URL 字符串，格式如 "data:audio/wav;base64,<base64_string>"
    
    返回:
        base64 字符串（不包含前缀）
    """
    if data_url.startswith("data:"):
        if "," in data_url:
            return data_url.split(",", 1)[1]
    return data_url
```

#### 4. 修改 `construct_message` 方法

根据模型类型使用不同的音频格式：

```python
def construct_message(self, system_prompt, question, text, image, audio):
    # ... 其他代码 ...
    
    # 添加音频内容
    if audio is not None:
        audio = [audio] if not isinstance(audio, list) else audio
        for aud in audio:
            data_url = normalize_audio_to_data_url(aud)
            
            # 对于 Gemini/Google 模型，使用 file 格式
            if self.api_provider and "google" in self.api_provider:
                base64_data = extract_base64_from_data_url(data_url)
                content.append({
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "audio/wav",
                    "data": base64_data
                })
            else:
                # 对于其他模型，使用 audio_url 格式
                content.append({
                    "type": "audio_url",
                    "audio_url": {
                        "url": data_url
                    }
                })
    
    # ... 其他代码 ...
```

## 完整代码示例

### 使用示例

```python
from src.models import load_model
from src.config import load_config

# 加载 Gemini 模型配置
config = load_config("configs/model_configs/gemini-2.0-flash.yaml")
model = load_model(config)

# 准备音频内容
content = {
    "system_prompt": "You are a helpful assistant.",
    "question": "Please summarize this audio.",
    "audio": "./media/audio.wav"  # 音频文件路径
}

# 调用模型
response = model.media_invoke(content)
print(response.content)
```

## 格式对比

### ❌ 错误的格式

```python
# 错误 1: audio_url 格式（OpenAI 风格）
{
    "type": "audio_url",
    "audio_url": {
        "url": "data:audio/wav;base64,..."
    }
}

# 错误 2: inline_data 格式
{
    "type": "inline_data",
    "inline_data": {
        "mime_type": "audio/wav",
        "data": "base64_string"
    }
}
```

### ✅ 正确的格式

```python
# Gemini API 正确格式
{
    "type": "file",
    "source_type": "base64",
    "mime_type": "audio/wav",
    "data": "base64_string"  # 纯 base64，无前缀
}
```

## 支持的音频格式

根据 `langchain_google_genai` 文档，支持的音频 MIME 类型包括：
- `audio/wav`
- `audio/mp3`
- `audio/mpeg`
- `audio/ogg`
- 其他常见音频格式

## 注意事项

1. **base64 数据格式**：必须提供纯 base64 字符串，不能包含 `data:audio/wav;base64,` 前缀
2. **MIME 类型**：确保 `mime_type` 与实际音频格式匹配
3. **向后兼容**：对于非 Gemini 模型，继续使用 `audio_url` 格式以保持兼容性
4. **文件大小限制**：注意 Gemini API 对音频文件大小的限制

## 参考文档

- [LangChain Google GenAI 文档](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
- [Gemini API 文档](https://ai.google.dev/docs)
- `langchain_google_genai/chat_models.py` 第 1711-1729 行（音频输入示例）

## 相关文件

- `src/models.py` - 主要实现文件
- `src/utils.py` - 包含 `normalize_audio_to_data_url` 函数
- `configs/model_configs/gemini-2.0-flash.yaml` - Gemini 模型配置示例

## 测试验证

修复后，可以通过以下方式验证：

```python
# 运行测试脚本
python test/vanilla_test.py --config_path test/config.yaml
```

如果配置了音频模态，应该能够成功处理音频输入而不会出现格式错误。

