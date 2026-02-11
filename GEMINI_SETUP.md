# Google Gemini 2.5-Flash 集成指南

本指南说明如何在 Agent-Omni 项目中使用 Google Gemini 2.5-Flash 模型。

## 前置要求

1. **Python 环境**：已安装 `agent-omni` conda 环境
2. **依赖包**：已安装 `langchain-google-genai`（如果未安装，运行 `pip install langchain-google-genai`）
3. **Google API Key**：需要从 Google AI Studio 获取

## 步骤 1：获取 Google API Key

1. 访问 [Google AI Studio](https://aistudio.google.com/)
2. 登录你的 Google 账号
3. 点击 "Get API Key" 或 "Create API Key"
4. 创建新的 API key 或使用现有的 key
5. 复制 API key（格式类似：`AIzaSy...`）

## 步骤 2：设置环境变量

将 API key 设置为环境变量：

```bash
# Linux/Mac
export GOOGLE_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your-api-key-here"

# Windows (CMD)
set GOOGLE_API_KEY=your-api-key-here
```

**永久设置（推荐）**：

在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

然后运行：
```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

## 步骤 3：配置文件说明

配置文件位于 `configs/model_configs/gemini-2.5-flash.yaml`：

```yaml
api_provider: google
model_id: gemini-2.0-flash-exp
api_key_name: GOOGLE_API_KEY
params:
  temperature: 0
  max_tokens: 8192
```

### 配置参数说明

- **api_provider**: 必须设置为 `google`
- **model_id**: Gemini 模型名称
  - `gemini-2.0-flash-exp` - Gemini 2.0 Flash（实验版，推荐）
  - `gemini-1.5-flash` - Gemini 1.5 Flash（稳定版）
  - `gemini-1.5-pro` - Gemini 1.5 Pro（更强大但更慢）
  - 注意：Gemini 2.5-Flash 可能尚未正式发布，请查看 [Google AI Studio](https://aistudio.google.com/) 获取最新模型名称
- **api_key_name**: 环境变量名称（默认：`GOOGLE_API_KEY`）
- **params**: 模型参数
  - `temperature`: 0-1，控制随机性（0=确定性，1=创造性）
  - `max_tokens`: 最大生成 token 数

## 步骤 4：在 Agent-Omni 中使用

### 作为 Master Agent

编辑 `test/config.yaml`：

```yaml
model:
  master_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  text_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  image_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  # ... 其他 agent
```

### 作为特定子 Agent

```yaml
model:
  master_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
  text_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  image_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  video_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  audio_agent: ../configs/model_configs/gemini-2.5-flash.yaml
```

## 步骤 5：运行测试

```bash
cd test
python agent_test.py
```

## 多模态支持

Gemini 2.0 Flash 支持以下模态：

- ✅ **文本**：完全支持
- ✅ **图像**：支持（通过 `image_url` 格式）
- ⚠️ **音频**：可能不支持或需要特殊处理（需要测试）
- ✅ **视频**：通过帧采样后作为图像处理

## 注意事项

1. **API 限制**：
   - Google API 有速率限制（RPM - Requests Per Minute）
   - 免费额度有限，超出后需要付费
   - 查看 [Google AI Studio](https://aistudio.google.com/) 了解当前配额

2. **成本**：
   - Gemini 2.0 Flash 是付费 API
   - 提供免费额度用于测试
   - 建议监控 API 使用量

3. **模型名称**：
   - 模型名称可能随时间变化
   - 如果 `gemini-2.0-flash-exp` 不可用，尝试：
     - `gemini-1.5-flash`
     - `gemini-1.5-pro`
     - 或查看 Google AI Studio 获取最新模型列表

4. **依赖冲突**：
   - `langchain-google-genai` 可能需要较新版本的 `langchain-core`
   - 如果遇到依赖冲突，代码仍可能运行，但建议关注警告信息

5. **错误处理**：
   - Agent-Omni 已内置重试机制（`retry_times: 10`）
   - API 错误会自动重试

## 故障排除

### 问题 1：API Key 未找到

**错误信息**：
```
ValueError: API key not found in environment variable: GOOGLE_API_KEY
```

**解决方案**：
1. 确认环境变量已设置：`echo $GOOGLE_API_KEY`
2. 确认在正确的 conda 环境中运行
3. 重启终端或重新加载环境变量

### 问题 2：模型名称无效

**错误信息**：
```
404: Model 'gemini-2.0-flash-exp' not found
```

**解决方案**：
1. 检查模型名称是否正确
2. 访问 [Google AI Studio](https://aistudio.google.com/) 查看可用模型
3. 尝试使用 `gemini-1.5-flash` 作为替代

### 问题 3：依赖冲突

**警告信息**：
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

**解决方案**：
- 这通常是警告，不影响运行
- 如果遇到运行时错误，可能需要升级相关包：
  ```bash
  pip install --upgrade langchain langchain-core
  ```

## 测试脚本

创建一个简单的测试脚本验证集成：

```python
# test_gemini.py
import os
import sys
sys.path.append("src")

from src.models import load_model
from src.config import load_config

# 加载配置
config = load_config("configs/model_configs/gemini-2.5-flash.yaml")

# 加载模型
model = load_model(config)

# 测试调用
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Hello, what is 2+2?")]
response = model.invoke(messages)
print(response.content)
```

运行：
```bash
python test_gemini.py
```

## 参考资源

- [Google AI Studio](https://aistudio.google.com/)
- [LangChain Google GenAI 文档](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
- [Gemini API 文档](https://ai.google.dev/docs)

