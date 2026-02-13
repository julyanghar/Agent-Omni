# 模型调用方法查询指南

## 为什么可以调用 `self.model.batch()`？

### 1. 继承链分析

在您的代码中，`self.model` 是 LangChain 的模型实例，它们都继承自 `Runnable` 接口：

```
Runnable (抽象基类)
  └── BaseLanguageModel
      └── BaseChatModel
          ├── ChatOpenAI (langchain_openai)
          ├── ChatGoogleGenerativeAI (langchain_google_genai)
          ├── ChatBedrockConverse (langchain_aws)
          └── 其他模型类...
```

### 2. Runnable 接口提供的方法

`Runnable` 接口定义了标准的调用方法：

- **`invoke(input)`**: 单次调用
- **`batch(inputs)`**: 批量调用（并行处理）
- **`stream(input)`**: 流式调用
- **`astream(input)`**: 异步流式调用
- **`abatch(inputs)`**: 异步批量调用

### 3. batch() 方法签名

根据 `langchain_core.runnables.base.Runnable` 的定义：

```python
def batch(
    self,
    inputs: list[Input],  # 输入列表
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]:  # 返回输出列表
```

## 如何查询调用模型的正确方法？

### 方法 1: 查看 LangChain 官方文档

1. **在线文档**：
   - LangChain Core: https://python.langchain.com/docs/reference/langchain-core/
   - Runnable 接口: https://python.langchain.com/docs/reference/langchain-core/runnables/base/

2. **搜索特定模型**：
   - ChatOpenAI: https://python.langchain.com/docs/integrations/chat/openai
   - ChatGoogleGenerativeAI: https://python.langchain.com/docs/integrations/chat/google_generative_ai

### 方法 2: 使用 Python 的 help() 和 dir()

```python
# 查看对象的所有方法和属性
print(dir(self.model))

# 查看特定方法的文档
help(self.model.batch)
help(self.model.invoke)
```

### 方法 3: 查看源码中的类型提示和文档字符串

```python
# 在 IDE 中查看类型提示
from langchain_openai import ChatOpenAI
model = ChatOpenAI()

# 查看方法的签名和文档
import inspect
print(inspect.signature(model.batch))
print(model.batch.__doc__)
```

### 方法 4: 检查继承关系

```python
# 查看类的继承关系
from langchain_openai import ChatOpenAI
print(ChatOpenAI.__mro__)  # Method Resolution Order

# 查看是否实现了特定方法
from langchain_core.runnables import Runnable
print(issubclass(ChatOpenAI, Runnable))  # True
```

### 方法 5: 查看 LangChain 源码

在您的环境中，源码位于：
```
/home/yilin/anaconda3/envs/agent-omni/lib/python3.10/site-packages/langchain_core/
```

关键文件：
- `runnables/base.py`: Runnable 基类定义
- `language_models/chat_models.py`: BaseChatModel 定义

### 方法 6: 使用类型检查工具

```python
# 使用 mypy 或 pyright 进行类型检查
# 这些工具会显示可用的方法和它们的签名

# 示例：在 IDE 中，将鼠标悬停在 self.model 上
# 会显示类型信息和可用方法
```

## 实际使用示例

### 单次调用 (invoke)

```python
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!")
]
response = self.model.invoke(messages)
```

### 批量调用 (batch)

```python
messages_batch = [
    [SystemMessage(content="You are a helpful assistant."), HumanMessage(content="Hello!")],
    [SystemMessage(content="You are a helpful assistant."), HumanMessage(content="Hi!")],
]
responses = self.model.batch(messages_batch)
```

### 带配置的批量调用

```python
from langchain_core.runnables.config import RunnableConfig

config = RunnableConfig(
    max_concurrency=5,  # 最大并发数
    tags=["batch-processing"],
    metadata={"source": "api"}
)

responses = self.model.batch(
    messages_batch,
    config=config,
    return_exceptions=False  # 如果为 True，异常会作为结果返回而不是抛出
)
```

## 常见问题

### Q: 为什么 batch() 可以接受消息列表？

A: 因为 `Runnable` 接口定义了 `batch()` 方法，它接受一个输入列表（`list[Input]`）。对于聊天模型，`Input` 类型是消息列表（`list[BaseMessage]`），所以 `batch()` 接受 `list[list[BaseMessage]]`。

### Q: batch() 和多次调用 invoke() 有什么区别？

A: 
- `batch()`: 并行处理多个请求，更高效
- 多次 `invoke()`: 串行处理，但可以更好地控制每个请求

### Q: 如何知道某个模型是否支持 batch()？

A: 所有继承自 `Runnable` 的模型都支持 `batch()`。如果模型类继承自 `BaseChatModel`，它一定支持这些方法。

## 参考资源

1. **LangChain Runnable 文档**: https://python.langchain.com/docs/concepts/#runnables
2. **LangChain API 参考**: https://api.python.langchain.com/en/latest/langchain_core.html
3. **源码位置**: `langchain_core/runnables/base.py`

