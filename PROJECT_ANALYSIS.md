# Agent-Omni 项目详细分析

## 项目概述

**Agent-Omni** 是一个基于多智能体协调的测试时多模态推理框架，通过主智能体（Master Agent）协调现有的基础模型，实现跨模态（文本、图像、视频、音频）的"理解一切"能力。

### 核心特点
- **无需微调**：完全在测试时运行，不需要对模型进行端到端微调
- **模块化设计**：主智能体 + 模态专用智能体的分层架构
- **可扩展性**：支持灵活配置不同模型用于不同模态
- **迭代推理**：支持多轮推理循环，逐步完善答案

---

## 项目架构

### 目录结构

```
Agent-Omni/
├── src/                    # 核心源代码
│   ├── graph.py           # LangGraph 工作流定义
│   ├── state.py           # 状态管理（State）
│   ├── nodes.py           # 主智能体节点（推理、调度、决策）
│   ├── models.py          # 模型加载和调用封装
│   ├── config.py          # 配置加载
│   ├── utils.py           # 工具函数（媒体处理、摘要合并等）
│   ├── subagents/         # 模态专用智能体
│   │   ├── text_agent.py
│   │   ├── image_agent.py
│   │   ├── video_agent.py
│   │   └── audio_agent.py
│   └── local_models/      # 本地模型支持
│       ├── base_local_model.py
│       ├── text_model.py
│       └── qwen2_audio.py
├── configs/               # 配置文件
│   ├── model_configs/     # 模型配置（支持 OpenAI、Bedrock、本地）
│   └── vllm_configs/      # vLLM 配置
├── test/                  # 测试脚本
│   ├── agent_test.py      # Agent-Omni 测试
│   ├── vanilla_test.py    # 单模型测试
│   └── config.yaml        # 测试配置
└── requirements.txt       # 依赖包

```

---

## 核心组件详解

### 1. 状态管理 (State)

**文件**: `src/state.py`

`State` 继承自 LangGraph 的 `MessagesState`，维护整个工作流的状态：

```python
class State(MessagesState):
    # 用户输入
    query: str                    # 用户查询
    text: List[Any]               # 文本输入
    image: List[Any]              # 图像输入
    video: List[Any]              # 视频输入
    audio: List[Any]              # 音频输入
    
    # 轮次控制
    cur_round_num: int            # 当前轮次
    max_round_num: int            # 最大轮次
    
    # 摘要（第一轮生成）
    text_summary: Any
    image_summary: Any
    video_summary: Any
    audio_summary: Any
    
    # 主智能体输出
    reasoning_result: Any         # 推理结果（包含用户意图、选中的智能体、问题列表）
    decision_result: Any          # 决策结果（最终答案、是否完成、下一轮建议）
    
    # 子智能体输出
    text_agent_result: Any
    image_agent_result: Any
    video_agent_result: Any
    audio_agent_result: Any
    
    # 历史记录（多轮）
    reasoning_result_list: Any
    decision_result_list: Any
    text_agent_result_list: Any
    image_agent_result_list: Any
    video_agent_result_list: Any
    audio_agent_result_list: Any
```

### 2. 工作流图 (Graph)

**文件**: `src/graph.py`

使用 LangGraph 构建有向无环图（DAG），定义执行流程：

#### 节点列表
- **摘要节点**：`text_summarize`, `image_summarize`, `video_summarize`, `audio_summarize`
- **主智能体节点**：
  - `master_dispatcher_1`：初始调度（汇总摘要）
  - `master_reasoning`：推理模块（分析意图、选择智能体、生成问题）
  - `master_dispatcher`：分发模块（路由到子智能体）
  - `master_decision`：决策模块（综合结果、判断是否完成）
- **子智能体节点**：`text_agent`, `image_agent`, `video_agent`, `audio_agent`

#### 边（Edge）定义

```
START
  ├─> text_summarize ──┐
  ├─> image_summarize ─┤
  ├─> video_summarize ─┤
  └─> audio_summarize ─┘
                        │
                        v
              master_dispatcher_1
                        │
                        v
              master_reasoning
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        v               v               v
   text_agent    image_agent    video_agent    audio_agent
        │               │               │               │
        └───────────────┼───────────────┘
                        │
                        v
              master_dispatcher
                        │
                        v
              master_decision
                        │
                        ├─> [条件判断: next_round]
                        │   ├─ True  ─> master_reasoning (下一轮)
                        │   └─ False ─> END
```

### 3. 主智能体节点 (Nodes)

**文件**: `src/nodes.py`

#### 3.1 master_reasoning（推理模块）

**功能**：分析用户意图，选择合适的子智能体，为每个智能体生成具体问题

**输入**：
- 用户查询 (`state["query"]`)
- 各模态摘要 (`text_summary`, `image_summary`, etc.)
- 历史消息（上一轮的建议和结果）
- 可用智能体信息

**输出**（Pydantic 结构化）：
```python
{
    "user_intent": str,              # 用户意图
    "agent_instructions": [          # 智能体指令列表
        {
            "agent_name": str,       # 智能体名称（text_agent/image_agent/etc.）
            "questions": List[str]   # 针对该智能体的具体问题列表
        }
    ]
}
```

**关键逻辑**：
- 使用 `PydanticOutputParser` 确保结构化输出
- 支持重试机制（`retry_times`）
- 问题必须包含完整上下文（子智能体看不到原始查询）

#### 3.2 master_dispatcher / master_dispatcher_1（调度模块）

**功能**：目前是透传节点，未来可扩展为智能路由

#### 3.3 master_decision（决策模块）

**功能**：综合所有子智能体的结果，生成最终答案，判断是否需要下一轮

**输入**：
- 所有子智能体的结果
- 推理结果
- 历史决策结果

**输出**（动态 Pydantic 模型）：
```python
{
    "final_answer": Any,                    # 最终答案（类型由 final_answer_structure 决定）
    "is_final": bool,                       # 是否完成
    "suggestions_for_next_round": List[str] # 下一轮改进建议（即使完成也要提供）
}
```

**关键逻辑**：
- 使用 `create_model` 动态创建 Pydantic 模型（支持自定义输出格式）
- 即使答案完成，也必须提供改进建议
- 根据 `is_final` 和 `cur_round_num < max_round_num` 决定是否继续

### 4. 子智能体 (Subagents)

#### 4.1 text_agent

**文件**: `src/subagents/text_agent.py`

**功能**：
- `text_summarize`：生成文本摘要（第一轮）
- `text_agent`：根据推理模块的问题，分析文本并回答

**处理流程**：
1. 检查 `state["text"]` 是否存在
2. 从 `reasoning_result` 中提取 `text_agent` 的指令
3. 批量调用模型（`model.media_batch`）处理多个问题
4. 返回 `{question, answer}` 列表

#### 4.2 image_agent

**文件**: `src/subagents/image_agent.py`

**功能**：分析图像并回答问题

**特殊处理**：
- 支持批量图像输入（`max_image_input` 限制）
- 图像自动转换为 Data URL 格式（base64 PNG）

#### 4.3 video_agent

**文件**: `src/subagents/video_agent.py`

**功能**：分析视频并回答问题

**特殊处理**：
- 使用 `decord.VideoReader` 读取视频
- 帧采样：根据 `max_frames_num` 和 `fps` 采样关键帧
- 将视频帧转换为图像序列，按批次处理

**采样逻辑**：
```python
# 如果帧数超过 max_frames_num，均匀采样
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
```

#### 4.4 audio_agent

**文件**: `src/subagents/audio_agent.py`

**功能**：分析音频并回答问题

**特殊处理**：
- 支持批量音频输入（`max_audio_input` 限制）
- 音频自动转换为 Data URL 格式（base64 WAV）

### 5. 模型管理 (Models)

**文件**: `src/models.py`

#### 5.1 模型加载 (`load_model`)

支持三种 API 提供者：

1. **OpenAI 兼容** (`api_provider: "openai"`)
   - 使用 `langchain_openai.ChatOpenAI`
   - 支持自定义 `base_url`（如 SiliconFlow）

2. **AWS Bedrock** (`api_provider: "bedrock"`)
   - 使用 `langchain_aws.ChatBedrockConverse`
   - 需要 boto3 配置

3. **本地模型** (`api_provider: "local"`)
   - 从 `local_models.MODEL_MAP` 加载
   - 支持 vLLM 等本地部署

#### 5.2 ModelInvokeWrapper

封装模型调用，提供统一接口：

- `construct_message`：构建多模态消息（文本、图像、音频）
- `media_invoke`：单次调用（支持重试）
- `media_batch`：批量调用（提高效率）

**媒体处理**：
- 图像：转换为 `data:image/png;base64,...`
- 音频：转换为 `data:audio/wav;base64,...`
- 支持本地文件路径、URL、PIL Image、bytes 等多种输入

### 6. 工具函数 (Utils)

**文件**: `src/utils.py`

#### 6.1 摘要合并

- `combine_summaries`：合并各模态摘要，用于推理模块
- `combine_historical_message`：合并历史消息，用于决策模块
- `combine_available_agent`：生成可用智能体描述

#### 6.2 媒体处理

- `normalize_image_to_data_url`：图像标准化（PIL/bytes/路径 → Data URL）
- `normalize_audio_to_data_url`：音频标准化（文件/bytes → WAV Data URL）
- `postprocessing`：后处理（移除 `<think>` 标签等）

---

## 数据流详解

### 完整执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户输入                                  │
│  query: "请总结提供材料的内容"                                    │
│  text: "文档内容..."                                             │
│  image: "image.jpeg"                                            │
│  video: "video.mp4"                                             │
│  audio: "audio.wav"                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│                   第一轮：摘要生成（并行）                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │text_sum  │  │image_sum │  │video_sum │  │audio_sum │     │
│  │marize    │  │marize    │  │marize    │  │marize    │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│       │              │              │              │            │
│       └──────────────┴──────────────┴──────────────┘            │
│                              │                                   │
│                              v                                   │
│                    master_dispatcher_1                           │
│                    (汇总摘要)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│                    master_reasoning（推理模块）                  │
│                                                                  │
│  输入：                                                          │
│  - 用户查询                                                      │
│  - 各模态摘要                                                    │
│  - 历史消息（第一轮为空）                                        │
│                                                                  │
│  处理：                                                          │
│  1. 分析用户意图                                                 │
│  2. 选择需要的智能体（text_agent, image_agent, etc.）          │
│  3. 为每个智能体生成具体问题                                     │
│                                                                  │
│  输出（结构化）：                                                │
│  {                                                               │
│    "user_intent": "总结多模态内容",                              │
│    "agent_instructions": [                                       │
│      {                                                           │
│        "agent_name": "text_agent",                              │
│        "questions": ["文档的主要主题是什么？", ...]              │
│      },                                                          │
│      {                                                           │
│        "agent_name": "image_agent",                             │
│        "questions": ["图像中显示了什么？", ...]                  │
│      }                                                           │
│    ]                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        v                     v                     v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ text_agent   │    │ image_agent  │    │ video_agent │    ...
│              │    │              │    │              │
│ 输入：        │    │ 输入：        │    │ 输入：        │
│ - text       │    │ - image      │    │ - video      │
│ - questions  │    │ - questions  │    │ - questions  │
│              │    │              │    │              │
│ 处理：        │    │ 处理：        │    │ 处理：        │
│ 调用文本模型  │    │ 调用视觉模型  │    │ 采样帧+视觉   │
│              │    │              │    │              │
│ 输出：        │    │ 输出：        │    │ 输出：        │
│ [{           │    │ [{           │    │ [{           │
│   question,  │    │   question,  │    │   question,  │
│   answer     │    │   answer     │    │   answer     │
│ }]           │    │ }]           │    │ }]           │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│                    master_dispatcher                             │
│                    (透传，未来可扩展)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│                    master_decision（决策模块）                   │
│                                                                  │
│  输入：                                                          │
│  - 所有子智能体的结果                                            │
│  - 推理结果                                                      │
│  - 历史决策结果（第一轮为空）                                    │
│                                                                  │
│  处理：                                                          │
│  1. 综合所有结果，生成最终答案                                   │
│  2. 评估完整性                                                   │
│  3. 判断是否需要下一轮                                           │
│  4. 生成改进建议                                                 │
│                                                                  │
│  输出（结构化）：                                                │
│  {                                                               │
│    "final_answer": "综合答案...",                                │
│    "is_final": false,                                           │
│    "suggestions_for_next_round": [                              │
│      "可以更详细地分析图像中的细节",                              │
│      "需要检查视频中的时间序列信息"                              │
│    ]                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
                    ┌─────────────────┐
                    │   条件判断       │
                    │ next_round()   │
                    └─────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
         is_final=False              is_final=True
         AND                         OR
         cur_round < max_round              cur_round >= max_round
                │                           │
                v                           v
        ┌───────────────┐          ┌───────────────┐
        │ 下一轮推理     │          │     结束       │
        │ (回到          │          │  返回最终答案  │
        │ master_       │          │                │
        │ reasoning)    │          │                │
        └───────────────┘          └───────────────┘
```

### 多轮迭代流程

```
Round 1:
  Summary → Reasoning → Agents → Decision
    │                                    │
    └────────────────────────────────────┘
              (suggestions)

Round 2:
  Reasoning (使用上一轮建议) → Agents → Decision
    │                                    │
    └────────────────────────────────────┘
              (suggestions)

Round 3:
  Reasoning → Agents → Decision → END
```

### 状态更新示例

```python
# 初始状态
state = State(
    query="请总结内容",
    text=["文档..."],
    image=["image.jpeg"],
    cur_round_num=0,
    max_round_num=3
)

# 第一轮：摘要生成后
state = {
    ...,
    "text_summary": "这是一份关于...的文档",
    "image_summary": "图像显示了一个...",
    "cur_round_num": 0
}

# 第一轮：推理后
state = {
    ...,
    "reasoning_result": {
        "user_intent": "总结多模态内容",
        "agent_instructions": [
            {"agent_name": "text_agent", "questions": ["主题是什么？"]},
            {"agent_name": "image_agent", "questions": ["显示了什么？"]}
        ]
    },
    "cur_round_num": 1
}

# 第一轮：子智能体处理后
state = {
    ...,
    "text_agent_result": [{"question": "...", "answer": "..."}],
    "image_agent_result": [{"question": "...", "answer": "..."}]
}

# 第一轮：决策后
state = {
    ...,
    "decision_result": {
        "final_answer": "初步答案...",
        "is_final": False,
        "suggestions_for_next_round": ["需要更详细分析..."]
    },
    "cur_round_num": 1
}

# 第二轮：推理（使用上一轮建议）
state = {
    ...,
    "reasoning_result": {
        "user_intent": "总结多模态内容（基于上一轮建议）",
        "agent_instructions": [
            {"agent_name": "image_agent", "questions": ["图像中的细节是什么？"]}
        ]
    },
    "cur_round_num": 2
}
```

---

## 配置系统

### 配置文件层级

1. **主配置** (`test/config.yaml` 或 `src/config.yaml`)
   ```yaml
   model:
     master_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
     text_agent: ...
     image_agent: ...
   
   agents:
     supported: ["text", "image", "video", "audio"]
     names: ["text_agent", "image_agent", "video_agent", "audio_agent"]
   
   system:
     retry_times: 10
   ```

2. **模型配置** (`configs/model_configs/*.yaml`)
   ```yaml
   api_provider: openai  # openai / bedrock / local
   model_id: Qwen/Qwen3-Omni-30B-A3B-Instruct
   base_url: https://api.siliconflow.cn/v1
   api_key_name: siliconflow_api_key
   params:
     max_tokens: 8192
     temperature: 0
   ```

3. **配置加载机制** (`src/config.py`)
   - 支持 YAML 文件包含（`include: other.yaml`）
   - 递归解析，防止循环引用
   - 环境变量 `CONFIG_PATH` 指定配置文件

---

## 关键技术点

### 1. LangGraph 工作流

- **状态管理**：使用 `MessagesState` 继承，自动处理消息历史
- **条件边**：`add_conditional_edges` 实现动态路由
- **节点并行**：摘要节点可并行执行

### 2. 结构化输出

- **Pydantic 模型**：确保 LLM 输出符合预期格式
- **OutputFixingParser**：自动修复格式错误
- **动态模型创建**：`create_model` 支持自定义输出结构

### 3. 多模态处理

- **统一接口**：所有媒体类型转换为 Data URL
- **批量处理**：支持批量图像/音频输入
- **视频采样**：智能帧采样，控制计算成本

### 4. 错误处理

- **重试机制**：所有模型调用支持重试（`retry_times`）
- **异常捕获**：详细的错误日志和堆栈跟踪
- **优雅降级**：缺失模态时返回 `None`，不中断流程

---

## 使用示例

### 基本使用

```python
from src.state import State
from src.graph import _graph

# 初始化状态
state = State(
    messages=[],
    max_round_num=3,
    query="请总结提供材料的内容",
    text=["文档内容..."],
    image=["image.jpeg"],
    video=["video.mp4"],
    audio=["audio.wav"]
)

# 执行工作流
end_state = _graph.batch([state])

# 获取结果
final_answer = end_state[0]["decision_result"]["final_answer"]
print(final_answer)
```

### 配置不同模型

```yaml
# test/config.yaml
model:
  master_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
  text_agent: ../configs/model_configs/qwen3-4b-instruct.yaml      # 小模型处理文本
  image_agent: ../configs/model_configs/llava-next-video-7b.yaml   # 视觉模型
  video_agent: ../configs/model_configs/llava-next-video-7b.yaml
  audio_agent: ../configs/model_configs/qwen2-audio-7b.yaml         # 音频模型
```

---

## 扩展点

### 1. 添加新模态

1. 在 `src/subagents/` 创建新智能体（如 `code_agent.py`）
2. 在 `src/graph.py` 注册节点
3. 在 `State` 中添加对应字段
4. 更新配置中的 `agents.supported`

### 2. 自定义决策逻辑

修改 `master_decision` 中的 `system_prompt`，调整决策策略

### 3. 本地模型集成

在 `src/local_models/` 实现 `BaseLocalModel` 子类，注册到 `MODEL_MAP`

---

## 性能优化建议

1. **批量处理**：子智能体使用 `media_batch` 批量调用模型
2. **并行摘要**：第一轮摘要节点可并行执行
3. **视频采样**：调整 `max_frames_num` 平衡精度和速度
4. **缓存机制**：可添加结果缓存，避免重复计算

---

## 总结

Agent-Omni 通过**主智能体协调 + 模态专用智能体**的架构，实现了无需微调的多模态理解。核心优势：

1. **模块化**：各组件职责清晰，易于扩展
2. **灵活性**：支持不同模型配置，适应不同场景
3. **可解释性**：结构化输出，便于调试和分析
4. **迭代优化**：多轮推理机制，逐步完善答案

该框架特别适合需要跨模态理解的应用场景，如文档分析、内容审核、智能问答等。

