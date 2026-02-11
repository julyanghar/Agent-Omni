# Agent-Omni 架构与数据流图

## 系统架构图

```mermaid
graph TB
    subgraph UserInput["用户输入层"]
        Query["用户查询<br/>query: str"]
        Text["文本输入<br/>text: List[str]"]
        Image["图像输入<br/>image: List[str]"]
        Video["视频输入<br/>video: List[str]"]
        Audio["音频输入<br/>audio: List[str]"]
    end

    subgraph SummaryPhase["摘要生成阶段（并行）"]
        TextSum["text_summarize<br/>生成文本摘要"]
        ImageSum["image_summarize<br/>生成图像摘要"]
        VideoSum["video_summarize<br/>生成视频摘要"]
        AudioSum["audio_summarize<br/>生成音频摘要"]
    end

    subgraph MasterAgent["主智能体"]
        Dispatcher1["master_dispatcher_1<br/>汇总摘要"]
        Reasoning["master_reasoning<br/>推理模块<br/>- 分析意图<br/>- 选择智能体<br/>- 生成问题"]
        Dispatcher["master_dispatcher<br/>分发模块"]
        Decision["master_decision<br/>决策模块<br/>- 综合结果<br/>- 判断完成<br/>- 生成建议"]
    end

    subgraph SubAgents["子智能体（模态专用）"]
        TextAgent["text_agent<br/>文本分析"]
        ImageAgent["image_agent<br/>图像分析"]
        VideoAgent["video_agent<br/>视频分析<br/>（帧采样）"]
        AudioAgent["audio_agent<br/>音频分析"]
    end

    subgraph Models["模型层"]
        MasterModel["Master Model<br/>（主模型）"]
        TextModel["Text Model<br/>（文本模型）"]
        ImageModel["Image Model<br/>（视觉模型）"]
        VideoModel["Video Model<br/>（视觉模型）"]
        AudioModel["Audio Model<br/>（音频模型）"]
    end

    subgraph State["状态管理"]
        StateObj["State Object<br/>- 输入数据<br/>- 中间结果<br/>- 历史记录"]
    end

    UserInput --> SummaryPhase
    Text --> TextSum
    Image --> ImageSum
    Video --> VideoSum
    Audio --> AudioSum

    SummaryPhase --> Dispatcher1
    Dispatcher1 --> Reasoning

    Reasoning --> Dispatcher
    Dispatcher --> TextAgent
    Dispatcher --> ImageAgent
    Dispatcher --> VideoAgent
    Dispatcher --> AudioAgent

    TextAgent --> Dispatcher
    ImageAgent --> Dispatcher
    VideoAgent --> Dispatcher
    AudioAgent --> Dispatcher

    Dispatcher --> Decision

    Decision -->|"is_final=False<br/>AND<br/>cur_round<max_round"| Reasoning
    Decision -->|"is_final=True<br/>OR<br/>cur_round>=max_round"| EndNode["END<br/>返回最终答案"]

    MasterModel --> Reasoning
    MasterModel --> Decision
    TextModel --> TextAgent
    ImageModel --> ImageAgent
    VideoModel --> VideoAgent
    AudioModel --> AudioAgent

    StateObj -.->|"读取/更新"| Reasoning
    StateObj -.->|"读取/更新"| Decision
    StateObj -.->|"读取/更新"| TextAgent
    StateObj -.->|"读取/更新"| ImageAgent
    StateObj -.->|"读取/更新"| VideoAgent
    StateObj -.->|"读取/更新"| AudioAgent

    style Reasoning fill:#e1f5ff
    style Decision fill:#fff4e1
    style TextAgent fill:#e8f5e9
    style ImageAgent fill:#e8f5e9
    style VideoAgent fill:#e8f5e9
    style AudioAgent fill:#e8f5e9
```

## 数据流详细图

```mermaid
sequenceDiagram
    participant User as 用户
    participant Graph as LangGraph工作流
    participant Summary as 摘要节点
    participant Master as 主智能体
    participant SubAgent as 子智能体
    participant Model as 模型API
    participant State as 状态对象

    User->>State: 初始化状态<br/>(query, text, image, video, audio)
    
    Note over Graph: 第一轮：摘要生成（并行）
    par 并行执行
        Graph->>Summary: text_summarize
        Summary->>Model: 调用文本模型<br/>"Summarize the text"
        Model-->>Summary: 文本摘要
        Summary->>State: 更新 text_summary
    and
        Graph->>Summary: image_summarize
        Summary->>Model: 调用视觉模型<br/>"Summarize the image"
        Model-->>Summary: 图像摘要
        Summary->>State: 更新 image_summary
    and
        Graph->>Summary: video_summarize
        Summary->>Model: 调用视觉模型<br/>"Summarize the video"
        Model-->>Summary: 视频摘要
        Summary->>State: 更新 video_summary
    and
        Graph->>Summary: audio_summarize
        Summary->>Model: 调用音频模型<br/>"Summarize the audio"
        Model-->>Summary: 音频摘要
        Summary->>State: 更新 audio_summary
    end

    Graph->>Master: master_dispatcher_1<br/>(汇总摘要)
    Master->>State: 读取所有摘要
    
    Note over Graph: 第一轮：推理
    Graph->>Master: master_reasoning
    Master->>State: 读取摘要和历史消息
    Master->>Model: 调用主模型<br/>系统提示 + 用户查询 + 摘要
    Model-->>Master: 结构化输出<br/>{user_intent, agent_instructions}
    Master->>State: 更新 reasoning_result<br/>cur_round_num=1

    Note over Graph: 第一轮：子智能体处理
    Graph->>SubAgent: text_agent
    SubAgent->>State: 读取 reasoning_result
    SubAgent->>State: 读取 text 输入
    loop 每个问题
        SubAgent->>Model: 调用文本模型<br/>问题 + 文本内容
        Model-->>SubAgent: 回答
    end
    SubAgent->>State: 更新 text_agent_result

    Graph->>SubAgent: image_agent
    SubAgent->>State: 读取 reasoning_result
    SubAgent->>State: 读取 image 输入
    loop 每个问题
        SubAgent->>Model: 调用视觉模型<br/>问题 + 图像
        Model-->>SubAgent: 回答
    end
    SubAgent->>State: 更新 image_agent_result

    Graph->>SubAgent: video_agent
    SubAgent->>State: 读取 reasoning_result
    SubAgent->>State: 读取 video 输入
    SubAgent->>SubAgent: 视频帧采样
    loop 每个问题
        SubAgent->>Model: 调用视觉模型<br/>问题 + 视频帧
        Model-->>SubAgent: 回答
    end
    SubAgent->>State: 更新 video_agent_result

    Graph->>SubAgent: audio_agent
    SubAgent->>State: 读取 reasoning_result
    SubAgent->>State: 读取 audio 输入
    loop 每个问题
        SubAgent->>Model: 调用音频模型<br/>问题 + 音频
        Model-->>SubAgent: 回答
    end
    SubAgent->>State: 更新 audio_agent_result

    Note over Graph: 第一轮：决策
    Graph->>Master: master_dispatcher<br/>(透传)
    Graph->>Master: master_decision
    Master->>State: 读取所有子智能体结果
    Master->>State: 读取历史决策结果
    Master->>Model: 调用主模型<br/>系统提示 + 所有结果
    Model-->>Master: 结构化输出<br/>{final_answer, is_final, suggestions}
    Master->>State: 更新 decision_result

    alt is_final=False AND cur_round<max_round
        Note over Graph: 第二轮：迭代推理
        Graph->>Master: master_reasoning<br/>(使用上一轮建议)
        Master->>State: 读取上一轮 suggestions
        Master->>Model: 调用主模型<br/>（包含改进建议）
        Model-->>Master: 新的 agent_instructions
        Master->>State: 更新 reasoning_result<br/>cur_round_num=2
        
        Graph->>SubAgent: 子智能体处理<br/>（基于新问题）
        SubAgent->>State: 更新结果
        
        Graph->>Master: master_decision
        Master->>State: 更新 decision_result
    end

    Graph->>User: 返回最终答案<br/>decision_result["final_answer"]
```

## 状态流转图

```mermaid
stateDiagram-v2
    [*] --> Initialized: 创建State对象
    
    Initialized --> Summarizing: 开始工作流
    
    state Summarizing {
        [*] --> TextSummary
        [*] --> ImageSummary
        [*] --> VideoSummary
        [*] --> AudioSummary
        TextSummary --> [*]
        ImageSummary --> [*]
        VideoSummary --> [*]
        AudioSummary --> [*]
    }
    
    Summarizing --> Reasoning: 所有摘要完成
    
    state Reasoning {
        [*] --> AnalyzeIntent
        AnalyzeIntent --> SelectAgents
        SelectAgents --> GenerateQuestions
        GenerateQuestions --> [*]
    }
    
    Reasoning --> Dispatching: 生成agent_instructions
    
    state Dispatching {
        [*] --> RouteToTextAgent
        [*] --> RouteToImageAgent
        [*] --> RouteToVideoAgent
        [*] --> RouteToAudioAgent
    }
    
    Dispatching --> Processing: 路由完成
    
    state Processing {
        [*] --> TextProcessing
        [*] --> ImageProcessing
        [*] --> VideoProcessing
        [*] --> AudioProcessing
        TextProcessing --> [*]
        ImageProcessing --> [*]
        VideoProcessing --> [*]
        AudioProcessing --> [*]
    }
    
    Processing --> Decision: 所有智能体完成
    
    state Decision {
        [*] --> SynthesizeAnswer
        SynthesizeAnswer --> EvaluateCompleteness
        EvaluateCompleteness --> GenerateSuggestions
        GenerateSuggestions --> [*]
    }
    
    Decision --> CheckFinal: 生成决策结果
    
    CheckFinal --> Reasoning: is_final=False<br/>AND<br/>cur_round<max_round
    CheckFinal --> [*]: is_final=True<br/>OR<br/>cur_round>=max_round
```

## 组件交互图

```mermaid
graph LR
    subgraph Config["配置层"]
        MainConfig["主配置<br/>config.yaml"]
        ModelConfig["模型配置<br/>*.yaml"]
    end

    subgraph Core["核心层"]
        Graph["graph.py<br/>工作流定义"]
        State["state.py<br/>状态定义"]
        ConfigLoader["config.py<br/>配置加载"]
    end

    subgraph Nodes["节点层"]
        MasterNodes["nodes.py<br/>- master_reasoning<br/>- master_decision"]
        SubAgentNodes["subagents/<br/>- text_agent<br/>- image_agent<br/>- video_agent<br/>- audio_agent"]
    end

    subgraph Model["模型层"]
        ModelLoader["models.py<br/>load_model()"]
        ModelWrapper["ModelInvokeWrapper<br/>- media_invoke<br/>- media_batch"]
        LocalModels["local_models/<br/>本地模型支持"]
    end

    subgraph Utils["工具层"]
        MediaUtils["utils.py<br/>- normalize_image<br/>- normalize_audio"]
        TextUtils["utils.py<br/>- combine_summaries<br/>- postprocessing"]
    end

    MainConfig --> ConfigLoader
    ModelConfig --> ConfigLoader
    ConfigLoader --> Graph
    ConfigLoader --> ModelLoader
    
    Graph --> State
    Graph --> MasterNodes
    Graph --> SubAgentNodes
    
    MasterNodes --> ModelWrapper
    SubAgentNodes --> ModelWrapper
    
    ModelLoader --> ModelWrapper
    LocalModels --> ModelLoader
    
    SubAgentNodes --> MediaUtils
    MasterNodes --> TextUtils
    SubAgentNodes --> TextUtils

    style Graph fill:#e1f5ff
    style State fill:#fff4e1
    style ModelWrapper fill:#e8f5e9
```

## 数据格式示例

### 输入数据格式

```python
State(
    query="请总结提供材料的内容",
    text=["文档内容..."],
    image=["image.jpeg"],
    video=["video.mp4"],
    audio=["audio.wav"],
    cur_round_num=0,
    max_round_num=3
)
```

### 推理结果格式

```json
{
    "user_intent": "总结多模态内容",
    "agent_instructions": [
        {
            "agent_name": "text_agent",
            "questions": [
                "文档的主要主题是什么？",
                "文档中提到了哪些关键点？"
            ]
        },
        {
            "agent_name": "image_agent",
            "questions": [
                "图像中显示了什么内容？",
                "图像中的关键元素是什么？"
            ]
        }
    ]
}
```

### 决策结果格式

```json
{
    "final_answer": "综合所有材料，主要内容包括...",
    "is_final": false,
    "suggestions_for_next_round": [
        "可以更详细地分析图像中的细节",
        "需要检查视频中的时间序列信息"
    ]
}
```

### 子智能体结果格式

```json
[
    {
        "question": "文档的主要主题是什么？",
        "answer": "文档主要讨论了..."
    },
    {
        "question": "文档中提到了哪些关键点？",
        "answer": "关键点包括..."
    }
]
```

