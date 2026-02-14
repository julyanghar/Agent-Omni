# Agent-Omni 评估模块

本模块提供了 Agent-Omni 在各种 benchmark 上的评估功能。每个 benchmark 都有自己独立的目录，包含所有相关的评估代码和脚本。

## 目录结构

```
eval/
├── __init__.py
├── README.md                   # 本文档
├── daily_omni/                 # Daily-Omni benchmark 评估模块
│   ├── __init__.py
│   ├── run_evaluation.py        # 评估入口脚本
│   ├── agent_omni_adapter.py   # Agent-Omni 适配器
│   ├── run_daily_omni.sh       # 评估启动脚本（推荐使用）
│   └── test_model_api/         # Daily-Omni 评估框架核心代码
│       ├── __init__.py
│       ├── main_tester.py       # 主测试框架
│       ├── test_utils.py       # 工具函数和模型调用接口
│       └── test_config.py      # 配置文件
└── [其他 benchmark 目录...]     # 未来可添加更多 benchmark
```

## 快速开始

### Daily-Omni Benchmark

#### 1. 准备数据

确保你有以下数据：
- Daily-Omni 的 QA JSON 文件（包含问题和答案）
- 视频文件目录（包含所有测试视频）

#### 2. 配置

编辑 `eval/daily_omni/test_model_api/test_config.py` 来设置：
- `BASE_VIDEO_DIR`: 视频文件基础目录
- `DEFAULT_QA_JSON_PATH`: 默认 QA JSON 文件路径
- `AGENT_OMNI_CONFIG_PATH`: Agent-Omni 配置文件路径（如果需要）
- `AGENT_OMNI_MAX_ROUNDS`: Agent-Omni 最大推理轮数

#### 3. 运行评估

**推荐方式：使用 bash 脚本**

```bash
# 使用默认配置
bash eval/daily_omni/run_daily_omni.sh

# 或使用环境变量自定义配置
MODEL_TYPE=agent_omni \
EXECUTION_MODE=sequential \
QA_FILE=/path/to/qa.json \
MAX_ITEMS=10 \
VIDEO_DIR=/path/to/videos \
bash eval/daily_omni/run_daily_omni.sh
```

**或者直接使用 Python 脚本**

```bash
# 使用默认配置
python eval/daily_omni/run_evaluation.py

# 指定 QA 文件
python eval/daily_omni/run_evaluation.py --qa_file /path/to/qa.json

# 限制处理数量（用于测试）
python eval/daily_omni/run_evaluation.py --qa_file /path/to/qa.json --max_items 10

# 使用并行模式（注意：Agent-Omni 可能不支持并行，建议使用 sequential）
python eval/daily_omni/run_evaluation.py --qa_file /path/to/qa.json --mode parallel
```

#### 4. 查看结果

评估完成后，会输出：
- 总体准确率
- 按 QA 类型的准确率
- 按视频类别的准确率
- 按视频时长的准确率
- 详细的统计信息

## 工作原理

### Agent-Omni 适配器

`daily_omni/agent_omni_adapter.py` 中的 `ask_agent_omni` 函数负责：

1. **输入转换**：将 Daily-Omni 格式的输入（question, choices, video_path）转换为 Agent-Omni 的 State 格式
2. **查询格式化**：将问题和选项格式化为明确的查询，要求只返回字母答案（A/B/C/D）
3. **调用 Graph**：调用 Agent-Omni 的 graph 执行多轮推理
4. **答案提取**：从 `decision_result["final_answer"]` 中提取字母答案

### 答案提取

适配器使用正则表达式从 Agent-Omni 的输出中提取答案字母：
- 首先尝试匹配独立的字母（A、B、C 或 D）
- 如果失败，尝试匹配字符串开头的字母
- 如果仍然失败，返回错误标识

### 错误处理

适配器会处理以下错误情况：
- 视频文件不存在
- Agent-Omni graph 执行失败
- 无法提取答案字母
- 其他异常情况

所有错误都会返回以 `error_` 开头的字符串，评估框架会将其视为失败。

## 配置说明

### Agent-Omni 配置

Agent-Omni 的配置文件路径通过以下方式确定（按优先级）：
1. 环境变量 `CONFIG_PATH`
2. `test/config.yaml`
3. `src/config.yaml`

适配器会自动尝试找到配置文件。

### Daily-Omni 配置

在 `eval/daily_omni/test_model_api/test_config.py` 中配置：
- `BASE_VIDEO_DIR`: 视频文件基础目录（默认: `./Videos`）
- `DEFAULT_QA_JSON_PATH`: 默认 QA JSON 文件路径（默认: `qa.json`）
- `DEFAULT_EXECUTION_MODE`: 默认执行模式（`sequential` 或 `parallel`）

## 注意事项

1. **性能**：Agent-Omni 是多轮推理系统，评估时间可能较长。建议：
   - 使用 `--max_items` 参数先进行小规模测试
   - 使用 `sequential` 模式（并行模式可能不稳定）

2. **答案格式**：Agent-Omni 的输出可能包含完整文本，适配器会尝试提取字母答案。如果提取失败，会返回错误。

3. **视频路径**：确保视频文件路径正确，视频文件应按照 Daily-Omni 的格式组织：
   ```
   Videos/
   ├── video_id_1/
   │   └── video_id_1_video.mp4
   ├── video_id_2/
   │   └── video_id_2_video.mp4
   ...
   ```

4. **依赖**：确保 Agent-Omni 的所有依赖都已安装，包括：
   - LangChain
   - LangGraph
   - 模型相关的依赖（根据配置）

## 故障排除

### 问题：无法导入 Agent-Omni 模块

**解决方案**：确保在 Agent-Omni 根目录下运行脚本，或检查 Python 路径设置。

### 问题：配置文件未找到

**解决方案**：设置环境变量 `CONFIG_PATH` 指向正确的配置文件路径。

### 问题：答案提取失败

**解决方案**：
- 检查 Agent-Omni 的输出格式
- 可能需要调整 `extract_answer_letter` 函数中的正则表达式
- 确保查询中明确要求只返回字母答案

### 问题：视频文件未找到

**解决方案**：
- 检查 `BASE_VIDEO_DIR` 配置
- 确保视频文件路径格式正确
- 使用 `--video_dir` 参数指定正确的视频目录

## 扩展

要添加新的 benchmark 支持：

1. 在 `eval/` 下创建新的子目录（如 `eval/new_benchmark/`）
2. 创建评估入口脚本（如 `run_evaluation.py`）
3. 实现适配器函数（参考 `daily_omni/agent_omni_adapter.py`）
4. 创建 bash 启动脚本（如 `run_new_benchmark.sh`）
5. 在评估框架中注册新的模型类型

## 参考

- [Daily-Omni 项目](https://github.com/lliar-liar/Daily-Omni)
- [Agent-Omni 文档](../README.md)
